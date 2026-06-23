#!/usr/bin/env python3
"""
测试 LongBench-v2 数据集 + Qwen3-8B + RetrievalAttention

数据集: LongBench-v2 (zai-org/LongBench-v2 或 THUDM/LongBench-v2)
模型: Qwen3-8B (支持 32K 原生 131K with YaRN)
KV Cache: Full_Flash_Attn, SuCo, PolarANN, PQ
评估: 多选题准确率
"""

import os
# 设置 CUDA 架构以避免编译警告；H20 为 compute capability 9.0。
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0;9.0')
import numpy as np
import torch
import json
import sys
import argparse
from termcolor import colored
import time
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset
import warnings

try:
    import nvtx
except ImportError:  # pragma: no cover - optional profiling dependency
    nvtx = None

try:
    import torch.cuda.nvtx as torch_nvtx
except ImportError:  # pragma: no cover
    torch_nvtx = None

try:
    import torch.cuda.cudart as cudart
except ImportError:  # pragma: no cover
    cudart = None

# 忽略 tokenizer 的长度警告（我们会在后续进行截断处理）
warnings.filterwarnings('ignore', message='Token indices sequence length is longer than the specified maximum')
import re

# decode 两半再拼回文本后 re-tokenize，接缝处 BPE 可能略增；预算内多留一些余量，并见 LENGTH_TOLERANCE。
RETOKENIZE_SAFETY_BUFFER = 32
# CoT：Stage1 prefill 在理论上限 (max_length - cot_max - answer_max) 上再收紧，避免边界/实现上的 off-by-one。
COT_PREFILL_SLACK = 50

# 添加路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# 导入模型
from model_hub.qwen import QwenModel

try:
    from model_hub.llama import LlamaModel
except ImportError:
    LlamaModel = None  # type: ignore


def _resolve_model_class(model_name: str):
    name = model_name.lower()
    if "llama" in name or "deepseek" in name:
        if LlamaModel is None:
            raise ImportError(
                "model_hub.llama 不可用，无法加载 Llama/DeepSeek 模型"
            )
        return LlamaModel, "llama"
    return QwenModel, "qwen"


def expand_batch_for_lockstep(input_ids, attention_masks, batch_size: int):
    """Repeat [1, L] tensors to [B, L] for multi-bs timing (identical sequences per row)."""
    if batch_size <= 1:
        return input_ids, attention_masks
    cur_b = input_ids.size(0)
    if cur_b == batch_size:
        return input_ids, attention_masks
    if cur_b != 1:
        raise ValueError(
            f"--batch_size={batch_size}: expected input row count 1 or {batch_size}, got {cur_b}; "
            f"shape={tuple(input_ids.shape)}"
        )
    return input_ids.repeat(batch_size, 1), attention_masks.repeat(batch_size, 1)


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Test LongBench-v2 with Qwen3-8B")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum new tokens to generate")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Data type")
    parser.add_argument("--model_name", type=str, 
                        default="Qwen/Qwen3-8B",
                        help="Model name or path (default: Qwen3-8B, supports 32K native, 131K with YaRN)")
    
    # LongBench-v2 数据集配置
    parser.add_argument("--dataset_name", type=str,
                        default="zai-org/LongBench-v2",
                        help="LongBench-v2 dataset name (zai-org/LongBench-v2 or THUDM/LongBench-v2)")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.environ.get(
            "LONG_BENCH_V2_PATH",
            "/data/ignore/LongBench-v2/data.json",
        ),
        help="Local LongBench-v2 json path; used when file exists (else HF hub).",
    )
    parser.add_argument("--split_name", type=str, default="train",
                        help="Dataset split name (default: train)")
    parser.add_argument("--num_samples", type=int, default=10, 
                        help="Number of samples to test (default: 10, -1 for all)")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Repeat the same prompt B times (lockstep decode) for throughput / multi-bs timing. "
             "Answers are still decoded from batch row 0 only; use mainly for speed comparison.",
    )
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index for testing")
    parser.add_argument("--sample_offset", type=int, default=None,
                        help="Alias for start_idx when resuming partially processed splits")
    parser.add_argument(
        "--sample_indices",
        type=str,
        default=None,
        help="Comma-separated **1-based** row indices after the same filters as this run "
             "(e.g. with --exclude_easy_short, index 1 = first of the 444 rows). "
             "Selects that subset for evaluation; use for OOM retries. Same filters as the parent run required.",
    )
    parser.add_argument(
        "--union_easy_short_plus_exclude_es_indices",
        type=str,
        default=None,
        help="Single run: all easy+short (59) ∪ rows whose **1-based** index is listed, where indexing follows "
             "the exclude_easy_short ordering (same 444 list as the main benchmark). "
             "Do not combine with --exclude_easy_short, --sample_indices, --domain, --difficulty, or --length.",
    )
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for results (auto-generate if None)")
    
    # 过滤选项
    parser.add_argument("--domain", type=str, default=None,
                        help="Filter by domain (e.g., single-document_qa)")
    parser.add_argument("--difficulty", type=str, default=None,
                        choices=["easy", "hard"],
                        help="Filter by difficulty")
    parser.add_argument("--length", type=str, default=None,
                        choices=["short", "medium", "long"],
                        help="Filter by length category")
    parser.add_argument(
        "--exclude_easy_short",
        action="store_true",
        help="Exclude samples with difficulty=easy AND length=short (all other train rows kept)",
    )
    parser.add_argument("--enable_cot", action="store_true",
                        help="Enable Chain-of-Thought reasoning")
    parser.add_argument("--enable_recall", type=lambda x: x.lower() == 'true', default=None,
                        help="Enable Recall@K computation (true/false, default: use config file)")
    parser.add_argument("--enable_decode_timing", action="store_true",
                        help="Enable decode timing statistics for Full_Flash_Attn")
    parser.add_argument("--max_context_tokens", type=int, default=None,
                        help="Maximum context tokens to process (skip samples exceeding this, default: no limit)")
    parser.add_argument("--force_process", action="store_true",
                        help="Force process samples even if they exceed max_context_tokens (will truncate)")
    parser.add_argument("--no_truncate", action="store_true",
                        help="Skip truncation and only test samples that fit within max context (overrides --force_process)")
    parser.add_argument("--sink_size", type=int, default=None,
                        help="Override sink zone size (default: use config value)")
    parser.add_argument("--local_size", type=int, default=None,
                        help="Override local zone size (default: use config value)")
    parser.add_argument("--dynamic_update_interval", type=int, default=None,
                        help="Override PolarANN dynamic update interval")
    parser.add_argument("--full_attention_threshold", type=int, default=None,
                        help="Override PolarANN full-attention threshold")
    
    # Attention 配置
    parser.add_argument("--attention_type", type=str, default="Full_Flash_Attn",
                        choices=["SuCo", "PQ", "PolarANN", "RetroInfer", "Full_Flash_Attn"],
                        help="Attention type")
    parser.add_argument("--collision_ratio", type=float, default=None,
                        help="Collision search ratio (default: use config file)")
    parser.add_argument("--candidate_ratio", type=float, default=None,
                        help="Candidate filtering ratio (default: use config file)")
    parser.add_argument("--num_subspaces", type=int, default=None,
                        help="Number of subspaces for SuCo (default: use config file)")
    parser.add_argument("--cluster_num_per_subspace", type=int, default=None,
                        help="Number of clusters per subspace for SuCo (default: use config file)")
    parser.add_argument("--static_pattern_end", type=int, default=None,
                        help="Static pattern end (local tokens size) for PolarANN (default: use config file)")
    
    # 采样参数（Qwen3 CoT 模式推荐：temperature=0.6, top_p=0.95, top_k=20）
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature (0.6 for CoT, 0.7 for non-CoT, DO NOT use 0.0 with Qwen3)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling top_p")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Top-k sampling")
    # PolarANN extended knobs (used by run/run_longbench_v2_polarV2.sh)
    parser.add_argument("--polar_cache_module", type=str, default=None,
                        help="Import path for polar_cache module; sets POLAR_CACHE_MODULE when set")
    parser.add_argument("--enable_offload", type=int, default=None, choices=[0, 1],
                        help="Override PolarANN enable_offload (1 = CPU-pinned KV offload)")
    parser.add_argument("--adaptive_topk_threshold", type=float, default=None,
                        help="Override adaptive topk mass threshold for adaptive-K cache modules")
    parser.add_argument("--adaptive_topk_min_k", type=int, default=None)
    parser.add_argument("--adaptive_topk_max_k", type=int, default=None)
    parser.add_argument("--adaptive_topk_enabled", type=int, default=None, choices=[0, 1],
                        help="Enable adaptive top-k path for cache modules that gate on adaptive_topk_enabled "
                             "(e.g. polar_cache_pure_rabitq_adaptiveK). 1=enable, 0=disable.")
    parser.add_argument("--legacy_stage2_concat", action="store_true",
                        help="Legacy CoT stage-2: concat original (long) context with stage-1 CoT and "
                             "re-prefill 130k tokens (old behavior). Default (off) follows LongBench-v2 "
                             "official 0shot_cot_ans.txt: stage-2 drops context, only feeds question + "
                             "choices + CoT + answer-format guard, ~hundreds of tokens.")
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Wrap the prompt with the model's chat template (tokenizer.apply_chat_template) "
                             "before generation. Required for reasoning models like DeepSeek-R1-Distill / "
                             "Qwen3-Thinking so generation starts in the trained <think> mode instead of "
                             "off-distribution raw text. Applied AFTER truncation so special tokens survive. "
                             "Stage-2 (drop-context) is wrapped with a pre-closed <think></think> so it goes "
                             "straight to the formatted answer.")
    parser.add_argument("--chat_template_thinking", type=str, default="auto",
                        choices=["auto", "on", "off"],
                        help="Passed to tokenizer.apply_chat_template as enable_thinking when supported. "
                             "'auto' leaves tokenizer default unchanged; 'on' enables hidden thinking; "
                             "'off' inserts the Qwen3 no-think prefill (<think></think>) when supported. "
                             "Use 'off' for Qwen3-8B LongBench-v2 to avoid hidden-think budget drift while "
                             "still using the model's chat roles.")
    parser.add_argument("--final_topk", type=int, default=None, help="Override PolarANN final_topk in config")
    # CDS pipeline (read by polar_cache_* at __init__ via os.environ)
    parser.add_argument(
        "--polar_cds_dense_ratio",
        type=float,
        default=None,
        help="POLAR_CDS_DENSE_RATIO: fraction of stage-1 chunks classified as dense",
    )
    parser.add_argument(
        "--polar_cds_dense_topk",
        type=int,
        default=None,
        help="POLAR_CDS_DENSE_TOPK: tokens kept per dense chunk in partial_rerank (≤16)",
    )
    parser.add_argument(
        "--polar_cds_sparse_topk",
        type=int,
        default=None,
        help="POLAR_CDS_SPARSE_TOPK: tokens kept per sparse chunk in partial_rerank (≤16)",
    )
    parser.add_argument(
        "--polar_topk_backend",
        type=str,
        default=None,
        choices=["torch", "float"],
        help="POLAR_TOPK_BACKEND: torch.topk vs float_topk_3d for quest/CDS top-k",
    )
    parser.add_argument(
        "--enable_coarse_recall_ablation",
        type=lambda x: x.lower() == 'true' or x == '1',
        default=None,
        help="Enable Stage-1 coarse collision recall ablation (true/1 to enable): "
             "prints R@K and R@C per sampled (layer, head). Tunable via env: "
             "POLAR_RECALL_ABLATION_LAYERS, POLAR_RECALL_ABLATION_HEADS, POLAR_RECALL_ABLATION_K, "
             "POLAR_RECALL_ABLATION_LOG_EVERY.",
    )
    parser.add_argument(
        "--enable_rerank_recall_ablation",
        type=lambda x: x.lower() == 'true' or x == '1',
        default=None,
        help="Enable rerank Recall@100 ablation (true/1 to enable; layer 4 head 4)",
    )
    parser.add_argument(
        "--hamming_candidate_ratio",
        type=float,
        default=None,
        help="Two-stage quantize Stage-1 candidate budget as a fraction of kv_len. "
             "N1 = clamp(ceil(ratio * kv_len), [final_topk, kv_len]). Default 0.05 (5%%). "
             "Only consumed by cache_hub.polar_cache_pure_rabitq_2stage_quantize.",
    )
    parser.add_argument("--rope_scaling_factor", type=float, default=None,
                        help="YaRN rope_scaling factor (e.g. 4.0). If set, also set --rope_scaling_orig_max.")
    parser.add_argument("--rope_scaling_orig_max", type=int, default=None,
                        help="YaRN rope_scaling original_max_position_embeddings (e.g. 32768)")
    parser.add_argument("--cot_max_new_tokens", type=int, default=1024,
                        help="CoT stage-1 max new tokens when --enable_cot")
    parser.add_argument("--answer_max_new_tokens", type=int, default=512,
                        help="CoT stage-2 max new tokens when --enable_cot")
    
    args = parser.parse_args()
    if getattr(args, "sample_offset", None) is not None:
        args.start_idx = args.sample_offset
    return args


def load_longbench_v2_data(
    dataset_name,
    split_name,
    num_samples=-1,
    start_idx=0,
    domain=None,
    difficulty=None,
    length=None,
    exclude_easy_short=False,
    dataset_path=None,
):
    """加载 LongBench-v2 数据集"""
    print(colored(f"\n加载 LongBench-v2 数据集", 'yellow'))
    print(colored(f"  数据集: {dataset_name}", 'cyan'))
    print(colored(f"  分割: {split_name}", 'cyan'))
    
    try:
        if dataset_path and os.path.exists(dataset_path):
            print(colored(f"  本地 JSON: {dataset_path}", 'cyan'))
            dataset = load_dataset("json", data_files=dataset_path, split=split_name)
        else:
            dataset = load_dataset(dataset_name, split=split_name)
        print(colored(f"  原始样本数: {len(dataset)}", 'cyan'))
        
        # 应用过滤
        if domain or difficulty or length:
            filtered_data = []
            for item in dataset:
                if domain and item.get('domain') != domain:
                    continue
                if difficulty and item.get('difficulty') != difficulty:
                    continue
                if length and item.get('length') != length:
                    continue
                filtered_data.append(item)
            dataset = dataset.from_list(filtered_data)
            print(colored(f"  过滤后样本数: {len(dataset)}", 'cyan'))

        if exclude_easy_short:
            items = list(dataset)
            filtered_data = [
                it
                for it in items
                if not (
                    it.get("difficulty") == "easy" and it.get("length") == "short"
                )
            ]
            dataset = dataset.from_list(filtered_data)
            print(
                colored(
                    f"  排除 easy+short 后样本数: {len(dataset)}", "cyan"
                )
            )

        # 限制样本数量
        if num_samples > 0:
            end_idx = min(start_idx + num_samples, len(dataset))
            dataset = dataset.select(range(start_idx, end_idx))
        elif start_idx > 0:
            dataset = dataset.select(range(start_idx, len(dataset)))
        
        if len(dataset) == 0:
            print(colored("  过滤或采样后没有可用样本，终止评估。", 'yellow'))
            sys.exit(0)
        
        print(colored(f"  加载了 {len(dataset)} 个样本", 'green'))
        
        # 显示样本示例
        if len(dataset) > 0:
            sample = dataset[0]
            print(colored(f"\n样本示例:", 'yellow'))
            print(f"  ID: {sample.get('_id', 'N/A')}")
            print(f"  Domain: {sample.get('domain', 'N/A')}")
            print(f"  Sub-domain: {sample.get('sub_domain', 'N/A')}")
            print(f"  Difficulty: {sample.get('difficulty', 'N/A')}")
            print(f"  Length: {sample.get('length', 'N/A')}")
            question = sample.get('question', '')
            print(f"  问题预览: {question[:100]}..." if len(question) > 100 else f"  问题: {question}")
            print(f"  正确答案: {sample.get('answer', 'N/A')}")
            context_len = len(sample.get('context', ''))
            print(f"  上下文长度: {context_len} 字符")
        
        return dataset
        
    except Exception as e:
        print(colored(f"加载数据集失败: {e}", 'red'))
        sys.exit(1)


def load_longbench_v2_union_easy_short_plus_exclude_es_indices(
    dataset_name,
    split_name,
    exclude_es_indices_1based_csv,
):
    """easy+short 全量 ∪ 「排除 easy+short」列表中按 1-based 指定的行（与 444 主跑序号一致）。"""
    print(colored(f"\n加载 LongBench-v2 数据集（union: easy+short ∪ 指定 exclude-ES 行）", 'yellow'))
    print(colored(f"  数据集: {dataset_name}", 'cyan'))
    print(colored(f"  分割: {split_name}", 'cyan'))

    parts = [p.strip() for p in exclude_es_indices_1based_csv.split(',') if p.strip()]
    try:
        one_based = sorted({int(p) for p in parts})
    except ValueError as e:
        print(colored(f"--union_easy_short_plus_exclude_es_indices 解析失败: {e}", "red"))
        sys.exit(1)

    try:
        dataset = load_dataset(dataset_name, split=split_name)
        rows = [dict(r) for r in dataset]
        print(colored(f"  原始样本数: {len(rows)}", 'cyan'))

        easy_short = [
            r for r in rows
            if r.get('difficulty') == 'easy' and r.get('length') == 'short'
        ]
        exclude_es = [
            r for r in rows
            if not (r.get('difficulty') == 'easy' and r.get('length') == 'short')
        ]
        print(colored(f"  easy+short: {len(easy_short)}", 'cyan'))
        print(colored(f"  排除 easy+short 列表长: {len(exclude_es)}", 'cyan'))

        bad = [k for k in one_based if k < 1 or k > len(exclude_es)]
        if bad:
            print(colored(f"  exclude-ES 1-based 越界 [1,{len(exclude_es)}]: {bad}", 'red'))
            sys.exit(1)

        extra = [exclude_es[k - 1] for k in one_based]

        seen = set()
        combined = []
        for r in easy_short + extra:
            sid = r.get('_id')
            if sid in seen:
                continue
            seen.add(sid)
            combined.append(r)

        out = Dataset.from_list(combined)
        print(colored(f"  union 去重后: {len(out)} (= {len(easy_short)} easy_short + {len(extra)} 指定行; 无重叠则 69)", 'green'))
        print(colored(f"  指定的 exclude-ES 行（1-based）: {one_based}", 'cyan'))

        if len(out) > 0:
            sample = out[0]
            print(colored(f"\n样本示例 (union 首条):", 'yellow'))
            print(f"  ID: {sample.get('_id', 'N/A')}")
            print(f"  Domain: {sample.get('domain', 'N/A')}, Difficulty: {sample.get('difficulty', 'N/A')}, Length: {sample.get('length', 'N/A')}")

        return out

    except Exception as e:
        print(colored(f"加载数据集失败: {e}", 'red'))
        sys.exit(1)


def build_longbench_v2_prompt(sample, enable_cot=False):
    """
    构建 LongBench-v2 的 prompt（多选题格式）
    使用官方 prompt 格式：
    - 标准: prompts/0shot.txt
    - CoT: prompts/0shot_cot.txt
    
    来源: https://github.com/THUDM/LongBench/tree/main/prompts
    """
    context = sample.get('context', '')
    question = sample.get('question', '')
    
    choice_a = sample.get('choice_A', '')
    choice_b = sample.get('choice_B', '')
    choice_c = sample.get('choice_C', '')
    choice_d = sample.get('choice_D', '')
    
    if enable_cot:
        # 官方 CoT prompt (0shot_cot.txt)
        prompt = f"""Please read the following text and answer the questions below.

<text>
{context}
</text>

What is the correct answer to this question: {question}
Choices:
(A) {choice_a}
(B) {choice_b}
(C) {choice_c}
(D) {choice_d}

Let's think step by step:"""
    else:
        # 官方标准 prompt (0shot.txt)
        prompt = f"""Please read the following text and answer the question below.

<text>
{context}
</text>

What is the correct answer to this question: {question}
Choices:
(A) {choice_a}
(B) {choice_b}
(C) {choice_c}
(D) {choice_d}

Format your response as follows: "The correct answer is (insert answer here)"."""
    
    return prompt


def build_longbench_v2_stage2_prompt(sample, cot_response: str) -> str:
    """构建 LongBench-v2 CoT Stage-2 prompt（与官方 0shot_cot_ans.txt 对齐）。

    关键点：Stage-2 不再带原始 context（130k+ tokens），而是仅由 question + choices +
    Stage-1 已生成的 CoT + 强制输出格式构成。这样：
      1. Stage-2 prefill 只有几百 token，不会再次踩中 context-driven 退化吸引子；
      2. 强制 ``"The correct answer is (X)"`` 格式，extract_answer 几乎 100% 命中；
      3. Stage-1 的 CoT 推理被显式喂给模型，模型只需做最终选择。

    源自 https://github.com/THUDM/LongBench/blob/main/prompts/0shot_cot_ans.txt
    """
    question = sample.get('question', '')
    choice_a = sample.get('choice_A', '')
    choice_b = sample.get('choice_B', '')
    choice_c = sample.get('choice_C', '')
    choice_d = sample.get('choice_D', '')

    return (
        "Please read the following text and answer the questions below.\n\n"
        "The text is too long and omitted here.\n\n"
        f"What is the correct answer to this question: {question}\n"
        "Choices:\n"
        f"(A) {choice_a}\n"
        f"(B) {choice_b}\n"
        f"(C) {choice_c}\n"
        f"(D) {choice_d}\n\n"
        f"Let's think step by step: {cot_response}\n\n"
        "Based on the above, what is the single, most likely answer choice? "
        'Format your response as follows: "The correct answer is (insert answer here)".'
    )


def detect_repetition(response):
    """
    检测响应是否存在重复模式（模型陷入循环生成）
    
    Returns:
        dict: {
            'is_repetitive': bool,  # 是否检测到重复
            'max_sentence_repetition': int,  # 最大句子重复次数
            'answer_pattern_count': int,  # "answer is (X)" 模式出现次数
            'uniqueness_ratio': float  # 独特字符比例
        }
    """
    if not response or len(response) < 100:
        return {
            'is_repetitive': False,
            'max_sentence_repetition': 0,
            'answer_pattern_count': 0,
            'uniqueness_ratio': 1.0
        }
    
    # 方法1: 检测完整句子的重复（以句号分割）
    sentences = response.split('.')
    sentence_counts = {}
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 20:  # 只检查长度>20的句子
            sentence_counts[sent] = sentence_counts.get(sent, 0) + 1
    
    max_repetition = max(sentence_counts.values()) if sentence_counts else 0
    
    # 方法2: 检测 "answer is (X)" 模式出现的频率
    answer_pattern_count = len(re.findall(
        r'(?:correct\s+)?answer\s+is\s+\([ABCD]\)', 
        response, 
        re.IGNORECASE
    ))
    
    # 方法3: 检查响应长度和独特内容的比例
    unique_chars = len(set(response))
    total_chars = len(response)
    uniqueness_ratio = unique_chars / total_chars if total_chars > 0 else 0
    
    # 判断是否重复：任一条件满足即认为是重复
    # - 有句子重复 >= 3 次
    # - "answer is (X)" 出现 >= 5 次
    # - 独特性比例 < 5%（太多重复字符） - 调整为更合理的阈值
    is_repetitive = (
        max_repetition >= 3 or 
        answer_pattern_count >= 5 or 
        (uniqueness_ratio < 0.05 and total_chars > 500)  # 只对长文本且独特性很低的情况判定为重复
    )
    
    return {
        'is_repetitive': is_repetitive,
        'max_sentence_repetition': max_repetition,
        'answer_pattern_count': answer_pattern_count,
        'uniqueness_ratio': uniqueness_ratio
    }


def extract_answer_from_response(response, enable_cot=False):
    """从模型响应中提取答案（A, B, C, 或 D）"""
    if not response:
        return None
    
    # 转换为大写方便匹配
    response_upper = response.upper()

    # 策略0: \boxed{X} 是 Qwen3 / DeepSeek 等 RL-trained 模型的最终答案标准格式,
    # 优先匹配并取最后一次出现 (循环输出 "**Final Answer** \boxed{D} ..." 时取末尾即可).
    boxed_matches = list(re.finditer(r'\\BOXED\s*\{\s*\(?\s*([ABCD])\s*\)?\s*\}', response_upper))
    if boxed_matches:
        return boxed_matches[-1].group(1)

    # 策略1: 优先匹配完整的答案声明 "answer is (X)" 或 "correct answer is (X)"
    # 这样可以避免提取到被截断的不完整括号
    answer_patterns = [
        r'(?:correct\s+)?(?:final\s+)?answer\s+is\s+\(([ABCD])\)',  # "answer is (X)" / "final answer is (X)"
        r'(?:correct\s+)?(?:final\s+)?answer\s*[:：]\s*\(([ABCD])\)',  # "answer: (X)"
        r'(?:correct\s+)?(?:final\s+)?answer\s+is\s+([ABCD])\b',  # "answer is X" (无括号)
        r'(?:correct\s+)?(?:final\s+)?answer\s*[:：]\s*([ABCD])\b',  # "answer: X" (无括号)
    ]
    for pattern in answer_patterns:
        matches = list(re.finditer(pattern, response_upper, re.IGNORECASE))
        if matches:
            # 策略：优先使用第一个"明确的答案声明"
            # 但如果第一个匹配距离开头很近（<200字符），可能是在复述问题
            # 此时寻找第一个在200字符之后的匹配
            first_match = matches[0]
            if first_match.start() < 200 and len(matches) > 1:
                # 寻找第一个在200字符之后的匹配
                for match in matches:
                    if match.start() >= 200:
                        return match.group(1)
                # 如果所有匹配都在200字符之前，使用第一个
                return first_match.group(1)
            return first_match.group(1)
    
    # 策略2: 如果没有完整的答案声明，再匹配孤立的括号 (A), (B), (C), (D)
    # 但要确保括号是完整的（后面有闭合括号）
    # 从后往前搜索，找最后一个完整的括号答案
    matches = list(re.finditer(r'\(([ABCD])\)', response_upper))
    if matches:
        # 检查最后一个匹配是否看起来像被截断的（例如在文本最后10个字符内）
        last_match = matches[-1]
        if len(response_upper) - last_match.end() < 10:
            # 可能被截断，尝试使用倒数第二个匹配
            if len(matches) > 1:
                return matches[-2].group(1)
        return last_match.group(1)
    
    # 策略3: 如果启用了 CoT，从最后几行中提取
    if enable_cot:
        lines = response.strip().split('\n')
        # 从后往前找，最多看最后10行
        for line in reversed(lines[-10:]):
            line = line.strip().upper()
            if not line:
                continue
            # 匹配括号内的字母
            match = re.search(r'\(([ABCD])\)', line)
            if match:
                return match.group(1)
            # 匹配独立的字母（但要避免匹配到 "a single" 这种情况）
            match = re.search(r'\b([ABCD])\b(?!\s+single)', line)
            if match:
                return match.group(1)
    
    # 策略4: 从整个响应的后半部分寻找独立的 A/B/C/D
    # 只在后半部分搜索，避免匹配到问题描述中的字母
    half_point = len(response_upper) // 2
    response_second_half = response_upper[half_point:]
    
    # 从后往前找独立的字母（避免 "a single" 等）
    matches = list(re.finditer(r'\b([ABCD])\b(?!\s+single)', response_second_half))
    if matches:
        return matches[-1].group(1)
    
    # 策略5: 最后的兜底，在整个响应中从后往前找
    matches = list(re.finditer(r'\b([ABCD])\b', response_upper))
    if matches:
        # 返回最后一个匹配，但要过滤掉 "a single" 这种情况
        for match in reversed(matches):
            # 检查这个字母后面是否跟着 "single"
            start = match.start()
            end = match.end()
            if end < len(response_upper) - 7:  # 有足够空间检查 " single"
                if response_upper[end:end+7] == ' SINGLE':
                    continue
            return match.group(1)
    
    # 如果都没匹配到，返回 None
    return None


def truncate_ids_decode_retokenize(tokenizer, input_ids_1d, max_len_budget, device):
    """超长序列：前半+后半 decode 再 re-tokenize；目标长度不超过 max_len_budget（硬保证）。"""
    row = input_ids_1d
    if row.dim() > 1:
        row = row[0]
    n = int(row.shape[0])
    if n <= max_len_budget:
        ids = row.unsqueeze(0).to(device)
        return ids, torch.ones_like(ids, device=device, dtype=torch.long)
    safe = max(max_len_budget - RETOKENIZE_SAFETY_BUFFER, 2)
    half = max(safe // 2, 1)
    cpu_row = row.detach().cpu()
    prefix_text = tokenizer.decode(cpu_row[:half], skip_special_tokens=True)
    suffix_text = tokenizer.decode(cpu_row[-half:], skip_special_tokens=True)
    truncated_text = prefix_text + suffix_text
    new = tokenizer([truncated_text], return_tensors="pt", padding=True)
    ids = new.input_ids.to(device)
    mask = new.attention_mask.to(device)
    if ids.shape[1] > max_len_budget:
        ids = ids[:, :max_len_budget]
        mask = mask[:, :max_len_budget]
    return ids, mask


def apply_chat_template_text(tokenizer, messages, thinking_mode="auto"):
    """Apply chat template while tolerating tokenizers without enable_thinking."""
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if thinking_mode == "on":
        kwargs["enable_thinking"] = True
    elif thinking_mode == "off":
        kwargs["enable_thinking"] = False

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Llama/older tokenizers do not accept enable_thinking; fall back to
        # plain chat template rather than failing the benchmark.
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def generate_config(args, context_len):
    """根据 attention 类型和上下文长度生成配置"""
    # 首先从配置文件加载完整配置
    import json
    import os
    
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
    MODEL_NAME = args.model_name.split("/")[-1] + '.json'
    CONFIG_FILE = os.path.join(CONFIG_DIR, MODEL_NAME)
    
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        # 如果配置文件不存在，返回空配置（将使用默认值）
        config = {}
    
    # 根据 attention 类型，覆盖动态参数
    if args.attention_type == "Full_Flash_Attn":
        # 为 Full_Flash_Attn 添加 decode timing 配置
        if "Full_Flash_Attn" not in config:
            config["Full_Flash_Attn"] = {}
        config["Full_Flash_Attn"]["enable_decode_timing"] = args.enable_decode_timing
        
    elif args.attention_type == "PolarANN" and "PolarANN" in config:
        # 只在用户显式提供参数时才覆盖配置文件
        if args.collision_ratio is not None:
            config["PolarANN"]["collision_ratio"] = args.collision_ratio
        if args.candidate_ratio is not None:
            config["PolarANN"]["candidate_ratio"] = args.candidate_ratio
        if args.sink_size is not None:
            config["PolarANN"]["sink_size"] = args.sink_size
            config["PolarANN"]["static_pattern_start"] = args.sink_size
        # static_pattern_start/end 的优先级：history flag -> legacy flag -> default
        if args.static_pattern_end is not None:
            # 保留历史参数名称，兼容旧脚本
            config["PolarANN"]["static_pattern_end"] = args.static_pattern_end
        if args.local_size is not None:
            config["PolarANN"]["local_size"] = args.local_size
            config["PolarANN"]["static_pattern_end"] = args.local_size
        if args.dynamic_update_interval is not None:
            config["PolarANN"]["dynamic_update_interval"] = args.dynamic_update_interval
        if args.full_attention_threshold is not None:
            config["PolarANN"]["full_attention_threshold"] = args.full_attention_threshold
        
        # 动态计算 nprobe（使用配置中的 candidate_ratio，与 polar_cache.py 逻辑一致）
        candidate_ratio = config["PolarANN"].get("candidate_ratio", 0.2)
        MIN_TOPK = 30
        calculated_topk = int(context_len * candidate_ratio)
        config["PolarANN"]["nprobe"] = max(MIN_TOPK, min(calculated_topk, context_len))
        
        # 如果命令行明确指定了 enable_recall，则覆盖配置文件
        if args.enable_recall is not None:
            config["PolarANN"]["enable_recall"] = args.enable_recall

        if getattr(args, "polar_cache_module", None):
            config["PolarANN"]["cache_module"] = args.polar_cache_module
        if getattr(args, "enable_offload", None) is not None:
            config["PolarANN"]["enable_offload"] = bool(args.enable_offload)
        if getattr(args, "adaptive_topk_threshold", None) is not None:
            config["PolarANN"]["adaptive_topk_threshold"] = args.adaptive_topk_threshold
        if getattr(args, "adaptive_topk_min_k", None) is not None:
            config["PolarANN"]["adaptive_topk_min_k"] = args.adaptive_topk_min_k
        if getattr(args, "adaptive_topk_max_k", None) is not None:
            config["PolarANN"]["adaptive_topk_max_k"] = args.adaptive_topk_max_k
        if getattr(args, "adaptive_topk_enabled", None) is not None:
            config["PolarANN"]["adaptive_topk_enabled"] = bool(args.adaptive_topk_enabled)
        if getattr(args, "final_topk", None) is not None:
            config["PolarANN"]["final_topk"] = args.final_topk
        if getattr(args, "enable_coarse_recall_ablation", None) is not None:
            config["PolarANN"]["enable_coarse_recall_ablation"] = bool(args.enable_coarse_recall_ablation)
        if getattr(args, "enable_rerank_recall_ablation", None) is not None:
            config["PolarANN"]["enable_rerank_recall_ablation"] = bool(args.enable_rerank_recall_ablation)
        if getattr(args, "hamming_candidate_ratio", None) is not None:
            config["PolarANN"]["hamming_candidate_ratio"] = float(args.hamming_candidate_ratio)
        
    elif args.attention_type == "SuCo" and "SuCo" in config:
        # 只在用户显式提供参数时才覆盖配置文件
        if args.collision_ratio is not None:
            config["SuCo"]["collision_ratio"] = args.collision_ratio
        if args.candidate_ratio is not None:
            config["SuCo"]["candidate_ratio"] = args.candidate_ratio
        
        # 动态计算 nprobe（使用配置中的 candidate_ratio）
        candidate_ratio = config["SuCo"].get("candidate_ratio", 0.2)
        MIN_TOPK = 30
        calculated_topk = int(context_len * candidate_ratio)
        config["SuCo"]["nprobe"] = max(MIN_TOPK, min(calculated_topk, context_len))
        
        # 如果命令行明确指定了参数（非 None），则覆盖配置文件；否则使用配置文件的值
        if args.num_subspaces is not None:
            config["SuCo"]["num_subspaces"] = args.num_subspaces
        if args.cluster_num_per_subspace is not None:
            config["SuCo"]["cluster_num_per_subspace"] = args.cluster_num_per_subspace
        if args.enable_recall is not None:
            config["SuCo"]["enable_recall"] = args.enable_recall
        
    elif args.attention_type == "PQ" and "PQ" in config:
        # 只在用户显式提供参数时才覆盖配置文件
        if args.collision_ratio is not None:
            config["PQ"]["collision_ratio"] = args.collision_ratio
        if args.candidate_ratio is not None:
            config["PQ"]["candidate_ratio"] = args.candidate_ratio
        
        # 动态计算 topk（使用配置中的 candidate_ratio）
        candidate_ratio = config["PQ"].get("candidate_ratio", 0.2)
        MIN_TOPK = 30
        calculated_topk = int(context_len * candidate_ratio)
        config["PQ"]["topk"] = max(MIN_TOPK, min(calculated_topk, context_len))
        
        # 如果命令行明确指定了参数，则覆盖配置文件
        if args.num_subspaces is not None:
            config["PQ"]["num_subspaces"] = args.num_subspaces
        if args.cluster_num_per_subspace is not None:
            config["PQ"]["cluster_num_per_subspace"] = args.cluster_num_per_subspace
        if args.enable_recall is not None:
            config["PQ"]["enable_recall"] = args.enable_recall
    
    return config


def apply_polar_cds_env_from_args(args):
    """Mirror CDS env knobs into os.environ before polar_cache __init__.

    polar_cache_pure_rabitq_quest_density* reads POLAR_CDS_* at instantiation;
  shell export alone works, but CLI makes the path explicit and overridable.
    """
    if getattr(args, "polar_cds_dense_ratio", None) is not None:
        os.environ["POLAR_CDS_DENSE_RATIO"] = str(args.polar_cds_dense_ratio)
    if getattr(args, "polar_cds_dense_topk", None) is not None:
        os.environ["POLAR_CDS_DENSE_TOPK"] = str(args.polar_cds_dense_topk)
    if getattr(args, "polar_cds_sparse_topk", None) is not None:
        os.environ["POLAR_CDS_SPARSE_TOPK"] = str(args.polar_cds_sparse_topk)
    if getattr(args, "polar_topk_backend", None) is not None:
        os.environ["POLAR_TOPK_BACKEND"] = str(args.polar_topk_backend)


def load_model(args):
    """加载 Qwen / Llama(DeepSeek-R1) 模型"""
    model_cls, model_backend = _resolve_model_class(args.model_name)
    print(colored(f"\n加载模型: {args.model_name}", 'yellow'))
    print(colored(f"  Backend: {model_backend}", 'cyan'))
    print(colored(f"  Attention Type: {args.attention_type}", 'cyan'))
    print(colored(f"  Device: {args.device}", 'cyan'))
    print(colored(f"  Data Type: {args.dtype}", 'cyan'))
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    max_length = 131072

    rope_scaling = None
    if getattr(args, 'rope_scaling_factor', None) is not None:
        # 显式传 --rope_scaling_factor 时构造 YaRN override（对 qwen 与 llama/deepseek 一视同仁）。
        # 不传则保持 None → 走模型 config 内置 rope（Llama-3.1/DeepSeek-R1 自带 llama3 rope，
        # 原生 128K，无需也不应叠加 YaRN —— 此分支仅供消融对比）。
        rope_scaling = {
            "rope_type": "yarn",
            "factor": float(args.rope_scaling_factor),
            "original_max_position_embeddings": int(
                args.rope_scaling_orig_max if args.rope_scaling_orig_max is not None else 32768
            ),
        }
        print(colored(f"  rope_scaling (override, backend={model_backend}): {rope_scaling}", 'cyan'))
        if model_backend == "llama":
            print(colored(
                "  [NOTE] Llama/DeepSeek 原生已是 llama3 rope(128K)；此处 YaRN override 仅用于消融实验。",
                "yellow",
            ))

    llm = model_cls(
        model_name=args.model_name,
        max_length=max_length,
        dtype=dtype,
        device_map=args.device,
        rope_scaling=rope_scaling,
    )
    print(colored("模型加载完成!", 'green'))
    return llm


def main():
    args = parse_args()
    if getattr(args, "polar_cache_module", None):
        os.environ["POLAR_CACHE_MODULE"] = args.polar_cache_module
    apply_polar_cds_env_from_args(args)
    set_seed(42)
    
    # 生成输出文件名
    if args.output_file is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"longbench_v2_{args.attention_type}_{timestamp}.jsonl"
    
    output_dir = os.path.dirname(args.output_file) or "../results/longbench_v2"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(args.output_file))
    _out_abs = os.path.abspath(output_dir)
    _jsonl_abs = os.path.abspath(output_path)

    print(colored("=" * 80, 'yellow'))
    print(colored("LongBench-v2 评估", 'yellow', attrs=['bold']))
    print(colored(f"模型: {args.model_name}", 'cyan'))
    print(colored(f"数据集: {args.dataset_name}", 'cyan'))
    print(colored(f"分割: {args.split_name}", 'cyan'))
    print(colored(f"Attention: {args.attention_type}", 'cyan'))
    if args.batch_size > 1:
        print(colored(f"batch_size: {args.batch_size} (same prompt repeated, lockstep timing)", 'cyan'))
    print(colored(f"输出目录 (绝对路径): {_out_abs}", 'cyan'))
    print(colored(f"JSONL (绝对路径): {_jsonl_abs}", 'cyan'))
    print(colored("若由 run_longbench_v2.sh 启动，tee 的 .log 与 JSONL 同目录", 'cyan'))
    
    union_csv = getattr(args, "union_easy_short_plus_exclude_es_indices", None)
    if union_csv:
        conflicts = []
        if args.sample_indices:
            conflicts.append("--sample_indices")
        if args.exclude_easy_short:
            conflicts.append("--exclude_easy_short")
        if args.domain:
            conflicts.append("--domain")
        if args.difficulty:
            conflicts.append("--difficulty")
        if args.length:
            conflicts.append("--length")
        if conflicts:
            print(
                colored(
                    f"与 --union_easy_short_plus_exclude_es_indices 不能同时使用: {', '.join(conflicts)}",
                    "red",
                )
            )
            sys.exit(1)
        if args.num_samples != -1:
            print(colored("union 模式请传 --num_samples -1（评测 union 全量子集）", "red"))
            sys.exit(1)
        if args.start_idx != 0:
            print(colored("union 模式不支持 start_idx/sample_offset != 0", "red"))
            sys.exit(1)

    # 加载数据
    if union_csv:
        data = load_longbench_v2_union_easy_short_plus_exclude_es_indices(
            args.dataset_name,
            args.split_name,
            union_csv,
        )
    else:
        data = load_longbench_v2_data(
            args.dataset_name,
            args.split_name,
            args.num_samples,
            args.start_idx,
            args.domain,
            args.difficulty,
            args.length,
            exclude_easy_short=args.exclude_easy_short,
            dataset_path=getattr(args, "dataset_path", None),
        )

    lbv2_row_rank_1based = None  # 全量跑中与 tqdm 序号一致；子集跑时映射回原始 1-based 下标
    n_filtered_full = len(data)  # 过滤后全量长度（--sample_indices 子集跑时作分母 如 444/444）
    if args.sample_indices:
        parts = [p.strip() for p in args.sample_indices.split(",") if p.strip()]
        try:
            one_based_sorted = sorted({int(p) for p in parts})
        except ValueError as e:
            print(colored(f"--sample_indices 解析失败: {e}", "red"))
            sys.exit(1)
        n_all = n_filtered_full
        out_of = [k for k in one_based_sorted if k < 1 or k > n_all]
        if out_of:
            print(colored(f"--sample_indices 超出当前数据范围 [1, {n_all}]: {out_of}", "red"))
            sys.exit(1)
        zero_based = [k - 1 for k in one_based_sorted]
        data = data.select(zero_based)
        lbv2_row_rank_1based = one_based_sorted
        print(
            colored(
                f"\n按 --sample_indices 子集评测：共 {len(data)} 条，对应全量 filtered 序（1-based）={lbv2_row_rank_1based}",
                "yellow",
                attrs=["bold"],
            )
        )
    
    # 加载模型
    llm = load_model(args)
    
    # 加载 tokenizer 并设置最大长度以支持超长文本
    from model_hub.tokenizer_utils import load_tokenizer
    tokenizer = load_tokenizer(args.model_name, trust_remote_code=True)
    # 与 QwenModel.max_length 对齐（本脚本默认 131072 = 128×1024，LongBench CoT 窗口）
    tokenizer.model_max_length = llm.max_length
    # CoT：同一 hard 上限须同时喂满 Stage1+Stage2。
    #   Stage1: prefill + CoT ≤ max_length
    #   Stage2: (prefill+CoT) + 答案 ≤ max_length  ⇒  Stage1 prefill ≤ max - CoT_max - Answer_max
    out_token_budget = (
        args.cot_max_new_tokens + args.answer_max_new_tokens
    ) if args.enable_cot else args.max_new_tokens
    if args.enable_cot:
        _cap = llm.max_length - out_token_budget - COT_PREFILL_SLACK
        print(
            colored(
                f"CoT 长度约束: model_max_length={llm.max_length} (~128K), "
                f"预留两阶段生成≤{out_token_budget} (stage1≤{args.cot_max_new_tokens} + stage2≤{args.answer_max_new_tokens}), "
                f"故 Stage1 prefill 上限={_cap} (= max_length - {out_token_budget} - slack{COT_PREFILL_SLACK})",
                "cyan",
            )
        )
    
    # 评估
    results = []
    correct = 0
    total = 0
    total_time = 0
    
    # 🆕 断点续跑逻辑
    processed_ids = set()
    mode = 'w'
    if os.path.exists(output_path):
        print(colored(f"发现输出文件 {output_path}，检查断点...", 'yellow'))
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        res = json.loads(line)
                        if 'sample_id' in res:
                            processed_ids.add(res['sample_id'])
                    except: pass
            if processed_ids:
                mode = 'a'
                print(colored(f"已处理 {len(processed_ids)} 个样本，将跳过这些样本 (模式: append)。", 'green'))
        except Exception as e:
            print(colored(f"读取输出文件出错: {e}", 'red'))

    output_fp = open(output_path, mode, encoding='utf-8')
    profiler_active = False
    
    try:
        # 使用 CUDA Profiler API 控制 nsys 采样范围（避免 NVTX 事件顺序问题）
        if cudart is not None:
            cudart.cudaProfilerStart()
            profiler_active = True

        print(colored(f"\n开始评估 {len(data)} 个样本...", 'yellow'))
        print(colored("=" * 80, 'yellow'))
        
        for idx, sample in enumerate(tqdm(data, desc="Testing", ncols=100)):
            # 🆕 跳过已处理样本
            if sample.get('_id') in processed_ids:
                continue

            display_idx_1based = (
                lbv2_row_rank_1based[idx]
                if lbv2_row_rank_1based is not None
                else idx + 1
            )
            try:
                # 在处理每个样本前清理 GPU 内存（特别是第一个样本）
                if idx == 0 and torch.cuda.is_available():
                    torch.cuda.synchronize()  # 先等 GPU 完成
                    torch.cuda.empty_cache()  # 再清理缓存

                # 构建 prompt
                prompt = build_longbench_v2_prompt(sample, enable_cot=args.enable_cot)
                correct_answer = sample.get('answer', '').strip().upper()
                
                # Tokenize
                inputs = tokenizer([prompt], return_tensors='pt', padding=True)
                input_ids = inputs.input_ids
                attention_masks = inputs.attention_mask
                input_ids = input_ids.to(llm.layers[0].device)
                attention_masks = attention_masks.to(llm.layers[0].device)
                context_len = input_ids.shape[1]
                
                # 检查是否超过最大长度限制
                # CoT：首段 prompt 最长 = max_length - cot_max - answer_max - COT_PREFILL_SLACK（两阶段生成额度 + 安全边际）
                # 非 CoT：max_input = max_length - max_new_tokens
                model_max_length = llm.max_length
                max_input_tokens = model_max_length - out_token_budget
                if args.enable_cot:
                    max_input_tokens -= COT_PREFILL_SLACK
                
                # 如果用户设置了 max_context_tokens，需要验证其合理性
                if args.max_context_tokens:
                    user_max_input = args.max_context_tokens - out_token_budget
                    if args.enable_cot:
                        user_max_input -= COT_PREFILL_SLACK
                    if user_max_input < out_token_budget or user_max_input <= 0:
                        # 用户设置的 max_context_tokens 太小或导致负数，发出警告并使用模型默认值
                        if idx == 0:  # 只在第一个样本时警告一次
                            print(colored(f"\n  警告: max_context_tokens({args.max_context_tokens}) 设置不合理", 'yellow'))
                            print(colored(f"    max_context_tokens - output_budget = {user_max_input} (需要 > 0)", 'yellow'))
                            print(colored(
                                f"    将使用模型侧 prefill 上限: {max_input_tokens} tokens "
                                f"(model_max_length={model_max_length}, 预留生成={out_token_budget}"
                                f"{', slack=' + str(COT_PREFILL_SLACK) if args.enable_cot else ''})",
                                'green',
                            ))
                    else:
                        # 使用用户设置和模型限制中较小的值
                        max_input_tokens = min(max_input_tokens, user_max_input)
                
                # 确保 max_input_tokens 是正数且合理
                if max_input_tokens <= 0:
                    print(colored(f"\n 错误: 计算出的 max_input_tokens = {max_input_tokens} <= 0", 'red'))
                    print(colored(f"    model_max_length={model_max_length}, output_budget={out_token_budget}", 'red'))
                    raise ValueError(f"Invalid max_input_tokens: {max_input_tokens}")
                
                if context_len > max_input_tokens:
                    # 🆕 如果设置了 --no_truncate，直接跳过超长样本
                    if args.no_truncate:
                        print(colored(f"\n样本 {display_idx_1based}/{n_filtered_full} ⏭️  跳过（--no_truncate: 不截断超长样本）", 'yellow'))
                        print(f"  上下文长度: {context_len:,} tokens (超过限制: {max_input_tokens:,})")
                        result = {
                            'sample_id': sample.get('_id', idx),
                            'domain': sample.get('domain', ''),
                            'sub_domain': sample.get('sub_domain', ''),
                            'difficulty': sample.get('difficulty', ''),
                            'length': sample.get('length', ''),
                            'question': sample.get('question', ''),
                            'correct_answer': correct_answer,
                            'predicted_answer': None,
                            'response': None,
                            'is_correct': False,
                            'context_length': context_len,
                            'num_output_tokens': 0,
                            'prefill_time': 0,
                            'decode_time': 0,
                            'generation_time': 0,
                            'tpot': 0,
                            'skipped': True,
                            'skip_reason': 'no_truncate_enabled'
                        }
                        results.append(result)
                        output_fp.write(json.dumps(result, ensure_ascii=False) + '\n')
                        output_fp.flush()
                        continue
                    
                    if args.force_process:
                        # 强制处理：使用 decode + re-tokenize 策略（类似 KVCache-Factory）
                        # 这样可以保证位置编码连续，避免 RoPE 错误
                        print(colored(f"\n样本 {display_idx_1based}/{n_filtered_full} ⚠ 警告：超过最大长度，使用截断（decode + re-tokenize）", 'yellow'))
                        print(f"  原始长度: {context_len:,} tokens")
                        
                        # 🔧 留 RETOKENIZE_SAFETY_BUFFER，防止 re-tokenize 后长度膨胀超过预算
                        safe_max_input = max_input_tokens - RETOKENIZE_SAFETY_BUFFER
                        print(
                            f"  截断策略: 保留前 {safe_max_input//2} + 后 {safe_max_input//2} tokens "
                            f"(预留 {out_token_budget} tokens + buffer={RETOKENIZE_SAFETY_BUFFER})"
                        )
                        
                        # 方法：decode 前半部分和后半部分文本，然后重新 tokenize
                        # 这样位置编码就是连续的 [0, 1, 2, ..., max_input_tokens-1]
                        half = safe_max_input // 2
                        
                        # Decode 前半部分和后半部分
                        prefix_text = tokenizer.decode(input_ids[0, :half], skip_special_tokens=True)
                        suffix_text = tokenizer.decode(input_ids[0, -half:], skip_special_tokens=True)
                        
                        # 拼接文本
                        truncated_text = prefix_text + suffix_text
                        
                        # 重新 tokenize（这样位置编码就是连续的）
                        new_inputs = tokenizer([truncated_text], return_tensors='pt', padding=True)
                        input_ids = new_inputs.input_ids.to(llm.layers[0].device)
                        attention_masks = new_inputs.attention_mask.to(llm.layers[0].device)
                        if input_ids.shape[1] > max_input_tokens:
                            input_ids = input_ids[:, :max_input_tokens]
                            attention_masks = attention_masks[:, :max_input_tokens]
                        
                        # 更新 context_len 为实际截断后的长度
                        context_len = input_ids.shape[1]
                        
                        print(f"  截断后长度: {context_len:,} tokens (位置编码连续: 0 到 {context_len-1})")
                        
                        # 验证截断结果（允许小范围偏差，避免频繁警告）
                        length_diff = abs(context_len - max_input_tokens)
                        LENGTH_TOLERANCE = RETOKENIZE_SAFETY_BUFFER  # 与截断 buffer 同量级
                        if length_diff > LENGTH_TOLERANCE:
                            print(colored(
                                f" 截断长度偏差较大: 实际 {context_len:,} vs 预期 {max_input_tokens:,} "
                                f"(差 {length_diff} tokens)", 'yellow'))
                        elif length_diff > 0:
                            print(f"  截断长度与预期略有差异 (±{length_diff} tokens)，属 re-tokenize 正常现象")
                        
                        # 最终验证：确保不会超过模型限制
                        if context_len + out_token_budget > model_max_length:
                            print(colored(f" 截断后仍超过限制: {context_len} + {out_token_budget} > {model_max_length}", 'red'))
                            raise ValueError(f"Truncation failed: {context_len} + {out_token_budget} > {model_max_length}")
                    else:
                        # 跳过该样本
                        print(colored(f"\n样本 {display_idx_1based}/{n_filtered_full}  跳过（超过最大长度限制）", 'yellow'))
                        print(f"  上下文长度: {context_len:,} tokens (输入限制: {max_input_tokens:,}, 需预留 {out_token_budget} tokens 用于输出)")
                        result = {
                            'sample_id': sample.get('_id', idx),
                            'domain': sample.get('domain', ''),
                            'sub_domain': sample.get('sub_domain', ''),
                            'difficulty': sample.get('difficulty', ''),
                            'length': sample.get('length', ''),
                            'question': sample.get('question', ''),
                            'correct_answer': correct_answer,
                            'predicted_answer': None,
                            'response': None,
                            'is_correct': False,
                            'context_length': context_len,
                            'num_output_tokens': 0,
                            'prefill_time': 0,
                            'decode_time': 0,
                            'generation_time': 0,
                            'tpot': 0,
                            'skipped': True,
                            'skip_reason': f'exceeds_max_length_{args.max_context_tokens}'
                        }
                        results.append(result)
                        continue

                # Chat template（R1 / Qwen thinking）：必须在截断之后再套，
                # 否则 hard/long 样本截断时 decode(skip_special_tokens) 会把
                # <｜User｜>/<｜Assistant｜>/<think> 等特殊 token 一起剥掉。
                # 此时 input_ids 已是最终（截断后）的纯 user 内容。
                if args.use_chat_template:
                    user_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    templated_text = apply_chat_template_text(
                        tokenizer,
                        [{"role": "user", "content": user_text}],
                        thinking_mode=args.chat_template_thinking,
                    )
                    ct_inputs = tokenizer(
                        [templated_text], return_tensors='pt', add_special_tokens=False
                    )
                    input_ids = ct_inputs.input_ids.to(llm.layers[0].device)
                    attention_masks = ct_inputs.attention_mask.to(llm.layers[0].device)
                    context_len = input_ids.shape[1]
                    if idx == 0:
                        print(colored(
                            f"  [chat_template] stage-1 已套模板, context_len={context_len:,}"
                            f" (template tail: {repr(templated_text[-32:])})",
                            "cyan",
                        ))

                # 在处理新样本前清理 GPU 内存，确保有足够空间
                if torch.cuda.is_available():
                    import gc
                    gc.collect()           # 清理 Python 对象
                    torch.cuda.synchronize()  # 等 GPU 完成所有操作
                    torch.cuda.empty_cache()  # 释放 PyTorch 缓存
                
                # 多 batch：同一 prompt 复制 B 行（锁步调测）
                if args.batch_size > 1:
                    input_ids, attention_masks = expand_batch_for_lockstep(
                        input_ids, attention_masks, args.batch_size
                    )
                    if idx == 0:
                        print(
                            colored(
                                f"  [batch_size={args.batch_size}] lockstep identical prompts (timing / throughput)",
                                "cyan",
                            )
                        )

                # 动态生成配置（基于实际的 input_len）
                config = generate_config(args, context_len)

                # 📊 Reset recall history for new sample (if using retrieval methods)
                if args.attention_type in ["PolarANN", "SuCo", "PQ"] and hasattr(llm.layers[0], 'kv_cache'):
                    if hasattr(llm.layers[0].kv_cache, 'reset_recall_history'):
                        llm.layers[0].kv_cache.reset_recall_history()
                
                # Generate（CoT：两阶段独立 KV；非 CoT：单次 generate）
                with torch.no_grad():
                    if args.enable_cot:
                        print(colored(
                            f"\n  [CoT Stage 1] 生成推理过程 (max_new_tokens={args.cot_max_new_tokens}) ...",
                            'cyan', attrs=['bold']))
                        out1, ts1 = llm.generate(
                            attention_type=args.attention_type,
                            inputs_ids=input_ids,
                            attention_masks=attention_masks,
                            max_new_length=args.cot_max_new_tokens,
                            attn_config=config,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                        )
                        gen_s1 = out1[0] if isinstance(out1[0], list) else out1[0].tolist()
                        if args.legacy_stage2_concat:
                            # Legacy: 拼回原始 context + stage1 CoT，重新 prefill 130k 长上下文。
                            # 已知问题：stage2 会沿用 stage1 的 context-driven 退化吸引子继续生成
                            # 垃圾循环（如 ", the village" / "-2022-2022"），predicted_answer=None
                            # 概率极高。仅为复现历史行为保留。
                            gen_tail = torch.tensor(
                                gen_s1,
                                device=input_ids.device,
                                dtype=input_ids.dtype,
                            ).unsqueeze(0)
                            if input_ids.shape[0] > 1:
                                gen_tail = gen_tail.repeat(input_ids.shape[0], 1)
                            s2_input_ids = torch.cat([input_ids, gen_tail], dim=1)
                            s2_attention_masks = torch.ones_like(
                                s2_input_ids, dtype=torch.long, device=input_ids.device
                            )
                            context_len_s2 = int(s2_input_ids.shape[1])
                            model_max_length = llm.max_length
                            max_s2_input = model_max_length - args.answer_max_new_tokens
                            if context_len_s2 > max_s2_input:
                                print(colored(
                                    f"  [CoT Stage 2] ⚠ 输入 {context_len_s2} tokens > {max_s2_input} "
                                    f"(model_max={model_max_length}, 预留 answer={args.answer_max_new_tokens})，"
                                    f"decode+re-tokenize 截断",
                                    "yellow",
                                ))
                                s2_input_ids, s2_attention_masks = truncate_ids_decode_retokenize(
                                    tokenizer,
                                    s2_input_ids[0],
                                    max_s2_input,
                                    llm.layers[0].device,
                                )
                                context_len_s2 = int(s2_input_ids.shape[1])
                        else:
                            # 官方 (THUDM/LongBench, 0shot_cot_ans.txt)：stage-2 不带原始 context
                            cot_text = tokenizer.decode(gen_s1, skip_special_tokens=True)
                            stage2_prompt = build_longbench_v2_stage2_prompt(sample, cot_text)
                            if args.use_chat_template:
                                # 套模板后，若模型把我们放进了 <think>（R1/Qwen-thinking），
                                # 立即补一个空的 </think> 关闭推理，让 stage-2 直接给格式化答案，
                                # 不再用 128 token 预算重新思考（否则常常 think 没闭合 → pred=None）。
                                templated_s2 = apply_chat_template_text(
                                    tokenizer,
                                    [{"role": "user", "content": stage2_prompt}],
                                    thinking_mode=args.chat_template_thinking,
                                )
                                # 仅当模板尾部留了一个“未闭合”的 <think>（R1 / Qwen enable_thinking=on）
                                # 才补 </think>。Qwen enable_thinking=off 的模板尾部已是
                                # <think>\n\n</think>\n\n（已闭合），不能再补，否则出现双重 </think>。
                                tail = templated_s2[-32:]
                                if "<think>" in tail and "</think>" not in tail:
                                    templated_s2 = templated_s2 + "</think>\n\n"
                                s2_inputs = tokenizer(
                                    [templated_s2], return_tensors='pt', padding=False,
                                    add_special_tokens=False,
                                )
                            else:
                                s2_inputs = tokenizer([stage2_prompt], return_tensors='pt', padding=False)
                            s2_input_ids = s2_inputs.input_ids.to(input_ids.device)
                            s2_attention_masks = s2_inputs.attention_mask.to(input_ids.device)
                            context_len_s2 = int(s2_input_ids.shape[1])
                            model_max_length = llm.max_length
                            max_s2_input = model_max_length - args.answer_max_new_tokens
                            if context_len_s2 > max_s2_input:
                                # Stage-2 prompt 仅几百到几千 token（取决于 CoT 长度），通常不会触发；
                                # 但若 CoT 异常长仍走 decode+retokenize 截断到上限。
                                print(colored(
                                    f"  [CoT Stage 2] ⚠ 输入 {context_len_s2} tokens > {max_s2_input}，"
                                    f"截断（应当极少发生）",
                                    "yellow",
                                ))
                                s2_input_ids, s2_attention_masks = truncate_ids_decode_retokenize(
                                    tokenizer,
                                    s2_input_ids[0],
                                    max_s2_input,
                                    llm.layers[0].device,
                                )
                                context_len_s2 = int(s2_input_ids.shape[1])
                        if args.batch_size > 1:
                            s2_input_ids, s2_attention_masks = expand_batch_for_lockstep(
                                s2_input_ids, s2_attention_masks, args.batch_size
                            )
                        config_s2 = generate_config(args, context_len_s2)
                        if args.attention_type in ["PolarANN", "SuCo", "PQ"] and hasattr(llm.layers[0], 'kv_cache'):
                            if hasattr(llm.layers[0].kv_cache, 'reset_recall_history'):
                                llm.layers[0].kv_cache.reset_recall_history()

                        print(colored(
                            f"  [CoT Stage 2] 生成最终答案 (max_new_tokens={args.answer_max_new_tokens}) ...",
                            'cyan', attrs=['bold']))
                        out2, ts2 = llm.generate(
                            attention_type=args.attention_type,
                            inputs_ids=s2_input_ids,
                            attention_masks=s2_attention_masks,
                            max_new_length=args.answer_max_new_tokens,
                            attn_config=config_s2,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                        )
                        gen_s2 = out2[0] if isinstance(out2[0], list) else out2[0].tolist()
                        response = tokenizer.decode(gen_s2, skip_special_tokens=True)
                        prefill_time = ts1['prefill_time'] + ts2['prefill_time']
                        decode_time = ts1['decode_time'] + ts2['decode_time']
                        gen_time = prefill_time + decode_time
                        timing_stats = {
                            'prefill_time': prefill_time,
                            'decode_time': decode_time,
                            'total_time': gen_time,
                        }
                        generated_ids = gen_s2
                    else:
                        output_ids, timing_stats = llm.generate(
                            attention_type=args.attention_type,
                            inputs_ids=input_ids,
                            attention_masks=attention_masks,
                            max_new_length=args.max_new_tokens,
                            attn_config=config,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                        )
                        prefill_time = timing_stats['prefill_time']
                        decode_time = timing_stats['decode_time']
                        gen_time = timing_stats['total_time']
                        if isinstance(output_ids[0], list):
                            generated_ids = output_ids[0]
                        else:
                            generated_ids = output_ids[0].tolist()
                        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

                total_time += gen_time
                
                # 检测重复生成
                repetition_info = detect_repetition(response)
                
                # 提取答案
                predicted_answer = extract_answer_from_response(response, enable_cot=args.enable_cot)
                is_correct = (predicted_answer == correct_answer) if predicted_answer else False
                
                if is_correct:
                    correct += 1
                total += 1
                
                if args.enable_cot:
                    num_output_tokens = len(gen_s1) + len(gen_s2)
                else:
                    num_output_tokens = len(generated_ids)
                tpot = (decode_time / num_output_tokens) if num_output_tokens > 0 else 0
                
                # 保存结果
                result = {
                    'sample_id': sample.get('_id', idx),
                    'domain': sample.get('domain', ''),
                    'sub_domain': sample.get('sub_domain', ''),
                    'difficulty': sample.get('difficulty', ''),
                    'length': sample.get('length', ''),
                    'question': sample.get('question', ''),
                    'correct_answer': correct_answer,
                    'predicted_answer': predicted_answer,
                    'response': response,
                    'is_correct': is_correct,
                    'context_length': context_len,
                    'num_output_tokens': num_output_tokens,
                    'prefill_time': prefill_time,  # Prefill 时间 (秒)
                    'decode_time': decode_time,    # Decode 时间 (秒)
                    'generation_time': gen_time,   # 总时间 (秒)
                    'tpot': tpot,  # Time Per Output Token (秒，只统计 decode)
                    'batch_size': args.batch_size,
                    'is_repetitive': repetition_info['is_repetitive'],  # 是否检测到重复
                    'repetition_stats': repetition_info  # 详细的重复统计
                }
                results.append(result)
                
                # 打印结果
                status = colored('✓', 'green') if is_correct else colored('✗', 'red')
                repetition_warning = colored(' [⚠ 重复生成]', 'yellow') if repetition_info['is_repetitive'] else ''
                print(f"\n样本 {display_idx_1based}/{n_filtered_full} {status}{repetition_warning}")
                print(f"  Domain: {sample.get('domain', 'N/A')}, Difficulty: {sample.get('difficulty', 'N/A')}, Length: {sample.get('length', 'N/A')}")
                print(f"  上下文长度: {context_len} tokens")
                print(f"  预测: {predicted_answer}, 正确: {correct_answer}")
                print(f"  生成时间: Prefill {prefill_time:.2f}s + Decode {decode_time:.2f}s = 总计 {gen_time:.2f}s")
                print(f"  生成 {num_output_tokens} tokens, TPOT (decode only): {tpot*1000:.2f}ms")
                if repetition_info['is_repetitive']:
                    print(colored(f"  ⚠ 检测到重复生成: 答案模式重复 {repetition_info['answer_pattern_count']} 次, "
                                 f"最大句子重复 {repetition_info['max_sentence_repetition']} 次", 'yellow'))
                
                # 📊 Print and plot recall curve for this sample (if PolarANN, SuCo, or PQ)
                if args.attention_type in ["PolarANN", "SuCo", "PQ"] and hasattr(llm.layers[0], 'kv_cache'):
                    kv_cache = llm.layers[0].kv_cache
                    # Print recall summary
                    if hasattr(kv_cache, 'print_recall_summary'):
                        kv_cache.print_recall_summary()
                    # Plot recall curve
                    if hasattr(kv_cache, 'plot_recall_curve'):
                        # Create recall plots directory
                        recall_plots_dir = os.path.join(os.path.dirname(args.output_file), "recall_plots")
                        os.makedirs(recall_plots_dir, exist_ok=True)
                        kv_cache.plot_recall_curve(save_dir=recall_plots_dir)
                
                # 强制清理 GPU 内存，减少碎片化
                # 在每个样本处理完后清理，避免内存累积
                if torch.cuda.is_available():
                    import gc
                    gc.collect()           # 清理 Python 对象
                    torch.cuda.synchronize()  # 等 GPU 完成
                    torch.cuda.empty_cache()  # 清理 CUDA 缓存
                
                output_fp.write(json.dumps(result, ensure_ascii=False) + '\n')
                output_fp.flush()
            
            except Exception as e:
                print(colored(f"\n错误处理样本 {display_idx_1based}: {e}", 'red'))
                import traceback
                traceback.print_exc()
                # 即使出错也要清理内存
                if torch.cuda.is_available():
                    import gc
                    gc.collect()           # 清理 Python 对象
                    torch.cuda.synchronize()  # 等 GPU 完成
                    torch.cuda.empty_cache()  # 清理 CUDA 缓存
                continue
    finally:
        # 停止 CUDA Profiler
        if profiler_active and cudart is not None:
            cudart.cudaProfilerStop()
        output_fp.close()
    
    # 打印汇总
    accuracy = (correct / total * 100) if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0
    
    print(colored("评估结果", 'yellow', attrs=['bold']))
    print(f"总样本数: {total}")
    print(f"正确数: {correct}")
    print(f"准确率: {accuracy:.2f}% ({correct}/{total})")
    print(f"平均生成时间: {avg_time:.2f}s")
    print(f"总时间: {total_time:.2f}s")
    print(colored("=" * 80, 'yellow'))
    print(f"\n结果已保存到: {output_path}")
    
    # 按 domain 统计（过滤掉 skipped 样本）
    if results:
        domain_stats = {}
        for r in results:
            # 跳过被跳过的样本
            if r.get('skipped', False):
                continue
            
            domain = r.get('domain', 'unknown')
            if domain not in domain_stats:
                domain_stats[domain] = {'correct': 0, 'total': 0}
            domain_stats[domain]['total'] += 1
            if r.get('is_correct'):
                domain_stats[domain]['correct'] += 1
        
        print(colored("\n按 Domain 统计:", 'cyan'))
        for domain, stats in sorted(domain_stats.items()):
            acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {domain}: {acc:.2f}% ({stats['correct']}/{stats['total']})")
    
    # 按 difficulty × length 统计（平均样本长度 + 平均 TPOT）
    if results:
        combo_stats = {}
        for r in results:
            # 跳过被跳过的样本
            if r.get('skipped', False):
                continue
            
            difficulty = r.get('difficulty', 'unknown')
            length = r.get('length', 'unknown')
            combo_key = f"{difficulty}_{length}"
            
            if combo_key not in combo_stats:
                combo_stats[combo_key] = {
                    'correct': 0, 
                    'total': 0,
                    'context_lengths': [],
                    'tpots': [],
                    'prefill_times': [],
                    'decode_times': [],
                    'gen_times': []
                }
            
            combo_stats[combo_key]['total'] += 1
            if r.get('is_correct'):
                combo_stats[combo_key]['correct'] += 1
            
            # 收集统计数据
            if r.get('context_length'):
                combo_stats[combo_key]['context_lengths'].append(r['context_length'])
            if r.get('tpot'):
                combo_stats[combo_key]['tpots'].append(r['tpot'])
            if r.get('prefill_time'):
                combo_stats[combo_key]['prefill_times'].append(r['prefill_time'])
            if r.get('decode_time'):
                combo_stats[combo_key]['decode_times'].append(r['decode_time'])
            if r.get('generation_time'):
                combo_stats[combo_key]['gen_times'].append(r['generation_time'])
        
        if combo_stats:
            print(colored("\n按 Difficulty X Length 统计:", 'cyan'))
            print(colored("=" * 80, 'cyan'))
            print(f"{'组合':<14}{'准确率':>10}{'样本数':>8}"
                  f"{'平均长度':>14}{'Prefill':>12}{'Decode':>12}{'TPOT(ms)':>12}")
            
            for combo_key in sorted(combo_stats.keys()):
                stats = combo_stats[combo_key]
                acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                
                avg_len = np.mean(stats['context_lengths']) if stats['context_lengths'] else 0
                avg_prefill = np.mean(stats['prefill_times']) if stats['prefill_times'] else 0
                avg_decode = np.mean(stats['decode_times']) if stats['decode_times'] else 0
                avg_tpot_ms = np.mean(stats['tpots']) * 1000 if stats['tpots'] else 0  # 转换为 ms
                
                print(f"{combo_key:<16}"
                      f"{acc:>10.2f}%({stats['correct']}/{stats['total']})"
                      f"{stats['total']:>8}"
                      f"{avg_len:>14,.0f}"
                      f"{avg_prefill:>12.2f}s"
                      f"{avg_decode:>12.2f}s"
                      f"{avg_tpot_ms:>12.2f}")
            
    
    # 输出 PolarANN timing 统计
    print(colored("\n" + "=" * 80, 'cyan'))
    if args.attention_type == 'PolarANN':
        print(colored("PolarANN 性能统计 (Timing Statistics)", 'cyan', attrs=['bold']))
        
        # LLM 层统计
        if hasattr(llm, 'timing_stats'):
            print(colored("\n[LLM Layer]", 'yellow'))
            print(f"  decode_update_kv_cache:  {llm.timing_stats.get('decode_update_kv_cache', 0):.3f}s")
            print(f"  decode_attention:        {llm.timing_stats.get('decode_attention', 0):.3f}s")
        
        # KV Cache 层统计
        if hasattr(llm.layers[0], 'kv_cache') and hasattr(llm.layers[0].kv_cache, 'timing_stats'):
            kv_stats = llm.layers[0].kv_cache.timing_stats
            print(colored("\n[PolarANN KV Cache]", 'yellow'))
            print(f"  collision_based_topk_batch:           {kv_stats.get('collision_based_topk_batch', 0):.3f}s")
            print(f"    ├─ get_q_score:                      {kv_stats.get('get_q_score', 0):.3f}s")
            print(f"    ├─ get_topk_clusters:                {kv_stats.get('get_topk_clusters', 0):.3f}s")
            print(f"    ├─ update_cache_cnt:                 {kv_stats.get('update_cache_cnt', 0):.3f}s")
            print(f"    └─ get_candidate_cache_rerank_slow:  {kv_stats.get('get_candidate_cache_rerank_slow', 0):.3f}s")
            print(f"  flash_attn_with_kvcache_compat:       {kv_stats.get('flash_attn_with_kvcache_compat', 0):.3f}s")
        
        print(colored("=" * 80, 'cyan'))
    
    # 输出 decode timing 统计（如果启用）
    if args.enable_decode_timing and args.attention_type == 'Full_Flash_Attn':
        if hasattr(llm.layers[0], 'kv_cache') and hasattr(llm.layers[0].kv_cache, 'decode_times') and llm.layers[0].kv_cache.decode_times:
            decode_times = llm.layers[0].kv_cache.decode_times
            
            print(colored("\n" + "=" * 80, 'cyan'))
            print(colored("Decode 阶段计时统计 (Full Flash Attention)", 'cyan', attrs=['bold']))
            print(f"总 decode 步数: {len(decode_times)}")
            print(f"平均每步时间: {np.mean(decode_times):.4f} ms")
            print(f"中位数时间: {np.median(decode_times):.4f} ms")
            print(f"最小时间: {np.min(decode_times):.4f} ms")
            print(f"最大时间: {np.max(decode_times):.4f} ms")
            print(f"标准差: {np.std(decode_times):.4f} ms")
            print(f"总 decode 时间: {np.sum(decode_times):.2f} ms ({np.sum(decode_times)/1000:.2f} s)")


if __name__ == "__main__":
    main()



# ./run_longbench_v2.sh [num_samples] [attention_type] [collision_ratio] [candidate_ratio] \
#                       [domain] [difficulty] [length] [device] [num_subspaces] [cluster_num] \
#                       [enable_recall] [static_pattern_end] [max_new_tokens] [sink_size] [local_size]
# ./run_longbench_v2.sh 1 PolarANN 0.3 0.2 "" easy long cuda:0 "" "" true 512 68 512 512
