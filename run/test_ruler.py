#!/usr/bin/env python3
"""
RULER benchmark test with PolarANN / Full_Flash_Attn on Qwen / Llama.

RULER data is pre-formatted with chat template in the 'input' field.
We tokenize directly and feed to the model.
"""

import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

import json
import sys
import argparse
import torch
import re
import string
import warnings
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


def _resolve_model_class(model_name: str):
    """Resolve backend model class by model name/path."""
    model_name_lower = model_name.lower()
    if "llama" in model_name_lower:
        from model_hub.llama import LlamaModel
        return LlamaModel, "llama"

    # Import QwenModel directly — model_hub/__init__.py may fail if LLM.py symlink is missing.
    # Some environments expose model_hub.qwen, while others expose model_hub.qwen_pai.
    import importlib
    try:
        _qwen_mod = importlib.import_module("model_hub.qwen_pai")
    except ModuleNotFoundError:
        _qwen_mod = importlib.import_module("model_hub.qwen")
    return _qwen_mod.QwenModel, "qwen"


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    p = argparse.ArgumentParser(description="RULER benchmark with PolarANN")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--data_dir", type=str, required=True,
                    help="RULER data root, e.g. .../ruler/data/qwen/131072 or .../ruler/data/llama-3/131072")
    p.add_argument("--tasks", type=str,
                    default="niah_single_1,niah_single_2,niah_single_3,"
                            "niah_multikey_1,niah_multikey_2,niah_multikey_3,"
                            "niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_1,qa_2",
                    help="Comma-separated task names (full ShadowKV RULER 13-task set)")
    p.add_argument("--max_gen", type=int, default=None,
                   help="Override decode length for every task. Default (None) "
                        "uses ShadowKV per-task lengths: niah=128, vt=30, cwe=120, "
                        "fwe=50, qa=32.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    p.add_argument("--out_dir", type=str, default=None,
                    help="Output directory (auto-generated if None)")
    p.add_argument("--no_think", action="store_true",
                    help="Inject /no_think into system prompt")

    # PolarANN / attention config
    p.add_argument("--attention_type", type=str, default="PolarANN",
                    choices=["PolarANN", "Full_Flash_Attn"])
    p.add_argument("--sink_size", type=int, default=64)
    p.add_argument("--local_size", type=int, default=256)
    p.add_argument("--final_topk", type=int, default=2048)
    p.add_argument("--collision_ratio", type=float, default=0.3)
    p.add_argument("--candidate_ratio", type=float, default=0.2)
    p.add_argument("--enable_offload", type=int, default=0, choices=[0, 1])
    p.add_argument("--polar_cache_module", type=str, default=None)

    p.add_argument("--rope_factor", type=str, default="auto",
                    help="YaRN rope_scaling factor, 'auto', or 'none'. "
                         "'auto' only scales when the dataset length exceeds the "
                         "model's native context (so Llama-3.1 131072 stays native). "
                         "A float (e.g. 4.0) forces explicit YaRN scaling.")
    p.add_argument("--rope_orig_max", type=int, default=None,
                    help="YaRN original_max_position_embeddings override. Default: "
                         "32768 for Qwen when scaling, else the model's native max.")
    p.add_argument("--data_idx", type=int, default=None, help="Single sample index")
    p.add_argument("--data_idx_to", type=int, default=None, help="End index (exclusive)")
    p.add_argument("--resume", action="store_true",
                    help="Resume from existing task jsonl files in out_dir")
    return p.parse_args()


def inject_no_think(text):
    tag = "<|im_start|>system\n"
    idx = text.find(tag)
    if idx >= 0:
        pos = idx + len(tag)
        return text[:pos] + "/no_think\n" + text[pos:]
    return text


# ----- ShadowKV-style per-task scoring (ported from data/metrics.py) -----
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(str(s).lower())))


def postprocess_pred(predict_str: str):
    predict_str = str(predict_str).strip()
    for tag in ('<|eot_id|>', '</s>', '</s', '</', '<|im_end|>', '<|endoftext|>'):
        predict_str = predict_str.replace(tag, '')
    return re.sub(r'[\x00-\x1f]', '\n', predict_str).strip()


def needle_score(prediction, ground_truth):
    prediction = normalize_answer(postprocess_pred(prediction))
    ground_truth = normalize_answer(ground_truth)
    min_length = len(ground_truth)
    score = float(prediction[:min_length] == ground_truth[:min_length])
    return max(float(ground_truth in prediction.split()), score)


def multi_number(prediction: str, ground_truth: list) -> float:
    prediction_list = re.findall(r'\d+', normalize_answer(prediction))
    hits = [str(item) for item in ground_truth if str(item) in prediction_list]
    return len(hits) / len(ground_truth) if ground_truth else 0.0


def multi_words(prediction: str, ground_truth: list) -> float:
    prediction = str(prediction).lower()
    ground_truth = [str(gt).lower() for gt in ground_truth]
    hits = [item for item in ground_truth if item in prediction]
    return len(hits) / len(ground_truth) if ground_truth else 0.0


def string_match_part(preds, refs):
    preds = postprocess_pred(preds)
    if isinstance(refs, str):
        refs = [refs]
    refs = [str(r) for r in refs]
    score_ref_in_pred = max([1.0 if r.lower() in preds.lower() else 0.0 for r in refs])
    score_pred_in_ref = max([1.0 if preds.lower() in r.lower() else 0.0 for r in refs])
    return round(max(score_ref_in_pred, score_pred_in_ref), 2)


def get_metric_fn(task_name):
    if 'multiquery' in task_name or 'multivalue' in task_name:
        return multi_number
    if 'niah' in task_name:
        return needle_score
    if 'vt' in task_name or 'cwe' in task_name or 'fwe' in task_name:
        return multi_words
    if 'qa' in task_name:
        return string_match_part
    raise ValueError(f"No metric for task: {task_name}")


def get_gen_len(task_name):
    """ShadowKV RULER per-task decode lengths (data/dataset.py get_gen_len)."""
    if 'niah' in task_name:
        return 128
    if 'vt' in task_name:
        return 30
    if 'cwe' in task_name:
        return 120
    if 'fwe' in task_name:
        return 50
    if 'qa' in task_name:
        return 32
    raise ValueError(f"No gen_len for task: {task_name}")


def coerce_ground_truth(answers, metric_fn):
    if answers is None:
        answers = []
    if not isinstance(answers, list):
        answers = [answers]
    if metric_fn in (multi_number, multi_words):
        return [str(a) for a in answers]
    if len(answers) == 1:
        return str(answers[0])
    return [str(a) for a in answers]


def infer_target_context_len(data_dir: str, fallback: int) -> int:
    name = os.path.basename(data_dir.rstrip("/"))
    try:
        return int(name)
    except ValueError:
        return fallback


def resolve_rope_scaling(args, cfg, model_backend, target_context_len: int):
    """Decide YaRN rope_scaling adaptively.

    Returns (rope_scaling_dict_or_None, effective_max_pos, base_max_pos).
    """
    base_max_pos = int(getattr(cfg, "max_position_embeddings", target_context_len))
    factor_arg = str(args.rope_factor).strip().lower()
    no_rope = factor_arg in {"", "none", "native", "false", "0"}

    if no_rope:
        print(f"[RoPE] native/extrapolate: config_max_pos={base_max_pos}, "
              f"target={target_context_len}")
        return None, max(base_max_pos, target_context_len), base_max_pos

    if factor_arg == "auto":
        if base_max_pos >= target_context_len:
            print(f"[RoPE] native: config_max_pos={base_max_pos} >= "
                  f"target={target_context_len} (no YaRN)")
            return None, base_max_pos, base_max_pos
        original_max = args.rope_orig_max
        if original_max is None:
            original_max = 32768 if model_backend == "qwen" else base_max_pos
        factor = target_context_len / float(original_max)
        mode = "auto"
    else:
        factor = float(factor_arg)
        original_max = args.rope_orig_max or base_max_pos
        mode = "explicit"

    rope_scaling = {
        "rope_type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": int(original_max),
    }
    effective_max_pos = max(target_context_len, int(round(original_max * factor)))
    print(f"[YaRN:{mode}] factor={factor}, orig_max_pos={original_max}, "
          f"config_max_pos={base_max_pos}, target={target_context_len}, "
          f"effective={effective_max_pos}")
    return rope_scaling, effective_max_pos, base_max_pos


def _guess_config_candidates(model_name: str):
    base = model_name.split("/")[-1]
    candidates = [base]
    model_name_lower = model_name.lower()

    # Local snapshot paths for Llama-3.1-8B typically end with a hash.
    # Prefer the canonical config filename when detected in the path.
    if "llama-3.1-8b-instruct" in model_name_lower:
        candidates.insert(0, "Llama-3.1-8B-Instruct")
    return candidates


def load_config(args, context_len):
    """Build attn_config dict compatible with generate()."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    config = {}
    for cand in _guess_config_candidates(args.model_name):
        config_path = os.path.join(project_root, "config", cand + ".json")
        try:
            with open(config_path) as f:
                config = json.load(f)
            break
        except FileNotFoundError:
            continue

    if args.attention_type == "PolarANN":
        cfg = config.get("PolarANN")
        if cfg is None:
            # Fallback defaults for models without a dedicated PolarANN block.
            cfg = {
                "core": 22,
                "cache_unit_size": 8,
                "cache_cluster_num": 450,
            }
            config["PolarANN"] = cfg

        cfg["sink_size"] = args.sink_size
        cfg["local_size"] = args.local_size
        cfg["static_pattern_start"] = args.sink_size
        cfg["static_pattern_end"] = args.local_size
        cfg["collision_ratio"] = args.collision_ratio
        cfg["candidate_ratio"] = args.candidate_ratio
        cfg["final_topk"] = args.final_topk
        cfg["enable_offload"] = (args.enable_offload == 1)

        # POLAR_ALIGN_OFFLOAD (default on) requires dynamic_update_interval to be
        # a multiple of the hierarchical large chunk. Models without a PolarANN
        # block in config/ (e.g. Llama-3.1) would otherwise fall back to llama.py's
        # default dui=128 and trip the assertion when large_chunk=256.
        if cfg.get("dynamic_update_interval") is None:
            large_chunk = int(os.environ.get("POLAR_QUEST_LARGE_CHUNK", "256"))
            base_dui = max(int(args.local_size), large_chunk)
            cfg["dynamic_update_interval"] = (
                (base_dui + large_chunk - 1) // large_chunk
            ) * large_chunk

        candidate_ratio = cfg.get("candidate_ratio", 0.2)
        calculated = int(context_len * candidate_ratio)
        cfg["nprobe"] = max(30, min(calculated, context_len))

    return config


def main():
    args = parse_args()
    set_seed(42)
    model_cls, model_backend = _resolve_model_class(args.model_name)

    if args.polar_cache_module:
        os.environ["POLAR_CACHE_MODULE"] = args.polar_cache_module

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # Determine output dir
    if args.out_dir is None:
        data_len = os.path.basename(args.data_dir.rstrip("/"))
        args.out_dir = os.path.join(PROJECT_ROOT, "results", "ruler",
                                     f"{args.model_name.split('/')[-1]}-{args.attention_type}-topk{args.final_topk}-{data_len}")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    print(f"\n{'='*80}")
    print("RULER Benchmark")
    print(f"  Model: {args.model_name}")
    print(f"  Backend: {model_backend}")
    print(f"  Attention: {args.attention_type}")
    print(f"  final_topk: {args.final_topk}")
    print(f"  sink: {args.sink_size}, local: {args.local_size}")
    print(f"  Data: {args.data_dir}")
    print(f"  Tasks: {task_list}")
    print(f"  no_think: {args.no_think}")
    print(f"{'='*80}\n")

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config_max_pos = int(getattr(cfg, "max_position_embeddings", 131072))
    target_context_len = infer_target_context_len(args.data_dir, config_max_pos)
    max_task_gen = max(
        args.max_gen if args.max_gen is not None else get_gen_len(t)
        for t in task_list
    )
    rope_scaling, effective_max_pos, base_max_pos = resolve_rope_scaling(
        args, cfg, model_backend, target_context_len
    )

    max_length = effective_max_pos + max_task_gen + 128
    print(f"[Context] max_length={max_length} "
          f"(effective_max_pos={effective_max_pos}, max_task_gen={max_task_gen})")
    if model_backend == "qwen":
        llm = model_cls(
            model_name=args.model_name,
            max_length=max_length,
            dtype=dtype,
            device_map=args.device,
            rope_scaling=rope_scaling,
        )
    else:
        llm = model_cls(
            model_name=args.model_name,
            max_length=max_length,
            dtype=dtype,
            device_map=args.device,
        )

    from model_hub.tokenizer_utils import load_tokenizer
    tokenizer = load_tokenizer(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    warnings.filterwarnings('ignore', message='Token indices sequence length')
    tokenizer.model_max_length = max_length

    for task_name in task_list:
        ds_path = os.path.join(args.data_dir, task_name, "validation.jsonl")
        if not os.path.exists(ds_path):
            print(f"[SKIP] {ds_path} not found")
            continue

        data = load_dataset("json", data_files=ds_path, split="train")
        if args.data_idx is not None:
            data = data.select(range(args.data_idx, args.data_idx + 1))
        elif args.data_idx_to is not None:
            data = data.select(range(0, args.data_idx_to))

        out_path = os.path.join(args.out_dir, f"{task_name}.jsonl")
        original_len = len(data)
        resume_from = 0
        file_mode = "w"
        if args.resume and os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as fin:
                resume_from = sum(1 for line in fin if line.strip())
            if resume_from >= original_len:
                print(f"\n--- {task_name}: already finished ({resume_from}/{original_len}), skip")
                continue
            data = data.select(range(resume_from, original_len))
            file_mode = "a"
            print(f"\n--- {task_name}: resume from #{resume_from}, remain {len(data)} samples → {out_path}")
        else:
            print(f"\n--- {task_name}: {original_len} samples → {out_path}")

        gen_len = args.max_gen if args.max_gen is not None else get_gen_len(task_name)
        metric_fn = get_metric_fn(task_name)

        with open(out_path, file_mode, encoding="utf-8") as fout:
            for di, sample in enumerate(tqdm(data, desc=task_name), start=resume_from):
                prompt_text = sample["input"]
                if args.no_think:
                    prompt_text = inject_no_think(prompt_text)

                inputs = tokenizer([prompt_text], return_tensors="pt", padding=True,
                                    add_special_tokens=False)
                input_ids = inputs.input_ids.to(llm.layers[0].device)
                attention_masks = inputs.attention_mask.to(llm.layers[0].device)
                context_len = input_ids.shape[1]

                if torch.cuda.is_available():
                    import gc
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                config = load_config(args, context_len)

                with torch.no_grad():
                    output_ids, timing = llm.generate(
                        attention_type=args.attention_type,
                        inputs_ids=input_ids,
                        attention_masks=attention_masks,
                        max_new_length=gen_len + 1,
                        attn_config=config,
                        temperature=0.0,
                        top_p=0.0,
                        top_k=0,
                        disable_early_stop=False,
                    )

                gen_ids = output_ids[0].tolist() if hasattr(output_ids[0], 'tolist') else output_ids[0]
                pred = tokenizer.decode(gen_ids, skip_special_tokens=True)
                answers = sample.get("outputs", [])
                gt_for_score = coerce_ground_truth(answers, metric_fn)
                score = float(metric_fn(pred, gt_for_score))

                rec = {
                    "qid": di,
                    "task": task_name,
                    "index": sample.get("index", di),
                    "input_len": context_len,
                    "output_len": len(gen_ids),
                    "gen_len": gen_len,
                    "pred": pred,
                    "answer": answers,
                    "score": score,
                    "prefill_time": timing.get("prefill_time", 0),
                    "decode_time": timing.get("decode_time", 0),
                }
                json.dump(rec, fout, ensure_ascii=False)
                fout.write("\n")
                fout.flush()

                status = "✓" if score > 0 else "✗"
                pred_preview = pred.replace("\n", "\\n").replace("\r", "\\r")
                if len(pred_preview) > 180:
                    pred_preview = pred_preview[:180] + "..."
                print(f"  [{status}] #{di} in={context_len} out={len(gen_ids)}/{gen_len} "
                      f"score={score:.3f} pred_preview={pred_preview!r} answer={answers}",
                      flush=True)

    print(f"\nDone. Results in {args.out_dir}")


if __name__ == "__main__":
    main()
