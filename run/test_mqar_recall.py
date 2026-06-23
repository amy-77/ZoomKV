#!/usr/bin/env python3
"""MQAR exact-QK recall harness for the KIVI Quest density (_seperate_kernel) cache.

This does NOT measure MQAR answer accuracy. It uses an MQAR sequence purely as a
*controlled long-context retrieval workload*, runs a real dense model
(Qwen3 by default) with the PolarANN sparse-decode cache
``cache_hub.polar_cache_kivi_quest_density_seperate_kernel`` and, at every
captured decode step / layer / head, compares the sparse selector's chosen
token set against the exact dense-attention top-K:

    GT_topK   = topk_K( q_t . K_retrieval )         # exact full-token q.k
    Selected  = cache.collision_based_topk_batch(...) indices  (retrieval-zone)
    recall@K  = |GT_topK  intersect  Selected| / K

i.e. "of the true exact-QK top-K tokens, how many fell inside the selector's
budget", aggregated over all layers/heads (same convention as the
``POLAR_PIPELINE_RECALL`` log in
``cache_hub.polar_cache_quest_density_single_layer_kivi``).

The measurement is done with an external monkeypatch around
``collision_based_topk_batch`` so the hot-path module source is untouched.

MQAR generation is a faithful inline port of zoology
``zoology/data/multiquery_ar.py`` (kept inline to avoid importing zoology's
pydantic/config dependency chain).

Example:
    conda activate retroinfer
    CUDA_VISIBLE_DEVICES=5 python run/test_mqar_recall.py \
        --model_name models/Qwen3-4B-Thinking-2507 \
        --input_seq_len 6144 --num_kv_pairs 512 \
        --num_queries 32 --final_topk 256 \
        --recall_ks 50,100,256 --log_every 1
"""
from __future__ import annotations

import os

# --- Cache-module env defaults (mirror run/test_pipeline_recall_density_easy_long.sh) ---
# Must be set before the cache module is imported / instantiated.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("POLAR_QUEST_STAGE1", "1")
os.environ.setdefault("POLAR_QUEST_CHUNK", "16")
os.environ.setdefault("POLAR_QUEST_MODE", "minmax_raw_cached")
os.environ.setdefault("POLAR_QUEST_HIERARCHICAL", "1")
os.environ.setdefault("POLAR_QUEST_LARGE_CHUNK", "256")
os.environ.setdefault("POLAR_QUEST_LARGE_RATIO", "0.4")
os.environ.setdefault("POLAR_QUEST_SMALL_RATIO", "0.3")
os.environ.setdefault("POLAR_TOPK_BACKEND", "float")
os.environ.setdefault("POLAR_CDS_ENABLED", "1")
os.environ.setdefault("POLAR_CDS_DENSE_RATIO", "0.3")
os.environ.setdefault("POLAR_CDS_DENSE_TOPK", "16")
os.environ.setdefault("POLAR_CDS_SPARSE_TOPK", "8")

import argparse
import importlib
import json
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

DEFAULT_CACHE_MODULE = "cache_hub.polar_cache_kivi_quest_density_seperate_kernel"


# ---------------------------------------------------------------------------
# MQAR generation (faithful port of zoology/data/multiquery_ar.py::multiquery_ar)
# ---------------------------------------------------------------------------
def multiquery_ar(
    vocab_size: int,
    input_seq_len: int,
    seed: int,
    num_kv_pairs: int = 8,
    power_a: float = 0.01,
    num_passes: int = 1,
    random_non_queries: bool = True,
):
    """Returns (inputs[input_seq_len] int64, label_positions list, label_values list).

    inputs: the MQAR token-id sequence (key/val/query layout).
    label_positions/label_values: for each query, the position in `inputs` whose
    *next* token should be the recalled value, and that value id (diagnostic only).
    """
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 2 * num_passes + num_kv_pairs * 2 <= input_seq_len

    np.random.seed(seed)
    context_size = num_kv_pairs * 2 * num_passes

    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys = np.random.choice(key_choices, size=num_kv_pairs, replace=False)
    values = np.random.choice(value_choices, size=num_kv_pairs, replace=False)

    kvs = np.zeros(context_size // num_passes, dtype=np.int64)
    kvs[0::2] = keys
    kvs[1::2] = values
    kvs = np.tile(kvs, num_passes)

    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()
    gaps = np.random.choice(space, size=num_kv_pairs, replace=False, p=p)

    queries = np.zeros(input_seq_len - context_size + 1, dtype=np.int64)
    np.put_along_axis(queries, gaps * 2, values=keys, axis=0)
    examples = np.concatenate([kvs, queries])  # len = input_seq_len + 1

    labels = np.full(input_seq_len + 1, -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=0)

    inputs = examples[:-1].copy()
    labels = labels[1:]
    if random_non_queries:
        zero_mask = inputs == 0
        inputs[zero_mask] = np.random.randint(vocab_size, size=zero_mask.sum())

    label_positions = np.where(labels != -100)[0].tolist()
    label_values = labels[labels != -100].tolist()
    return inputs.astype(np.int64), label_positions, label_values


# ---------------------------------------------------------------------------
# Recall recorder (external monkeypatch around collision_based_topk_batch)
# ---------------------------------------------------------------------------
class RecallRecorder:
    def __init__(self, ks, log_every, layer_num):
        self.ks = sorted({int(k) for k in ks if int(k) > 0})
        self.log_every = max(1, int(log_every))
        self.layer_num = int(layer_num)
        self.step = 0
        self.log_this = False
        self.accum = {}            # K -> [hits, total]
        self.kv_len = 0
        self.history = []          # list of per-step dict for json dump
        self.warned = False

    def _flush(self):
        if not self.accum:
            return
        parts, rec = [], {"step": self.step, "kv_len": self.kv_len, "recall": {}}
        for K in sorted(self.accum):
            hits, tot = self.accum[K]
            r = (hits / tot * 100.0) if tot > 0 else 0.0
            parts.append(f"R@{K}={r:.1f}%")
            rec["recall"][K] = round(r, 3)
        self.history.append(rec)
        print(f"[MQAR Recall] step={self.step} kv={self.kv_len} (agg all layers/heads) | "
              + " ".join(parts), flush=True)
        self.accum = {}

    @torch.no_grad()
    def record(self, cache, query, topk_indices, layer_idx):
        if layer_idx == 0:
            if self.log_this and self.accum:
                self._flush()
            self.step += 1
            self.log_this = (self.step % self.log_every == 0)
            if self.log_this:
                self.accum = {}
                self.kv_len = 0

        if not self.log_this:
            return

        cpu_len = int(cache.cpu_valid_end[layer_idx])
        if cpu_len <= 0:
            return
        self.kv_len = cpu_len

        keys = cache.unified_keys_cpu[layer_idx][:, :, :cpu_len, :].to(cache.device).float()
        q = query.squeeze(2).float()  # [bs, kv_heads, head_dim]

        # Strip the tail "skipped" tokens that _cds_dense_sparse_select cat's on,
        # so we measure the pure stage-1+rerank selection (sibling convention).
        sel = topk_indices
        sk = int(cache._chunk_skipped_count[layer_idx]) if hasattr(cache, "_chunk_skipped_count") else 0
        if sk > 0 and sel.shape[-1] > sk:
            sel = sel[..., :-sk]

        nh = keys.shape[1]
        for h in range(nh):
            scores = keys[0, h] @ q[0, h]               # [cpu_len]
            sel_set = set(sel[0, h].tolist())
            for K in self.ks:
                k = min(K, cpu_len)
                if k <= 0:
                    continue
                gt = torch.topk(scores, k).indices.tolist()
                inter = sum(1 for g in gt if g in sel_set)
                acc = self.accum.setdefault(K, [0, 0])
                acc[0] += inter
                acc[1] += k

        if layer_idx == self.layer_num - 1 and self.accum:
            self._flush()
            self.log_this = False


def install_recall_hook(cache_module_name, recorder):
    mod = importlib.import_module(cache_module_name)
    cls = mod.polar_cache
    orig = cls.collision_based_topk_batch

    def wrapped(self, *args, **kwargs):
        topk = orig(self, *args, **kwargs)
        try:
            query = kwargs.get("query")
            if query is None and args:
                query = args[0]
            layer_idx = kwargs.get("layer_idx")
            recorder.record(self, query, topk, layer_idx)
        except Exception as e:  # never break decode because of the recall probe
            if not recorder.warned:
                print(f"[MQAR Recall] WARN: recall probe disabled: {e!r}", flush=True)
                recorder.warned = True
        return topk

    cls.collision_based_topk_batch = wrapped
    return cls, orig


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="MQAR exact-QK recall for PolarANN KIVI cache")
    p.add_argument("--model_name", type=str, default="models/Qwen3-4B-Thinking-2507")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--polar_cache_module", type=str, default=DEFAULT_CACHE_MODULE)
    # MQAR workload
    p.add_argument("--input_seq_len", type=int, default=6144,
                   help="MQAR sequence length (== prefill length). Must be > sink+local+final_topk*2 to trigger retrieval.")
    p.add_argument("--num_kv_pairs", type=int, default=512)
    p.add_argument("--mqar_vocab", type=int, default=0,
                   help="MQAR vocab size. 0 => input_seq_len*2 (must be > input_seq_len).")
    p.add_argument("--id_offset", type=int, default=2000,
                   help="Offset added to MQAR ids to avoid model special-token ids.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_queries", type=int, default=32, help="Decode steps to run.")
    # PolarANN config
    p.add_argument("--final_topk", type=int, default=256)
    p.add_argument("--sink_size", type=int, default=64)
    p.add_argument("--local_size", type=int, default=256)
    p.add_argument("--candidate_ratio", type=float, default=0.2)
    p.add_argument("--full_attention_threshold", type=int, default=2000)
    # recall logging
    p.add_argument("--recall_ks", type=str, default="50,100,256")
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--out", type=str, default=None, help="Optional json path for per-step recall.")
    return p.parse_args()


def build_polar_config(args, context_len):
    cand = args.candidate_ratio
    nprobe = max(30, min(int(context_len * cand), context_len))
    large_chunk = int(os.environ.get("POLAR_QUEST_LARGE_CHUNK", "256"))
    base_dui = max(args.local_size, large_chunk)
    dui = ((base_dui + large_chunk - 1) // large_chunk) * large_chunk
    codebook_path = os.path.join(
        PROJECT_ROOT,
        "turboquant",
        "codebooks",
        "codebook_d128_m8_Kr1_Kw256_rabitq_sign.json",
    )
    return {
        "PolarANN": {
            "core": 22,
            "nprobe": nprobe,
            "cache_unit_size": 8,
            "cache_cluster_num": 450,
            "sink_size": args.sink_size,
            "local_size": args.local_size,
            "static_pattern_start": args.sink_size,
            "static_pattern_end": args.local_size,
            "collision_ratio": 0.3,
            "candidate_ratio": cand,
            "codebook_path": codebook_path,
            "final_topk": args.final_topk,
            "full_attention_threshold": args.full_attention_threshold,
            "enable_offload": True,
            "dynamic_update_interval": dui,
        }
    }


def main():
    args = parse_args()
    os.environ["POLAR_CACHE_MODULE"] = args.polar_cache_module
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    mqar_vocab = args.mqar_vocab if args.mqar_vocab > 0 else args.input_seq_len * 2
    inputs_np, label_pos, label_val = multiquery_ar(
        vocab_size=mqar_vocab,
        input_seq_len=args.input_seq_len,
        seed=args.seed,
        num_kv_pairs=args.num_kv_pairs,
    )

    print(f"[mqar-recall] module={args.polar_cache_module}", flush=True)
    print(f"[mqar-recall] MQAR: input_seq_len={args.input_seq_len} num_kv_pairs={args.num_kv_pairs} "
          f"vocab={mqar_vocab} queries={len(label_pos)}", flush=True)

    # Load model
    from model_hub.qwen import QwenModel
    max_length = args.input_seq_len + args.num_queries + 16
    print(f"[mqar-recall] loading model {args.model_name} (max_length={max_length}) ...", flush=True)
    llm = QwenModel(
        model_name=args.model_name,
        max_length=max_length,
        dtype=dtype,
        device_map=args.device,
        rope_scaling=None,
    )
    print(f"[mqar-recall]   layers={llm.num_layers} heads={llm.num_heads} "
          f"kv_heads={llm.num_key_value_heads} head_dim={llm.head_dim} vocab={llm.vocab_size}",
          flush=True)

    # Map MQAR ids into a safe model-vocab band.
    safe_max = llm.vocab_size - 1
    ids = (inputs_np.astype(np.int64) + args.id_offset) % safe_max
    input_ids = torch.from_numpy(ids).unsqueeze(0).to(llm.layers[0].device)
    attention_masks = torch.ones_like(input_ids)

    # Install recall hook
    recorder = RecallRecorder(
        ks=[int(x) for x in args.recall_ks.split(",") if x.strip()],
        log_every=args.log_every,
        layer_num=llm.num_layers,
    )
    install_recall_hook(args.polar_cache_module, recorder)

    config = build_polar_config(args, context_len=args.input_seq_len)

    min_total = max(args.full_attention_threshold,
                    args.sink_size + args.local_size + args.final_topk * 2)
    print(f"[mqar-recall] retrieval activates when total_len >= {min_total} "
          f"(prefill={args.input_seq_len}) → {'OK' if args.input_seq_len >= min_total else 'TOO SHORT'}",
          flush=True)

    with torch.no_grad():
        output_ids, timing = llm.generate(
            attention_type="PolarANN",
            inputs_ids=input_ids,
            attention_masks=attention_masks,
            max_new_length=args.num_queries + 1,
            attn_config=config,
            temperature=0.0,
            top_p=0.0,
            top_k=0,
            disable_early_stop=True,
        )

    recorder._flush()

    if recorder.history:
        agg = {}
        for rec in recorder.history:
            for K, v in rec["recall"].items():
                agg.setdefault(K, []).append(v)
        print("\n[mqar-recall] ===== mean recall over "
              f"{len(recorder.history)} logged step(s) =====", flush=True)
        for K in sorted(agg):
            print(f"   R@{K}: {np.mean(agg[K]):.2f}%", flush=True)
    else:
        print("[mqar-recall] no recall logged — sequence likely too short to trigger retrieval.",
              flush=True)

    print(f"[mqar-recall] decode timing: prefill={timing.get('prefill_time', 0):.2f}s "
          f"decode={timing.get('decode_time', 0):.2f}s", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "args": vars(args),
                "mqar": {"vocab": mqar_vocab, "num_queries": len(label_pos)},
                "history": recorder.history,
            }, f, indent=2)
        print(f"[mqar-recall] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
