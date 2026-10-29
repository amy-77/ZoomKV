#!/usr/bin/env python3
"""Calibrate the per-level fixed overhead C0 of ZoomKV's chunk selection.

One hierarchy level does two things per decode step: score every chunk of the
level with the Quest min/max bound (``quest_chunk_score``) and keep the top
chunks (``float_topk_3d``). This script times that pair for A = 128 ... 8192
scored chunks, separately for the parent level (top-k keeps 40% of the chunks)
and the child level (30%), and fits t = intercept + slope * A by least squares.
intercept / slope is the fixed cost of one level expressed in chunks; this is
the constant C0 of the cost model (Section 4.7; response to Reviewer 6, D7).

Run from a ZoomKV checkout (needs ``cache_hub`` importable, e.g.
``PYTHONPATH=/path/to/ZoomKV``) on one GPU:

    CUDA_VISIBLE_DEVICES=0 python calibrate_level_overhead.py --out calibration.json

Output rows are ``[scored_chunks, selected_chunks, mean_us, min_us, max_us]``
(mean/min/max over --repeats blocks of --iters timed launches each), the same
layout as revision_artifacts/cost_overhead_calibration_kernel_h20.json.
"""
import argparse
import datetime
import json
import subprocess

import torch

from cache_hub.quest.quest_ops import load_kernel_module
from cache_hub.topk.float_topk_ops import float_topk_3d

LEVELS = {"parent": 0.40, "child": 0.30}  # fraction of scored chunks kept by top-k
SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]
TOPK = "float"  # --topk float: the production float_topk_3d kernel (used for the paper); torch and score_only are diagnostics


def time_level(kernel, n_chunks, keep_ratio, *, kv_heads, head_dim, iters, repeats, device):
    k = max(1, round(keep_ratio * n_chunks))
    q = torch.randn(1, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    lo = torch.randn(1, kv_heads, n_chunks, head_dim, device=device, dtype=torch.bfloat16)
    hi = lo + torch.rand_like(lo)
    valid = torch.ones(1, kv_heads, n_chunks, dtype=torch.bool, device=device)
    scores = torch.empty(1, kv_heads, n_chunks, dtype=torch.float32, device=device)

    if TOPK == "torch":
        def step():
            kernel.quest_chunk_score(q, lo, hi, scores, n_chunks, valid)
            return torch.topk(scores, k, dim=-1).indices
    elif TOPK == "score_only":
        def step():
            kernel.quest_chunk_score(q, lo, hi, scores, n_chunks, valid)
    else:
        def step():
            kernel.quest_chunk_score(q, lo, hi, scores, n_chunks, valid)
            return float_topk_3d(scores, k)

    for _ in range(50):  # warm-up (JIT, allocator, clocks)
        step()
    torch.cuda.synchronize()

    per_iter_us = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            step()
        end.record()
        torch.cuda.synchronize()
        per_iter_us.append(start.elapsed_time(end) * 1000.0 / iters)
    return [n_chunks, k, sum(per_iter_us) / len(per_iter_us), min(per_iter_us), max(per_iter_us)]


def fit(rows):
    xs = [r[0] for r in rows]
    ys = [r[2] for r in rows]
    n = len(rows)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    intercept = my - slope * mx
    return {"slope_us_per_chunk": slope, "intercept_us": intercept,
            "intercept_equiv_chunks": intercept / slope}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--kv-heads", type=int, default=8)     # Qwen3-4B / Llama-3.1-8B
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--note", default="")
    ap.add_argument("--topk", default="float", choices=["float", "torch", "score_only"])
    args = ap.parse_args()
    global TOPK
    TOPK = args.topk

    device = torch.device("cuda")
    kernel = load_kernel_module()
    result = {
        "device": torch.cuda.get_device_name(device),
        "date": datetime.date.today().isoformat(),
        "config": {"batch": 1, "kv_heads": args.kv_heads, "head_dim": args.head_dim,
                   "dtype": "bfloat16", "topk": args.topk, "iters": args.iters, "repeats": args.repeats,
                   "keep_ratio": LEVELS},
        "note": args.note,
    }
    try:
        result["gpu_utilization_before"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    for level, ratio in LEVELS.items():
        rows = [time_level(kernel, n, ratio, kv_heads=args.kv_heads, head_dim=args.head_dim,
                           iters=args.iters, repeats=args.repeats, device=device) for n in SIZES]
        result[level] = rows
        result[f"{level}_fit"] = fit(rows)
        f = result[f"{level}_fit"]
        print(f"{level}: t ~ {f['intercept_us']:.2f} us + {f['slope_us_per_chunk']:.5f} us * A"
              f"  -> C0 ~ {f['intercept_equiv_chunks']:.0f} chunks")
        for r in rows:
            print(f"  A={r[0]:5d} k={r[1]:5d} mean={r[2]:7.2f} min={r[3]:7.2f} max={r[4]:7.2f} us")
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
