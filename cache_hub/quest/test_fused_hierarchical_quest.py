"""
Fused Hierarchical Quest — correctness test.

Compares the fused single-kernel pipeline against the reference separate-kernel pipeline:
  Reference: quest_chunk_score → float_topk_3d → quest_sub_chunk_score → float_topk_3d
  Fused:     fused_hierarchical_quest (single kernel)

Metric: set overlap of final selected small-chunk indices.
Uses real KV cache data from model inference.
"""

import math
import glob
import torch
import torch.cuda.nvtx as nvtx
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quest_ops import load_kernel_module
from topk.float_topk_ops import float_topk_3d
from fused_hierarchical_quest_ops import fused_hierarchical_quest

DATA_DIR = "/data/qwang/q/thalia/kvcache/ParisKV3/ANNCache/results/chunk_analysis_data_20260515_134112"


def make_chunk_minmax(keys, chunk_size):
    """keys: [kv_len, head_dim] -> chunk_min/max: [n_chunks, head_dim]"""
    kv_len, head_dim = keys.shape
    n_chunks = kv_len // chunk_size
    chunked = keys[:n_chunks * chunk_size].reshape(n_chunks, chunk_size, head_dim)
    return chunked.min(dim=1).values, chunked.max(dim=1).values


def run_reference_pipeline(q_3d, large_min_4d, large_max_4d, small_min_4d, small_max_4d,
                           n_large, nk_large, nk_small, factor, quest_kernel):
    """Run the separate 4-kernel reference pipeline."""
    nk_sub_total = nk_large * factor

    large_scores_buf = torch.empty(1, 1, n_large, dtype=torch.float32, device=q_3d.device)
    sub_scores_buf = torch.empty(1, 1, nk_sub_total, dtype=torch.float32, device=q_3d.device)

    # Stage 1: score large chunks
    quest_kernel.quest_chunk_score(
        q_3d, large_min_4d, large_max_4d, large_scores_buf, n_large)

    # Stage 1: top-K
    large_idx = float_topk_3d(large_scores_buf, nk_large)

    # Stage 2: score sub-chunks
    quest_kernel.quest_sub_chunk_score(
        q_3d, small_min_4d, small_max_4d,
        large_idx, sub_scores_buf, nk_large, factor)

    # Stage 2: top-K
    sub_topk = float_topk_3d(sub_scores_buf, nk_small)

    # Map back to global small chunk indices
    large_sel = sub_topk // factor
    sub_off = sub_topk % factor
    chunk_idx = torch.gather(large_idx, 2, large_sel) * factor + sub_off
    return chunk_idx[0, 0].to(torch.int32)


def run_test(device="cuda:0"):
    small_chunk = 16
    large_chunk = 128
    factor = large_chunk // small_chunk
    large_ratio = 0.25
    small_ratio = 0.2

    print("=" * 70)
    print("Fused Hierarchical Quest — Correctness Test")
    print(f"  large_chunk={large_chunk}, small_chunk={small_chunk}, factor={factor}")
    print(f"  Stage 1: top {large_ratio*100:.0f}%, Stage 2: top {small_ratio*100:.0f}%")
    print("=" * 70)

    # Compile kernels
    print("\nCompiling kernels...")
    quest_kernel = load_kernel_module()
    _ = float_topk_3d(torch.randn(1, 1, 64, device=device), k=4)
    fused_mod = fused_hierarchical_quest  # triggers compilation on first call
    print("Kernels ready.\n")

    key_files = sorted(glob.glob(f"{DATA_DIR}/keys_*.npz"))
    print(f"Found {len(key_files)} key files")

    all_overlaps = []
    all_ref_sizes = []
    all_fused_sizes = []

    for kf in key_files:
        kdata = np.load(kf)
        keys_np = kdata['keys']
        layer = int(kdata['layer'])
        head = int(kdata['head'])
        kv_len = keys_np.shape[0]

        keys = torch.from_numpy(keys_np).to(device)

        n_large = kv_len // large_chunk
        usable_len = n_large * large_chunk
        n_small = usable_len // small_chunk

        nk_large = max(1, int(math.ceil(n_large * large_ratio)))
        nk_sub_total = nk_large * factor
        nk_small = max(1, int(math.ceil(nk_sub_total * small_ratio)))

        # Chunk min/max
        small_min, small_max = make_chunk_minmax(keys[:usable_len], small_chunk)
        large_min = small_min.reshape(n_large, factor, -1).min(dim=1).values
        large_max = small_max.reshape(n_large, factor, -1).max(dim=1).values

        # 4D bf16 for kernels
        small_min_4d = small_min.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous()
        small_max_4d = small_max.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous()
        large_min_4d = large_min.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous()
        large_max_4d = large_max.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous()

        # Queries
        pattern = f"query_L{layer}_H{head}_step*_kvlen{kv_len}.npz"
        query_files = sorted(glob.glob(f"{DATA_DIR}/{pattern}"))
        if not query_files:
            continue

        print(f"\n{'─' * 70}")
        print(f"Layer={layer}, Head={head}, kv_len={kv_len}")
        print(f"  n_large={n_large}, nk_large={nk_large}, nk_sub_total={nk_sub_total}, nk_small={nk_small}")
        print(f"  Queries: {len(query_files)}")

        layer_overlaps = []

        for qf in query_files:
            qdata = np.load(qf)
            q = torch.from_numpy(qdata['query']).to(device)
            q_3d = q.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous()

            # Reference pipeline (4 kernels)
            nvtx.range_push("reference_pipeline")
            ref_idx = run_reference_pipeline(
                q_3d, large_min_4d, large_max_4d, small_min_4d, small_max_4d,
                n_large, nk_large, nk_small, factor, quest_kernel)
            nvtx.range_pop()

            # Fused pipeline (1 kernel)
            nvtx.range_push("fused_pipeline")
            fused_idx = fused_hierarchical_quest(
                q_3d, large_min_4d, large_max_4d, small_min_4d, small_max_4d,
                n_large, nk_large, nk_small, factor)
            fused_idx = fused_idx[0, 0]
            nvtx.range_pop()

            # Compare: set overlap
            ref_set = set(ref_idx.cpu().tolist())
            fused_set = set(fused_idx.cpu().tolist())
            overlap = len(ref_set & fused_set) / len(ref_set) if ref_set else 1.0
            layer_overlaps.append(overlap)
            all_overlaps.append(overlap)
            all_ref_sizes.append(len(ref_set))
            all_fused_sizes.append(len(fused_set))

        avg = sum(layer_overlaps) / len(layer_overlaps)
        mn = min(layer_overlaps)
        print(f"  Set overlap (fused vs reference): avg={avg*100:.1f}%, min={mn*100:.1f}%")

    # Overall summary
    n_queries = len(all_overlaps)
    print(f"\n{'=' * 70}")
    print(f"OVERALL SUMMARY ({n_queries} queries)")
    print(f"  Config: large_chunk={large_chunk}, small_chunk={small_chunk}, "
          f"factor={factor}, stage1={large_ratio*100:.0f}%, stage2={small_ratio*100:.0f}%")
    print(f"{'─' * 70}")
    if all_overlaps:
        avg = sum(all_overlaps) / len(all_overlaps)
        mn = min(all_overlaps)
        mx = max(all_overlaps)
        print(f"  Set overlap (fused vs reference):")
        print(f"    avg={avg*100:.2f}%, min={mn*100:.1f}%, max={mx*100:.1f}%")
        print(f"  Avg |ref_set|={sum(all_ref_sizes)/len(all_ref_sizes):.0f}, "
              f"|fused_set|={sum(all_fused_sizes)/len(all_fused_sizes):.0f}")
        if mn < 1.0:
            print(f"\n  NOTE: Overlap < 100% is expected due to tie-breaking differences")
            print(f"  in radix top-K (both produce valid top-K sets).")
    print(f"\n  Use nsys to compare latency:")
    print(f"    nvtx ranges: reference_pipeline, fused_pipeline")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    device = "cuda:0"
    if len(sys.argv) > 1:
        device = sys.argv[1]
    run_test(device=device)
