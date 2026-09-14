"""
Hierarchical Quest — CUDA kernel correctness + latency test.

Verifies each stage of the kernel pipeline against PyTorch:
  1. quest_chunk_score: kernel scores vs PyTorch reference scores (max|diff|)
  2. float_topk_3d: kernel topk ids vs torch.topk ids (set overlap)
  3. quest_sub_chunk_score: kernel sub-scores vs PyTorch reference (max|diff|)
  4. float_topk_3d (stage2): kernel topk ids vs torch.topk ids (set overlap)
  5. Final chunk_idx set overlap (kernel pipeline vs torch pipeline)
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

DATA_DIR = "/data/qwang/q/thalia/kvcache/ParisKV3/ANNCache/results/chunk_analysis_data_20260515_134112"


def make_chunk_minmax(keys, chunk_size):
    """keys: [kv_len, head_dim] -> chunk_min/max: [n_chunks, head_dim]"""
    kv_len, head_dim = keys.shape
    n_chunks = kv_len // chunk_size
    chunked = keys[:n_chunks * chunk_size].reshape(n_chunks, chunk_size, head_dim)
    return chunked.min(dim=1).values, chunked.max(dim=1).values


def quest_score_ref_3d(q_3d, chunk_min_4d, chunk_max_4d):
    """PyTorch reference score for [1,1,d] query and [1,1,n,d] min/max -> [1,1,n]"""
    q_f = q_3d.float().unsqueeze(2)  # [1,1,1,d]
    return torch.maximum(
        q_f * chunk_min_4d.float(), q_f * chunk_max_4d.float()
    ).sum(dim=-1)  # [1,1,n]


def run_test(device="cuda:0"):
    # ── Config ──
    small_chunk = 16
    large_chunk = 256
    factor = large_chunk // small_chunk  # 16
    large_ratio = 0.25
    small_ratio = 0.2

    print("=" * 70)
    print("Hierarchical Quest — Kernel Correctness + Latency")
    print(f"  large_chunk={large_chunk}, small_chunk={small_chunk}, factor={factor}")
    print(f"  Stage 1: top {large_ratio*100:.0f}%, Stage 2: top {small_ratio*100:.0f}%")
    print("=" * 70)

    # ── Load kernels ──
    print("\nCompiling kernels...")
    quest_kernel = load_kernel_module()
    _ = float_topk_3d(torch.randn(1, 1, 64, device=device), k=4)
    print("Kernels ready.\n")

    key_files = sorted(glob.glob(f"{DATA_DIR}/keys_*.npz"))
    print(f"Found {len(key_files)} key files")

    # Accumulators
    all_s1_score_diffs = []
    all_s1_topk_recalls = []
    all_s2_score_diffs = []
    all_s2_topk_recalls = []
    all_final_recalls = []

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

        # 4D bf16 for kernel
        small_min_4d = small_min.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous()
        small_max_4d = small_max.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous()
        large_min_4d = large_min.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous()
        large_max_4d = large_max.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous()

        # Buffers
        large_scores_buf = torch.empty(1, 1, n_large, dtype=torch.float32, device=device)
        sub_scores_buf = torch.empty(1, 1, nk_sub_total, dtype=torch.float32, device=device)

        # Queries
        pattern = f"query_L{layer}_H{head}_step*_kvlen{kv_len}.npz"
        query_files = sorted(glob.glob(f"{DATA_DIR}/{pattern}"))
        if not query_files:
            continue

        print(f"\n{'─' * 70}")
        print(f"Layer={layer}, Head={head}, kv_len={kv_len}")
        print(f"  n_large={n_large}, n_small={n_small}, nk_large={nk_large}, nk_small={nk_small}")

        layer_s1_score = []
        layer_s1_topk = []
        layer_s2_score = []
        layer_s2_topk = []
        layer_final = []

        for qf in query_files:
            qdata = np.load(qf)
            q = torch.from_numpy(qdata['query']).to(device)
            q_3d = q.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous()  # [1,1,128]

            nvtx.range_push("hierarchical_quest_full")

            # ═══════════════════════════════════════════════════════════
            # Stage 1: Large chunk scoring — Kernel
            # ═══════════════════════════════════════════════════════════
            nvtx.range_push("s1_chunk_score_kernel")
            quest_kernel.quest_chunk_score(
                q_3d, large_min_4d, large_max_4d,
                large_scores_buf, n_large,
            )
            nvtx.range_pop()

            # Stage 1: Large chunk scoring — Torch reference
            nvtx.range_push("s1_chunk_score_torch")
            ref_large_scores = quest_score_ref_3d(q_3d, large_min_4d, large_max_4d)
            nvtx.range_pop()

            # [Check 1] Score diff
            s1_diff = (large_scores_buf - ref_large_scores).abs().max().item()
            layer_s1_score.append(s1_diff)
            all_s1_score_diffs.append(s1_diff)

            # ═══════════════════════════════════════════════════════════
            # Stage 1: TopK — Kernel (float_topk_3d)
            # ═══════════════════════════════════════════════════════════
            nvtx.range_push("s1_topk_kernel")
            large_idx_kernel = float_topk_3d(large_scores_buf, nk_large)
            nvtx.range_pop()

            # Stage 1: TopK — Torch
            nvtx.range_push("s1_topk_torch")
            _, large_idx_torch = torch.topk(large_scores_buf, k=nk_large, dim=-1)
            nvtx.range_pop()

            # [Check 2] TopK id overlap
            kset = set(large_idx_kernel[0, 0].tolist())
            tset = set(large_idx_torch[0, 0].tolist())
            s1_topk_recall = len(kset & tset) / nk_large
            layer_s1_topk.append(s1_topk_recall)
            all_s1_topk_recalls.append(s1_topk_recall)

            # ═══════════════════════════════════════════════════════════
            # Stage 2: Sub-chunk scoring — Kernel
            # ═══════════════════════════════════════════════════════════
            nvtx.range_push("s2_sub_chunk_score_kernel")
            quest_kernel.quest_sub_chunk_score(
                q_3d, small_min_4d, small_max_4d,
                large_idx_kernel, sub_scores_buf,
                nk_large, factor,
            )
            nvtx.range_pop()

            # Stage 2: Sub-chunk scoring — Torch reference
            sub_offsets = torch.arange(factor, device=device)
            all_small_idx = (large_idx_kernel[0, 0].unsqueeze(-1) * factor + sub_offsets).reshape(-1)
            gathered_min = small_min_4d[0, 0, all_small_idx].unsqueeze(0).unsqueeze(0)
            gathered_max = small_max_4d[0, 0, all_small_idx].unsqueeze(0).unsqueeze(0)

            nvtx.range_push("s2_sub_chunk_score_torch")
            ref_sub_scores = quest_score_ref_3d(q_3d, gathered_min, gathered_max)
            nvtx.range_pop()

            # [Check 3] Sub-chunk score diff
            s2_diff = (sub_scores_buf - ref_sub_scores).abs().max().item()
            layer_s2_score.append(s2_diff)
            all_s2_score_diffs.append(s2_diff)

            # ═══════════════════════════════════════════════════════════
            # Stage 2: TopK — Kernel (float_topk_3d)
            # ═══════════════════════════════════════════════════════════
            nvtx.range_push("s2_topk_kernel")
            sub_topk_kernel = float_topk_3d(sub_scores_buf, nk_small)
            nvtx.range_pop()

            # Stage 2: TopK — Torch
            nvtx.range_push("s2_topk_torch")
            _, sub_topk_torch = torch.topk(sub_scores_buf, k=nk_small, dim=-1)
            nvtx.range_pop()

            nvtx.range_pop()  # hierarchical_quest_full

            # [Check 4] TopK id overlap
            kset2 = set(sub_topk_kernel[0, 0].tolist())
            tset2 = set(sub_topk_torch[0, 0].tolist())
            s2_topk_recall = len(kset2 & tset2) / nk_small
            layer_s2_topk.append(s2_topk_recall)
            all_s2_topk_recalls.append(s2_topk_recall)

            # ═══════════════════════════════════════════════════════════
            # [Check 5] Final chunk_idx overlap
            # ═══════════════════════════════════════════════════════════
            # Kernel pipeline
            large_sel_k = sub_topk_kernel // factor
            sub_off_k = sub_topk_kernel % factor
            chunk_idx_kernel = torch.gather(large_idx_kernel, 2, large_sel_k) * factor + sub_off_k
            chunk_idx_kernel = chunk_idx_kernel.clamp(min=0, max=n_small - 1)

            # Torch pipeline (same scores, torch.topk)
            large_sel_t = sub_topk_torch // factor
            sub_off_t = sub_topk_torch % factor
            chunk_idx_torch = torch.gather(large_idx_kernel, 2, large_sel_t) * factor + sub_off_t
            chunk_idx_torch = chunk_idx_torch.clamp(min=0, max=n_small - 1)

            k_final = set(chunk_idx_kernel[0, 0].tolist())
            t_final = set(chunk_idx_torch[0, 0].tolist())
            final_recall = len(k_final & t_final) / len(t_final) if t_final else 1.0
            layer_final.append(final_recall)
            all_final_recalls.append(final_recall)

        # Per-layer summary
        print(f"  Queries: {len(query_files)}")
        print(f"\n  [Check 1] quest_chunk_score max|diff|:")
        print(f"    avg={sum(layer_s1_score)/len(layer_s1_score):.2e}, max={max(layer_s1_score):.2e}")
        print(f"  [Check 2] Stage1 float_topk_3d vs torch.topk overlap:")
        print(f"    avg={sum(layer_s1_topk)/len(layer_s1_topk)*100:.1f}%, min={min(layer_s1_topk)*100:.1f}%")
        print(f"  [Check 3] quest_sub_chunk_score max|diff|:")
        print(f"    avg={sum(layer_s2_score)/len(layer_s2_score):.2e}, max={max(layer_s2_score):.2e}")
        print(f"  [Check 4] Stage2 float_topk_3d vs torch.topk overlap:")
        print(f"    avg={sum(layer_s2_topk)/len(layer_s2_topk)*100:.1f}%, min={min(layer_s2_topk)*100:.1f}%")
        print(f"  [Check 5] Final chunk_idx set overlap:")
        print(f"    avg={sum(layer_final)/len(layer_final)*100:.1f}%, min={min(layer_final)*100:.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # Overall summary
    # ═══════════════════════════════════════════════════════════════
    n_queries = len(all_s1_score_diffs)
    print(f"\n{'=' * 70}")
    print(f"OVERALL SUMMARY ({n_queries} queries)")
    print(f"  Config: large_chunk={large_chunk}, small_chunk={small_chunk}, "
          f"stage1={large_ratio*100:.0f}%, stage2={small_ratio*100:.0f}%")
    print(f"  n_large={n_large}, nk_large={nk_large}, nk_sub_total={nk_sub_total}, nk_small={nk_small}")
    print(f"{'─' * 70}")

    print(f"\n  Correctness:")
    print(f"  [Check 1] quest_chunk_score max|diff|: avg={sum(all_s1_score_diffs)/n_queries:.2e}, "
          f"max={max(all_s1_score_diffs):.2e}")
    print(f"  [Check 2] Stage1 topk id overlap: avg={sum(all_s1_topk_recalls)/n_queries*100:.2f}%, "
          f"min={min(all_s1_topk_recalls)*100:.1f}%")
    print(f"  [Check 3] quest_sub_chunk_score max|diff|: avg={sum(all_s2_score_diffs)/n_queries:.2e}, "
          f"max={max(all_s2_score_diffs):.2e}")
    print(f"  [Check 4] Stage2 topk id overlap: avg={sum(all_s2_topk_recalls)/n_queries*100:.2f}%, "
          f"min={min(all_s2_topk_recalls)*100:.1f}%")
    print(f"  [Check 5] Final chunk_idx overlap: avg={sum(all_final_recalls)/n_queries*100:.2f}%, "
          f"min={min(all_final_recalls)*100:.1f}%")
    print(f"\n  Use nsys to profile latency. NVTX ranges:")
    print(f"    hierarchical_quest_full, s1_chunk_score_kernel, s1_chunk_score_torch,")
    print(f"    s1_topk_kernel, s1_topk_torch, s2_sub_chunk_score_kernel,")
    print(f"    s2_sub_chunk_score_torch, s2_topk_kernel, s2_topk_torch")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    device = "cuda:0"
    if len(sys.argv) > 1:
        device = sys.argv[1]
    run_test(device=device)
