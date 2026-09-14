"""
Hierarchical Quest vs Single-Layer Quest — Recall & Latency Comparison.

Both methods select the same total budget = 5% of tokens as candidates.
  - Hierarchical: large_chunk=256, small_chunk=16, stage1=25%, stage2=20% → 5%
  - Single-layer (flat): chunk_size=16, top 5% of all chunks directly

Measures:
  - Recall@500, Recall@100 against true top-K (brute-force QK)
  - chunk_score_kernel latency (CUDA events, averaged over queries)
"""

import math
import glob
import torch
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


def run_test(device="cuda:0"):
    # ── Config ──
    small_chunk = 16
    large_chunk = 256
    factor = large_chunk // small_chunk  # 16
    total_budget = 0.05

    # Hierarchical: 25% * 20% = 5%
    large_ratio = 0.25
    small_ratio = 0.20

    # Single-layer: flat 5% of all chunk-16 chunks
    flat_ratio = total_budget

    topk_values = [500, 100]
    warmup_iters = 10

    print("=" * 70)
    print("Hierarchical vs Single-Layer Quest — Budget = 5%")
    print(f"  Hierarchical: large_chunk={large_chunk}, small_chunk={small_chunk}")
    print(f"    stage1={large_ratio*100:.0f}%, stage2={small_ratio*100:.0f}%")
    print(f"  Single-layer: chunk_size={small_chunk}, top {flat_ratio*100:.0f}%")
    print("=" * 70)

    # ── Load kernels ──
    print("\nCompiling kernels...")
    quest_kernel = load_kernel_module()
    _ = float_topk_3d(torch.randn(1, 1, 64, device=device), k=4)
    print("Kernels ready.\n")

    key_files = sorted(glob.glob(f"{DATA_DIR}/keys_*.npz"))
    print(f"Found {len(key_files)} key files")

    # Accumulators
    hier_recalls = {k: [] for k in topk_values}
    flat_recalls = {k: [] for k in topk_values}
    hier_times_score = []
    hier_times_topk1 = []
    hier_times_sub_score = []
    hier_times_topk2 = []
    hier_times_total = []
    flat_times_score = []
    flat_times_topk = []
    flat_times_total = []

    for kf in key_files:
        kdata = np.load(kf)
        keys_np = kdata['keys']
        layer = int(kdata['layer'])
        head = int(kdata['head'])
        kv_len = keys_np.shape[0]

        keys = torch.from_numpy(keys_np).to(device)

        n_large = kv_len // large_chunk
        usable_len = n_large * large_chunk
        n_small = usable_len // small_chunk  # same for both methods

        nk_large = max(1, int(math.ceil(n_large * large_ratio)))
        nk_sub_total = nk_large * factor
        nk_small_hier = max(1, int(math.ceil(nk_sub_total * small_ratio)))
        nk_flat = max(1, int(math.ceil(n_small * flat_ratio)))

        # Build min/max for both methods
        small_min, small_max = make_chunk_minmax(keys[:usable_len], small_chunk)
        large_min = small_min.reshape(n_large, factor, -1).min(dim=1).values
        large_max = small_max.reshape(n_large, factor, -1).max(dim=1).values

        # 4D bf16 tensors
        small_min_4d = small_min.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous()
        small_max_4d = small_max.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous()
        large_min_4d = large_min.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous()
        large_max_4d = large_max.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous()

        # Buffers
        large_scores_buf = torch.empty(1, 1, n_large, dtype=torch.float32, device=device)
        sub_scores_buf = torch.empty(1, 1, nk_sub_total, dtype=torch.float32, device=device)
        flat_scores_buf = torch.empty(1, 1, n_small, dtype=torch.float32, device=device)

        # Queries
        pattern = f"query_L{layer}_H{head}_step*_kvlen{kv_len}.npz"
        query_files = sorted(glob.glob(f"{DATA_DIR}/{pattern}"))
        if not query_files:
            continue

        print(f"\n{'─' * 70}")
        print(f"Layer={layer}, Head={head}, kv_len={kv_len}")
        print(f"  n_large={n_large}, n_small={n_small}")
        print(f"  Hierarchical: nk_large={nk_large}, nk_sub_total={nk_sub_total}, "
              f"nk_small={nk_small_hier}, tokens={nk_small_hier * small_chunk}")
        print(f"  Flat:          nk_flat={nk_flat}, tokens={nk_flat * small_chunk}")
        print(f"  Queries: {len(query_files)}")

        # Warmup
        q_warmup = torch.randn(1, 1, keys.shape[1], dtype=torch.bfloat16, device=device)
        for _ in range(warmup_iters):
            quest_kernel.quest_chunk_score(q_warmup, large_min_4d, large_max_4d, large_scores_buf, n_large)
            quest_kernel.quest_chunk_score(q_warmup, small_min_4d, small_max_4d, flat_scores_buf, n_small)

        # CUDA events for timing
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        for qf in query_files:
            qdata = np.load(qf)
            q = torch.from_numpy(qdata['query']).to(device)
            q_3d = q.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous()

            # Ground truth
            true_scores = keys[:usable_len].float() @ q.float()

            # ═══════════════════════════════════════════════════════════
            # Hierarchical pipeline
            # ═══════════════════════════════════════════════════════════
            torch.cuda.synchronize()

            # Stage 1: score large chunks
            start_evt.record()
            quest_kernel.quest_chunk_score(
                q_3d, large_min_4d, large_max_4d, large_scores_buf, n_large
            )
            end_evt.record()
            end_evt.synchronize()
            t_hier_score = start_evt.elapsed_time(end_evt)

            # Stage 1: topk
            start_evt.record()
            large_idx = float_topk_3d(large_scores_buf, nk_large)
            end_evt.record()
            end_evt.synchronize()
            t_hier_topk1 = start_evt.elapsed_time(end_evt)

            # Stage 2: sub-chunk scoring
            start_evt.record()
            quest_kernel.quest_sub_chunk_score(
                q_3d, small_min_4d, small_max_4d,
                large_idx, sub_scores_buf,
                nk_large, factor,
            )
            end_evt.record()
            end_evt.synchronize()
            t_hier_sub_score = start_evt.elapsed_time(end_evt)

            # Stage 2: topk
            start_evt.record()
            sub_topk_idx = float_topk_3d(sub_scores_buf, nk_small_hier)
            end_evt.record()
            end_evt.synchronize()
            t_hier_topk2 = start_evt.elapsed_time(end_evt)

            t_hier_total = t_hier_score + t_hier_topk1 + t_hier_sub_score + t_hier_topk2

            # Reconstruct token candidates
            large_sel = sub_topk_idx // factor
            sub_off = sub_topk_idx % factor
            chunk_idx_hier = torch.gather(large_idx, 2, large_sel) * factor + sub_off
            chunk_idx_hier = chunk_idx_hier.clamp(min=0, max=n_small - 1)

            token_offsets = torch.arange(small_chunk, device=device)
            hier_tokens = (chunk_idx_hier[0, 0].unsqueeze(-1) * small_chunk + token_offsets).reshape(-1)
            hier_set = set(hier_tokens.clamp(0, usable_len - 1).tolist())

            # ═══════════════════════════════════════════════════════════
            # Single-layer (flat) pipeline
            # ═══════════════════════════════════════════════════════════
            torch.cuda.synchronize()

            # Score all small chunks
            start_evt.record()
            quest_kernel.quest_chunk_score(
                q_3d, small_min_4d, small_max_4d, flat_scores_buf, n_small
            )
            end_evt.record()
            end_evt.synchronize()
            t_flat_score = start_evt.elapsed_time(end_evt)

            # TopK
            start_evt.record()
            flat_idx = float_topk_3d(flat_scores_buf, nk_flat)
            end_evt.record()
            end_evt.synchronize()
            t_flat_topk = start_evt.elapsed_time(end_evt)

            t_flat_total = t_flat_score + t_flat_topk

            # Token candidates
            flat_tokens = (flat_idx[0, 0].unsqueeze(-1) * small_chunk + token_offsets).reshape(-1)
            flat_set = set(flat_tokens.clamp(0, usable_len - 1).tolist())

            # ═══════════════════════════════════════════════════════════
            # Recall
            # ═══════════════════════════════════════════════════════════
            for k in topk_values:
                _, true_topk_idx = torch.topk(true_scores, k=k)
                true_set = set(true_topk_idx.tolist())
                hier_recalls[k].append(len(hier_set & true_set) / k)
                flat_recalls[k].append(len(flat_set & true_set) / k)

            # Timing
            hier_times_score.append(t_hier_score)
            hier_times_topk1.append(t_hier_topk1)
            hier_times_sub_score.append(t_hier_sub_score)
            hier_times_topk2.append(t_hier_topk2)
            hier_times_total.append(t_hier_total)
            flat_times_score.append(t_flat_score)
            flat_times_topk.append(t_flat_topk)
            flat_times_total.append(t_flat_total)

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    n_queries = len(hier_times_total)
    print(f"\n{'=' * 70}")
    print(f"OVERALL SUMMARY ({n_queries} queries)")
    print(f"  Budget = {total_budget*100:.0f}% of tokens")
    print(f"  Hierarchical: 256→16, stage1=25%, stage2=20%")
    print(f"  Flat: chunk=16, top 5%")
    print(f"{'─' * 70}")

    # Recall comparison
    print(f"\n  {'Metric':<18}{'Hierarchical':>14}{'Flat':>14}{'Diff':>10}")
    for k in topk_values:
        h_avg = sum(hier_recalls[k]) / len(hier_recalls[k]) * 100
        f_avg = sum(flat_recalls[k]) / len(flat_recalls[k]) * 100
        print(f"  Recall@{k:<9}{h_avg:>13.1f}%{f_avg:>13.1f}%{f_avg - h_avg:>+9.1f}%")

    # Latency comparison
    print(f"\n  {'Latency (ms)':<28}{'Mean':>10}{'Median':>10}{'P95':>10}")
    print(f"  {'─' * 58}")

    def stats(vals):
        s = sorted(vals)
        n = len(s)
        return sum(s) / n, s[n // 2], s[int(n * 0.95)]

    h_score = stats(hier_times_score)
    h_topk1 = stats(hier_times_topk1)
    h_sub = stats(hier_times_sub_score)
    h_topk2 = stats(hier_times_topk2)
    h_total = stats(hier_times_total)
    f_score = stats(flat_times_score)
    f_topk = stats(flat_times_topk)
    f_total = stats(flat_times_total)

    print(f"  Hier chunk_score (large) {h_score[0]:>10.3f}{h_score[1]:>10.3f}{h_score[2]:>10.3f}")
    print(f"  Hier topk1               {h_topk1[0]:>10.3f}{h_topk1[1]:>10.3f}{h_topk1[2]:>10.3f}")
    print(f"  Hier sub_chunk_score     {h_sub[0]:>10.3f}{h_sub[1]:>10.3f}{h_sub[2]:>10.3f}")
    print(f"  Hier topk2               {h_topk2[0]:>10.3f}{h_topk2[1]:>10.3f}{h_topk2[2]:>10.3f}")
    print(f"  Hier TOTAL               {h_total[0]:>10.3f}{h_total[1]:>10.3f}{h_total[2]:>10.3f}")
    print(f"  {'─' * 58}")
    print(f"  Flat chunk_score (all)   {f_score[0]:>10.3f}{f_score[1]:>10.3f}{f_score[2]:>10.3f}")
    print(f"  Flat topk                {f_topk[0]:>10.3f}{f_topk[1]:>10.3f}{f_topk[2]:>10.3f}")
    print(f"  Flat TOTAL               {f_total[0]:>10.3f}{f_total[1]:>10.3f}{f_total[2]:>10.3f}")
    print(f"  {'─' * 58}")
    speedup = f_total[0] / h_total[0] if h_total[0] > 0 else float('inf')
    print(f"\n  Speedup (Flat/Hier): {speedup:.2f}x")
    print(f"  Hier is {'faster' if speedup > 1 else 'slower'} by "
          f"{abs(f_total[0] - h_total[0]):.3f} ms on average")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    device = "cuda:0"
    if len(sys.argv) > 1:
        device = sys.argv[1]
    run_test(device=device)
