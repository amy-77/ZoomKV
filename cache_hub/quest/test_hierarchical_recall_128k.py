"""
Hierarchical Quest single-sided recall test — real KV cache data.

Config:
  - ~130K tokens (real data from model inference)
  - chunk1 (large) = 256, chunk2 (small) = 16, factor = 16
  - Stage 1: keep top 40% of large chunks
  - Stage 2: keep top 30% of sub-chunks within selected large chunks
  - Measure: recall of true top-500 and top-100 tokens
"""

import math
import glob
import torch
import numpy as np
import sys
import os

DATA_DIR = "/data/qwang/q/thalia/kvcache/ParisKV3/ANNCache/results/chunk_analysis_data_20260515_134112"


def make_chunk_minmax(keys, chunk_size):
    """keys: [kv_len, head_dim] -> chunk_min/max: [n_chunks, head_dim]"""
    kv_len, head_dim = keys.shape
    n_chunks = kv_len // chunk_size
    chunked = keys[:n_chunks * chunk_size].reshape(n_chunks, chunk_size, head_dim)
    return chunked.min(dim=1).values, chunked.max(dim=1).values


def quest_score(q, chunk_min, chunk_max):
    """score = sum_d max(q_d * min_d, q_d * max_d). q: [d], chunk_min/max: [n, d]"""
    q_f = q.float().unsqueeze(0)  # [1, d]
    return torch.maximum(
        q_f * chunk_min.float(), q_f * chunk_max.float()
    ).sum(dim=-1)  # [n]


def run_test(device="cuda:0"):
    print("=" * 70)
    print("Hierarchical Quest Recall Test — Real KV Cache Data")
    print("  large_chunk=128, small_chunk=16, factor=8")
    print("  Stage 1: top 25% large chunks")
    print("  Stage 2: top 20% sub-chunks")
    print("  Metric: recall of true top-500 and top-100 tokens")
    print("=" * 70)

    small_chunk = 16
    large_chunk = 128
    factor = large_chunk // small_chunk  # 8
    large_ratio = 0.25
    small_ratio = 0.2
    topk_values = [500, 100]

    # Load all available key files
    key_files = sorted(glob.glob(f"{DATA_DIR}/keys_*.npz"))
    print(f"\nFound {len(key_files)} key files")

    all_results = {k: [] for k in topk_values}

    for kf in key_files:
        kdata = np.load(kf)
        keys_np = kdata['keys']
        layer = int(kdata['layer'])
        head = int(kdata['head'])
        kv_len = keys_np.shape[0]

        keys = torch.from_numpy(keys_np).to(device)  # [kv_len, head_dim]

        n_small = kv_len // small_chunk
        n_large = kv_len // large_chunk
        usable_len = n_large * large_chunk  # tokens that fit into full large chunks

        nk_large = max(1, int(math.ceil(n_large * large_ratio)))
        nk_sub_total = nk_large * factor
        nk_small = max(1, int(math.ceil(nk_sub_total * small_ratio)))

        # Compute chunk min/max
        small_min, small_max = make_chunk_minmax(keys[:usable_len], small_chunk)
        large_min = small_min.reshape(n_large, factor, -1).min(dim=1).values
        large_max = small_max.reshape(n_large, factor, -1).max(dim=1).values

        # Load corresponding queries
        pattern = f"query_L{layer}_H{head}_step*_kvlen{kv_len}.npz"
        query_files = sorted(glob.glob(f"{DATA_DIR}/{pattern}"))

        if not query_files:
            continue

        print(f"\n{'─' * 70}")
        print(f"Layer={layer}, Head={head}, kv_len={kv_len}")
        print(f"  n_large={n_large}, n_small(usable)={n_large*factor}")
        print(f"  nk_large(40%)={nk_large}, nk_sub_total={nk_sub_total}, nk_small(30%)={nk_small}")
        print(f"  Candidate tokens = {nk_small * small_chunk} ({nk_small * small_chunk / usable_len * 100:.1f}% of usable)")
        print(f"  Queries: {len(query_files)}")

        layer_results = {k: [] for k in topk_values}

        for qf in query_files:
            qdata = np.load(qf)
            q = torch.from_numpy(qdata['query']).to(device)  # [head_dim]

            # Ground truth: actual QK scores on usable tokens
            true_scores = (keys[:usable_len].float() @ q.float())  # [usable_len]

            # Stage 1: score large chunks → keep top 40%
            large_scores = quest_score(q, large_min, large_max)
            _, large_idx = torch.topk(large_scores, k=nk_large)

            # Stage 2: score sub-chunks within selected large chunks → keep top 30%
            sub_offsets = torch.arange(factor, device=device)
            all_small_indices = (large_idx.unsqueeze(-1) * factor + sub_offsets).reshape(-1)

            gathered_min = small_min[all_small_indices]
            gathered_max = small_max[all_small_indices]
            sub_scores = quest_score(q, gathered_min, gathered_max)

            _, sub_topk_pos = torch.topk(sub_scores, k=nk_small)
            selected_small_chunks = all_small_indices[sub_topk_pos]

            # Expand to token indices
            token_offsets = torch.arange(small_chunk, device=device)
            candidate_tokens = (selected_small_chunks.unsqueeze(-1) * small_chunk + token_offsets).reshape(-1)
            candidate_set = set(candidate_tokens.tolist())

            # Compute recall
            for k in topk_values:
                _, true_topk_idx = torch.topk(true_scores, k=k)
                true_set = set(true_topk_idx.tolist())
                recall = len(candidate_set & true_set) / k
                layer_results[k].append(recall)
                all_results[k].append(recall)

        # Print per-layer summary
        print(f"\n  {'Metric':<14}{'Mean':>8}{'Min':>8}{'Max':>8}{'Median':>8}")
        for k in topk_values:
            vals = layer_results[k]
            if vals:
                avg = sum(vals) / len(vals)
                mn = min(vals)
                mx = max(vals)
                md = sorted(vals)[len(vals) // 2]
                print(f"  Recall@{k:<5}{avg*100:>7.1f}%{mn*100:>7.1f}%{mx*100:>7.1f}%{md*100:>7.1f}%")

    # Overall summary
    print(f"\n{'=' * 70}")
    print("OVERALL SUMMARY (all layers, all queries)")
    print(f"  Total queries evaluated: {len(all_results[topk_values[0]])}")
    print(f"  Config: large_chunk=256, small_chunk=16")
    print(f"  Stage1=25%, Stage2=20%")
    print(f"{'─' * 70}")
    print(f"  {'Metric':<14}{'Mean':>8}{'Min':>8}{'Max':>8}{'Median':>8}")
    for k in topk_values:
        vals = all_results[k]
        if vals:
            avg = sum(vals) / len(vals)
            mn = min(vals)
            mx = max(vals)
            md = sorted(vals)[len(vals) // 2]
            print(f"  Recall@{k:<5}{avg*100:>7.1f}%{mn*100:>7.1f}%{mx*100:>7.1f}%{md*100:>7.1f}%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    device = "cuda:0"
    if len(sys.argv) > 1:
        device = sys.argv[1]
    run_test(device=device)
