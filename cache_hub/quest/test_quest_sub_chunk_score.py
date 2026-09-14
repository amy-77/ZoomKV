"""
Unit test for quest_sub_chunk_score_kernel — hierarchical Quest stage-2.

Simulates 128k tokens with large_chunk=256, small_chunk=32 (factor=8).
Verifies:
  1. Index mapping: large_id * factor + sub_offset → correct small chunk min/max
  2. Score accuracy: kernel output matches Python reference (exact for bf16 d128)
  3. Output layout: scores[bh, sel*factor + sub] placed correctly
  4. Two-level recall: hierarchical (30% large → 20% small) vs flat all-chunk scoring
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quest_ops import load_kernel_module


# ── Helpers ───────────────────────────────────────────────────────────────

def make_chunk_minmax(keys, chunk_size):
    """Compute per-chunk min/max from raw keys.

    keys: [bs, kv_heads, kv_len, head_dim] bf16
    Returns: chunk_min, chunk_max  [bs, kv_heads, n_chunks, head_dim] bf16
    """
    bs, kv_heads, kv_len, head_dim = keys.shape
    n_chunks = kv_len // chunk_size
    chunked = keys[:, :, :n_chunks * chunk_size].reshape(
        bs, kv_heads, n_chunks, chunk_size, head_dim
    )
    return chunked.min(dim=3).values, chunked.max(dim=3).values


def python_quest_score(q, chunk_min, chunk_max):
    """Python reference: score = Σ_d max(q_d * min_d, q_d * max_d)."""
    q_f = q.unsqueeze(2).float()
    return torch.maximum(
        q_f * chunk_min.float(), q_f * chunk_max.float()
    ).sum(dim=-1)


def python_sub_chunk_scores(q, small_min, small_max, large_ids, factor):
    """Python reference for sub-chunk scoring.

    For each selected large chunk, compute score for each of its sub-chunks.
    Returns: [bs, kv_heads, n_selected * factor] float32
    """
    bs, kv_heads, n_selected = large_ids.shape
    head_dim = q.shape[-1]

    sub_offsets = torch.arange(factor, device=q.device)
    # [bs, kv_heads, n_selected, factor]
    small_indices = large_ids.unsqueeze(-1) * factor + sub_offsets
    small_indices = small_indices.reshape(bs, kv_heads, n_selected * factor)

    idx_expand = small_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    gathered_min = torch.gather(small_min, 2, idx_expand)
    gathered_max = torch.gather(small_max, 2, idx_expand)

    q_f = q.unsqueeze(2).float()
    scores = torch.maximum(
        q_f * gathered_min.float(), q_f * gathered_max.float()
    ).sum(dim=-1)
    return scores


def run_cuda_sub_chunk(q, small_min, small_max, large_ids, n_selected, factor,
                       kernel):
    """Call quest_sub_chunk_score CUDA kernel."""
    bs, kv_heads, _ = q.shape
    scores = torch.empty(
        bs, kv_heads, n_selected * factor,
        dtype=torch.float32, device=q.device,
    )
    kernel.quest_sub_chunk_score(
        q.contiguous(),
        small_min.contiguous(),
        small_max.contiguous(),
        large_ids.contiguous(),
        scores,
        n_selected,
        factor,
    )
    torch.cuda.synchronize()
    return scores


# ── Tests ─────────────────────────────────────────────────────────────────

def test_index_and_score_128k(device, kernel):
    """128k tokens, large=256, small=32. Verify index mapping and scores."""
    print("\n" + "=" * 70)
    print("test_index_and_score_128k: 128k tokens, large=256, small=32, factor=8")
    print("=" * 70)

    bs, kv_heads, head_dim = 1, 4, 128
    kv_len = 131072
    small_chunk, large_chunk = 32, 256
    factor = large_chunk // small_chunk  # 8
    n_small = kv_len // small_chunk      # 4096
    n_large = kv_len // large_chunk      # 512

    torch.manual_seed(42)
    keys = torch.randn(bs, kv_heads, kv_len, head_dim,
                       dtype=torch.bfloat16, device=device)
    q = torch.randn(bs, kv_heads, head_dim,
                    dtype=torch.bfloat16, device=device)

    small_min, small_max = make_chunk_minmax(keys, small_chunk)
    print(f"  small_min/max shape: {list(small_min.shape)}")

    # Stage-1: score large chunks → select top 30%
    large_min = small_min.reshape(bs, kv_heads, n_large, factor, head_dim).min(dim=3).values
    large_max = small_max.reshape(bs, kv_heads, n_large, factor, head_dim).max(dim=3).values
    large_scores = python_quest_score(q, large_min, large_max)
    n_selected = max(1, int(n_large * 0.3))  # 154
    _, large_ids = torch.topk(large_scores, k=n_selected, dim=-1)
    large_ids = large_ids.to(torch.int64)
    print(f"  Stage-1: {n_large} large chunks → selected {n_selected} (30%)")
    print(f"  large_ids range: [{large_ids.min().item()}, {large_ids.max().item()}]")

    # Stage-2: CUDA sub-chunk scoring
    cuda_scores = run_cuda_sub_chunk(
        q, small_min, small_max, large_ids, n_selected, factor, kernel,
    )

    # Python reference
    ref_scores = python_sub_chunk_scores(q, small_min, small_max, large_ids, factor)

    # Compare
    diff = (cuda_scores - ref_scores).abs()
    max_diff = diff.max().item()
    match = torch.allclose(cuda_scores, ref_scores, atol=1e-5)

    print(f"  cuda_scores shape: {list(cuda_scores.shape)}")
    print(f"  max |cuda - ref|: {max_diff:.6e}")
    if not match:
        n_diff = (diff > 1e-5).sum().item()
        print(f"  {n_diff}/{cuda_scores.numel()} entries exceed atol=1e-5")

    # Spot check: verify a few specific index mappings
    print("  Spot-checking index mapping (first 3 selected large chunks):")
    for sel in range(min(3, n_selected)):
        for sub in range(min(2, factor)):
            lid = large_ids[0, 0, sel].item()
            expected_small = lid * factor + sub
            out_pos = sel * factor + sub
            ref_val = ref_scores[0, 0, out_pos].item()
            cuda_val = cuda_scores[0, 0, out_pos].item()
            ok = abs(ref_val - cuda_val) < 1e-5
            print(f"    large_id={lid} sub={sub} → small_chunk={expected_small}"
                  f"  ref={ref_val:.4f}  cuda={cuda_val:.4f}  {'OK' if ok else 'MISMATCH'}")

    print(f"[test_index_and_score_128k] {'PASSED' if match else 'FAILED'}")
    assert match


def test_score_exact_bf16_d128(device, kernel):
    """Exact score match for bf16 + head_dim=128 vectorized path."""
    print("\n" + "=" * 70)
    print("test_score_exact_bf16_d128: exact score match (vectorized path)")
    print("=" * 70)

    bs, kv_heads, head_dim = 1, 8, 128
    n_small = 1024
    factor = 8
    n_selected = 50

    torch.manual_seed(99)
    q = torch.randn(bs, kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    small_min = torch.randn(bs, kv_heads, n_small, head_dim,
                            dtype=torch.bfloat16, device=device)
    small_max = torch.randn(bs, kv_heads, n_small, head_dim,
                            dtype=torch.bfloat16, device=device)
    # Ensure min <= max per element
    real_min = torch.minimum(small_min, small_max)
    real_max = torch.maximum(small_min, small_max)
    small_min, small_max = real_min, real_max

    n_large = n_small // factor
    large_ids = torch.stack([
        torch.randperm(n_large, device=device)[:n_selected]
        for _ in range(bs * kv_heads)
    ]).reshape(bs, kv_heads, n_selected).to(torch.int64)

    cuda_scores = run_cuda_sub_chunk(
        q, small_min, small_max, large_ids, n_selected, factor, kernel,
    )
    ref_scores = python_sub_chunk_scores(q, small_min, small_max, large_ids, factor)

    diff = (cuda_scores - ref_scores).abs()
    max_diff = diff.max().item()
    # Warp-shuffle tree reduction vs PyTorch .sum() differ in float32 accumulation
    # order, so allow small tolerance instead of bit-exact.
    match = torch.allclose(cuda_scores, ref_scores, atol=1e-4)

    n_diff = (diff > 1e-4).sum().item()
    print(f"  max |cuda - ref|: {max_diff:.6e}  ({n_diff}/{cuda_scores.numel()} exceed 1e-4)")

    print(f"[test_score_exact_bf16_d128] {'PASSED' if match else 'FAILED'}")
    assert match


def test_output_layout(device, kernel):
    """Verify output[sel*factor + sub] maps to large_ids[sel] * factor + sub."""
    print("\n" + "=" * 70)
    print("test_output_layout: verify output placement order")
    print("=" * 70)

    bs, kv_heads, head_dim = 1, 2, 128
    factor = 8

    # Make each small chunk have a unique, identifiable score by setting
    # chunk_min = chunk_max = constant per chunk (so score = q · chunk_val)
    n_small = 256
    n_large = n_small // factor
    n_selected = 10

    torch.manual_seed(77)
    q = torch.randn(bs, kv_heads, head_dim, dtype=torch.bfloat16, device=device)

    # Each small chunk i gets constant value i/n_small across all dims
    chunk_vals = torch.arange(n_small, device=device, dtype=torch.bfloat16
                              ).reshape(1, 1, n_small, 1).expand(bs, kv_heads, -1, head_dim)
    small_min = chunk_vals.clone()
    small_max = chunk_vals.clone()

    # Select specific large chunks: [0, 5, 10, 15, 20, 25, 30, 1, 2, 3]
    large_ids = torch.tensor(
        [[0, 5, 10, 15, 20, 25, 30, 1, 2, 3]],
        device=device, dtype=torch.int64
    ).unsqueeze(0).expand(bs, kv_heads, -1).contiguous()

    cuda_scores = run_cuda_sub_chunk(
        q, small_min, small_max, large_ids, n_selected, factor, kernel,
    )
    ref_scores = python_sub_chunk_scores(q, small_min, small_max, large_ids, factor)

    all_ok = True
    for sel in range(n_selected):
        lid = large_ids[0, 0, sel].item()
        for sub in range(factor):
            out_pos = sel * factor + sub
            expected_small_idx = lid * factor + sub
            cuda_val = cuda_scores[0, 0, out_pos].item()
            ref_val = ref_scores[0, 0, out_pos].item()
            # Large-magnitude scores (hundreds) → allow 1e-3 absolute tolerance
            if abs(cuda_val - ref_val) > 1e-3:
                print(f"  MISMATCH: sel={sel} sub={sub} "
                      f"large_id={lid} small={expected_small_idx} "
                      f"cuda={cuda_val:.4f} ref={ref_val:.4f}")
                all_ok = False

    match = torch.allclose(cuda_scores, ref_scores, atol=1e-3)
    print(f"  Checked {n_selected * factor} output positions")
    print(f"[test_output_layout] {'PASSED' if all_ok and match else 'FAILED'}")
    assert all_ok and match


def test_two_level_recall(device, kernel):
    """Recall of hierarchical (30% large → 20% small) vs flat all-chunk scoring.

    128k tokens, large=256, small=32. Compare which small chunks are selected
    by the two-level approach vs scoring all small chunks directly.
    """
    print("\n" + "=" * 70)
    print("test_two_level_recall: hierarchical vs flat recall")
    print("=" * 70)

    bs, kv_heads, head_dim = 1, 4, 128
    kv_len = 131072
    small_chunk, large_chunk = 32, 256
    factor = large_chunk // small_chunk
    n_small = kv_len // small_chunk
    n_large = kv_len // large_chunk

    torch.manual_seed(123)
    keys = torch.randn(bs, kv_heads, kv_len, head_dim,
                       dtype=torch.bfloat16, device=device)
    q = torch.randn(bs, kv_heads, head_dim,
                    dtype=torch.bfloat16, device=device)

    small_min, small_max = make_chunk_minmax(keys, small_chunk)

    # --- Flat: score all small chunks, select top 20% ---
    flat_scores = python_quest_score(q, small_min, small_max)
    n_final = max(1, int(n_small * 0.2))  # 819
    _, flat_topk = torch.topk(flat_scores, k=n_final, dim=-1)

    # --- Hierarchical: large chunks (30%) → sub-chunk score → top 20% ---
    large_min = small_min.reshape(bs, kv_heads, n_large, factor, head_dim).min(dim=3).values
    large_max = small_max.reshape(bs, kv_heads, n_large, factor, head_dim).max(dim=3).values
    large_scores = python_quest_score(q, large_min, large_max)
    n_selected = max(1, int(n_large * 0.3))
    _, large_ids = torch.topk(large_scores, k=n_selected, dim=-1)
    large_ids = large_ids.to(torch.int64)

    # Sub-chunk scoring via kernel
    cuda_sub_scores = run_cuda_sub_chunk(
        q, small_min, small_max, large_ids, n_selected, factor, kernel,
    )

    # Select top n_final from sub-chunk scores
    n_sub_total = n_selected * factor
    actual_final = min(n_final, n_sub_total)
    _, sub_topk_local = torch.topk(cuda_sub_scores, k=actual_final, dim=-1)

    # Convert local sub-chunk positions → global small chunk indices
    sub_offsets = torch.arange(factor, device=device)
    all_small_indices = large_ids.unsqueeze(-1) * factor + sub_offsets
    all_small_indices = all_small_indices.reshape(bs, kv_heads, n_sub_total)
    hierarchical_topk = torch.gather(all_small_indices, 2, sub_topk_local)

    # Compute per-head recall
    recalls = []
    for h in range(kv_heads):
        flat_set = set(flat_topk[0, h].tolist())
        hier_set = set(hierarchical_topk[0, h].tolist())
        recall = len(flat_set & hier_set) / len(flat_set) if flat_set else 1.0
        recalls.append(recall)
        print(f"  Head {h}: recall = {recall*100:.1f}%  "
              f"(flat={len(flat_set)}, hier={len(hier_set)}, "
              f"overlap={len(flat_set & hier_set)})")

    avg_recall = sum(recalls) / len(recalls)
    print(f"  Average recall: {avg_recall*100:.1f}%")

    # With 30% large chunk selection, candidate pool is ~30% of all small chunks.
    # Recall is bounded by pool coverage; 35% is a reasonable floor for random data.
    ok = avg_recall >= 0.35
    print(f"[test_two_level_recall] {'PASSED' if ok else 'FAILED'}"
          f" (threshold: >=35%)")
    assert ok, f"avg recall {avg_recall*100:.1f}% < 35%"


def test_boundary_large_id(device, kernel):
    """Edge case: large_id at the last valid position."""
    print("\n" + "=" * 70)
    print("test_boundary_large_id: last valid large chunk")
    print("=" * 70)

    bs, kv_heads, head_dim = 1, 2, 128
    factor = 8
    n_small = 512
    n_large = n_small // factor  # 64
    n_selected = 3

    torch.manual_seed(200)
    q = torch.randn(bs, kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    small_min = torch.randn(bs, kv_heads, n_small, head_dim,
                            dtype=torch.bfloat16, device=device)
    small_max = torch.randn(bs, kv_heads, n_small, head_dim,
                            dtype=torch.bfloat16, device=device)
    real_min = torch.minimum(small_min, small_max)
    real_max = torch.maximum(small_min, small_max)
    small_min, small_max = real_min, real_max

    # Include the last valid large chunk (63) and first (0)
    large_ids = torch.tensor(
        [[[0, n_large - 1, n_large // 2]] * kv_heads],
        device=device, dtype=torch.int64
    )

    cuda_scores = run_cuda_sub_chunk(
        q, small_min, small_max, large_ids, n_selected, factor, kernel,
    )
    ref_scores = python_sub_chunk_scores(q, small_min, small_max, large_ids, factor)

    diff = (cuda_scores - ref_scores).abs()
    max_diff = diff.max().item()
    match = max_diff < 1e-5

    # Verify the last large chunk's sub-chunks
    last_large = n_large - 1
    last_sub = last_large * factor + (factor - 1)
    print(f"  Last large chunk: {last_large}")
    print(f"  Its last sub-chunk global index: {last_sub} (should be < {n_small})")
    print(f"  max |cuda - ref|: {max_diff:.6e}")

    ok = match and last_sub < n_small
    print(f"[test_boundary_large_id] {'PASSED' if ok else 'FAILED'}")
    assert ok


def test_multi_batch_head(device, kernel):
    """Multiple heads with distinct data to catch bh indexing bugs."""
    print("\n" + "=" * 70)
    print("test_multi_batch_head: 1 batch, 8 heads, distinct per-head data")
    print("=" * 70)

    bs, kv_heads, head_dim = 1, 8, 128
    factor = 8
    n_small = 2048
    n_large = n_small // factor
    n_selected = 30

    torch.manual_seed(456)
    q = torch.randn(bs, kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    small_min = torch.randn(bs, kv_heads, n_small, head_dim,
                            dtype=torch.bfloat16, device=device)
    small_max = small_min + torch.rand(bs, kv_heads, n_small, head_dim,
                                       dtype=torch.bfloat16, device=device).abs()

    # Each head selects different large chunks
    large_ids = torch.stack([
        torch.randperm(n_large, device=device)[:n_selected]
        for _ in range(bs * kv_heads)
    ]).reshape(bs, kv_heads, n_selected).to(torch.int64)

    cuda_scores = run_cuda_sub_chunk(
        q, small_min, small_max, large_ids, n_selected, factor, kernel,
    )
    ref_scores = python_sub_chunk_scores(q, small_min, small_max, large_ids, factor)

    per_head_ok = []
    for h in range(kv_heads):
        diff = (cuda_scores[0, h] - ref_scores[0, h]).abs()
        max_diff = diff.max().item()
        ok = max_diff < 1e-5
        per_head_ok.append(ok)
        if not ok:
            n = (diff > 1e-5).sum().item()
            print(f"  Head {h}: FAIL  max_diff={max_diff:.6e}  {n} values differ")

    all_ok = all(per_head_ok)
    if all_ok:
        print(f"  All {kv_heads} heads match exactly")
    print(f"[test_multi_batch_head] {'PASSED' if all_ok else 'FAILED'}")
    assert all_ok


def test_hierarchical_recall_pipeline(device, kernel):
    """End-to-end recall of _quest_pick_coarse_indices_hierarchical.

    Replicates the exact production pipeline:
      Stage 1: quest_chunk_score on large chunks → topk (large_ratio=0.3)
      Stage 2: quest_sub_chunk_score on sub-chunks → topk (small_ratio=0.3)
      Map-back: sub_topk_pos // factor, % factor, gather → global small chunk idx

    Baseline: flat quest_chunk_score on ALL small chunks → topk (same nk_small).

    Config: 128k tokens, large_chunk=256, small_chunk=32, factor=8.
    """
    import math

    print("\n" + "=" * 70)
    print("test_hierarchical_recall_pipeline")
    print("  128k tokens, large=256, small=32, factor=8")
    print("  large_ratio=0.3, small_ratio=0.3")
    print("=" * 70)

    bs, kv_heads, head_dim = 1, 4, 128
    kv_len = 131072
    small_chunk, large_chunk = 32, 256
    factor = large_chunk // small_chunk   # 8
    large_ratio, small_ratio = 0.3, 0.3

    n_small = kv_len // small_chunk       # 4096
    n_large = n_small // factor           # 512
    nk_large = min(max(1, int(math.ceil(n_large * large_ratio))), n_large)  # 154
    nk_sub_total = nk_large * factor      # 1232
    nk_small = min(max(1, int(math.ceil(nk_sub_total * small_ratio))), nk_sub_total)  # 370

    print(f"  n_small={n_small}, n_large={n_large}")
    print(f"  nk_large={nk_large}, nk_sub_total={nk_sub_total}, nk_small={nk_small}")

    torch.manual_seed(42)
    keys = torch.randn(bs, kv_heads, kv_len, head_dim,
                       dtype=torch.bfloat16, device=device)
    q = torch.randn(bs, kv_heads, head_dim,
                    dtype=torch.bfloat16, device=device)

    # ── Compute small chunk min/max ──
    small_min, small_max = make_chunk_minmax(keys, small_chunk)

    # ── Compute large chunk min/max from small chunk min/max ──
    large_min = small_min.reshape(bs, kv_heads, n_large, factor, head_dim).min(dim=3).values
    large_max = small_max.reshape(bs, kv_heads, n_large, factor, head_dim).max(dim=3).values

    # ══════════════════════════════════════════════════════════════════
    # Hierarchical pipeline (replicating _quest_pick_coarse_indices_hierarchical)
    # ══════════════════════════════════════════════════════════════════

    # Stage 1: score all large chunks → select top nk_large
    large_scores = torch.empty(bs, kv_heads, n_large,
                               dtype=torch.float32, device=device)
    kernel.quest_chunk_score(q, large_min, large_max, large_scores, n_large)
    _, large_idx = torch.topk(large_scores, k=nk_large, dim=-1)
    large_idx = large_idx.to(torch.int64).clamp(min=0, max=n_large - 1)

    # Stage 2: score sub-chunks within selected large chunks
    sub_scores = torch.empty(bs, kv_heads, nk_sub_total,
                             dtype=torch.float32, device=device)
    kernel.quest_sub_chunk_score(
        q, small_min, small_max,
        large_idx, sub_scores,
        nk_large, factor,
    )
    _, sub_topk_pos = torch.topk(sub_scores, k=nk_small, dim=-1)

    # Map back to global small chunk indices
    # (exact same logic as _quest_pick_coarse_indices_hierarchical)
    large_sel = sub_topk_pos // factor
    sub_off = sub_topk_pos % factor
    hier_chunk_idx = torch.gather(large_idx, 2, large_sel) * factor + sub_off
    hier_chunk_idx = hier_chunk_idx.clamp(min=0, max=n_small - 1)

    # ══════════════════════════════════════════════════════════════════
    # Flat baseline: score ALL small chunks → select top nk_small
    # ══════════════════════════════════════════════════════════════════
    flat_scores = torch.empty(bs, kv_heads, n_small,
                              dtype=torch.float32, device=device)
    kernel.quest_chunk_score(q, small_min, small_max, flat_scores, n_small)
    _, flat_topk = torch.topk(flat_scores, k=nk_small, dim=-1)

    # ══════════════════════════════════════════════════════════════════
    # Recall measurement
    # ══════════════════════════════════════════════════════════════════
    recalls = []
    for h in range(kv_heads):
        flat_set = set(flat_topk[0, h].tolist())
        hier_set = set(hier_chunk_idx[0, h].tolist())
        overlap = len(flat_set & hier_set)
        recall = overlap / len(flat_set) if flat_set else 1.0
        recalls.append(recall)
        print(f"  Head {h}: recall = {recall*100:.1f}%  "
              f"(flat={len(flat_set)}, hier={len(hier_set)}, overlap={overlap})")

    avg_recall = sum(recalls) / len(recalls)
    print(f"  ── Average recall: {avg_recall*100:.1f}% ──")
    print(f"  (theoretical random baseline: "
          f"{nk_sub_total}/{n_small} = {nk_sub_total/n_small*100:.1f}%)")

    # Hierarchical should substantially beat random
    random_baseline = nk_sub_total / n_small
    ok = avg_recall > random_baseline * 1.2
    print(f"[test_hierarchical_recall_pipeline] {'PASSED' if ok else 'FAILED'}"
          f" (must beat random×1.2 = {random_baseline*1.2*100:.1f}%)")
    assert ok


# ── main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda:0"
    if len(sys.argv) > 1:
        device = sys.argv[1]

    print(f"Using device: {device}")
    print("Compiling quest_chunk_score kernel …")
    kernel = load_kernel_module("quest_chunk_score.cu", "quest_ops")
    print("Kernel compiled.\n")

    tests = [
        test_index_and_score_128k,
        test_score_exact_bf16_d128,
        test_output_layout,
        test_two_level_recall,
        test_boundary_large_id,
        test_multi_batch_head,
        test_hierarchical_recall_pipeline,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t(device, kernel)
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  !! {t.__name__} FAILED: {e}")
        except Exception as e:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"  !! {t.__name__} ERROR: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        sys.exit(1)
