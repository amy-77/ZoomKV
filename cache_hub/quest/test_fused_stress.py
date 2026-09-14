"""
Stress test for fused_hierarchical_quest with various batch sizes.
Tests boundary conditions: last large chunk with incomplete sub-chunks.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fused_hierarchical_quest_ops import fused_hierarchical_quest


def test_batch_sizes(device="cuda:0"):
    """Test with increasing batch sizes including edge cases."""
    factor = 8
    head_dim = 128
    kv_heads = 8

    # n_large=1000 so last large chunk maps to sub-chunks 7992..7999
    # but n_small = 1000*8 = 8000 which is exactly full — no boundary issue
    # To trigger boundary: use n_large where last chunk's sub-chunks go past n_small_max
    configs = [
        # (n_large, n_small_max, nk_large, nk_small, description)
        (100, 800, 25, 50, "normal, fully aligned"),
        (100, 795, 25, 50, "n_small_max < n_large*factor (boundary!)"),
        (200, 1590, 50, 100, "larger, boundary case"),
        (500, 3990, 125, 250, "large, boundary case"),
        (1000, 8000, 250, 500, "1k chunks, aligned"),
        (1000, 7980, 250, 500, "1k chunks, boundary"),
    ]

    batch_sizes = [1, 2, 4, 8, 16, 32]

    print("=" * 70)
    print("Fused Hierarchical Quest — Batch Size Stress Test")
    print(f"  kv_heads={kv_heads}, factor={factor}")
    print("=" * 70)

    for n_large, n_small_max, nk_large, nk_small, desc in configs:
        n_large_max = n_large
        print(f"\n{'─' * 70}")
        print(f"Config: {desc}")
        print(f"  n_large={n_large}, n_small_max={n_small_max}, "
              f"nk_large={nk_large}, nk_small={nk_small}")

        for bs in batch_sizes:
            try:
                q = torch.randn(bs, kv_heads, head_dim, dtype=torch.bfloat16, device=device)
                large_min = torch.randn(bs, kv_heads, n_large_max, head_dim, dtype=torch.bfloat16, device=device)
                large_max = torch.randn(bs, kv_heads, n_large_max, head_dim, dtype=torch.bfloat16, device=device)
                small_min = torch.randn(bs, kv_heads, n_small_max, head_dim, dtype=torch.bfloat16, device=device)
                small_max = torch.randn(bs, kv_heads, n_small_max, head_dim, dtype=torch.bfloat16, device=device)

                out = fused_hierarchical_quest(
                    q, large_min, large_max, small_min, small_max,
                    n_large, nk_large, nk_small, factor)

                torch.cuda.synchronize()

                # Validate output indices are in valid range
                max_idx = out.max().item()
                min_idx = out.min().item()
                valid = (min_idx >= 0 and max_idx < n_small_max)

                status = "OK" if valid else f"FAIL (idx range [{min_idx}, {max_idx}], max valid={n_small_max-1})"
                print(f"  bs={bs:3d}: {status}  out.shape={list(out.shape)}")

                if not valid:
                    print(f"    WARNING: output indices out of range!")

                del q, large_min, large_max, small_min, small_max, out
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  bs={bs:3d}: ERROR — {e}")
                torch.cuda.empty_cache()

    print(f"\n{'=' * 70}")
    print("Stress test complete.")
    print("=" * 70)


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    test_batch_sizes(device)
