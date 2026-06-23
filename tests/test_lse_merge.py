import argparse
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cache_hub.lse_merge import lse_merge_cuda, lse_merge_torch


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def check_operator(dtype: torch.dtype, bs: int, heads: int, dim: int) -> None:
    torch.manual_seed(0)
    attn_a = torch.randn(bs, 1, heads, dim, device="cuda", dtype=dtype)
    attn_b = torch.randn(bs, 1, heads, dim, device="cuda", dtype=dtype)
    lse_a = torch.randn(bs, heads, 1, device="cuda", dtype=torch.float32) * 3.0
    lse_b = torch.randn(bs, heads, 1, device="cuda", dtype=torch.float32) * 3.0

    ref = lse_merge_torch(attn_a, lse_a, attn_b, lse_b)
    got = lse_merge_cuda(attn_a, lse_a, attn_b, lse_b)
    torch.cuda.synchronize()

    max_abs = _max_abs(ref, got)
    print(f"[operator] dtype={dtype} max_abs={max_abs:.6g}")
    tol = 3e-3 if dtype in (torch.float16, torch.bfloat16) else 2e-6
    assert max_abs <= tol, f"dtype={dtype} max_abs={max_abs} > tol={tol}"


def check_flash_equivalence(dtype: torch.dtype, bs: int, q_heads: int, kv_heads: int, dim: int) -> None:
    from attn_hub.flash_attn_compat import flash_attn_with_kvcache_compat

    torch.manual_seed(1)
    len_a = 260
    len_b = 100
    q = torch.randn(bs, 1, q_heads, dim, device="cuda", dtype=dtype)
    k_a = torch.randn(bs, len_a, kv_heads, dim, device="cuda", dtype=dtype)
    v_a = torch.randn(bs, len_a, kv_heads, dim, device="cuda", dtype=dtype)
    k_b = torch.randn(bs, len_b, kv_heads, dim, device="cuda", dtype=dtype)
    v_b = torch.randn(bs, len_b, kv_heads, dim, device="cuda", dtype=dtype)

    full = flash_attn_with_kvcache_compat(
        q=q,
        k_cache=torch.cat([k_a, k_b], dim=1),
        v_cache=torch.cat([v_a, v_b], dim=1),
        causal=False,
    )
    out_a, lse_a = flash_attn_with_kvcache_compat(
        q=q,
        k_cache=k_a,
        v_cache=v_a,
        causal=False,
        return_softmax_lse=True,
    )
    out_b, lse_b = flash_attn_with_kvcache_compat(
        q=q,
        k_cache=k_b,
        v_cache=v_b,
        causal=False,
        return_softmax_lse=True,
    )

    merged_torch = lse_merge_torch(out_a, lse_a, out_b, lse_b)
    merged_cuda = lse_merge_cuda(out_a, lse_a, out_b, lse_b)
    torch.cuda.synchronize()

    print(f"[flash] torch_merge_vs_full max_abs={_max_abs(merged_torch, full):.6g}")
    print(f"[flash] cuda_merge_vs_full  max_abs={_max_abs(merged_cuda, full):.6g}")
    assert _max_abs(merged_torch, full) <= 6e-3
    assert _max_abs(merged_cuda, full) <= 6e-3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--flash", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        check_operator(dtype, args.bs, args.heads, args.dim)

    if args.flash:
        check_flash_equivalence(torch.bfloat16, args.bs, args.heads, args.kv_heads, args.dim)

    print("PASS")


if __name__ == "__main__":
    main()
