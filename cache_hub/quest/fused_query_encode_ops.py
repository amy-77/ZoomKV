"""Python wrapper for the fused query encode kernel.

Fuses normalize + sign multiply + FWHT (Hadamard) + reshape into ONE CUDA
kernel launch, replacing the previous 3-step pipeline:

    Y, q_norm = fused_normalize_sign(query, signs)   # kernel 1
    Y = hadamard_transform(Y, scale=...)             # kernel 2
    q_blocks = Y.view(bs, kv_heads, B, m).contiguous()

Saves ~8-12 μs per call (one kernel launch + internal sync), totalling
~30-45 ms across one decode pass.

Usage:
    from cache_hub.quest.fused_query_encode_ops import fused_query_encode
    q_blocks, q_norm = fused_query_encode(query, signs, B=16, m=8)
"""

from __future__ import annotations

import math
import os
import warnings
import importlib.util
from functools import lru_cache
from typing import Any, Iterable

import torch


@lru_cache()
def _prepare_for_load() -> str:
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="torch.utils.cpp_extension"
    )
    return os.path.dirname(os.path.abspath(__file__))


@lru_cache()
def load_kernel_module(
    path: str = "fused_query_encode.cu",
    name: str = "fused_query_encode_ext",
    *,
    build: str = "build",
    cflags: Iterable[str] | None = None,
    cuda_flags: Iterable[str] | None = None,
) -> Any:
    from torch.utils.cpp_extension import load

    abs_path = _prepare_for_load()
    build_dir = f"{abs_path}/{build}"
    os.makedirs(build_dir, exist_ok=True)
    source_paths = [f"{abs_path}/{path}"]
    module_path = os.path.join(build_dir, f"{name}.so")

    if os.path.exists(module_path):
        module_mtime = os.path.getmtime(module_path)
        latest_source_mtime = max(os.path.getmtime(src) for src in source_paths)
        if module_mtime >= latest_source_mtime:
            spec = importlib.util.spec_from_file_location(name, module_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

    return load(
        name=name,
        sources=source_paths,
        extra_cflags=list(cflags or []) or ["-O3", "-std=c++17"],
        extra_cuda_cflags=list(cuda_flags or []) or ["-O3", "-std=c++17", "--use_fast_math"],
        build_directory=build_dir,
        verbose=True,
    )


def fused_query_encode(
    query: torch.Tensor,
    signs: torch.Tensor,
    B: int = 16,
    m: int = 8,
    *,
    q_blocks_out: torch.Tensor | None = None,
    q_norm_out: torch.Tensor | None = None,
    kernel: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused normalize + sign multiply + FWHT for query encoding.

    Equivalent to the 3-step pipeline:
        Y, q_norm = fused_normalize_sign(query, signs)
        Y         = hadamard_transform(Y, scale=1/sqrt(head_dim))
        q_blocks  = Y.view(bs, kv_heads, B, m)

    but in a single kernel launch with FWHT done inline (strides 1..16 via
    __shfl_xor_sync within warps, strides 32 and 64 via shared memory).

    Args:
        query: [bs, kv_heads, head_dim] bf16 — raw query (un-normalized).
        signs: [head_dim] bf16 — diagonal signs for SRHT.
        B, m:  block decomposition; must satisfy B * m == 128.
        q_blocks_out, q_norm_out: optional pre-allocated output buffers to
            avoid per-call allocation. Shapes/dtypes must match the docs.
        kernel: pre-loaded kernel module (from ``load_kernel_module()``).

    Returns:
        q_blocks: [bs, kv_heads, B, m] bf16 — encoded query blocks.
        q_norm:   [bs, kv_heads, 1] fp32 — L2 norm of the raw query.
    """
    assert query.is_cuda, "query must be on CUDA"
    assert signs.is_cuda, "signs must be on CUDA"
    bs, kv_heads, head_dim = query.shape
    assert head_dim == B * m == 128, (
        f"head_dim must be 128 (got {head_dim}); B*m={B*m}"
    )

    if kernel is None:
        kernel = load_kernel_module()

    query = query.contiguous()
    signs = signs.contiguous()

    if q_blocks_out is None:
        q_blocks_flat = torch.empty(
            bs, kv_heads, head_dim, dtype=torch.bfloat16, device=query.device,
        )
    else:
        assert q_blocks_out.dtype == torch.bfloat16
        assert q_blocks_out.shape in ((bs, kv_heads, head_dim),
                                      (bs, kv_heads, B, m))
        q_blocks_flat = q_blocks_out.view(bs, kv_heads, head_dim)

    if q_norm_out is None:
        q_norm_flat = torch.empty(
            bs, kv_heads, dtype=torch.float32, device=query.device,
        )
    else:
        assert q_norm_out.dtype == torch.float32
        # Accept either [bs, kv, 1] or [bs, kv]
        q_norm_flat = q_norm_out.view(bs, kv_heads)

    hadamard_scale = 1.0 / math.sqrt(head_dim)
    kernel.fused_query_encode(query, signs, q_blocks_flat, q_norm_flat, hadamard_scale)

    q_blocks = q_blocks_flat.view(bs, kv_heads, B, m)
    q_norm = q_norm_flat.unsqueeze(-1)  # [bs, kv_heads, 1]
    return q_blocks, q_norm
