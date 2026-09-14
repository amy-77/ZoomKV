"""
Fused Hierarchical Quest CUDA ops.

Single kernel that combines: large chunk scoring + top-K + sub-chunk scoring + top-K.
All intermediate results stay in shared memory.

Usage:
    from cache_hub.quest.fused_hierarchical_quest_ops import fused_hierarchical_quest
    chunk_indices = fused_hierarchical_quest(
        q, large_min, large_max, small_min, small_max,
        n_large, nk_large, nk_small, factor
    )
"""

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
def load_fused_hq_module(
    path: str = "fused_hierarchical_quest.cu",
    name: str = "fused_hq_ext",
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


def fused_hierarchical_quest(
    q: torch.Tensor,
    large_min: torch.Tensor,
    large_max: torch.Tensor,
    small_min: torch.Tensor,
    small_max: torch.Tensor,
    n_large: int,
    nk_large: int,
    nk_small: int,
    factor: int,
) -> torch.Tensor:
    """
    Fused hierarchical Quest: score large chunks → top-K → score sub-chunks → top-K.
    Single kernel launch, all intermediates in shared memory.

    Args:
        q:         [bs, kv_heads, 128] bf16
        large_min: [bs, kv_heads, n_large_max, 128] bf16
        large_max: [bs, kv_heads, n_large_max, 128] bf16
        small_min: [bs, kv_heads, n_small_max, 128] bf16
        small_max: [bs, kv_heads, n_small_max, 128] bf16
        n_large:   actual number of large chunks
        nk_large:  number of large chunks to select (stage 1)
        nk_small:  number of sub-chunks to output (stage 2)
        factor:    large_chunk_size // small_chunk_size

    Returns:
        indices: [bs, kv_heads, nk_small] int32 — global small chunk indices
    """
    bs, kv_heads, _ = q.shape
    n_large_max = large_min.shape[2]
    n_small_max = small_min.shape[2]

    out = torch.empty(
        (bs, kv_heads, nk_small),
        dtype=torch.int32,
        device=q.device,
    )

    kernel = load_fused_hq_module()
    kernel.fused_hierarchical_quest(
        q.contiguous(),
        large_min.contiguous(),
        large_max.contiguous(),
        small_min.contiguous(),
        small_max.contiguous(),
        out,
        n_large,
        n_large_max,
        n_small_max,
        nk_large,
        nk_small,
        factor,
    )

    return out
