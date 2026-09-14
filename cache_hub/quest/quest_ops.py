"""
Quest Stage-1 CUDA ops: chunk-level upper bound scoring.

Usage:
    from cache_hub.quest.quest_ops import quest_chunk_score
    scores = quest_chunk_score(q, chunk_min, chunk_max)
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
def load_kernel_module(
    path: str = "quest_chunk_score.cu",
    name: str = "quest_ops",
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


def quest_chunk_score(
    q: torch.Tensor,
    chunk_min: torch.Tensor,
    chunk_max: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Quest chunk upper bound scores using a fused CUDA kernel.

    Args:
        q:         [bs, kv_heads, head_dim] - RoPE'd query (bf16/fp16/fp32)
        chunk_min: [bs, kv_heads, n_chunks, head_dim] - per-chunk per-dim key min
        chunk_max: [bs, kv_heads, n_chunks, head_dim] - per-chunk per-dim key max

    Returns:
        scores:    [bs, kv_heads, n_chunks] float32 - upper bound attention score per chunk
    """
    bs, kv_heads, head_dim = q.shape
    n_chunks = chunk_min.shape[2]

    scores = torch.empty(
        (bs, kv_heads, n_chunks),
        dtype=torch.float32,
        device=q.device,
    )

    kernel = load_kernel_module()
    kernel.quest_chunk_score(
        q.contiguous(),
        chunk_min.contiguous(),
        chunk_max.contiguous(),
        scores,
        n_chunks,
    )

    return scores
