"""
Hierarchical Quest filtering — multi-kernel CUDA wrapper.

Usage:
    from cache_hub.quest.hierarchical_quest_ops import hierarchical_quest_batch
    alive_bits = hierarchical_quest_batch(level_scores, stages, n_base, n_tokens)
"""

import os
import warnings
import importlib.util
from functools import lru_cache
from typing import Any, Iterable

import torch


BASE_CHUNK = 8


@lru_cache()
def _prepare_for_load() -> str:
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="torch.utils.cpp_extension"
    )
    return os.path.dirname(os.path.abspath(__file__))


@lru_cache()
def load_hierarchical_quest_module(
    path: str = "hierarchical_quest.cu",
    name: str = "hierarchical_quest_ext",
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


def hierarchical_quest_batch(
    level_scores: dict[int, torch.Tensor],
    stages: list[tuple[int, float]],
    n_base: int,
    n_tokens: int,
    alive_buf: torch.Tensor | None = None,
    group_buf: torch.Tensor | None = None,
    ws_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Batched hierarchical Quest filtering for all queries at once.

    Args:
        level_scores: {chunk_size: [n_queries, n_chunks]} precomputed bbox scores
        stages: [(chunk_size, ratio), ...] coarse-to-fine stage config
        n_base: number of base chunks (usable // BASE_CHUNK)
        n_tokens: total number of tokens
        alive_buf: optional pre-allocated [n_queries, alive_words] int32
        group_buf: optional pre-allocated [n_queries, max_group_words] int32
        ws_buf: optional pre-allocated [n_queries, 2 * max_groups] int32

    Returns:
        alive_bits: [n_queries, alive_words] int32 (uint32 bit array)
                    bit b of word w = whether base chunk (w*32+b) is alive.
                    Tokens beyond usable are implicitly alive.
    """
    score_list = [level_scores[cs].contiguous() for cs, _ in stages]
    chunk_sizes = [cs for cs, _ in stages]
    ratios = [r for _, r in stages]

    kernel = load_hierarchical_quest_module()
    return kernel.hierarchical_quest(
        score_list, chunk_sizes, ratios, n_base, n_tokens,
        alive_buf, group_buf, ws_buf)


def alive_bits_to_token_mask(
    alive_bits: torch.Tensor,
    n_base: int,
    n_tokens: int,
) -> torch.Tensor:
    """Expand compact alive_bits to dense [n_queries, n_tokens] bool mask."""
    Q = alive_bits.shape[0]
    device = alive_bits.device

    # Reinterpret int32 words as bytes (little-endian preserves bit order)
    flat = alive_bits.contiguous().view(torch.uint8).reshape(Q, -1)
    # Unpack each byte to 8 bits
    bits = flat.unsqueeze(-1).bitwise_right_shift(torch.arange(8, device=device, dtype=torch.uint8)).bitwise_and(1).bool()
    base_mask = bits.reshape(Q, -1)[:, :n_base]
    token_mask = base_mask.repeat_interleave(BASE_CHUNK, dim=1)

    if n_tokens > n_base * BASE_CHUNK:
        pad = torch.ones(Q, n_tokens - n_base * BASE_CHUNK,dtype=torch.bool, device=device)
        token_mask = torch.cat([token_mask, pad], dim=1)

    return token_mask
