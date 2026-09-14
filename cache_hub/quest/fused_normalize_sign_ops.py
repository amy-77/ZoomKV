"""
Fused normalize + sign multiply CUDA op.

Replaces 3 PyTorch kernels (reduce_kernel, vectorized_elementwise, elementwise)
with a single fused kernel.

Usage:
    from cache_hub.quest.fused_normalize_sign_ops import fused_normalize_sign
    Y, q_norm = fused_normalize_sign(query, signs)
"""

import os
import warnings
import importlib.util
from functools import lru_cache
from typing import Any

import torch


@lru_cache()
def _prepare_for_load() -> str:
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="torch.utils.cpp_extension"
    )
    return os.path.dirname(os.path.abspath(__file__))


@lru_cache()
def _load_module() -> Any:
    from torch.utils.cpp_extension import load

    abs_path = _prepare_for_load()
    build_dir = f"{abs_path}/build"
    os.makedirs(build_dir, exist_ok=True)
    source_path = f"{abs_path}/fused_normalize_sign.cu"
    module_path = os.path.join(build_dir, "fused_normalize_sign_ext.so")

    if os.path.exists(module_path):
        module_mtime = os.path.getmtime(module_path)
        source_mtime = os.path.getmtime(source_path)
        if module_mtime >= source_mtime:
            spec = importlib.util.spec_from_file_location("fused_normalize_sign_ext", module_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

    return load(
        name="fused_normalize_sign_ext",
        sources=[source_path],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math"],
        build_directory=build_dir,
        verbose=True,
    )


def fused_normalize_sign(
    query: torch.Tensor,
    signs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused: Y = (query / ||query||) * signs

    Args:
        query: [bs, kv_heads, head_dim] bf16
        signs: [head_dim] bf16 (+1/-1 diagonal signs)

    Returns:
        Y:      [bs, kv_heads, head_dim] bf16
        q_norm: [bs, kv_heads, 1] float32
    """
    assert query.is_cuda, "query must be on CUDA"
    assert query.dim() == 3, "query must be 3D [bs, kv_heads, head_dim]"
    assert signs.dim() == 1, "signs must be 1D [head_dim]"

    kernel = _load_module()
    results = kernel.fused_normalize_sign(query.contiguous(), signs.contiguous())
    return results[0], results[1]
