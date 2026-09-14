"""Test fused_normalize_sign correctness and timing."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from cache_hub.quest.fused_normalize_sign_ops import fused_normalize_sign
from fast_hadamard_transform import hadamard_transform

torch.manual_seed(42)

bs, kv_heads, head_dim = 1, 8, 128
query = torch.randn(bs, kv_heads, head_dim, dtype=torch.bfloat16, device='cuda')
signs = (torch.randint(0, 2, (head_dim,), device='cuda', dtype=torch.bfloat16) * 2 - 1)

# Reference: 3 separate ops
q_norm_ref = torch.linalg.vector_norm(query.float(), ord=2, dim=-1, keepdim=True)
q_normalized_ref = query.float() / q_norm_ref.clamp(min=1e-8)
Y_ref = (q_normalized_ref * signs.float()).to(torch.bfloat16)

# Fused kernel
Y_fused, q_norm_fused = fused_normalize_sign(query, signs)

# Check correctness
norm_err = (q_norm_fused.float() - q_norm_ref.float()).abs().max().item()
y_err = (Y_fused.float() - Y_ref.float()).abs().max().item()
print(f"Norm max error: {norm_err:.6e}")
print(f"Y max error:    {y_err:.6e}")

assert norm_err < 1e-3, f"Norm error too large: {norm_err}"
assert y_err < 1e-2, f"Y error too large: {y_err}"
print("Correctness OK")

# Timing comparison
import time

torch.cuda.synchronize()

# Warmup
for _ in range(100):
    fused_normalize_sign(query, signs)
torch.cuda.synchronize()

# Fused timing
start = time.perf_counter()
for _ in range(1000):
    fused_normalize_sign(query, signs)
torch.cuda.synchronize()
fused_time = (time.perf_counter() - start) / 1000 * 1e6

# Reference timing (3 kernels)
for _ in range(100):
    q_norm_ref = torch.linalg.vector_norm(query, ord=2, dim=-1, keepdim=True)
    q_normalized_ref = query / q_norm_ref.clamp(min=1e-8)
    Y_ref = q_normalized_ref * signs
torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(1000):
    q_norm_ref = torch.linalg.vector_norm(query, ord=2, dim=-1, keepdim=True)
    q_normalized_ref = query / q_norm_ref.clamp(min=1e-8)
    Y_ref = q_normalized_ref * signs
torch.cuda.synchronize()
ref_time = (time.perf_counter() - start) / 1000 * 1e6

print(f"\nFused:     {fused_time:.1f} µs")
print(f"Reference: {ref_time:.1f} µs")
print(f"Speedup:   {ref_time / fused_time:.2f}x")
