"""
Quest chunk score CUDA kernel 单元测试

测试 quest_chunk_score CUDA kernel 是否返回正确的结果。
对比 CUDA kernel 输出和 PyTorch 参考实现。
"""

import torch
import sys
import os

# 添加路径以便导入 quest_ops
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quest_ops import quest_chunk_score


def quest_chunk_score_reference(
    q: torch.Tensor,
    chunk_min: torch.Tensor,
    chunk_max: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch 参考实现：score(chunk) = Σ_d max(q_d * min_d, q_d * max_d)

    Args:
        q:         [bs, kv_heads, head_dim]
        chunk_min: [bs, kv_heads, n_chunks, head_dim]
        chunk_max: [bs, kv_heads, n_chunks, head_dim]

    Returns:
        scores:    [bs, kv_heads, n_chunks] float32
    """
    q_expanded = q.unsqueeze(2).float()  # [bs, kv_heads, 1, head_dim]
    scores = torch.maximum(
        q_expanded * chunk_min.float(),
        q_expanded * chunk_max.float()
    ).sum(dim=-1)  # [bs, kv_heads, n_chunks]
    return scores.to(torch.float32)


def test_quest_chunk_score_basic():
    """基础测试：小规模数据"""
    print("\n" + "="*80)
    print("测试 1: 基础功能测试 (小规模)")
    print("="*80)

    bs, kv_heads, head_dim = 1, 2, 128
    n_chunks = 10

    # 构造输入
    torch.manual_seed(42)
    q = torch.randn(bs, kv_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    chunk_min = torch.randn(bs, kv_heads, n_chunks, head_dim, dtype=torch.bfloat16, device='cuda')
    chunk_max = torch.randn(bs, kv_heads, n_chunks, head_dim, dtype=torch.bfloat16, device='cuda')

    # 确保 min <= max
    chunk_min, chunk_max = torch.minimum(chunk_min, chunk_max), torch.maximum(chunk_min, chunk_max)

    # CUDA kernel
    scores_cuda = quest_chunk_score(q, chunk_min, chunk_max)

    # PyTorch 参考
    scores_ref = quest_chunk_score_reference(q, chunk_min, chunk_max)

    # 比较
    max_diff = (scores_cuda - scores_ref).abs().max().item()
    mean_diff = (scores_cuda - scores_ref).abs().mean().item()

    print(f"输入形状: q={q.shape}, chunk_min={chunk_min.shape}, chunk_max={chunk_max.shape}")
    print(f"输出形状: scores_cuda={scores_cuda.shape}, scores_ref={scores_ref.shape}")
    print(f"最大差异: {max_diff:.6f}")
    print(f"平均差异: {mean_diff:.6f}")
    print(f"CUDA scores[:5]: {scores_cuda[0, 0, :5].cpu().tolist()}")
    print(f"Ref  scores[:5]: {scores_ref[0, 0, :5].cpu().tolist()}")

    # 验证
    tolerance = 1e-2  # bf16 精度较低
    assert max_diff < tolerance, f"最大差异 {max_diff} 超过容差 {tolerance}"
    print("✓ 测试通过")


def test_quest_chunk_score_large_scale():
    """大规模测试：接近实际使用场景"""
    print("\n" + "="*80)
    print("测试 2: 大规模测试 (接近实际场景)")
    print("="*80)

    bs, kv_heads, head_dim = 1, 8, 128
    n_chunks = 4743  # 37951 tokens / 8 chunk_size

    # 构造输入
    torch.manual_seed(123)
    q = torch.randn(bs, kv_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    chunk_min = torch.randn(bs, kv_heads, n_chunks, head_dim, dtype=torch.bfloat16, device='cuda') * 2
    chunk_max = torch.randn(bs, kv_heads, n_chunks, head_dim, dtype=torch.bfloat16, device='cuda') * 2

    # 确保 min <= max
    chunk_min, chunk_max = torch.minimum(chunk_min, chunk_max), torch.maximum(chunk_min, chunk_max)

    # CUDA kernel
    scores_cuda = quest_chunk_score(q, chunk_min, chunk_max)

    # PyTorch 参考
    scores_ref = quest_chunk_score_reference(q, chunk_min, chunk_max)

    # 比较
    max_diff = (scores_cuda - scores_ref).abs().max().item()
    mean_diff = (scores_cuda - scores_ref).abs().mean().item()
    rel_diff = ((scores_cuda - scores_ref).abs() / (scores_ref.abs() + 1e-6)).max().item()

    print(f"输入形状: q={q.shape}, chunk_min={chunk_min.shape}, chunk_max={chunk_max.shape}")
    print(f"输出形状: scores_cuda={scores_cuda.shape}, scores_ref={scores_ref.shape}")
    print(f"最大绝对差异: {max_diff:.6f}")
    print(f"平均绝对差异: {mean_diff:.6f}")
    print(f"最大相对差异: {rel_diff:.6f}")
    print(f"CUDA scores[:10]: {scores_cuda[0, 0, :10].cpu().tolist()}")
    print(f"Ref  scores[:10]: {scores_ref[0, 0, :10].cpu().tolist()}")

    # Top-k 一致性检查
    k = 100
    _, topk_cuda = torch.topk(scores_cuda[0, 0], k=k)
    _, topk_ref = torch.topk(scores_ref[0, 0], k=k)
    topk_overlap = len(set(topk_cuda.cpu().tolist()) & set(topk_ref.cpu().tolist()))
    print(f"Top-{k} 重叠率: {topk_overlap}/{k} ({topk_overlap/k*100:.1f}%)")

    # 验证
    tolerance = 1e-2
    assert max_diff < tolerance, f"最大差异 {max_diff} 超过容差 {tolerance}"
    assert topk_overlap >= k * 0.95, f"Top-{k} 重叠率 {topk_overlap/k*100:.1f}% 低于 95%"
    print("✓ 测试通过")


def test_quest_chunk_score_edge_cases():
    """边界情况测试"""
    print("\n" + "="*80)
    print("测试 3: 边界情况")
    print("="*80)

    bs, kv_heads, head_dim = 1, 4, 128

    # 测试 3.1: 单个 chunk
    print("\n[3.1] 单个 chunk")
    n_chunks = 1
    q = torch.randn(bs, kv_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    chunk_min = torch.randn(bs, kv_heads, n_chunks, head_dim, dtype=torch.bfloat16, device='cuda')
    chunk_max = chunk_min + torch.rand_like(chunk_min)

    scores_cuda = quest_chunk_score(q, chunk_min, chunk_max)
    scores_ref = quest_chunk_score_reference(q, chunk_min, chunk_max)
    max_diff = (scores_cuda - scores_ref).abs().max().item()
    print(f"  n_chunks={n_chunks}, max_diff={max_diff:.6f}")
    assert max_diff < 1e-2
    print("  ✓ 通过")

    # 测试 3.2: 全零 query
    print("\n[3.2] 全零 query")
    n_chunks = 100
    q = torch.zeros(bs, kv_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    chunk_min = torch.randn(bs, kv_heads, n_chunks, head_dim, dtype=torch.bfloat16, device='cuda')
    chunk_max = chunk_min + torch.rand_like(chunk_min)

    scores_cuda = quest_chunk_score(q, chunk_min, chunk_max)
    scores_ref = quest_chunk_score_reference(q, chunk_min, chunk_max)
    max_diff = (scores_cuda - scores_ref).abs().max().item()
    print(f"  全零 query, max_diff={max_diff:.6f}")
    assert max_diff < 1e-6
    assert scores_cuda.abs().max().item() < 1e-6, "全零 query 应该产生全零 scores"
    print("  ✓ 通过")

    # 测试 3.3: min == max (退化为精确内积)
    print("\n[3.3] min == max (chunk 内所有 key 相同)")
    n_chunks = 50
    q = torch.randn(bs, kv_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    chunk_min = torch.randn(bs, kv_heads, n_chunks, head_dim, dtype=torch.bfloat16, device='cuda')
    chunk_max = chunk_min.clone()

    scores_cuda = quest_chunk_score(q, chunk_min, chunk_max)
    scores_ref = quest_chunk_score_reference(q, chunk_min, chunk_max)
    max_diff = (scores_cuda - scores_ref).abs().max().item()
    print(f"  min==max, max_diff={max_diff:.6f}")
    assert max_diff < 1e-2
    print("  ✓ 通过")

    # 测试 3.4: 负值 query 和 chunk
    print("\n[3.4] 负值 query 和 chunk")
    n_chunks = 100
    q = -torch.rand(bs, kv_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    chunk_min = -torch.rand(bs, kv_heads, n_chunks, head_dim, dtype=torch.bfloat16, device='cuda') * 5
    chunk_max = -torch.rand(bs, kv_heads, n_chunks, head_dim, dtype=torch.bfloat16, device='cuda')
    chunk_min, chunk_max = torch.minimum(chunk_min, chunk_max), torch.maximum(chunk_min, chunk_max)

    scores_cuda = quest_chunk_score(q, chunk_min, chunk_max)
    scores_ref = quest_chunk_score_reference(q, chunk_min, chunk_max)
    max_diff = (scores_cuda - scores_ref).abs().max().item()
    print(f"  负值输入, max_diff={max_diff:.6f}")
    assert max_diff < 1e-2
    print("  ✓ 通过")

    print("\n✓ 所有边界情况测试通过")


def test_quest_chunk_score_fp16():
    """测试 fp16 数据类型"""
    print("\n" + "="*80)
    print("测试 4: FP16 数据类型")
    print("="*80)

    bs, kv_heads, head_dim = 1, 4, 128
    n_chunks = 500

    torch.manual_seed(456)
    q = torch.randn(bs, kv_heads, head_dim, dtype=torch.float16, device='cuda')
    chunk_min = torch.randn(bs, kv_heads, n_chunks, head_dim, dtype=torch.float16, device='cuda')
    chunk_max = torch.randn(bs, kv_heads, n_chunks, head_dim, dtype=torch.float16, device='cuda')
    chunk_min, chunk_max = torch.minimum(chunk_min, chunk_max), torch.maximum(chunk_min, chunk_max)

    scores_cuda = quest_chunk_score(q, chunk_min, chunk_max)
    scores_ref = quest_chunk_score_reference(q, chunk_min, chunk_max)

    max_diff = (scores_cuda - scores_ref).abs().max().item()
    mean_diff = (scores_cuda - scores_ref).abs().mean().item()

    print(f"数据类型: {q.dtype}")
    print(f"最大差异: {max_diff:.6f}")
    print(f"平均差异: {mean_diff:.6f}")

    tolerance = 1e-2
    assert max_diff < tolerance, f"最大差异 {max_diff} 超过容差 {tolerance}"
    print("✓ 测试通过")


def test_quest_chunk_score_batch():
    """测试 batch 维度"""
    print("\n" + "="*80)
    print("测试 5: Batch 维度")
    print("="*80)

    bs, kv_heads, head_dim = 4, 8, 128
    n_chunks = 1000

    torch.manual_seed(789)
    q = torch.randn(bs, kv_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    chunk_min = torch.randn(bs, kv_heads, n_chunks, head_dim, dtype=torch.bfloat16, device='cuda')
    chunk_max = torch.randn(bs, kv_heads, n_chunks, head_dim, dtype=torch.bfloat16, device='cuda')
    chunk_min, chunk_max = torch.minimum(chunk_min, chunk_max), torch.maximum(chunk_min, chunk_max)

    scores_cuda = quest_chunk_score(q, chunk_min, chunk_max)
    scores_ref = quest_chunk_score_reference(q, chunk_min, chunk_max)

    max_diff = (scores_cuda - scores_ref).abs().max().item()
    mean_diff = (scores_cuda - scores_ref).abs().mean().item()

    print(f"Batch size: {bs}")
    print(f"最大差异: {max_diff:.6f}")
    print(f"平均差异: {mean_diff:.6f}")

    # 检查每个 batch 的差异
    for b in range(bs):
        batch_max_diff = (scores_cuda[b] - scores_ref[b]).abs().max().item()
        print(f"  Batch {b}: max_diff={batch_max_diff:.6f}")

    tolerance = 1e-2
    assert max_diff < tolerance, f"最大差异 {max_diff} 超过容差 {tolerance}"
    print("✓ 测试通过")


def test_quest_chunk_score_valid_mask():
    """测试 valid mask 融合：invalid chunk 应输出 -inf"""
    print("\n" + "="*80)
    print("测试 6: valid mask 融合测试")
    print("="*80)

    bs, kv_heads, head_dim = 2, 4, 128
    n_chunks = 32

    torch.manual_seed(123)
    q = torch.randn(bs, kv_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    chunk_min = torch.randn(bs, kv_heads, n_chunks, head_dim, dtype=torch.bfloat16, device='cuda')
    chunk_max = torch.randn(bs, kv_heads, n_chunks, head_dim, dtype=torch.bfloat16, device='cuda')
    chunk_min, chunk_max = torch.minimum(chunk_min, chunk_max), torch.maximum(chunk_min, chunk_max)

    # 构造 valid mask：随机让一些 chunk invalid
    valid = torch.ones(bs, kv_heads, n_chunks, dtype=torch.bool, device='cuda')
    valid[:, :, -5:] = False  # 最后 5 个 chunk invalid
    valid[0, 1, 3] = False    # 随机再 invalid 一个

    # CUDA kernel with valid mask
    scores_out = torch.empty(bs, kv_heads, n_chunks, dtype=torch.float32, device='cuda')
    from quest_ops import load_kernel_module
    kernel = load_kernel_module()
    kernel.quest_chunk_score(q, chunk_min, chunk_max, scores_out, n_chunks, valid)

    # 参考实现：先算 score 再 masked_fill
    scores_ref = quest_chunk_score_reference(q, chunk_min, chunk_max)
    scores_ref = scores_ref.masked_fill(~valid, float("-inf"))

    # 比较 valid 位置
    valid_scores_cuda = scores_out[valid]
    valid_scores_ref = scores_ref[valid]
    max_diff_valid = (valid_scores_cuda - valid_scores_ref).abs().max().item()

    # 比较 invalid 位置应为 -inf
    invalid_scores_cuda = scores_out[~valid]
    all_neg_inf = torch.all(invalid_scores_cuda == float("-inf")).item()

    print(f"输入形状: q={q.shape}, n_chunks={n_chunks}")
    print(f"Valid chunks: {valid.sum().item()}, Invalid chunks: {(~valid).sum().item()}")
    print(f"Valid 位置最大差异: {max_diff_valid:.6f}")
    print(f"Invalid 位置全为 -inf: {all_neg_inf}")

    tolerance = 1e-2
    assert max_diff_valid < tolerance, f"Valid 位置最大差异 {max_diff_valid} 超过容差 {tolerance}"
    assert all_neg_inf, "Invalid 位置未全部为 -inf"
    print("✓ 测试通过")


def test_quest_chunk_score_no_valid_mask():
    """测试不传 valid mask 时行为不变（向后兼容）"""
    print("\n" + "="*80)
    print("测试 7: 不传 valid mask（向后兼容）")
    print("="*80)

    bs, kv_heads, head_dim = 1, 2, 128
    n_chunks = 16

    torch.manual_seed(77)
    q = torch.randn(bs, kv_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    chunk_min = torch.randn(bs, kv_heads, n_chunks, head_dim, dtype=torch.bfloat16, device='cuda')
    chunk_max = torch.randn(bs, kv_heads, n_chunks, head_dim, dtype=torch.bfloat16, device='cuda')
    chunk_min, chunk_max = torch.minimum(chunk_min, chunk_max), torch.maximum(chunk_min, chunk_max)

    # 不传 valid — 应该和之前行为一致
    scores_out = torch.empty(bs, kv_heads, n_chunks, dtype=torch.float32, device='cuda')
    from quest_ops import load_kernel_module
    kernel = load_kernel_module()
    kernel.quest_chunk_score(q, chunk_min, chunk_max, scores_out, n_chunks)

    scores_ref = quest_chunk_score_reference(q, chunk_min, chunk_max)
    max_diff = (scores_out - scores_ref).abs().max().item()

    print(f"最大差异: {max_diff:.6f}")
    tolerance = 1e-2
    assert max_diff < tolerance, f"最大差异 {max_diff} 超过容差 {tolerance}"
    print("✓ 测试通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*80)
    print("Quest Chunk Score CUDA Kernel 单元测试")
    print("="*80)

    try:
        test_quest_chunk_score_basic()
        test_quest_chunk_score_large_scale()
        test_quest_chunk_score_edge_cases()
        test_quest_chunk_score_fp16()
        test_quest_chunk_score_batch()
        test_quest_chunk_score_valid_mask()
        test_quest_chunk_score_no_valid_mask()

        print("\n" + "="*80)
        print("✓ 所有测试通过！")
        print("="*80)
        return True
    except Exception as e:
        print("\n" + "="*80)
        print(f"✗ 测试失败: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
