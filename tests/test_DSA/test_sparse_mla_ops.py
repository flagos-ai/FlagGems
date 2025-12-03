import math
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import pytest
import torch
import triton

from flag_gems.fused.DSA.sparse_mla import triton_sparse_mla_fwd_interface
from benchmark.sparse_mla_fwd import ref_sparse_mla_fwd_interface #, sparse_mla_fwd_interface

# 精度检查函数
def gems_assert_close(actual, expected, dtype, equal_nan=False):
    # 对于bfloat16和float16，使用更宽松的容差
    if dtype in [torch.bfloat16, torch.float16]:
        atol = 1e-2
        rtol = 1e-2
    else:
        atol = 1e-4
        rtol = 1e-4
    
    print(f"Actual shape: {actual.shape}, Expected shape: {expected.shape}")
    print(f"Actual dtype: {actual.dtype}, Expected dtype: {expected.dtype}")
    
    # 计算差异统计
    diff = torch.abs(actual - expected)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")
    print(f"Actual range: [{torch.min(actual).item():.6f}, {torch.max(actual).item():.6f}]")
    print(f"Expected range: [{torch.min(expected).item():.6f}, {torch.max(expected).item():.6f}]")
    
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, equal_nan=equal_nan)

def init_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def to_reference(tensor, requires_grad=False):
    result = tensor.detach().clone()
    if requires_grad:
        result.requires_grad_()
    return result

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_sparse_mla_input(
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_heads: int,
    num_kv_heads: int,
    qk_dim: int,
    topk: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool = False
):
    """为稀疏MLA算子创建输入数据"""
    init_seed(42)
    B = batch_size
    S = seq_len_q
    H = num_heads
    DQK = qk_dim
    SKV = seq_len_kv
    HKV = num_kv_heads

    q = torch.randn((B, S, H, DQK), dtype=dtype, device=device).requires_grad_(requires_grad)
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device=device).requires_grad_(requires_grad)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device=device)
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, :len(i_i)] = i_i
        
    return q, kv, indices

def reference_sparse_mla_implementation(q, kv, indices, sm_scale=None, d_v=512):
    """参考实现 - 使用提供的参考函数"""
    return ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=sm_scale, d_v=d_v)

@pytest.mark.sparse_mla_forward
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len_q", [64, 128, 512])
@pytest.mark.parametrize("seq_len_kv", [1024, 2048, 4096])
@pytest.mark.parametrize("num_heads", [64, 128, 256])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("qk_dim", [576])  # 你的算子固定为576
@pytest.mark.parametrize("d_v", [512])     # 输出维度
@pytest.mark.parametrize("topk", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_sparse_mla_forward(
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_heads: int,
    num_kv_heads: int,
    qk_dim: int,
    d_v: int,
    topk: int,
    dtype: torch.dtype
):
    """稀疏MLA前向传播测试"""
    # 跳过不支持的情况
    if num_heads % num_kv_heads != 0:
        pytest.skip("num_heads must be divisible by num_kv_heads")
    
    if topk > seq_len_kv:
        pytest.skip("topk cannot be larger than seq_len_kv")
    
    # 创建输入
    q, kv, indices = make_sparse_mla_input(
        batch_size, seq_len_q, seq_len_kv, num_heads, num_kv_heads, 
        qk_dim, topk, dtype, device
    )
    
    # 参考实现
    ref_q = to_reference(q, False)
    ref_kv = to_reference(kv, False)
    ref_indices = to_reference(indices, False)
    
    ref_output = reference_sparse_mla_implementation(ref_q, ref_kv, ref_indices, d_v=d_v)
    
    # 你的算子实现
    your_output, your_lse = triton_sparse_mla_fwd_interface(q, kv, indices, d_v=d_v)
    
    # 精度比较
    gems_assert_close(your_output, ref_output, dtype)

@pytest.mark.sparse_mla_forward_edge_cases
@pytest.mark.parametrize("config", [
    # 边界情况测试
    {"batch_size": 1, "seq_len_q": 1, "seq_len_kv": 1, "num_heads": 1, "num_kv_heads": 1, "topk": 1},
    {"batch_size": 1, "seq_len_q": 2, "seq_len_kv": 100, "num_heads": 4, "num_kv_heads": 1, "topk": 50},
    {"batch_size": 1, "seq_len_q": 17, "seq_len_kv": 1030, "num_heads": 8, "num_kv_heads": 1, "topk": 256},
])
def test_sparse_mla_forward_edge_cases(config):
    """稀疏MLA边界情况测试"""
    dtype = torch.bfloat16
    qk_dim = 576
    d_v = 512
    
    q, kv, indices = make_sparse_mla_input(
        config["batch_size"], config["seq_len_q"], config["seq_len_kv"],
        config["num_heads"], config["num_kv_heads"], qk_dim, config["topk"], 
        dtype, device
    )
    
    # 参考实现
    ref_output = reference_sparse_mla_implementation(
        to_reference(q), to_reference(kv), to_reference(indices), d_v=d_v
    )
    
    # 你的算子实现
    your_output, your_lse = triton_sparse_mla_fwd_interface(q, kv, indices, d_v=d_v)
    
    gems_assert_close(your_output, ref_output, dtype)


# @pytest.mark.sparse_mla_performance
# @pytest.mark.parametrize("config", [
#     {"batch_size": 1, "seq_len_q": 4096, "seq_len_kv": 32768, "num_heads": 128, "num_kv_heads": 1, "topk": 2048},
#     {"batch_size": 2, "seq_len_q": 2048, "seq_len_kv": 16384, "num_heads": 128, "num_kv_heads": 1, "topk": 2048},
# ])
# def test_sparse_mla_performance(config):
#     """稀疏MLA性能测试"""
#     dtype = torch.bfloat16
#     qk_dim = 576
#     d_v = 512
    
#     q, kv, indices = make_sparse_mla_input(
#         config["batch_size"], config["seq_len_q"], config["seq_len_kv"],
#         config["num_heads"], config["num_kv_heads"], qk_dim, config["topk"], 
#         dtype, device
#     )
    
#     # 性能测试 - Triton实现
#     def triton_fn():
#         return triton_sparse_mla_fwd_interface(q, kv, indices, d_v=d_v)
    
#     # 性能测试 - Tilelang实现  
#     def tilelang_fn():
#         return sparse_mla_fwd_interface(q, kv, indices, d_v=d_v)
    
#     # 这里可以添加性能基准测试逻辑
#     # 例如使用 do_bench 或其他性能测试工具
    
#     # 确保结果正确性
#     triton_out, triton_lse = triton_fn()
#     tilelang_out, tilelang_lse = tilelang_fn()
    
#     # 验证两种实现结果一致
#     gems_assert_close(triton_out, tilelang_out, dtype)

# 设备兼容性测试
@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要CUDA设备")
@pytest.mark.sparse_mla_device
def test_sparse_mla_device_compatibility():
    """测试设备兼容性"""
    config = {
        "batch_size": 1, 
        "seq_len_q": 128, 
        "seq_len_kv": 1024, 
        "num_heads": 8, 
        "num_kv_heads": 1, 
        "topk": 64
    }
    dtype = torch.bfloat16
    qk_dim = 576
    d_v = 512
    
    q, kv, indices = make_sparse_mla_input(
        config["batch_size"], config["seq_len_q"], config["seq_len_kv"],
        config["num_heads"], config["num_kv_heads"], qk_dim, config["topk"], 
        dtype, device
    )
    
    # 在CUDA设备上运行
    your_output, your_lse = triton_sparse_mla_fwd_interface(q, kv, indices, d_v=d_v)
    
    # 验证输出形状正确
    expected_shape = (config["batch_size"], config["seq_len_q"], config["num_heads"], d_v)
    assert your_output.shape == expected_shape, f"输出形状不正确: {your_output.shape} != {expected_shape}"

if __name__ == "__main__":
    # 可以直接运行这个文件进行测试
    pytest.main([__file__, "-v"])