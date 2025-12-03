import math
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import pytest
import torch
import triton

# 导入你的算子
from flag_gems.fused.DSA.bin_topk import bucket_sort_topk # 替换为实际模块名

def gems_assert_close(actual, expected, dtype, equal_nan=False):
    """精度检查函数，针对topk测试进行优化"""
    print(f"Actual shape: {actual.shape}, Expected shape: {expected.shape}")
    print(f"Actual dtype: {actual.dtype}, Expected dtype: {expected.dtype}")
    
    # 对于topk索引，我们主要关心选择的元素是否正确
    if actual.dtype == torch.int32:
        # 计算交集比例
        batch_size = actual.shape[0]
        total_intersection = 0
        total_elements = 0
        
        for i in range(batch_size):
            actual_set = set(actual[i].cpu().numpy())
            expected_set = set(expected[i].cpu().numpy())
            intersection = actual_set & expected_set
            intersection_ratio = len(intersection) / len(expected_set)
            total_intersection += len(intersection)
            total_elements += len(expected_set)
            
            print(f"Batch {i}: Intersection ratio = {intersection_ratio:.4f}")
            
            # 要求至少95%的topk元素匹配
            assert intersection_ratio >= 0.95, f"Batch {i}: Only {intersection_ratio:.4f} intersection, expected at least 0.95"
        
        overall_ratio = total_intersection / total_elements
        print(f"Overall intersection ratio: {overall_ratio:.4f}")
        return
        
    # 对于浮点数，使用标准比较
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2, equal_nan=equal_nan)

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

# """为topk算子创建输入数据"""
def make_topk_input(
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
):
    init_seed(1234)
    inputs = torch.randn((batch_size, seq_len), dtype=dtype, device=device).requires_grad_(False)
    
    # 创建starts和ends（支持变长序列）
    starts = torch.zeros(batch_size, dtype=torch.int32, device=device)
    
    # 模拟变长序列：每个batch的序列长度不同
    min_len = max(1, seq_len // 2)
    ends = torch.randint(min_len, seq_len + 1, (batch_size,), dtype=torch.int32, device=device)
    
    return inputs, starts, ends

def reference_topk_implementation(inputs, starts, ends, topk):
    """参考实现 - 使用torch.topk"""
    batch_size, seq_len = inputs.shape
    ref_indices = torch.zeros(batch_size, topk, dtype=torch.int32, device=inputs.device)
    
    for i in range(batch_size):
        start = starts[i].item()
        end = ends[i].item()
        seq_slice = inputs[i, start:end]
        
        if len(seq_slice) > 0:
            # 获取topk索引
            _, topk_indices = torch.topk(seq_slice, min(topk, len(seq_slice)))
            # 转换为全局索引
            global_indices = topk_indices + start
            ref_indices[i, :len(global_indices)] = global_indices
    
    return ref_indices

def debug_topk_results(actual, expected, inputs, test_name=""):
    """调试topk结果"""
    print(f"\n=== {test_name} ===")
    batch_size = actual.shape[0]
    
    m = 20
    for i in range(min(16, batch_size)):  # 只检查前2个batch
        actual_set = set(actual[i].cpu().numpy())
        expected_set = set(expected[i].cpu().numpy())
        intersection = actual_set & expected_set
        print(f"Batch {i}:")
        print(f"  Actual indices: {sorted(actual_set)[:m]}...")  # 只显示前10个
        print(f"  Expected indices: {sorted(expected_set)[:m]}...")
        print(f"  Intersection: {len(intersection)}/{len(expected_set)} = {len(intersection)/len(expected_set):.4f}")
        
        # 检查实际选择的值的质量
        actual_values = inputs[i, actual[i]].cpu().numpy()
        expected_values = inputs[i, expected[i]].cpu().numpy()
        
        print(f"  Actual top values: {np.sort(actual_values)[-m:][::-1]}")  # 最大的5个值
        print(f"  Expected top values: {np.sort(expected_values)[-m:][::-1]}")

@pytest.mark.bucket_sort_topk_forward
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("seq_len", [256, 1024, 8192])
@pytest.mark.parametrize("topk", [16, 64, 256])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_bucket_sort_topk_forward(
    batch_size: int,
    seq_len: int,
    topk: int,
    dtype: torch.dtype
):
    """bucket sort topk前向传播测试"""
    if topk > seq_len:
        pytest.skip("topk cannot be larger than seq_len")
    
    # 创建输入
    inputs, starts, ends = make_topk_input(batch_size, seq_len, dtype, device)
    
    # 参考实现
    ref_indices = reference_topk_implementation(
        to_reference(inputs), to_reference(starts), to_reference(ends), topk
    )
    
    # 你的算子实现
    your_indices = bucket_sort_topk(inputs, starts, ends, topk)
    
    # 调试输出
    debug_topk_results(your_indices, ref_indices, inputs, 
                      f"forward_b{batch_size}_s{seq_len}_k{topk}")
    
    # 精度比较 - 使用自定义的topk比较逻辑
    gems_assert_close(your_indices, ref_indices, dtype)

@pytest.mark.bucket_sort_topk_edge_cases
@pytest.mark.parametrize("config", [
    # 边界情况测试
    {"batch_size": 1, "seq_len": 1, "topk": 1},
    {"batch_size": 1, "seq_len": 10, "topk": 10},  # topk等于序列长度
    {"batch_size": 2, "seq_len": 100, "topk": 50},
    {"batch_size": 8, "seq_len": 17, "topk": 8},   # 小序列
])
def test_bucket_sort_topk_edge_cases(config):
    """bucket sort topk边界情况测试"""
    dtype = torch.float32
    
    inputs, starts, ends = make_topk_input(
        config["batch_size"], config["seq_len"], dtype, device
    )
    
    # 参考实现
    ref_indices = reference_topk_implementation(
        to_reference(inputs), to_reference(starts), to_reference(ends), config["topk"]
    )
    
    # 你的算子实现
    your_indices = bucket_sort_topk(inputs, starts, ends, config["topk"])
    
    debug_topk_results(your_indices, ref_indices, inputs, "edge_case")
    
    gems_assert_close(your_indices, ref_indices, dtype)

@pytest.mark.bucket_sort_topk_large_scale
@pytest.mark.parametrize("config", [
    # 大规模测试 - 使用你原始测试的参数
    {"batch_size": 64, "seq_len": 32768, "topk": 2048},
    {"batch_size": 32, "seq_len": 65536, "topk": 4096},
    {"batch_size": 96, "seq_len": 32768, "topk": 2048},  # 你的原始测试参数
])
def test_bucket_sort_topk_large_scale(config):
    """bucket sort topk大规模测试"""
    dtype = torch.float32
    
    inputs, starts, ends = make_topk_input(
        config["batch_size"], config["seq_len"], dtype, device
    )
    
    # 参考实现
    ref_indices = reference_topk_implementation(
        to_reference(inputs), to_reference(starts), to_reference(ends), config["topk"]
    )
    
    # 你的算子实现
    your_indices = bucket_sort_topk(inputs, starts, ends, config["topk"])
    
    debug_topk_results(your_indices, ref_indices, inputs, "large_scale")
    
    gems_assert_close(your_indices, ref_indices, dtype)

@pytest.mark.bucket_sort_topk_performance
@pytest.mark.parametrize("config", [
    {"batch_size": 96, "seq_len": 32768, "topk": 2048},
])
def test_bucket_sort_topk_performance(config):
    """bucket sort topk性能测试"""
    dtype = torch.float32
    
    inputs, starts, ends = make_topk_input(
        config["batch_size"], config["seq_len"], dtype, device
    )
    
    # 性能测试 - 你的算子
    def your_topk_fn():
        return bucket_sort_topk(inputs, starts, ends, config["topk"])
    
    # 性能测试 - torch.topk
    def torch_topk_fn():
        return reference_topk_implementation(inputs, starts, ends, config["topk"])
    
    # 预热
    for _ in range(10):
        your_indices = your_topk_fn()
        torch_indices = torch_topk_fn()
    
    # 测量你的算子性能
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    n_iters = 100
    start_event.record()
    for _ in range(n_iters):
        your_indices = your_topk_fn()
    end_event.record()
    torch.cuda.synchronize()
    your_time_ms = start_event.elapsed_time(end_event) / n_iters
    
    # 测量torch.topk性能
    start_event.record()
    for _ in range(n_iters):
        torch_indices = torch_topk_fn()
    end_event.record()
    torch.cuda.synchronize()
    torch_time_ms = start_event.elapsed_time(end_event) / n_iters
    
    print(f"Your topk average time: {your_time_ms:.3f} ms")
    print(f"Torch topk average time: {torch_time_ms:.3f} ms")
    print(f"Speedup: {torch_time_ms/your_time_ms:.2f}x")
    
    # 验证结果正确性
    debug_topk_results(your_indices, torch_indices, inputs, "performance")
    gems_assert_close(your_indices, torch_indices, dtype)

@pytest.mark.bucket_sort_topk_variable_length
def test_bucket_sort_topk_variable_length():
    """测试变长序列处理"""
    batch_size = 4
    max_seq_len = 1024
    topk = 64
    dtype = torch.float32
    
    # 创建输入，但使用不同的序列长度
    inputs = torch.randn(batch_size, max_seq_len, dtype=dtype, device=device)
    starts = torch.zeros(batch_size, dtype=torch.int32, device=device)
    
    # 每个batch使用不同的序列长度
    ends = torch.tensor([100, 500, 800, 1024], dtype=torch.int32, device=device)
    
    # 参考实现
    ref_indices = reference_topk_implementation(
        to_reference(inputs), to_reference(starts), to_reference(ends), topk
    )
    
    # 你的算子实现
    your_indices = bucket_sort_topk(inputs, starts, ends, topk)
    
    debug_topk_results(your_indices, ref_indices, inputs, "variable_length")
    
    gems_assert_close(your_indices, ref_indices, dtype)

@pytest.mark.bucket_sort_topk_correctness
def test_bucket_sort_topk_correctness():
    """正确性测试 - 使用你原始的测试逻辑"""
    batch_size = 96
    seq_len = 32768
    topk = 2048
    
    # torch.manual_seed(1)
    inputs = torch.randn(batch_size, seq_len, dtype=torch.float32, device=device)
    starts = torch.zeros(batch_size, dtype=torch.int32, device=device)
    ends = torch.ones(batch_size, dtype=torch.int32, device=device) * seq_len

    # 你的算子
    your_indices = bucket_sort_topk(inputs, starts, ends, topk)
    
    # 参考实现
    ref_indices = torch.topk(inputs, topk, dim=-1)[1]
    
    # 计算交集比例
    total_intersection = 0
    total_elements = 0
    
    for i in range(batch_size):
        your_set = set(your_indices[i].cpu().numpy())
        ref_set = set(ref_indices[i].cpu().numpy())
        intersection = your_set & ref_set
        intersection_ratio = len(intersection) / len(ref_set)
        total_intersection += len(intersection)
        total_elements += len(ref_set)
        
        print(f"Batch {i}: Intersection ratio = {intersection_ratio:.4f}")
        
        # 要求至少95%的topk元素匹配
        assert intersection_ratio >= 0.95, f"Batch {i}: Only {intersection_ratio:.4f} intersection, expected at least 0.95"
    
    overall_ratio = total_intersection / total_elements
    print(f"Overall intersection ratio: {overall_ratio:.4f}")

if __name__ == "__main__":
    # 可以直接运行这个文件进行测试
    pytest.main([__file__, "-v"])