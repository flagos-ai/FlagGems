import torch
import pytest
import time
import pandas as pd
from flag_gems.ops.asinh import asinh, asinh_, asinh_out

# 1. 定义测试配置 (覆盖 4.1.4 输入规模要求)
TEST_CONFIGS = [
    # (shape, label)
    ((1, 1), "Small (1x1)"),
    ((8, 8), "Small (8x8)"),
    ((64, 64), "Medium (64x64)"),
    ((256, 256), "Medium (256x256)"),
    ((1024, 1024), "Large (1024x1024)"),
    ((4096, 4096), "Large (4096x4096)"),
    ((16, 3, 224, 224), "4D (Image Batch)"),
    ((2, 4, 8, 16, 32), "5D (High Dimension)"),
]

# 2. 精度评价标准定义 (对应 4.1.1)
# Float32 atol: 1.30e-06, rtol: 1e-04
FLOAT32_ATOL = 1.30e-06
FLOAT32_RTOL = 1.0e-04

def benchmark_op(func, input_tensor, iterations=100):
    # Warm up
    for _ in range(10):
        func(input_tensor)
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iterations):
        func(input_tensor)
    end_event.record()
    
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / iterations

def test_asinh_comprehensive():
    results = []
    print("\n" + "="*80)
    print(f"{'Shape':<25} | {'Speedup':<10} | {'Prec. (F32)':<12} | {'Status':<10}")
    print("-"*80)

    for shape, label in TEST_CONFIGS:
        # 准备数据
        x = torch.randn(shape, device="cuda", dtype=torch.float32)
        
        # 1. 精度验证 (4.1.1)
        ref_out = torch.asinh(x)
        tri_out = asinh(x)
        
        # 计算最大绝对误差
        max_abs_diff = torch.max(torch.abs(tri_out - ref_out)).item()
        is_passed = torch.allclose(tri_out, ref_out, atol=FLOAT32_ATOL, rtol=FLOAT32_RTOL)
        
        # 2. 性能验证 (提交要求)
        torch_time = benchmark_op(torch.asinh, x)
        gems_time = benchmark_op(asinh, x)
        speedup = torch_time / gems_time
        
        status = "✅ PASS" if is_passed else "❌ FAIL"
        results.append({
            "Shape": str(shape),
            "Label": label,
            "Speedup": f"{speedup:.2f}x",
            "Max Abs Diff": f"{max_abs_diff:.2e}",
            "Status": status
        })
        
        print(f"{str(shape):<25} | {speedup:>9.2f}x | {max_abs_diff:>12.2e} | {status:<10}")

    # 3. 边界值与功能完整性验证 (4.1.4)
    print("-"*80)
    special_cases = torch.tensor([0.0, -0.0, 1e10, -1e10, float('inf'), float('-inf')], device="cuda")
    special_passed = torch.allclose(asinh(special_cases), torch.asinh(special_cases), equal_nan=True)
    print(f"Special Values (0, Inf, Large): {'✅ PASS' if special_passed else '❌ FAIL'}")
    
    # 4. 原位运算验证 (4.1.4)
    x_inplace = torch.randn((100, 100), device="cuda")
    x_ref = x_inplace.clone()
    torch.asinh_(x_ref)
    asinh_(x_inplace)
    inplace_passed = torch.allclose(x_inplace, x_ref, atol=FLOAT32_ATOL)
    print(f"Inplace Operation (asinh_):   {'✅ PASS' if inplace_passed else '❌ FAIL'}")

    return results

if __name__ == "__main__":
    test_asinh_comprehensive()