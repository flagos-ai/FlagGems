import torch
import pytest
import numpy as np
from flag_gems.ops.asinh import asinh, asinh_, asinh_out

# ==============================================================================
# 1. 严格遵循 4.1.1 精度评价标准定义
# ==============================================================================
ACCURACY_CFG = {
    torch.float32: {"atol": 1.30e-06, "rtol": 1e-04},
    torch.float16: {"atol": 1.00e-03, "rtol": 1e-04},
    torch.bfloat16: {"atol": 0.016, "rtol": 1e-04},
}

# ==============================================================================
# 2. 定义测试规模 (覆盖 4.1.4 输入规模与维数要求)
# ==============================================================================
TEST_SHAPES = [
    (),                          # 标量 (0D)
    (1, 1),                      # 小尺寸 (1x1)
    (8, 8),                      # 小尺寸 (8x8)
    (64, 64),                    # 常规尺寸
    (256, 256),                  # 常规尺寸
    (1024, 1024),                # 大尺寸
    (4096, 4096),                # 大尺寸
    (16, 3, 224, 224),           # 4D 维数覆盖
    (2, 4, 8, 16, 32),           # 5D 维数覆盖
]

# ==============================================================================
# 工具函数：性能测量
# ==============================================================================
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

# ==============================================================================
# 3. 核心功能测试 (支持精度、连续性、Dtype) - 供 pytest 使用
# ==============================================================================
@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("contiguous", [True, False])
def test_asinh_accuracy(shape, dtype, contiguous):
    # 准备数据
    if len(shape) == 0:
        x = torch.randn((), device="cuda", dtype=dtype)
    else:
        x = torch.randn(shape, device="cuda", dtype=dtype)
    
    # 覆盖非连续内存场景 (4.1.4)
    if not contiguous and x.ndim >= 2:
        x = x.transpose(-1, -2)

    # 计算
    ref_out = torch.asinh(x)
    tri_out = asinh(x)
    
    # 验证精度 (4.1.1)
    cfg = ACCURACY_CFG[dtype]
    torch.testing.assert_close(tri_out, ref_out, **cfg, msg=f"Failed at shape:{shape}, dtype:{dtype}")

# ==============================================================================
# 4. 功能完整性验证 (4.1.4: 原位、Out 变体、特殊值)
# ==============================================================================
def test_asinh_inplace():
    """验证原位运算 asinh_"""
    x = torch.randn((1024, 1024), device="cuda", dtype=torch.float32)
    x_ref = x.clone()
    
    torch.asinh_(x_ref)
    asinh_(x)
    
    torch.testing.assert_close(x, x_ref, **ACCURACY_CFG[torch.float32])

def test_asinh_out():
    """验证 Out 变体 asinh_out"""
    x = torch.randn((1024, 1024), device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    
    ref = torch.asinh(x)
    asinh_out(x, out=out)
    
    torch.testing.assert_close(out, ref, **ACCURACY_CFG[torch.float32])

def test_asinh_special_values():
    """验证特殊值 (4.1.4: 边界值、特殊值)"""
    special_cases = torch.tensor(
        [0.0, -0.0, 1e10, -1e10, float('inf'), float('-inf'), float('nan')], 
        device="cuda", dtype=torch.float32
    )
    ref = torch.asinh(special_cases)
    res = asinh(special_cases)
    
    torch.testing.assert_close(res, ref, equal_nan=True)

# ==============================================================================
# 5. 性能与精度对比工具 (用于生成提交要求的完整对比表)
# ==============================================================================
def test_asinh_performance_table():
    """打印符合提交要求的性能与精度对比表"""
    print("\n" + "="*105)
    print(f"{'Shape':<25} | {'Dtype':<10} | {'PyTorch(ms)':<12} | {'Gems(ms)':<12} | {'Speedup':<10} | {'Max Abs Diff'}")
    print("-"*105)
    
    for shape in TEST_SHAPES:
        if len(shape) == 0: continue
        
        dtype = torch.float32
        x = torch.randn(shape, device="cuda", dtype=dtype)
        
        # 1. 计算精度误差 (Prec.)
        with torch.no_grad():
            ref_out = torch.asinh(x)
            tri_out = asinh(x)
            max_diff = torch.max(torch.abs(tri_out - ref_out)).item()
        
        # 2. 测量性能
        t_torch = benchmark_op(torch.asinh, x)
        t_gems = benchmark_op(asinh, x)
        speedup = t_torch / t_gems
        
        # 3. 打印行
        dtype_str = str(dtype).split('.')[-1]
        print(f"{str(shape):<25} | {dtype_str:<10} | {t_torch:>10.4f}   | {t_gems:>10.4f}   | {speedup:>8.2f}x  | {max_diff:.2e}")
    
    print("="*105 + "\n")

if __name__ == "__main__":
    # 执行该函数以生成 PR 所需的表格
    test_asinh_performance_table()