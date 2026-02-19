#!/usr/bin/env python3
"""
直接测试各个操作的性能并收集加速比
"""
import torch
import time
import flag_gems
import numpy as np
from pathlib import Path

def benchmark_op(op_func, ref_func, inputs, warmup=50, iterations=100):
    """运行性能基准测试"""
    device = flag_gems.device
    
    # Warmup
    for _ in range(warmup):
        _ = op_func(*inputs)
    torch.cuda.synchronize()
    
    # Test GEMS
    start = time.perf_counter()
    for _ in range(iterations):
        with flag_gems.use_gems():
            result = op_func(*inputs)
    torch.cuda.synchronize()
    gems_time = (time.perf_counter() - start) / iterations * 1000  # ms
    
    # Test PyTorch baseline
    start = time.perf_counter()
    for _ in range(iterations):
        ref_result = ref_func(*inputs)
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) / iterations * 1000  # ms
    
    speedup = torch_time / gems_time if gems_time > 0 else 0.0
    
    return {
        "torch_latency_ms": torch_time,
        "gems_latency_ms": gems_time,
        "speedup": speedup,
        "shape": inputs[0].shape if isinstance(inputs[0], torch.Tensor) else str(inputs[0])
    }

def test_cosh():
    """测试 cosh"""
    print("Testing cosh...")
    shapes = [
        (1024, 1024),
        (4096, 4096),
    ]
    results = []
    for shape in shapes:
        for dtype in [torch.float16, torch.float32]:
            inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
            result = benchmark_op(
                lambda x: torch.cosh(x),
                lambda x: torch.cosh(x),
                (inp,)
            )
            result["dtype"] = str(dtype)
            results.append(result)
    return results

def test_max_pool3d():
    """测试 max_pool3d"""
    print("Testing max_pool3d...")
    shapes = [
        (2, 4, 8, 8, 8),
        (1, 3, 16, 16, 16),
    ]
    results = []
    for shape in shapes:
        for dtype in [torch.float16, torch.float32]:
            inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
            result = benchmark_op(
                lambda x: flag_gems.max_pool3d_with_indices(x, kernel_size=3, stride=2, padding=1)[0],
                lambda x: torch.nn.functional.max_pool3d(x, kernel_size=3, stride=2, padding=1),
                (inp,)
            )
            result["dtype"] = str(dtype)
            results.append(result)
    return results

def test_avg_pool3d():
    """测试 avg_pool3d"""
    print("Testing avg_pool3d...")
    shapes = [
        (2, 4, 8, 8, 8),
        (1, 3, 16, 16, 16),
    ]
    results = []
    for shape in shapes:
        for dtype in [torch.float16, torch.float32]:
            inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
            result = benchmark_op(
                lambda x: flag_gems.avg_pool3d(x, kernel_size=3, stride=2, padding=1),
                lambda x: torch.nn.functional.avg_pool3d(x, kernel_size=3, stride=2, padding=1),
                (inp,)
            )
            result["dtype"] = str(dtype)
            results.append(result)
    return results

def test_grid_sample():
    """测试 grid_sample"""
    print("Testing grid_sample...")
    shapes = [
        ((1, 3, 8, 8), (1, 6, 6, 2)),
        ((2, 4, 16, 16), (2, 8, 8, 2)),
    ]
    results = []
    for input_shape, grid_shape in shapes:
        for dtype in [torch.float16, torch.float32]:
            inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
            grid = torch.empty(grid_shape, dtype=dtype, device=flag_gems.device).uniform_(-1, 1)
            result = benchmark_op(
                lambda x, g: torch.nn.functional.grid_sample(x, g, mode='bilinear', padding_mode='zeros', align_corners=False),
                lambda x, g: torch.nn.functional.grid_sample(x, g, mode='bilinear', padding_mode='zeros', align_corners=False),
                (inp, grid)
            )
            result["dtype"] = str(dtype)
            results.append(result)
    return results

def test_svd():
    """测试 svd"""
    print("Testing svd...")
    shapes = [
        (4, 3),
        (8, 8),
    ]
    results = []
    for shape in shapes:
        dtype = torch.float32  # SVD typically uses float32
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        result = benchmark_op(
            lambda x: torch.svd(x, some=True, compute_uv=True),
            lambda x: torch.svd(x, some=True, compute_uv=True),
            (inp,)
        )
        result["dtype"] = str(dtype)
        results.append(result)
    return results

def test_ctc_loss():
    """测试 ctc_loss"""
    print("Testing ctc_loss...")
    # CTC loss requires specific input format
    results = []
    try:
        T, N, C = 50, 2, 20
        log_probs = torch.randn(T, N, C, dtype=torch.float32, device=flag_gems.device).log_softmax(2)
        targets = torch.randint(1, C, (N, 30), dtype=torch.long, device=flag_gems.device)
        input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)
        target_lengths = torch.randint(10, 30, (N,), dtype=torch.long, device=flag_gems.device)
        
        result = benchmark_op(
            lambda lp, t, il, tl: torch.nn.functional.ctc_loss(lp, t, il, tl, reduction='mean'),
            lambda lp, t, il, tl: torch.nn.functional.ctc_loss(lp, t, il, tl, reduction='mean'),
            (log_probs, targets, input_lengths, target_lengths)
        )
        result["dtype"] = "float32"
        results.append(result)
    except Exception as e:
        print(f"Error testing ctc_loss: {e}")
    return results

def main():
    all_results = {}
    
    # Test each operation
    try:
        all_results["cosh"] = test_cosh()
    except Exception as e:
        print(f"Error testing cosh: {e}")
        all_results["cosh"] = []
    
    try:
        all_results["max_pool3d"] = test_max_pool3d()
    except Exception as e:
        print(f"Error testing max_pool3d: {e}")
        all_results["max_pool3d"] = []
    
    try:
        all_results["avg_pool3d"] = test_avg_pool3d()
    except Exception as e:
        print(f"Error testing avg_pool3d: {e}")
        all_results["avg_pool3d"] = []
    
    try:
        all_results["grid_sample"] = test_grid_sample()
    except Exception as e:
        print(f"Error testing grid_sample: {e}")
        all_results["grid_sample"] = []
    
    try:
        all_results["svd"] = test_svd()
    except Exception as e:
        print(f"Error testing svd: {e}")
        all_results["svd"] = []
    
    try:
        all_results["ctc_loss"] = test_ctc_loss()
    except Exception as e:
        print(f"Error testing ctc_loss: {e}")
        all_results["ctc_loss"] = []
    
    # Print summary
    print("\n" + "="*80)
    print("Performance Benchmark Results Summary")
    print("="*80)
    
    for op_name, results in all_results.items():
        if not results:
            print(f"\n{op_name}: No results")
            continue
        
        print(f"\n{op_name}:")
        print(f"{'Shape':<30} {'Dtype':<15} {'Torch (ms)':<15} {'GEMS (ms)':<15} {'Speedup':<10}")
        print("-" * 85)
        
        avg_speedup = []
        for r in results:
            shape_str = str(r["shape"])[:28]
            print(f"{shape_str:<30} {r['dtype']:<15} {r['torch_latency_ms']:<15.4f} {r['gems_latency_ms']:<15.4f} {r['speedup']:<10.3f}")
            avg_speedup.append(r["speedup"])
        
        if avg_speedup:
            print(f"Average Speedup: {np.mean(avg_speedup):.3f}")
    
    # Save to file
    output_file = Path("/home/qinhaiyan/FlagGems/perf_results_summary.txt")
    with open(output_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("FlagGems Performance Benchmark Results\n")
        f.write("="*80 + "\n\n")
        
        for op_name, results in all_results.items():
            if not results:
                f.write(f"{op_name}: No results\n\n")
                continue
            
            f.write(f"{op_name}:\n")
            f.write(f"{'Shape':<30} {'Dtype':<15} {'Torch (ms)':<15} {'GEMS (ms)':<15} {'Speedup':<10}\n")
            f.write("-" * 85 + "\n")
            
            avg_speedup = []
            for r in results:
                shape_str = str(r["shape"])[:28]
                f.write(f"{shape_str:<30} {r['dtype']:<15} {r['torch_latency_ms']:<15.4f} {r['gems_latency_ms']:<15.4f} {r['speedup']:<10.3f}\n")
                avg_speedup.append(r["speedup"])
            
            if avg_speedup:
                f.write(f"Average Speedup: {np.mean(avg_speedup):.3f}\n")
            f.write("\n")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
