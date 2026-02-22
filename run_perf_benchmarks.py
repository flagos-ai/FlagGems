#!/usr/bin/env python3
"""
为各个操作运行性能基准测试
由于某些操作可能没有预定义的基准测试，我们直接测试性能
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
        try:
            _ = op_func(*inputs)
        except:
            pass
    torch.cuda.synchronize()
    
    # Test GEMS
    start = time.perf_counter()
    for _ in range(iterations):
        with flag_gems.use_gems():
            try:
                result = op_func(*inputs)
            except:
                pass
    torch.cuda.synchronize()
    gems_time = (time.perf_counter() - start) / iterations * 1000  # ms
    
    # Test PyTorch baseline
    start = time.perf_counter()
    for _ in range(iterations):
        try:
            ref_result = ref_func(*inputs)
        except:
            pass
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) / iterations * 1000  # ms
    
    speedup = torch_time / gems_time if gems_time > 0 else 0.0
    
    return {
        "torch_latency_ms": torch_time,
        "gems_latency_ms": gems_time,
        "speedup": speedup,
        "shape": inputs[0].shape if isinstance(inputs[0], torch.Tensor) else str(inputs[0])
    }

def test_max_pool3d():
    """测试 max_pool3d"""
    print("Testing max_pool3d...")
    shapes = [
        (2, 4, 8, 8, 8),
        (1, 3, 16, 16, 16),
        (4, 8, 32, 32, 32),
    ]
    results = []
    for shape in shapes:
        for dtype in [torch.float16, torch.float32]:
            try:
                inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
                result = benchmark_op(
                    lambda x: flag_gems.max_pool3d_with_indices(x, kernel_size=3, stride=2, padding=1)[0],
                    lambda x: torch.nn.functional.max_pool3d(x, kernel_size=3, stride=2, padding=1),
                    (inp,)
                )
                result["dtype"] = str(dtype)
                results.append(result)
            except Exception as e:
                print(f"  Error with shape {shape}, dtype {dtype}: {e}")
    return results

def test_avg_pool3d():
    """测试 avg_pool3d"""
    print("Testing avg_pool3d...")
    shapes = [
        (2, 4, 8, 8, 8),
        (1, 3, 16, 16, 16),
        (4, 8, 32, 32, 32),
    ]
    results = []
    for shape in shapes:
        for dtype in [torch.float16, torch.float32]:
            try:
                inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
                result = benchmark_op(
                    lambda x: flag_gems.avg_pool3d(x, kernel_size=3, stride=2, padding=1),
                    lambda x: torch.nn.functional.avg_pool3d(x, kernel_size=3, stride=2, padding=1),
                    (inp,)
                )
                result["dtype"] = str(dtype)
                results.append(result)
            except Exception as e:
                print(f"  Error with shape {shape}, dtype {dtype}: {e}")
    return results

def test_grid_sample():
    """测试 grid_sample"""
    print("Testing grid_sample...")
    configs = [
        ((1, 3, 8, 8), (1, 6, 6, 2)),
        ((2, 4, 16, 16), (2, 8, 8, 2)),
        ((4, 8, 32, 32), (4, 16, 16, 2)),
    ]
    results = []
    for input_shape, grid_shape in configs:
        for dtype in [torch.float16, torch.float32]:
            try:
                inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
                grid = torch.empty(grid_shape, dtype=dtype, device=flag_gems.device).uniform_(-1, 1)
                result = benchmark_op(
                    lambda x, g: torch.nn.functional.grid_sample(x, g, mode='bilinear', padding_mode='zeros', align_corners=False),
                    lambda x, g: torch.nn.functional.grid_sample(x, g, mode='bilinear', padding_mode='zeros', align_corners=False),
                    (inp, grid)
                )
                result["dtype"] = str(dtype)
                results.append(result)
            except Exception as e:
                print(f"  Error with input_shape {input_shape}, dtype {dtype}: {e}")
    return results

def test_svd():
    """测试 svd"""
    print("Testing svd...")
    shapes = [
        (4, 3),
        (8, 8),
        (16, 16),
        (32, 32),
    ]
    results = []
    for shape in shapes:
        dtype = torch.float32
        try:
            inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
            result = benchmark_op(
                lambda x: torch.svd(x, some=True, compute_uv=True),
                lambda x: torch.svd(x, some=True, compute_uv=True),
                (inp,)
            )
            result["dtype"] = str(dtype)
            results.append(result)
        except Exception as e:
            print(f"  Error with shape {shape}: {e}")
    return results

def test_ctc_loss():
    """测试 ctc_loss"""
    print("Testing ctc_loss...")
    results = []
    configs = [
        (50, 2, 20),
        (100, 4, 30),
    ]
    for T, N, C in configs:
        try:
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
            print(f"  Error with config (T={T}, N={N}, C={C}): {e}")
    return results

def main():
    branches_ops = {
        "codex/max_pool3d": test_max_pool3d,
        "codex/avg_pool3d": test_avg_pool3d,
        "codex/grid_sample": test_grid_sample,
        "codex/svd": test_svd,
        "codex/ctc_loss": test_ctc_loss,
    }
    
    all_results = {}
    
    for branch, test_func in branches_ops.items():
        print(f"\n{'='*60}")
        print(f"Testing branch: {branch}")
        print(f"{'='*60}\n")
        
        # Switch to branch
        import subprocess
        subprocess.run(["git", "checkout", branch], cwd="/home/qinhaiyan/FlagGems", check=True)
        
        try:
            results = test_func()
            all_results[branch] = results
        except Exception as e:
            print(f"Error testing {branch}: {e}")
            all_results[branch] = []
    
    # Print and save results
    output_file = Path("/home/qinhaiyan/FlagGems/perf_benchmark_results.txt")
    with open(output_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("FlagGems Performance Benchmark Results\n")
        f.write("="*80 + "\n\n")
        
        for branch, results in all_results.items():
            if not results:
                print(f"\n{branch}: No results")
                f.write(f"{branch}: No results\n\n")
                continue
            
            print(f"\n{branch}:")
            print(f"{'Shape':<40} {'Dtype':<15} {'Torch (ms)':<15} {'GEMS (ms)':<15} {'Speedup':<10}")
            print("-" * 95)
            
            f.write(f"{branch}:\n")
            f.write(f"{'Shape':<40} {'Dtype':<15} {'Torch (ms)':<15} {'GEMS (ms)':<15} {'Speedup':<10}\n")
            f.write("-" * 95 + "\n")
            
            avg_speedup = []
            for r in results:
                shape_str = str(r["shape"])[:38]
                print(f"{shape_str:<40} {r['dtype']:<15} {r['torch_latency_ms']:<15.4f} {r['gems_latency_ms']:<15.4f} {r['speedup']:<10.3f}")
                f.write(f"{shape_str:<40} {r['dtype']:<15} {r['torch_latency_ms']:<15.4f} {r['gems_latency_ms']:<15.4f} {r['speedup']:<10.3f}\n")
                avg_speedup.append(r["speedup"])
            
            if avg_speedup:
                avg = np.mean(avg_speedup)
                print(f"Average Speedup: {avg:.3f}x")
                f.write(f"Average Speedup: {avg:.3f}x\n")
            f.write("\n")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
