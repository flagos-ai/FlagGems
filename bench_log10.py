"""
Benchmark suite for log10 operator.
"""

import torch
import triton
import triton.testing
import numpy as np
import sys
sys.path.append('.')
from log10 import log10


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28, 2)],  # 4096 to 268435456
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='ms',
        plot_name='log10-performance',
        args={},
    )
)
def benchmark_log10(size, provider):
    """Benchmark log10 performance."""
    # Create input
    x = torch.rand(size, device='cuda', dtype=torch.float32) + 0.1
    
    # Warmup
    for _ in range(10):
        if provider == 'triton':
            log10(x)
        else:
            torch.log10(x)
    
    # Measure
    torch.cuda.synchronize()
    
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: log10(x))
    else:
        ms = triton.testing.do_bench(lambda: torch.log10(x))
    
    return ms


def test_speedup():
    """Verify speedup >= 0.9."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return
    
    sizes = [4096, 65536, 1048576, 16777216]
    speedups = []
    
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK RESULTS")
    print("="*60)
    print(f"{'Size':>12} | {'Triton(ms)':>10} | {'PyTorch(ms)':>10} | {'Speedup':>8}")
    print("-" * 50)
    
    for size in sizes:
        x = torch.rand(size, device='cuda', dtype=torch.float32) + 0.1
        
        # Warmup
        for _ in range(10):
            log10(x)
            torch.log10(x)
        
        # Benchmark
        torch.cuda.synchronize()
        time_custom = triton.testing.do_bench(lambda: log10(x))
        time_torch = triton.testing.do_bench(lambda: torch.log10(x))
        
        speedup = time_torch / time_custom
        speedups.append(speedup)
        
        print(f"{size:12d} | {time_custom:10.3f} | {time_torch:10.3f} | {speedup:8.3f}")
        
        assert speedup >= 0.9, f"Speedup {speedup:.3f} < 0.9 for size {size}"
    
    avg_speedup = sum(speedups) / len(speedups)
    print("-" * 50)
    print(f"Average speedup: {avg_speedup:.3f}")
    print(f"✅ Meets requirement (>=0.9)")


if __name__ == "__main__":
    # Run benchmark
    benchmark_log10.run(print_data=True)
    
    # Verify speedup
    test_speedup()
