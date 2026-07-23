# benchmark/benchmark_meshgrid.py
import os
import sys
import time

import numpy as np
import torch

from flag_gems.ops.meshgrid import meshgrid

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)


def get_device():
    """Auto-select available device"""
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return "npu:0"
    except Exception:
        pass
    return "cpu"


def format_shape(shape):
    """Format shape tuple as string"""
    return "x".join(str(s) for s in shape)


def benchmark(shape, indexing="ij", warmup=200, runs=1000):
    """Run performance benchmark"""
    device = get_device()
    shape_str = format_shape(shape)
    print(f"\n{shape_str} {indexing} on {device}:")

    tensors = [torch.randn(s, device=device) for s in shape]

    # Warmup
    for _ in range(warmup):
        _ = meshgrid(tensors, indexing=indexing)
        _ = torch.meshgrid(tensors, indexing=indexing)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Use CUDA Event for precise timing
    if device == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
    else:
        starter = ender = None

    our_times = []
    torch_times = []

    # Test separately to avoid interference
    # Test FlagGems first
    for _ in range(runs):
        if device == "cuda":
            starter.record()
            result = meshgrid(tensors, indexing=indexing)
            ender.record()
            torch.cuda.synchronize()
            _ = result
            our_times.append(starter.elapsed_time(ender))
        else:
            start = time.perf_counter()
            result = meshgrid(tensors, indexing=indexing)
            _ = result
            our_times.append((time.perf_counter() - start) * 1000)

    # Then test PyTorch
    for _ in range(runs):
        if device == "cuda":
            starter.record()
            result = torch.meshgrid(tensors, indexing=indexing)
            ender.record()
            torch.cuda.synchronize()
            _ = result
            torch_times.append(starter.elapsed_time(ender))
        else:
            start = time.perf_counter()
            result = torch.meshgrid(tensors, indexing=indexing)
            _ = result
            torch_times.append((time.perf_counter() - start) * 1000)

    # Statistics
    our_median = np.median(our_times)
    torch_median = np.median(torch_times)
    speedup = torch_median / our_median

    # Calculate percentage difference
    pct_diff = (our_median - torch_median) / torch_median * 100

    print(f"  FlagGems (median): {our_median:.4f} ms")
    print(f"  PyTorch  (median): {torch_median:.4f} ms")
    print(f"  Speedup:           {speedup:.2f}x")
    print(f"  Difference:        {pct_diff:+.1f}%")
    print(f"  (FlagGems mean±std: {np.mean(our_times):.4f}±{np.std(our_times):.4f} ms)")
    print(
        f"  (PyTorch mean±std:  {np.mean(torch_times):.4f}±{np.std(torch_times):.4f} ms)"
    )

    return our_median, torch_median, speedup


def main():
    """Main test function"""
    print("=" * 70)
    print("Meshgrid Performance Benchmark")
    print(f"Device: {get_device()}")
    print(f"PyTorch Version: {torch.__version__}")
    print("=" * 70)

    # Test cases
    cases = [
        ((10, 10), "ij"),
        ((10, 10), "xy"),
        ((20, 20), "ij"),
        ((50, 50), "ij"),
        ((100, 100), "ij"),
        ((200, 200), "ij"),
        ((500, 500), "ij"),
        ((1000, 1000), "ij"),
        ((32, 32, 32), "ij"),
        ((32, 32, 32), "xy"),
        ((10, 10, 10, 10), "ij"),  # 4D
        ((5, 5, 5, 5, 5), "ij"),  # 5D
    ]

    all_results = []

    for shape, indexing in cases:
        try:
            our_time, torch_time, speedup = benchmark(shape, indexing)
            all_results.append((shape, indexing, speedup, our_time, torch_time))
        except Exception as e:
            print(f"  ⚠️  Benchmark failed for {shape}: {e}")
        print("-" * 70)

    # Summary
    print("\n" + "=" * 70)
    print("Summary of Results:")
    print("=" * 70)
    print(f"{'Shape':<15} {'Mode':<6} {'Speedup':<10} {'Status'}")
    print("-" * 70)

    total_speedup = 0
    count = 0
    fast_count = 0

    for shape, indexing, speedup, our_time, torch_time in all_results:
        shape_str = format_shape(shape)
        if speedup >= 1.2:
            status = "🚀 EXCELLENT"
            fast_count += 1
        elif speedup >= 1.05:
            status = "✅ GOOD"
            fast_count += 1
        elif speedup >= 0.95:
            status = "✅ PASS"
        else:
            status = "⚠️  SLOW"
        print(f"{shape_str:<15} {indexing:<6} {speedup:>6.2f}x     {status}")
        if speedup > 0.5:
            total_speedup += speedup
            count += 1

    if count > 0:
        avg_speedup = total_speedup / count
        print("-" * 70)
        print(f"Average Speedup:     {avg_speedup:.2f}x")
        print(f"Fast Cases (>1.05x): {fast_count}/{len(all_results)}")

    print("=" * 70)
    print("Note: Speedup = PyTorch_time / FlagGems_time")
    print("Speedup > 1.0 means FlagGems is faster than PyTorch")
    print("=" * 70)


if __name__ == "__main__":
    main()

# meshgrid benchmark complete

# meshgrid benchmark complete
