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


def benchmark(shape, indexing="ij", warmup=50, runs=500):
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
            # Ensure result is computed
            _ = result[0].cpu() if result else None
            our_times.append(starter.elapsed_time(ender))
        else:
            start = time.perf_counter()
            result = meshgrid(tensors, indexing=indexing)
            _ = result
            our_times.append((time.perf_counter() - start) * 1000)

    # Clear cache
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Then test PyTorch
    for _ in range(runs):
        if device == "cuda":
            starter.record()
            result = torch.meshgrid(tensors, indexing=indexing)
            ender.record()
            torch.cuda.synchronize()
            _ = result[0].cpu() if result else None
            torch_times.append(starter.elapsed_time(ender))
        else:
            start = time.perf_counter()
            result = torch.meshgrid(tensors, indexing=indexing)
            _ = result
            torch_times.append((time.perf_counter() - start) * 1000)

    # Filter outliers (remove top and bottom 5%)
    our_times_sorted = sorted(our_times)
    torch_times_sorted = sorted(torch_times)
    cut = int(0.05 * len(our_times_sorted))
    our_times_filtered = our_times_sorted[cut:-cut] if cut > 0 else our_times_sorted
    torch_times_filtered = torch_times_sorted[cut:-cut] if cut > 0 else torch_times_sorted

    # Statistics
    our_median = np.median(our_times_filtered) if our_times_filtered else np.median(our_times)
    torch_median = np.median(torch_times_filtered) if torch_times_filtered else np.median(torch_times)
    speedup = torch_median / our_median if our_median > 0 else 0

    # Calculate percentage difference
    pct_diff = (our_median - torch_median) / torch_median * 100 if torch_median > 0 else 0

    print(f"  FlagGems (median): {our_median:.4f} ms")
    print(f"  PyTorch  (median): {torch_median:.4f} ms")
    print(f"  Speedup:           {speedup:.2f}x")
    print(f"  Difference:        {pct_diff:+.1f}%")
    print(f"  (FlagGems mean+-std: {np.mean(our_times):.4f}+-{np.std(our_times):.4f} ms)")
    print(f"  (PyTorch mean+-std:  {np.mean(torch_times):.4f}+-{np.std(torch_times):.4f} ms)")

    return our_median, torch_median, speedup


def main():
    """Main test function"""
    print("=" * 70)
    print("Meshgrid Performance Benchmark")
    print(f"Device: {get_device()}")
    print(f"PyTorch Version: {torch.__version__}")
    print("=" * 70)

    # Test cases focusing on common usage patterns
    cases = [
        # 2D cases
        ((10, 10), "ij"),
        ((10, 10), "xy"),
        ((32, 32), "ij"),
        ((64, 64), "ij"),
        ((128, 128), "ij"),
        ((256, 256), "ij"),
        ((512, 512), "ij"),
        ((1024, 1024), "ij"),
        # 3D cases
        ((16, 16, 16), "ij"),
        ((32, 32, 32), "ij"),
        ((64, 64, 64), "ij"),
        # 4D cases
        ((8, 8, 8, 8), "ij"),
        ((16, 16, 16, 16), "ij"),
    ]

    all_results = []

    for shape, indexing in cases:
        try:
            our_time, torch_time, speedup = benchmark(shape, indexing)
            all_results.append((shape, indexing, speedup, our_time, torch_time))
        except Exception as e:
            print(f"  [WARNING] Benchmark failed for {shape}: {e}")
            import traceback
            traceback.print_exc()
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
    excellent_count = 0

    for shape, indexing, speedup, our_time, torch_time in all_results:
        shape_str = format_shape(shape)
        if speedup >= 1.5:
            status = "EXCELLENT"
            excellent_count += 1
            fast_count += 1
        elif speedup >= 1.1:
            status = "GOOD"
            fast_count += 1
        elif speedup >= 0.95:
            status = "PASS"
        else:
            status = "SLOW"
        print(f"{shape_str:<15} {indexing:<6} {speedup:>6.2f}x     {status}")
        if speedup > 0.01:
            total_speedup += speedup
            count += 1

    if count > 0:
        avg_speedup = total_speedup / count
        print("-" * 70)
        print(f"Average Speedup:     {avg_speedup:.2f}x")
        print(f"Fast Cases (>1.1x):  {fast_count}/{len(all_results)}")
        print(f"Excellent (>1.5x):   {excellent_count}/{len(all_results)}")

    print("=" * 70)
    print("Note: Speedup = PyTorch_time / FlagGems_time")
    print("Speedup > 1.0 means FlagGems is faster than PyTorch")
    print("=" * 70)


if __name__ == "__main__":
    main()
