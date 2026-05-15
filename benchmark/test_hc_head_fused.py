"""Benchmark: Triton hc_head_fused kernel vs PyTorch reference (naive multi-op baseline).

Baseline: Pure PyTorch implementation using separate operations:
  - torch.matmul for dot products
  - torch.rsqrt for RMS normalization
  - torch.sigmoid for activation
  - Broadcasting + torch.sum for weighted sum

This baseline represents the naive approach without operator fusion, which is
what a typical PyTorch user would write. The Triton kernel fuses all these
operations into a single kernel launch, eliminating intermediate tensor
allocations and redundant memory traffic.

Production MHC shapes for DeepSeek-V4:
  - hidden_size: 1280, 2560, 4096 (varies by model variant)
  - hc_mult: 2 or 4 (number of heads to compress)
  - num_tokens: 1-4096 (decode=1, prefill=up to 4096)

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmark/benchmark_hc_head_fused.py
"""

import time

import torch

from flag_gems.fused.mhc.hc_head_fused_kernel import hc_head_fused_kernel, hc_head_fused_kernel_ref


def benchmark_fn(fn, warmup=10, rep=100):
    """Benchmark a function, return median time in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1e6)

    times.sort()
    return times[len(times) // 2]


def run_benchmark(n, hidden_size, hc_mult, dtype, rms_eps=1e-6, hc_eps=1e-6):
    """Benchmark one configuration."""
    torch.manual_seed(42)

    hs_flat = torch.randn(n, hc_mult, hidden_size, device="cuda", dtype=dtype)
    fn = (
        torch.randn(hc_mult, hc_mult * hidden_size, device="cuda", dtype=torch.float32)
        * 0.01
    )
    hc_scale = torch.randn(1, device="cuda", dtype=torch.float32) * 0.1
    hc_base = torch.randn(hc_mult, device="cuda", dtype=torch.float32) * 0.1
    out = torch.empty(n, hidden_size, device="cuda", dtype=dtype)

    def triton_fn():
        hc_head_fused_kernel(
            hs_flat, fn, hc_scale, hc_base, out, hidden_size, rms_eps, hc_eps, hc_mult
        )

    def ref_fn():
        hc_head_fused_kernel_ref(
            hs_flat, fn, hc_scale, hc_base, out, hidden_size, rms_eps, hc_eps, hc_mult
        )

    triton_us = benchmark_fn(triton_fn)
    ref_us = benchmark_fn(ref_fn)
    speedup = ref_us / triton_us if triton_us > 0 else 0.0

    return triton_us, ref_us, speedup


def main():
    # DeepSeek-V4 MHC production shapes
    test_configs = [
        # (num_tokens, hidden_size, hc_mult)
        (256, 1280, 2),
        (256, 1280, 4),
        (512, 1280, 2),
        (512, 1280, 4),
        (512, 2560, 2),
        (512, 2560, 4),
        (1024, 2560, 2),
        (1024, 2560, 4),
        (2048, 4096, 2),
        (2048, 4096, 4),
        (4096, 4096, 2),
        (4096, 4096, 4),
    ]
    dtypes = [torch.float16, torch.bfloat16]

    print("=" * 90)
    print("HC Head Fused Kernel Benchmark (Triton vs PyTorch Baseline)")
    print("Baseline: naive PyTorch (matmul + rsqrt + sigmoid + broadcast sum)")
    print("=" * 90)
    print(
        f"{'n':>6} {'h':>6} {'hc':>4} {'dtype':>8} "
        f"{'triton_us':>12} {'baseline_us':>12} {'speedup':>10}"
    )
    print("-" * 90)

    for n, h, hc in test_configs:
        for dtype in dtypes:
            triton_us, ref_us, speedup = run_benchmark(n, h, hc, dtype)
            dtype_str = str(dtype).split(".")[-1]
            print(
                f"{n:>6} {h:>6} {hc:>4} {dtype_str:>8} "
                f"{triton_us:>12.1f} {ref_us:>12.1f} {speedup:>10.4f}x"
            )

    print("-" * 90)


if __name__ == "__main__":
    main()
