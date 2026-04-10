"""
Performance benchmark for logaddexp (FlagGems · 赛道一 · 初级算子)

Timing method: torch.cuda.Event (GPU async safe)
Speedup requirement: ≥ 0.9 (competition threshold)
"""

from __future__ import annotations

import os
import sys

import torch

import flag_gems
from flag_gems.experimental_ops.logaddexp import logaddexp

# ---------------------------------------------------------------------------
# Try to use FlagGems official benchmark framework
# ---------------------------------------------------------------------------
_FLAGGEMS_ROOT = os.path.join(os.path.dirname(__file__), "..", "FlagGems")
_FLAGGEMS_BENCH = os.path.join(_FLAGGEMS_ROOT, "benchmark")
if os.path.isdir(_FLAGGEMS_BENCH) and _FLAGGEMS_ROOT not in sys.path:
    sys.path.insert(0, _FLAGGEMS_ROOT)

try:
    from benchmark.performance_utils import GenericBenchmark

    _HAS_OFFICIAL_BENCH = True
except ImportError:
    _HAS_OFFICIAL_BENCH = False

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------
BENCHMARK_SHAPES = [
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    (1024 * 1024,),
    (64, 64, 64),
    (2, 1024, 1024),
]

FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


# ===========================================================================
# CUDA Event precise timing (median latency)
# ===========================================================================
def _bench(
    op, x: torch.Tensor, y: torch.Tensor, warmup: int = 50, rep: int = 200
) -> float:
    """Return median latency (ms). Must use CUDA Event, time.time() is inaccurate for GPU async."""
    for _ in range(warmup):
        op(x, y)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    lats = []
    for _ in range(rep):
        start.record()
        op(x, y)
        end.record()
        torch.cuda.synchronize()
        lats.append(start.elapsed_time(end))

    lats.sort()
    return lats[len(lats) // 2]


# ===========================================================================
# Official framework path
# ===========================================================================
def run_official():
    def binary_input_fn(shape, dtype, device):
        """Input generation for binary operator."""
        x = torch.rand(shape, dtype=dtype, device=device).abs() + 1e-3
        y = torch.rand(shape, dtype=dtype, device=device).abs() + 1e-3
        yield x, y

    bench = GenericBenchmark(
        op_name="logaddexp",
        torch_op=torch.logaddexp,
        input_fn=binary_input_fn,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(logaddexp)
    bench.run()


# ===========================================================================
# Standalone benchmark (fallback when official framework unavailable)
# ===========================================================================
def run_standalone():
    W = 72
    print("=" * W)
    print("  logaddexp Benchmark — Iluvatar BI-V150 / corex-4.4.0")
    print("  Timing: torch.cuda.Event  |  Stats: 200 runs median latency")
    print("=" * W)
    print(
        f"  {'Shape':<20} {'Dtype':<12} {'PyTorch':>10}  {'FlagGems':>10}  {'Speedup':>8}  Status"
    )
    print("-" * W)

    speedups = []
    for dtype in FLOAT_DTYPES:
        for shape in BENCHMARK_SHAPES:
            x = (torch.rand(shape, dtype=torch.float32, device="cuda").abs() + 1e-3).to(
                dtype
            )
            y = (torch.rand(shape, dtype=torch.float32, device="cuda").abs() + 1e-3).to(
                dtype
            )

            t_ref = _bench(torch.logaddexp, x, y)
            t_gems = _bench(logaddexp, x, y)
            sp = t_ref / t_gems
            speedups.append(sp)
            status = "✅" if sp >= 0.9 else "❌"

            shape_s = "×".join(map(str, shape))
            print(
                f"  {shape_s:<20} {str(dtype):<12} "
                f"{t_ref:>9.4f}ms  {t_gems:>9.4f}ms  {sp:>7.3f}x  {status}"
            )
        print()

    avg = sum(speedups) / len(speedups)
    mn = min(speedups)
    print("=" * W)
    print(f"  Average speedup: {avg:.3f}x")
    print(f"  Minimum speedup: {mn:.3f}x")
    print(f"  Overall result  : {'✅ PASS (≥0.9)' if mn >= 0.9 else '❌ FAIL (<0.9)'}")
    print("=" * W)

    # Memory bandwidth utilization (float32 1024×1024)
    x32 = torch.rand(1024, 1024, dtype=torch.float32, device="cuda").abs() + 1e-3
    y32 = torch.rand(1024, 1024, dtype=torch.float32, device="cuda").abs() + 1e-3
    t_ms = _bench(logaddexp, x32, y32)
    # Read 2*N elements + Write N elements
    gbps = 3 * x32.numel() * x32.element_size() / (t_ms * 1e-3) / 1e9
    print(f"\n  Memory bandwidth (fp32, 1024×1024): {gbps:.1f} GB/s  [{t_ms:.4f} ms]")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA unavailable, please check GPU environment.")
        sys.exit(1)

    print(f"Device   : {torch.cuda.get_device_name(0)}")
    print(f"FlagGems : {flag_gems.__version__}")
    print(
        f"Mode     : {'Official GenericBenchmark' if _HAS_OFFICIAL_BENCH else 'Standalone CUDA-Event benchmark'}\n"
    )

    if _HAS_OFFICIAL_BENCH:
        run_official()
    else:
        run_standalone()
