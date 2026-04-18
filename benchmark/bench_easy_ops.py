"""
Performance benchmarks for Easy-difficulty operators.
Compares FlagGems implementations vs PyTorch native.

Usage:
    python benchmark/bench_easy_ops.py
"""

import time
from typing import Callable, List, Tuple

import torch

import flag_gems
from flag_gems.ops.log10 import log10
from flag_gems.ops.logaddexp import logaddexp
from flag_gems.ops.cosh import cosh
from flag_gems.ops.gcd import gcd
from flag_gems.ops.roll import roll
from flag_gems.ops.leaky_relu import leaky_relu
from flag_gems.ops.asinh import asinh

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ops.tril import tril

DEVICE = flag_gems.device
WARMUP = 10
REPEATS = 100
DTYPE = torch.float32


def _bench(fn: Callable, *args, warmup: int = WARMUP, repeats: int = REPEATS) -> float:
    """Return median latency in milliseconds."""
    for _ in range(warmup):
        fn(*args)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        if DEVICE.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(*args)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            fn(*args)
            times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


SHAPES: List[Tuple] = [
    (1, 1),
    (8, 8),
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
]


def run_unary(name: str, gems_fn: Callable, torch_fn: Callable, shapes=SHAPES):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"{'Shape':<20} {'PyTorch (ms)':>14} {'FlagGems (ms)':>14} {'Speedup':>10}")
    print(f"{'-'*60}")
    for shape in shapes:
        x = torch.rand(shape, dtype=DTYPE, device=DEVICE) + 1e-3
        t_torch = _bench(torch_fn, x)
        t_gems = _bench(gems_fn, x)
        speedup = t_torch / t_gems if t_gems > 0 else float("inf")
        print(f"{str(shape):<20} {t_torch:>14.4f} {t_gems:>14.4f} {speedup:>10.3f}x")


def run_binary(name: str, gems_fn: Callable, torch_fn: Callable, shapes=SHAPES):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"{'Shape':<20} {'PyTorch (ms)':>14} {'FlagGems (ms)':>14} {'Speedup':>10}")
    print(f"{'-'*60}")
    for shape in shapes:
        a = torch.randn(shape, dtype=DTYPE, device=DEVICE)
        b = torch.randn(shape, dtype=DTYPE, device=DEVICE)
        t_torch = _bench(torch_fn, a, b)
        t_gems = _bench(gems_fn, a, b)
        speedup = t_torch / t_gems if t_gems > 0 else float("inf")
        print(f"{str(shape):<20} {t_torch:>14.4f} {t_gems:>14.4f} {speedup:>10.3f}x")


def bench_roll():
    print(f"\n{'='*60}")
    print(f"  roll")
    print(f"{'='*60}")
    print(f"{'Shape':<20} {'PyTorch (ms)':>14} {'FlagGems (ms)':>14} {'Speedup':>10}")
    print(f"{'-'*60}")
    for shape in SHAPES:
        x = torch.randn(shape, dtype=DTYPE, device=DEVICE)
        t_torch = _bench(torch.roll, x, [shape[0] // 4, shape[1] // 4], [0, 1])
        t_gems = _bench(roll, x, [shape[0] // 4, shape[1] // 4], [0, 1])
        speedup = t_torch / t_gems if t_gems > 0 else float("inf")
        print(f"{str(shape):<20} {t_torch:>14.4f} {t_gems:>14.4f} {speedup:>10.3f}x")


def bench_tril():
    print(f"\n{'='*60}")
    print(f"  tril")
    print(f"{'='*60}")
    print(f"{'Shape':<20} {'PyTorch (ms)':>14} {'FlagGems (ms)':>14} {'Speedup':>10}")
    print(f"{'-'*60}")
    for shape in SHAPES:
        x = torch.randn(shape, dtype=DTYPE, device=DEVICE)
        t_torch = _bench(torch.tril, x)
        t_gems = _bench(tril, x)
        speedup = t_torch / t_gems if t_gems > 0 else float("inf")
        print(f"{str(shape):<20} {t_torch:>14.4f} {t_gems:>14.4f} {speedup:>10.3f}x")


def bench_gcd():
    print(f"\n{'='*60}")
    print(f"  gcd")
    print(f"{'='*60}")
    print(f"{'Shape':<20} {'PyTorch (ms)':>14} {'FlagGems (ms)':>14} {'Speedup':>10}")
    print(f"{'-'*60}")
    for shape in SHAPES:
        a = torch.randint(1, 1000, shape, dtype=torch.int32, device=DEVICE)
        b = torch.randint(1, 1000, shape, dtype=torch.int32, device=DEVICE)
        t_torch = _bench(torch.gcd, a, b)
        t_gems = _bench(gcd, a, b)
        speedup = t_torch / t_gems if t_gems > 0 else float("inf")
        print(f"{str(shape):<20} {t_torch:>14.4f} {t_gems:>14.4f} {speedup:>10.3f}x")


if __name__ == "__main__":
    print(f"Device: {DEVICE}  |  dtype: {DTYPE}")
    run_unary("log10", log10, torch.log10)
    run_binary("logaddexp", logaddexp, torch.logaddexp)
    run_unary("cosh", cosh, torch.cosh)
    bench_gcd()
    bench_tril()
    bench_roll()
    run_unary("leaky_relu", leaky_relu, lambda x: torch.nn.functional.leaky_relu(x))
    run_unary("asinh", asinh, torch.asinh)
