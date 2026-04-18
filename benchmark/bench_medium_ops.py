"""
Performance benchmarks for Medium-difficulty operators.
Compares FlagGems implementations vs PyTorch native.

Usage:
    python benchmark/bench_medium_ops.py
"""

import time
from typing import Callable

import torch
import torch.nn.functional as F

import flag_gems

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ops.upsample_nearest2d import upsample_nearest2d
from ops.median import median
from ops.smooth_l1_loss import smooth_l1_loss, _MEAN
from ops.pixel_shuffle import pixel_shuffle
from ops.avg_pool3d import avg_pool3d
from ops.max_pool3d import max_pool3d

DEVICE = flag_gems.device
WARMUP = 10
REPEATS = 100
DTYPE = torch.float32


def _bench(fn: Callable, *args, warmup=WARMUP, repeats=REPEATS) -> float:
    for _ in range(warmup):
        fn(*args)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        if DEVICE.type == "cuda":
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(*args); e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter(); fn(*args)
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def _header(name):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    print(f"{'Config':<30} {'PyTorch (ms)':>14} {'FlagGems (ms)':>14} {'Speedup':>10}")
    print("-" * 60)


def bench_upsample():
    _header("upsample_nearest2d")
    configs = [
        ((1, 3, 8, 8), [16, 16]),
        ((1, 3, 32, 32), [64, 64]),
        ((2, 16, 64, 64), [256, 256]),
        ((4, 32, 128, 128), [512, 512]),
    ]
    for (shape, out_size) in configs:
        x = torch.randn(shape, dtype=DTYPE, device=DEVICE)
        t_torch = _bench(lambda: F.interpolate(x, size=out_size, mode="nearest"))
        t_gems = _bench(upsample_nearest2d, x, out_size)
        sp = t_torch / t_gems if t_gems > 0 else float("inf")
        print(f"{str(shape)+' -> '+str(out_size):<30} {t_torch:>14.4f} {t_gems:>14.4f} {sp:>10.3f}x")


def bench_median():
    _header("median")
    configs = [
        (64, 64, 0),
        (256, 256, 1),
        (1024, 1024, 0),
        (4096, 4096, 1),
    ]
    for (r, c, dim) in configs:
        x = torch.randn(r, c, dtype=DTYPE, device=DEVICE)
        t_torch = _bench(lambda: torch.median(x, dim=dim))
        t_gems = _bench(median, x, dim)
        sp = t_torch / t_gems if t_gems > 0 else float("inf")
        print(f"{f'({r},{c}) dim={dim}':<30} {t_torch:>14.4f} {t_gems:>14.4f} {sp:>10.3f}x")


def bench_smooth_l1():
    _header("smooth_l1_loss")
    shapes = [(64,), (256, 256), (1024, 1024)]
    for shape in shapes:
        x = torch.randn(shape, dtype=DTYPE, device=DEVICE)
        y = torch.randn(shape, dtype=DTYPE, device=DEVICE)
        t_torch = _bench(lambda: F.smooth_l1_loss(x, y))
        t_gems = _bench(smooth_l1_loss, x, y, _MEAN, 1.0)
        sp = t_torch / t_gems if t_gems > 0 else float("inf")
        print(f"{str(shape):<30} {t_torch:>14.4f} {t_gems:>14.4f} {sp:>10.3f}x")


def bench_pixel_shuffle():
    _header("pixel_shuffle")
    configs = [(1, 4, 32, 32, 2), (2, 9, 64, 64, 3), (4, 4, 128, 128, 2)]
    for (n, c, h, w, r) in configs:
        x = torch.randn(n, c * r * r, h, w, dtype=DTYPE, device=DEVICE)
        t_torch = _bench(lambda: F.pixel_shuffle(x, r))
        t_gems = _bench(pixel_shuffle, x, r)
        sp = t_torch / t_gems if t_gems > 0 else float("inf")
        label = f"({n},{c*r*r},{h},{w}) r={r}"
        print(f"{label:<30} {t_torch:>14.4f} {t_gems:>14.4f} {sp:>10.3f}x")


def bench_avg_pool3d():
    _header("avg_pool3d")
    configs = [
        ((1, 4, 8, 8, 8), 2, 2),
        ((2, 8, 16, 16, 16), 3, 1),
        ((2, 16, 32, 32, 32), 2, 2),
    ]
    for (shape, k, s) in configs:
        x = torch.randn(shape, dtype=DTYPE, device=DEVICE)
        t_torch = _bench(lambda: F.avg_pool3d(x, k, stride=s))
        t_gems = _bench(avg_pool3d, x, k, s)
        sp = t_torch / t_gems if t_gems > 0 else float("inf")
        print(f"{str(shape):<30} {t_torch:>14.4f} {t_gems:>14.4f} {sp:>10.3f}x")


def bench_max_pool3d():
    _header("max_pool3d")
    configs = [
        ((1, 4, 8, 8, 8), 2, 2),
        ((2, 8, 16, 16, 16), 3, 1),
        ((2, 16, 32, 32, 32), 2, 2),
    ]
    for (shape, k, s) in configs:
        x = torch.randn(shape, dtype=DTYPE, device=DEVICE)
        t_torch = _bench(lambda: F.max_pool3d(x, k, stride=s))
        t_gems = _bench(max_pool3d, x, k, s)
        sp = t_torch / t_gems if t_gems > 0 else float("inf")
        print(f"{str(shape):<30} {t_torch:>14.4f} {t_gems:>14.4f} {sp:>10.3f}x")


if __name__ == "__main__":
    print(f"Device: {DEVICE}  |  dtype: {DTYPE}")
    bench_upsample()
    bench_median()
    bench_smooth_l1()
    bench_pixel_shuffle()
    bench_avg_pool3d()
    bench_max_pool3d()
