"""
Performance benchmarks for Medium-difficulty operators.
Compares FlagGems implementations vs PyTorch native.

Usage:
    pytest benchmark/bench_medium_ops.py
    # or run directly:
    python -m pytest benchmark/bench_medium_ops.py -v
"""

import pytest
import torch
import torch.nn.functional as F

import flag_gems

DEVICE = flag_gems.device


# ---------------------------------------------------------------------------
# upsample_nearest2d
# ---------------------------------------------------------------------------
UPSAMPLE_CONFIGS = [
    ((1, 3, 8, 8), [16, 16]),
    ((1, 3, 32, 32), [64, 64]),
    ((2, 16, 64, 64), [256, 256]),
    ((4, 32, 128, 128), [512, 512]),
]


@pytest.mark.parametrize("shape, out_size", UPSAMPLE_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_perf_upsample_nearest2d(shape, out_size, dtype, benchmark):
    inp = torch.randn(shape, dtype=dtype, device=DEVICE)
    with flag_gems.use_gems():
        benchmark(F.interpolate, inp, size=out_size, mode="nearest")


# ---------------------------------------------------------------------------
# pixel_shuffle
# ---------------------------------------------------------------------------
PIXEL_SHUFFLE_CONFIGS = [
    (1, 4, 32, 32, 2),
    (2, 9, 64, 64, 3),
    (4, 4, 128, 128, 2),
    (1, 16, 64, 64, 4),
]


@pytest.mark.parametrize("n, c, h, w, r", PIXEL_SHUFFLE_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_perf_pixel_shuffle(n, c, h, w, r, dtype, benchmark):
    inp = torch.randn(n, c * r * r, h, w, dtype=dtype, device=DEVICE)
    with flag_gems.use_gems():
        benchmark(torch.pixel_shuffle, inp, r)


# ---------------------------------------------------------------------------
# scatter_reduce
# ---------------------------------------------------------------------------
SCATTER_REDUCE_CONFIGS = [
    (64, 64),
    (256, 256),
    (1024, 1024),
]


@pytest.mark.parametrize("rows, cols", SCATTER_REDUCE_CONFIGS)
@pytest.mark.parametrize("reduce", ["sum", "amax"])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_perf_scatter_reduce(rows, cols, reduce, dtype, benchmark):
    src = torch.randn(rows, cols, dtype=dtype, device=DEVICE)
    idx = torch.randint(0, rows, (rows, cols), device=DEVICE)
    base = torch.zeros(rows, cols, dtype=dtype, device=DEVICE)
    with flag_gems.use_gems():
        benchmark(torch.scatter_reduce, base, 0, idx, src, reduce=reduce)


# ---------------------------------------------------------------------------
# median
# ---------------------------------------------------------------------------
MEDIAN_CONFIGS = [
    (64, 64, 0),
    (256, 256, 1),
    (1024, 1024, 0),
]


@pytest.mark.parametrize("rows, cols, dim", MEDIAN_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_perf_median(rows, cols, dim, dtype, benchmark):
    inp = torch.randn(rows, cols, dtype=dtype, device=DEVICE)
    with flag_gems.use_gems():
        benchmark(torch.median, inp, dim=dim)


# ---------------------------------------------------------------------------
# smooth_l1_loss
# ---------------------------------------------------------------------------
SMOOTH_L1_SHAPES = [
    (64,),
    (256, 256),
    (1024, 1024),
]


@pytest.mark.parametrize("shape", SMOOTH_L1_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_perf_smooth_l1_loss(shape, dtype, benchmark):
    inp = torch.randn(shape, dtype=dtype, device=DEVICE)
    target = torch.randn(shape, dtype=dtype, device=DEVICE)
    with flag_gems.use_gems():
        benchmark(F.smooth_l1_loss, inp, target)


# ---------------------------------------------------------------------------
# avg_pool3d
# ---------------------------------------------------------------------------
AVGPOOL3D_CONFIGS = [
    ((1, 4, 8, 8, 8), 2, 2),
    ((2, 8, 16, 16, 16), 3, 1),
    ((2, 16, 32, 32, 32), 2, 2),
]


@pytest.mark.parametrize("shape, k, s", AVGPOOL3D_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_perf_avg_pool3d(shape, k, s, dtype, benchmark):
    inp = torch.randn(shape, dtype=dtype, device=DEVICE)
    with flag_gems.use_gems():
        benchmark(F.avg_pool3d, inp, k, stride=s)


# ---------------------------------------------------------------------------
# max_pool3d
# ---------------------------------------------------------------------------
MAXPOOL3D_CONFIGS = [
    ((1, 4, 8, 8, 8), 2, 2),
    ((2, 8, 16, 16, 16), 3, 1),
    ((2, 16, 32, 32, 32), 2, 2),
]


@pytest.mark.parametrize("shape, k, s", MAXPOOL3D_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_perf_max_pool3d(shape, k, s, dtype, benchmark):
    inp = torch.randn(shape, dtype=dtype, device=DEVICE)
    with flag_gems.use_gems():
        benchmark(F.max_pool3d, inp, k, stride=s)


# ---------------------------------------------------------------------------
# conv_transpose2d
# ---------------------------------------------------------------------------
CONV_T2D_CONFIGS = [
    ((1, 4, 16, 16), (4, 2, 3, 3), 1),
    ((2, 8, 32, 32), (8, 4, 3, 3), 2),
    ((1, 16, 64, 64), (16, 8, 3, 3), 2),
]


@pytest.mark.parametrize("in_shape, w_shape, stride", CONV_T2D_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_perf_conv_transpose2d(in_shape, w_shape, stride, dtype, benchmark):
    inp = torch.randn(in_shape, dtype=dtype, device=DEVICE)
    weight = torch.randn(w_shape, dtype=dtype, device=DEVICE)
    with flag_gems.use_gems():
        benchmark(F.conv_transpose2d, inp, weight, stride=stride)
