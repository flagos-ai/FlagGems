"""
Performance benchmarks for Easy-difficulty operators.
Compares FlagGems implementations vs PyTorch native.

Usage:
    pytest benchmark/bench_easy_ops.py
    # or run directly:
    python -m pytest benchmark/bench_easy_ops.py -v
"""

import pytest
import torch

import flag_gems

DEVICE = flag_gems.device
WARMUP = 10
REPEATS = 100

SHAPES = [
    (1, 1),
    (8, 8),
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
]


# ---------------------------------------------------------------------------
# log10
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_perf_log10(shape, dtype, benchmark):
    inp = torch.rand(shape, dtype=dtype, device=DEVICE) + 1e-3
    with flag_gems.use_gems():
        benchmark(torch.log10, inp)


# ---------------------------------------------------------------------------
# logaddexp
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_perf_logaddexp(shape, dtype, benchmark):
    a = torch.randn(shape, dtype=dtype, device=DEVICE)
    b = torch.randn(shape, dtype=dtype, device=DEVICE)
    with flag_gems.use_gems():
        benchmark(torch.logaddexp, a, b)


# ---------------------------------------------------------------------------
# cosh
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_perf_cosh(shape, dtype, benchmark):
    inp = torch.randn(shape, dtype=dtype, device=DEVICE)
    with flag_gems.use_gems():
        benchmark(torch.cosh, inp)


# ---------------------------------------------------------------------------
# gcd
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_perf_gcd(shape, dtype, benchmark):
    a = torch.randint(1, 1000, shape, dtype=dtype, device=DEVICE)
    b = torch.randint(1, 1000, shape, dtype=dtype, device=DEVICE)
    with flag_gems.use_gems():
        benchmark(torch.gcd, a, b)


# ---------------------------------------------------------------------------
# tril
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_perf_tril(shape, dtype, benchmark):
    inp = torch.randn(shape, dtype=dtype, device=DEVICE)
    with flag_gems.use_gems():
        benchmark(torch.tril, inp)


# ---------------------------------------------------------------------------
# roll
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_perf_roll(shape, dtype, benchmark):
    inp = torch.randn(shape, dtype=dtype, device=DEVICE)
    shifts = [shape[0] // 4, shape[1] // 4]
    with flag_gems.use_gems():
        benchmark(torch.roll, inp, shifts, [0, 1])


# ---------------------------------------------------------------------------
# leaky_relu
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_perf_leaky_relu(shape, dtype, benchmark):
    inp = torch.randn(shape, dtype=dtype, device=DEVICE)
    with flag_gems.use_gems():
        benchmark(torch.nn.functional.leaky_relu, inp)


# ---------------------------------------------------------------------------
# asinh
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_perf_asinh(shape, dtype, benchmark):
    inp = torch.randn(shape, dtype=dtype, device=DEVICE)
    with flag_gems.use_gems():
        benchmark(torch.asinh, inp)
