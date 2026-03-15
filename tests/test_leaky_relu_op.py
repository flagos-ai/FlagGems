# [FlagGems Operator Development Competition]
# Tests for : leaky_relu
# Run with  : pytest tests/test_leaky_relu_op.py -v

import torch
import pytest
from flag_gems.ops.leaky_relu import leaky_relu

device          = "cuda"
shapes          = [(64, 64), (256, 256), (1024, 1024), (4096, 4096)]
N_WARMUP        = 50
N_RUNS          = 500
FALLBACK_THRESH = 2_097_152  # must match leaky_relu.py


def allclose(out, ref, dtype):
    atol = {
        torch.float32:  1.3e-6,
        torch.float16:  1e-3,
        torch.bfloat16: 0.016,
    }[dtype]
    return torch.allclose(out.float(), ref.float(), rtol=1e-4, atol=atol)


# ------------------------------------------------------------------
# Correctness — dtype
# ------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_correctness_dtype(dtype):
    x   = torch.randn(1024, 1024, device=device, dtype=dtype)
    ref = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
    out = leaky_relu(x, negative_slope=0.01)
    assert allclose(out, ref, dtype), f"FAIL dtype={dtype}"


# ------------------------------------------------------------------
# Correctness — shapes
# ------------------------------------------------------------------
@pytest.mark.parametrize("shape", shapes)
def test_correctness_shape(shape):
    x   = torch.randn(shape, device=device, dtype=torch.float32)
    ref = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
    out = leaky_relu(x, negative_slope=0.01)
    assert allclose(out, ref, torch.float32), f"FAIL shape={shape}"


# ------------------------------------------------------------------
# Correctness — slopes
# ------------------------------------------------------------------
@pytest.mark.parametrize("slope", [0.01, 0.1, 0.2, 0.5, -0.1])
def test_correctness_slope(slope):
    x   = torch.randn(1024, 1024, device=device)
    ref = torch.nn.functional.leaky_relu(x, negative_slope=slope)
    out = leaky_relu(x, negative_slope=slope)
    assert allclose(out, ref, torch.float32), f"FAIL slope={slope}"


# ------------------------------------------------------------------
# Correctness — inplace
# ------------------------------------------------------------------
def test_inplace():
    x     = torch.randn(1024, 1024, device=device)
    x_ref = x.clone()
    leaky_relu(x, inplace=True)
    ref = torch.nn.functional.leaky_relu(x_ref)
    assert allclose(x, ref, torch.float32)


# ------------------------------------------------------------------
# Correctness — edge cases
# ------------------------------------------------------------------
def test_all_positive():
    x = torch.abs(torch.randn(256, 256, device=device)) + 1e-3
    assert allclose(leaky_relu(x), x, torch.float32)


def test_all_negative():
    x     = -torch.abs(torch.randn(256, 256, device=device)) - 1e-3
    slope = 0.1
    assert allclose(leaky_relu(x, negative_slope=slope), x * slope, torch.float32)


def test_zeros():
    x = torch.zeros(256, 256, device=device)
    assert torch.all(leaky_relu(x, negative_slope=0.5) == 0)


def test_nan_inf():
    x   = torch.tensor([float('nan'), float('inf'), float('-inf'), 1.0, -1.0], device=device)
    out = leaky_relu(x)
    ref = torch.nn.functional.leaky_relu(x)
    assert torch.isnan(out[0])
    assert out[1] == ref[1]
    assert out[2] == ref[2]


# ------------------------------------------------------------------
# Correctness — dimensions
# ------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(1,), (8,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_dimensions(shape):
    x   = torch.randn(shape, device=device)
    ref = torch.nn.functional.leaky_relu(x)
    assert allclose(leaky_relu(x), ref, torch.float32)


# ------------------------------------------------------------------
# Gradient
# ------------------------------------------------------------------
def test_backward():
    x1 = torch.randn(128, 128, device=device, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    leaky_relu(x1, negative_slope=0.1).sum().backward()
    torch.nn.functional.leaky_relu(x2, negative_slope=0.1).sum().backward()
    assert allclose(x1.grad, x2.grad, torch.float32)


# ------------------------------------------------------------------
# Speedup — must be >= 0.9x vs PyTorch for all shapes
# ------------------------------------------------------------------
@pytest.mark.parametrize("shape", shapes)
def test_speedup(shape):
    x = torch.randn(shape, device=device, dtype=torch.float32)

    for _ in range(N_WARMUP):
        torch.nn.functional.leaky_relu(x)
        leaky_relu(x)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)

    s.record()
    for _ in range(N_RUNS):
        torch.nn.functional.leaky_relu(x)
    e.record()
    torch.cuda.synchronize()
    pt_ms = s.elapsed_time(e) / N_RUNS

    s.record()
    for _ in range(N_RUNS):
        leaky_relu(x)
    e.record()
    torch.cuda.synchronize()
    fg_ms = s.elapsed_time(e) / N_RUNS

    speedup = pt_ms / fg_ms
    n       = shape[0] * shape[1]
    mode    = "Triton" if n >= FALLBACK_THRESH else "PyTorch fallback"
    print(f"\n  {shape} [{mode}] PyTorch={pt_ms:.4f}ms "
          f"FlagGems={fg_ms:.4f}ms Speedup={speedup:.2f}x")
    assert speedup >= 0.9, f"Speedup {speedup:.2f}x < 0.9x for shape={shape}"
