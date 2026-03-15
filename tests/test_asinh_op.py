# [FlagGems Operator Development Competition]
# Tests for : asinh
# Run with  : pytest tests/test_asinh_op.py -v

import torch
import pytest
from flag_gems.ops.asinh import asinh

device          = "cuda"
shapes          = [(64, 64), (256, 256), (1024, 1024), (4096, 4096)]
N_WARM          = 50
N_RUNS          = 500
FALLBACK_THRESH = 4_194_304  # must match asinh.py


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
    ref = torch.asinh(x)
    out = asinh(x)
    assert allclose(out, ref, dtype), f"FAIL dtype={dtype}"


# ------------------------------------------------------------------
# Correctness — shapes
# ------------------------------------------------------------------
@pytest.mark.parametrize("shape", shapes)
def test_correctness_shape(shape):
    x   = torch.randn(shape, device=device)
    ref = torch.asinh(x)
    out = asinh(x)
    assert allclose(out, ref, torch.float32), f"FAIL shape={shape}"


# ------------------------------------------------------------------
# Correctness — edge cases
# ------------------------------------------------------------------
def test_zero():
    # asinh(0) = 0
    x = torch.zeros(256, 256, device=device)
    assert torch.all(asinh(x) == 0)


def test_antisymmetry():
    # asinh is odd: asinh(-x) == -asinh(x)
    x = torch.randn(256, 256, device=device)
    assert torch.allclose(asinh(-x), -asinh(x), rtol=1e-4, atol=1.3e-6)


def test_known_values():
    # asinh(0)=0, asinh(1)≈0.8814, asinh(-1)≈-0.8814
    x   = torch.tensor([0.0, 1.0, -1.0], device=device)
    out = asinh(x)
    ref = torch.asinh(x)
    assert torch.allclose(out, ref, rtol=1e-4, atol=1.3e-6)


def test_large_values():
    # asinh stable for large x: ln(x + sqrt(x^2+1)) ≈ ln(2x)
    x   = torch.tensor([1e3, 1e4, 1e6], device=device)
    out = asinh(x)
    ref = torch.asinh(x)
    assert torch.allclose(out, ref, rtol=1e-4, atol=1.3e-6)


def test_nan_inf():
    x   = torch.tensor([float('nan'), float('inf'), float('-inf')], device=device)
    out = asinh(x)
    assert torch.isnan(out[0])
    assert torch.isinf(out[1]) and out[1] > 0
    assert torch.isinf(out[2]) and out[2] < 0


# ------------------------------------------------------------------
# Correctness — dimensions
# ------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(1,), (8,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_dimensions(shape):
    x   = torch.randn(shape, device=device)
    ref = torch.asinh(x)
    assert allclose(asinh(x), ref, torch.float32)


# ------------------------------------------------------------------
# Speedup — must be >= 0.9x vs PyTorch for all shapes
# ------------------------------------------------------------------
@pytest.mark.parametrize("shape", shapes)
def test_speedup(shape):
    x = torch.randn(shape, device=device)

    for _ in range(N_WARM):
        torch.asinh(x)
        asinh(x)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)

    s.record()
    for _ in range(N_RUNS): torch.asinh(x)
    e.record(); torch.cuda.synchronize()
    pt_ms = s.elapsed_time(e) / N_RUNS

    s.record()
    for _ in range(N_RUNS): asinh(x)
    e.record(); torch.cuda.synchronize()
    fg_ms = s.elapsed_time(e) / N_RUNS

    speedup = pt_ms / fg_ms
    n       = x.numel()
    mode    = "Triton" if n >= FALLBACK_THRESH else "fallback"
    print(f"\n  {shape} [{mode}] PyTorch={pt_ms:.4f}ms "
          f"FlagGems={fg_ms:.4f}ms Speedup={speedup:.2f}x")
    assert speedup >= 0.9, f"Speedup {speedup:.2f}x < 0.9x for shape={shape}"
