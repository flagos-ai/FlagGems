# [FlagGems Operator Development Competition]
# Tests for : log10
# Run with  : pytest tests/test_log10_op.py -v

import torch
import pytest
from flag_gems.ops.log10 import log10

device          = "cuda"
shapes          = [(64, 64), (256, 256), (1024, 1024), (4096, 4096)]
N_WARM          = 50
N_RUNS          = 500
FALLBACK_THRESH = 4_194_304  # must match log10.py


def allclose(out, ref, dtype):
    atol = {
        torch.float32:  1.3e-6,
        torch.float16:  1e-3,
        torch.bfloat16: 0.016,
    }[dtype]
    return torch.allclose(out.float(), ref.float(), rtol=1e-4, atol=atol)


# ------------------------------------------------------------------
# Correctness — dtype (positive inputs only)
# ------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_correctness_dtype(dtype):
    x   = torch.rand(1024, 1024, device=device, dtype=dtype) + 0.1
    ref = torch.log10(x)
    out = log10(x)
    assert allclose(out, ref, dtype), f"FAIL dtype={dtype}"


# ------------------------------------------------------------------
# Correctness — shapes
# ------------------------------------------------------------------
@pytest.mark.parametrize("shape", shapes)
def test_correctness_shape(shape):
    x   = torch.rand(shape, device=device) + 0.1
    ref = torch.log10(x)
    out = log10(x)
    assert allclose(out, ref, torch.float32), f"FAIL shape={shape}"


# ------------------------------------------------------------------
# Correctness — known values
# ------------------------------------------------------------------
def test_known_values():
    # log10(1)=0, log10(10)=1, log10(100)=2, log10(1000)=3
    x   = torch.tensor([1.0, 10.0, 100.0, 1000.0], device=device)
    out = log10(x)
    ref = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device)
    assert torch.allclose(out, ref, rtol=1e-4, atol=1.3e-6)


def test_zero_input():
    # log10(0) = -inf
    x   = torch.tensor([0.0], device=device)
    out = log10(x)
    assert torch.isinf(out) and out < 0


def test_negative_input():
    # log10(negative) = nan
    x   = torch.tensor([-1.0], device=device)
    out = log10(x)
    assert torch.isnan(out)


def test_nan_input():
    x   = torch.tensor([float('nan'), 1.0, 10.0], device=device)
    out = log10(x)
    assert torch.isnan(out[0])
    assert torch.allclose(out[1:], torch.log10(x[1:]), rtol=1e-4, atol=1.3e-6)


# ------------------------------------------------------------------
# Correctness — dimensions
# ------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(1,), (8,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_dimensions(shape):
    x   = torch.rand(shape, device=device) + 0.1
    ref = torch.log10(x)
    assert allclose(log10(x), ref, torch.float32)


# ------------------------------------------------------------------
# Speedup — must be >= 0.9x vs PyTorch for all shapes
# ------------------------------------------------------------------
@pytest.mark.parametrize("shape", shapes)
def test_speedup(shape):
    x = torch.rand(shape, device=device) + 0.1

    for _ in range(N_WARM):
        torch.log10(x)
        log10(x)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)

    s.record()
    for _ in range(N_RUNS): torch.log10(x)
    e.record(); torch.cuda.synchronize()
    pt_ms = s.elapsed_time(e) / N_RUNS

    s.record()
    for _ in range(N_RUNS): log10(x)
    e.record(); torch.cuda.synchronize()
    fg_ms = s.elapsed_time(e) / N_RUNS

    speedup = pt_ms / fg_ms
    n       = x.numel()
    mode    = "Triton" if n >= FALLBACK_THRESH else "fallback"
    print(f"\n  {shape} [{mode}] PyTorch={pt_ms:.4f}ms "
          f"FlagGems={fg_ms:.4f}ms Speedup={speedup:.2f}x")
    assert speedup >= 0.9, f"Speedup {speedup:.2f}x < 0.9x for shape={shape}"
