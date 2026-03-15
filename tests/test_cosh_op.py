# [FlagGems Operator Development Competition]
# Tests for : cosh
# Run with  : pytest tests/test_cosh_op.py -v

import torch
import pytest
from flag_gems.ops.cosh import cosh

device          = "cuda"
shapes          = [(64, 64), (256, 256), (1024, 1024), (4096, 4096)]
N_WARM          = 50
N_RUNS          = 500
FALLBACK_THRESH = 4_194_304  # must match cosh.py


def allclose(out, ref, dtype):
    atol = {
        torch.float32:  1.3e-6,
        torch.float16:  1e-3,
        torch.bfloat16: 0.016,
    }[dtype]
    return torch.allclose(out.float(), ref.float(), rtol=1e-4, atol=atol)


# ------------------------------------------------------------------
# Correctness — dtype
# cosh(x) overflows fp16 for |x| > 11.09 — clamp to safe range
# ------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_correctness_dtype(dtype):
    x   = torch.randn(1024, 1024, device=device, dtype=dtype).clamp(-8, 8)
    ref = torch.cosh(x)
    out = cosh(x)
    assert allclose(out, ref, dtype), f"FAIL dtype={dtype}"


# ------------------------------------------------------------------
# Correctness — shapes
# ------------------------------------------------------------------
@pytest.mark.parametrize("shape", shapes)
def test_correctness_shape(shape):
    x   = torch.randn(shape, device=device).clamp(-8, 8)
    ref = torch.cosh(x)
    out = cosh(x)
    assert allclose(out, ref, torch.float32), f"FAIL shape={shape}"


# ------------------------------------------------------------------
# Correctness — edge cases
# ------------------------------------------------------------------
def test_zero():
    # cosh(0) = 1.0
    x = torch.zeros(256, 256, device=device)
    assert torch.allclose(cosh(x), torch.ones_like(x))


def test_symmetry():
    # cosh is even: cosh(x) == cosh(-x)
    x = torch.randn(256, 256, device=device).clamp(-8, 8)
    assert torch.allclose(cosh(x), cosh(-x), rtol=1e-4, atol=1.3e-6)


def test_nan():
    x   = torch.tensor([float('nan'), 0.0, 1.0], device=device)
    out = cosh(x)
    assert torch.isnan(out[0])
    assert torch.allclose(out[1:].float(), torch.cosh(x[1:]).float(),
                          rtol=1e-4, atol=1.3e-6)


# ------------------------------------------------------------------
# Correctness — dimensions
# ------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(1,), (8,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_dimensions(shape):
    x   = torch.randn(shape, device=device).clamp(-8, 8)
    ref = torch.cosh(x)
    assert allclose(cosh(x), ref, torch.float32)


# ------------------------------------------------------------------
# Speedup — must be >= 0.9x vs PyTorch for all shapes
# ------------------------------------------------------------------
@pytest.mark.parametrize("shape", shapes)
def test_speedup(shape):
    x = torch.randn(shape, device=device).clamp(-8, 8)

    for _ in range(N_WARM):
        torch.cosh(x)
        cosh(x)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)

    s.record()
    for _ in range(N_RUNS): torch.cosh(x)
    e.record(); torch.cuda.synchronize()
    pt_ms = s.elapsed_time(e) / N_RUNS

    s.record()
    for _ in range(N_RUNS): cosh(x)
    e.record(); torch.cuda.synchronize()
    fg_ms = s.elapsed_time(e) / N_RUNS

    speedup = pt_ms / fg_ms
    n       = x.numel()
    mode    = "Triton" if n >= FALLBACK_THRESH else "fallback"
    print(f"\n  {shape} [{mode}] PyTorch={pt_ms:.4f}ms "
          f"FlagGems={fg_ms:.4f}ms Speedup={speedup:.2f}x")
    assert speedup >= 0.9, f"Speedup {speedup:.2f}x < 0.9x for shape={shape}"
