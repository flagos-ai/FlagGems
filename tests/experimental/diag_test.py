# DIAG operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    # Fallback values when running outside pytest
    
    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)

import pytest
import triton

import flag_gems
from flag_gems.experimental_ops.diag import diag as gems_diag, diag_out as gems_diag_out

import torch

@pytest.mark.diag
@pytest.mark.parametrize("shape", [(5,), (2, 3), (128,), (128, 256), (512,), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("diagonal", [-1, 0, 2])
def test_diag_tensor(shape, dtype, diagonal):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    ref_out = torch.ops.aten.diag(ref_x, diagonal)

    with flag_gems.use_gems():
        act_out = gems_diag(x, diagonal)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.diag
@pytest.mark.parametrize("shape", [(5,), (2, 3), (128,), (128, 256), (512,), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("diagonal", [-1, 0, 2])
def test_diag_out(shape, dtype, diagonal):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    # compute out shape
    if len(shape) == 1:
        n = shape[0]
        m = n + abs(diagonal)
        out_shape = (m, m)
    else:
        m, n = shape
        if diagonal >= 0:
            l = max(0, min(m, n - diagonal))
        else:
            l = max(0, min(m + diagonal, n))
        out_shape = (l,)

    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten.diag.out(ref_x, diagonal, out=ref_out_buf)

    with flag_gems.use_gems():
        act_out = gems_diag_out(x, diagonal, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)

@pytest.mark.diag
@pytest.mark.parametrize("shape", [(5,), (2, 3), (128,), (128, 256), (512,), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("diagonal", [-1, 0, 2])
def test_diag_benchmark_tensor(shape, dtype, diagonal):
    import torch.utils.benchmark as benchmark

    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.diag(ref_x, diagonal),
        rep=100,
        quantiles=quantiles
    )


    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_diag(x, diagonal),
            rep=100,
            quantiles=quantiles
        )


    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"diag {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.diag
@pytest.mark.parametrize("shape", [(5,), (2, 3), (128,), (128, 256), (512,), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("diagonal", [-1, 0, 2])
def test_diag_benchmark_out(shape, dtype, diagonal):

    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    # compute out shape
    if len(shape) == 1:
        n = shape[0]
        m = n + abs(diagonal)
        out_shape = (m, m)
    else:
        m, n = shape
        if diagonal >= 0:
            l = max(0, min(m, n - diagonal))
        else:
            l = max(0, min(m + diagonal, n))
        out_shape = (l,)

    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.diag.out(ref_x, diagonal, out=ref_out_buf),
        rep=100,
        quantiles=quantiles
    )


    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_diag_out(x, diagonal, act_out_buf),
            rep=100,
            quantiles=quantiles
        )


    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"diag {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")