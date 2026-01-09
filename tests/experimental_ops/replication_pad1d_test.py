# REPLICATION_PAD1D operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.replication_pad1d import (
    replication_pad1d as gems_replication_pad1d,
)
from flag_gems.experimental_ops.replication_pad1d import (
    replication_pad1d_out as gems_replication_pad1d_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import gems_assert_close, TO_CPU
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    if TO_CPU:
        ref_inp = inp.to("cpu")
    else:
        ref_inp = inp.clone()
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.replication_pad1d
@pytest.mark.parametrize("shape", [(2, 3, 7), (4, 16, 64), (8, 32, 256), (32, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding", [(0, 0), (1, 2), (3, 1)])
def test_replication_pad1d_tensor(shape, dtype, padding):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    act_inp = inp.clone()

    ref_out = torch.ops.aten.replication_pad1d(ref_inp, padding)

    with flag_gems.use_gems():
        act_out = gems_replication_pad1d(act_inp, padding)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.replication_pad1d
@pytest.mark.parametrize("shape", [(2, 3, 7), (4, 16, 64), (8, 32, 256), (32, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding", [(0, 0), (1, 2), (3, 1)])
def test_replication_pad1d_out(shape, dtype, padding):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    act_inp = inp.clone()

    pl, pr = padding
    w_out = shape[-1] + pl + pr
    if len(shape) == 3:
        N, C, _ = shape
        out_shape = (N, C, w_out)
    else:
        C, _ = shape
        out_shape = (C, w_out)

    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=ref_inp.device)
    ref_out = torch.ops.aten.replication_pad1d.out(ref_inp, padding, out=ref_out_buf)

    with flag_gems.use_gems():
        act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
        act_out = gems_replication_pad1d_out(act_inp, padding, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.replication_pad1d
@pytest.mark.parametrize("shape", [(2, 3, 7), (4, 16, 64), (8, 32, 256), (32, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding", [(0, 0), (1, 2), (3, 1)])
def test_replication_pad1d_benchmark_tensor(shape, dtype, padding):
    quantiles = [0.5, 0.2, 0.8]

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = inp.clone()
    act_inp = inp.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.replication_pad1d(ref_inp, padding),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_replication_pad1d(act_inp, padding),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"replication_pad1d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.replication_pad1d
@pytest.mark.parametrize("shape", [(2, 3, 7), (4, 16, 64), (8, 32, 256), (32, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding", [(0, 0), (1, 2), (3, 1)])
def test_replication_pad1d_benchmark_out(shape, dtype, padding):
    quantiles = [0.5, 0.2, 0.8]

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = inp.clone()
    act_inp = inp.clone()

    pl, pr = padding
    w_out = shape[-1] + pl + pr
    if len(shape) == 3:
        N, C, _ = shape
        out_shape = (N, C, w_out)
    else:
        C, _ = shape
        out_shape = (C, w_out)

    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.replication_pad1d.out(ref_inp, padding, out=ref_out_buf),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_replication_pad1d_out(act_inp, padding, act_out_buf),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"replication_pad1d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
