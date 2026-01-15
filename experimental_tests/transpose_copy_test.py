# TRANSPOSE_COPY operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.transpose_copy import (  # noqa: E402
    transpose_copy_int,
    transpose_copy_int_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close  # noqa: E402
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


@pytest.mark.transpose_copy
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims", [(0, 1), (1, 0)])
def test_transpose_copy_int(shape, dtype, dims):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    d0, d1 = dims

    ref_out = torch.ops.aten.transpose_copy(ref_inp, d0, d1)

    with flag_gems.use_gems():
        act_out = transpose_copy_int(inp, d0, d1)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.transpose_copy
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims", [(0, 1), (1, 0)])
def test_transpose_copy_int_out(shape, dtype, dims):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    d0, d1 = dims

    out_shape = list(shape)
    out_shape[d0], out_shape[d1] = out_shape[d1], out_shape[d0]
    out_shape = tuple(out_shape)

    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=ref_inp.device)
    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten.transpose_copy(ref_inp, d0, d1, out=ref_out_buf)

    with flag_gems.use_gems():
        act_out = transpose_copy_int_out(inp, d0, d1, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.transpose_copy
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims", [(0, 1), (1, 0)])
def test_transpose_copy_int_performance(shape, dtype, dims):
    import torch.utils.benchmark as benchmark  # noqa: E402, F401, F401

    quantiles = [0.5, 0.2, 0.8]

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = inp.clone()
    d0, d1 = dims

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.transpose_copy(ref_inp, d0, d1),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: transpose_copy_int(inp, d0, d1), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"transpose_copy {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.transpose_copy
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims", [(0, 1), (1, 0)])
def test_transpose_copy_int_benchmark_out(shape, dtype, dims):
    quantiles = [0.5, 0.2, 0.8]

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = inp.clone()
    d0, d1 = dims

    out_shape = list(shape)
    out_shape[d0], out_shape[d1] = out_shape[d1], out_shape[d0]
    out_shape = tuple(out_shape)

    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.transpose_copy(ref_inp, d0, d1, out=ref_out_buf),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: transpose_copy_int_out(inp, d0, d1, act_out_buf),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"transpose_copy {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
