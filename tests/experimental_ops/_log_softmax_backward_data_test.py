# _LOG_SOFTMAX_BACKWARD_DATA operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops._log_softmax_backward_data import (
    _log_softmax_backward_data as gems__log_softmax_backward_data,
)
from flag_gems.experimental_ops._log_softmax_backward_data import (
    _log_softmax_backward_data_out as gems__log_softmax_backward_data_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False

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


@pytest.mark.log_softmax_backward_data
@pytest.mark.parametrize(
    "shape_dim",
    [((2, 3), 1), ((2, 3), 0), ((128, 256), 1), ((8, 16, 32), -1), ((512, 1024), 1)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test__log_softmax_backward_data_tensor(shape_dim, dtype):
    shape, dim = shape_dim
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    output = torch.log_softmax(x, dim=dim)
    grad_output = torch.randn_like(output)

    ref_grad_output = to_reference(grad_output)
    ref_output = to_reference(output)

    ref_out = torch.ops.aten._log_softmax_backward_data(
        ref_grad_output, ref_output, dim, dtype
    )

    with flag_gems.use_gems():
        act_out = gems__log_softmax_backward_data(grad_output, output, dim, dtype)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.log_softmax_backward_data
@pytest.mark.parametrize(
    "shape_dim",
    [((2, 3), 1), ((2, 3), 0), ((128, 256), 1), ((8, 16, 32), -1), ((512, 1024), 1)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test__log_softmax_backward_data_out(shape_dim, dtype):
    shape, dim = shape_dim
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    output = torch.log_softmax(x, dim=dim)
    grad_output = torch.randn_like(output)

    ref_grad_output = to_reference(grad_output)
    ref_output = to_reference(output)

    ref_out_buf = torch.empty_like(ref_grad_output)
    ref_out = torch.ops.aten._log_softmax_backward_data.out(
        ref_grad_output, ref_output, dim, dtype, out=ref_out_buf
    )

    act_out_buf = torch.empty_like(x)
    with flag_gems.use_gems():
        act_out = gems__log_softmax_backward_data_out(
            grad_output, output, dim, dtype, act_out_buf
        )

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.log_softmax_backward_data
@pytest.mark.parametrize(
    "shape_dim",
    [((2, 3), 1), ((2, 3), 0), ((128, 256), 1), ((8, 16, 32), -1), ((512, 1024), 1)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test__log_softmax_backward_data_benchmark_tensor(shape_dim, dtype):
    quantiles = [0.5, 0.2, 0.8]

    shape, dim = shape_dim
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    output = torch.log_softmax(x, dim=dim)
    grad_output = torch.randn_like(output)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten._log_softmax_backward_data(
            grad_output, output, dim, dtype
        ),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems__log_softmax_backward_data(grad_output, output, dim, dtype),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"_log_softmax_backward_data {shape_dim} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.log_softmax_backward_data
@pytest.mark.parametrize(
    "shape_dim",
    [((2, 3), 1), ((2, 3), 0), ((128, 256), 1), ((8, 16, 32), -1), ((512, 1024), 1)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test__log_softmax_backward_data_benchmark_out(shape_dim, dtype):
    quantiles = [0.5, 0.2, 0.8]

    shape, dim = shape_dim
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    output = torch.log_softmax(x, dim=dim)
    grad_output = torch.randn_like(output)

    ref_out_buf = torch.empty_like(x)
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten._log_softmax_backward_data.out(
            grad_output, output, dim, dtype, out=ref_out_buf
        ),
        rep=100,
        quantiles=quantiles,
    )

    act_out_buf = torch.empty_like(x)

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems__log_softmax_backward_data_out(
                grad_output, output, dim, dtype, act_out_buf
            ),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"_log_softmax_backward_data {shape_dim} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
