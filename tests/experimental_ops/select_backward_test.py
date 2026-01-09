# SELECT_BACKWARD operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.select_backward import (  # noqa: E402
    select_backward as gems_select_backward,
)
from flag_gems.experimental_ops.select_backward import (  # noqa: E402
    select_backward_out as gems_select_backward_out,
)


def to_reference(inp):
    """Convert tensor to reference device (CPU if TO_CPU is True)."""
    if TO_CPU:
        return inp.to("cpu")
    return inp.clone()


@pytest.mark.select_backward
@pytest.mark.parametrize("input_sizes", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("index_mode", ["first", "mid", "neg1"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_select_backward_default(input_sizes, dim, index_mode, dtype):
    ndim = len(input_sizes)
    dim_c = dim % ndim
    size_d = input_sizes[dim_c]
    if index_mode == "first":
        index = 0
    elif index_mode == "mid":
        index = size_d // 2
    else:
        index = -1
    out_shape = list(input_sizes)
    del out_shape[dim_c]
    out_shape = tuple(out_shape) if len(out_shape) > 0 else ()
    grad = torch.randn(out_shape, dtype=dtype, device=flag_gems.device)
    grad_ref = to_reference(grad)
    grad_act = grad.clone()

    ref_out = torch.ops.aten.select_backward(grad_ref, list(input_sizes), dim, index)

    with flag_gems.use_gems():
        act_out = gems_select_backward(grad_act, list(input_sizes), dim, index)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.select_backward
@pytest.mark.parametrize("input_sizes", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("index_mode", ["first", "mid", "neg1"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_select_backward_out(input_sizes, dim, index_mode, dtype):
    ndim = len(input_sizes)
    dim_c = dim % ndim
    size_d = input_sizes[dim_c]
    if index_mode == "first":
        index = 0
    elif index_mode == "mid":
        index = size_d // 2
    else:
        index = -1
    out_shape = list(input_sizes)
    del out_shape[dim_c]
    out_shape = tuple(out_shape) if len(out_shape) > 0 else ()
    grad = torch.randn(out_shape, dtype=dtype, device=flag_gems.device)
    grad_ref = to_reference(grad)
    grad_act = grad.clone()

    ref_out_buf = torch.empty(input_sizes, dtype=dtype, device=flag_gems.device)
    ref_out = torch.ops.aten.select_backward.out(
        grad_ref, list(input_sizes), dim, index, out=ref_out_buf
    )

    with flag_gems.use_gems():
        act_out_buf = torch.empty(input_sizes, dtype=dtype, device=flag_gems.device)
        act_out = gems_select_backward_out(
            grad_act, list(input_sizes), dim, index, act_out_buf
        )

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.select_backward
@pytest.mark.parametrize("input_sizes", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("index_mode", ["first", "mid", "neg1"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_select_backward_default_performance(input_sizes, dim, index_mode, dtype):
    import torch.utils.benchmark as benchmark  # noqa: E402, F401

    quantiles = [0.5, 0.2, 0.8]

    ndim = len(input_sizes)
    dim_c = dim % ndim
    size_d = input_sizes[dim_c]
    if index_mode == "first":
        index = 0
    elif index_mode == "mid":
        index = size_d // 2
    else:
        index = -1
    out_shape = list(input_sizes)
    del out_shape[dim_c]
    out_shape = tuple(out_shape) if len(out_shape) > 0 else ()
    grad_ref = torch.randn(out_shape, dtype=dtype, device=flag_gems.device)
    grad_act = grad_ref.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.select_backward(grad_ref, list(input_sizes), dim, index),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_select_backward(grad_act, list(input_sizes), dim, index),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"select_backward {input_sizes} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.select_backward
@pytest.mark.parametrize("input_sizes", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("index_mode", ["first", "mid", "neg1"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_select_backward_benchmark_out(input_sizes, dim, index_mode, dtype):
    quantiles = [0.5, 0.2, 0.8]

    ndim = len(input_sizes)
    dim_c = dim % ndim
    size_d = input_sizes[dim_c]
    if index_mode == "first":
        index = 0
    elif index_mode == "mid":
        index = size_d // 2
    else:
        index = -1
    out_shape = list(input_sizes)
    del out_shape[dim_c]
    out_shape = tuple(out_shape) if len(out_shape) > 0 else ()
    grad_ref = torch.randn(out_shape, dtype=dtype, device=flag_gems.device)
    grad_act = grad_ref.clone()

    ref_out_buf = torch.empty(input_sizes, dtype=dtype, device=flag_gems.device)
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.select_backward.out(
            grad_ref, list(input_sizes), dim, index, out=ref_out_buf
        ),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        act_out_buf = torch.empty(input_sizes, dtype=dtype, device=flag_gems.device)
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_select_backward_out(
                grad_act, list(input_sizes), dim, index, act_out_buf
            ),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"select_backward {input_sizes} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
