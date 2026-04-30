import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference
from .conftest import QUICK_MODE

FLOAT_DTYPES = [torch.float32] if QUICK_MODE else FLOAT_DTYPES


AVGPOOL3D_CONFIGS = [
    # (shape, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    # Basic 3x3x3 kernel
    ((2, 3, 16, 16, 16), 3, 2, 1, False, True, None),
    # count_include_pad=False
    ((2, 3, 16, 16, 16), 3, 2, 1, False, False, None),
    # Non-cubic kernel and stride
    ((4, 8, 12, 16, 20), (2, 3, 4), (1, 2, 2), (0, 1, 1), False, True, None),
    # ceil_mode
    ((2, 4, 15, 15, 15), 3, 2, 1, True, True, None),
    # divisor_override
    ((1, 1, 7, 7, 7), 2, 1, 0, False, True, 1),
    # Typical CNN shapes
    ((1, 64, 16, 56, 56), 3, 2, 1, False, True, None),
    # No padding
    ((2, 8, 8, 16, 16), 2, 2, 0, False, False, None),
    # Non-symmetric padding
    ((2, 8, 10, 16, 20), 2, 2, (0, 1, 0), False, True, None),
    # Small input
    ((1, 1, 4, 4, 4), 2, 1, 0, False, True, None),
    # Large batch
    ((8, 16, 8, 8, 8), 3, 1, 1, False, True, None),
]


@pytest.mark.avg_pool3d
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override",
    AVGPOOL3D_CONFIGS,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_avg_pool3d(
    shape,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
    dtype,
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.ops.aten.avg_pool3d(
        ref_inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.avg_pool3d(
            inp,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.avg_pool3d_backward
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override",
    AVGPOOL3D_CONFIGS,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_avg_pool3d_backward(
    shape,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
    dtype,
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.ops.aten.avg_pool3d(
        ref_inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )
    out_grad = torch.randn_like(ref_out, dtype=inp.dtype, device=flag_gems.device)
    ref_out_grad = to_reference(out_grad, True)
    ref_inp_grad = torch.ops.aten.avg_pool3d_backward(
        ref_out_grad,
        ref_inp,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )

    with flag_gems.use_gems():
        res_inp_grad = torch.ops.aten.avg_pool3d_backward(
            out_grad,
            inp,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )
    # 3D backward accumulates over kernel_d * kernel_h * kernel_w elements per
    # input position. With stride < kernel, each input can receive gradient
    # contributions from many overlapping output windows, amplifying fp error.
    # Use kernel_volume^2 as reduce_dim to account for the compounded error
    # from both the accumulation and the per-element division.
    if isinstance(kernel_size, int):
        kd = kh = kw = kernel_size
    else:
        kd, kh, kw = kernel_size
    kernel_volume = kd * kh * kw
    gems_assert_close(
        res_inp_grad, ref_inp_grad, dtype, reduce_dim=kernel_volume * kernel_volume
    )
