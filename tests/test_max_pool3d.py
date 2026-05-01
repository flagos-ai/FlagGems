import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference
from .conftest import QUICK_MODE

FLOAT_DTYPES = [torch.float32] if QUICK_MODE else FLOAT_DTYPES


MAXPOOL3D_CONFIGS = [
    # Classic 3x3x3 kernel, stride 2, padding 1
    ((4, 3, 16, 16, 16), 3, 2, 1, 1, False),
    # Non-cubic kernel and stride
    ((8, 16, 12, 14, 14), (2, 3, 3), (1, 2, 2), (0, 1, 1), 1, False),
    # ceil_mode
    ((2, 4, 15, 15, 15), 3, 2, 1, 1, True),
    # dilation
    ((1, 1, 9, 9, 9), 2, 1, 0, 2, False),
    # Typical 3D CNN shape
    ((1, 64, 8, 28, 28), 3, 2, 1, 1, False),
    # No padding
    ((2, 8, 8, 16, 16), 2, 2, 0, 1, False),
    # Non-symmetric padding
    ((2, 8, 10, 16, 20), 2, 2, (0, 1, 0), 1, False),
    # Small input
    ((1, 1, 5, 5, 5), 2, 1, 0, 1, False),
    # Large batch
    ((8, 16, 8, 8, 8), 3, 1, 1, 1, False),
]


@pytest.mark.max_pool3d
@pytest.mark.skip(reason="#2865: this test always fail.")
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode", MAXPOOL3D_CONFIGS
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_pool3d(shape, kernel_size, stride, padding, dilation, ceil_mode, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.max_pool3d(
        ref_inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    with flag_gems.use_gems():
        res_out = torch.nn.functional.max_pool3d(
            inp,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.max_pool3d_backward
@pytest.mark.skip(reason="#2865: this test always fail.")
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode", MAXPOOL3D_CONFIGS
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_pool3d_backward(
    shape, kernel_size, stride, padding, dilation, ceil_mode, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp, upcast=True)

    ref_out = torch.nn.functional.max_pool3d(
        ref_inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    with flag_gems.use_gems():
        res_out = torch.nn.functional.max_pool3d(
            inp,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )

    out_grad = torch.randn_like(res_out, device=flag_gems.device)
    ref_grad = to_reference(out_grad, upcast=True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    with flag_gems.use_gems():
        (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)

    gems_assert_close(res_in_grad, ref_in_grad, dtype)
