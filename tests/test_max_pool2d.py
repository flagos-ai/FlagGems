import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

MAXPOOL2D_CONFIGS = [
    # Classic case: 3x3 kernel, stride 2, padding 1
    ((4, 3, 32, 32), 3, 2, 1, 1, False),
    # Non-square kernel and stride
    ((8, 16, 28, 28), (3, 5), (1, 2), 1, 1, False),
    # Test ceil_mode
    ((2, 4, 15, 15), 3, 2, 1, 1, True),
    # Test dilation
    ((1, 1, 7, 7), 2, 1, 0, 2, False),
    # Larger case from ResNet
    ((1, 64, 56, 56), 3, 2, 1, 1, False),
    # No padding
    ((2, 8, 16, 16), 2, 2, 0, 1, False),
    # Non-square padding
    ((2, 8, 16, 20), 2, 2, (1, 0), 1, False),
]

# Make sure every thread has same seed.
random.seed(time.time() // 100)


@pytest.mark.max_pool2d_with_indices
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode", MAXPOOL2D_CONFIGS
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_pool2d_with_indices(
    shape, kernel_size, stride, padding, dilation, ceil_mode, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp, True)

    ref_out, _ = torch.nn.functional.max_pool2d_with_indices(
        ref_inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    res_out, _ = flag_gems.max_pool2d_with_indices(
        inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.max_pool2d_backward
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode", MAXPOOL2D_CONFIGS
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_pool2d_backward(
    shape, kernel_size, stride, padding, dilation, ceil_mode, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp, upcast=True)
    ref_out, _ = torch.nn.functional.max_pool2d_with_indices(
        ref_inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    res_out, res_indices = flag_gems.max_pool2d_with_indices(
        inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    out_grad = torch.randn_like(res_out, device=flag_gems.device)
    ref_grad = utils.to_reference(out_grad, upcast=True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    res_in_grad = flag_gems.max_pool2d_backward(
        out_grad,
        inp,
        res_indices,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
