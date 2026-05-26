import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Custom shapes for fractional_max_pool2d: (batch, channels, height, width), kernel_size, output_size
FRACTIONAL_MAXPOOL2D_CONFIGS = [
    # Classic case: 2x2 kernel, output size 16x16
    ((4, 3, 32, 32), 2, (16, 16)),
    # Different output sizes
    ((8, 16, 28, 28), 2, (14, 14)),
    # Test different output sizes
    ((2, 4, 20, 20), 2, (10, 10)),
    # Larger case
    ((1, 64, 56, 56), 2, (28, 28)),
    # No padding, different kernel sizes
    ((2, 8, 16, 16), 2, (8, 8)),
    # Non-square output
    ((2, 8, 16, 20), 2, (8, 10)),
]


@pytest.mark.fractional_max_pool2d
@pytest.mark.parametrize(
    "shape, kernel_size, output_size", FRACTIONAL_MAXPOOL2D_CONFIGS
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fractional_max_pool2d(shape, kernel_size, output_size, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp, True)

    ref_out, ref_indices = torch.nn.functional.fractional_max_pool2d(
        ref_inp,
        kernel_size=kernel_size,
        output_size=output_size,
        return_indices=True,
    )

    res_out, res_indices = flag_gems.fractional_max_pool2d(
        inp,
        kernel_size=kernel_size,
        output_size=output_size,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.fractional_max_pool2d_backward
@pytest.mark.parametrize(
    "shape, kernel_size, output_size", FRACTIONAL_MAXPOOL2D_CONFIGS
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fractional_max_pool2d_backward(shape, kernel_size, output_size, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp, upcast=True)

    ref_out, ref_indices = torch.nn.functional.fractional_max_pool2d(
        ref_inp,
        kernel_size=kernel_size,
        output_size=output_size,
        return_indices=True,
    )
    res_out, res_indices = flag_gems.fractional_max_pool2d(
        inp,
        kernel_size=kernel_size,
        output_size=output_size,
    )
    out_grad = torch.randn_like(res_out, device=flag_gems.device)
    ref_grad = utils.to_reference(out_grad, upcast=True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    res_in_grad = flag_gems.fractional_max_pool2d_backward(
        out_grad,
        inp,
        res_indices,
        kernel_size=kernel_size,
        output_size=output_size,
    )

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
