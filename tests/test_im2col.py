import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Cover 3D input, small/large 4D inputs for representative im2col testing
IM2COL_SHAPES = [(3, 8, 8), (1, 3, 16, 16), (16, 64, 64), (32, 128, 128)]
IM2COL_CONFIGS = [
    ((3, 3), (1, 1), (1, 1), (1, 1)),
    ((3, 3), (1, 1), (0, 0), (2, 2)),
    ((5, 4), (2, 2), (2, 1), (1, 2)),
    ((1, 1), (1, 1), (0, 0), (1, 1)),
]


@pytest.mark.im2col
@pytest.mark.parametrize("shape", IM2COL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("kernel_size, dilation, padding, stride", IM2COL_CONFIGS)
def test_im2col(shape, dtype, kernel_size, dilation, padding, stride):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)

    ref_out = torch.ops.aten.im2col(ref_x, kernel_size, dilation, padding, stride)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.im2col(x, kernel_size, dilation, padding, stride)

    utils.gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.im2col_out
@pytest.mark.parametrize("shape", IM2COL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("kernel_size, dilation, padding, stride", IM2COL_CONFIGS)
def test_im2col_out(shape, dtype, kernel_size, dilation, padding, stride):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)

    # Build the reference through the .out overload as well: for 3D inputs the
    # aten im2col.out variant keeps the leading batch dim (shape (1, C*kH*kW, L)),
    # unlike the functional variant which returns a 2D tensor. Comparing against
    # the .out reference keeps the shapes consistent.
    ref_out = torch.empty(0, dtype=ref_x.dtype, device=ref_x.device)
    torch.ops.aten.im2col.out(
        ref_x, kernel_size, dilation, padding, stride, out=ref_out
    )
    out = torch.empty(0, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.ops.aten.im2col.out(x, kernel_size, dilation, padding, stride, out=out)

    utils.gems_assert_close(out, ref_out, dtype=dtype)
