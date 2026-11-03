import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

FLOAT_DTYPES = utils.FLOAT_DTYPES


MAXPOOL1D_CONFIGS = [
    # (shape, kernel_size, stride, padding, dilation, ceil_mode)
    # Classic case: stride defaults to kernel_size
    ((4, 8, 64), 2, None, 0, 1, False),
    # Explicit stride, padding
    ((2, 16, 100), 3, 2, 1, 1, False),
    # ceil_mode
    ((2, 4, 15), 3, 2, 1, 1, True),
    # dilation
    ((1, 1, 32), 2, 1, 0, 2, False),
    # Larger channel/length
    ((8, 32, 224), 3, 2, 1, 1, False),
    # No padding, stride 1
    ((3, 8, 50), 4, 1, 0, 1, False),
    # 2D input (C, L)
    ((16, 128), 3, 2, 1, 1, False),
    # kernel_size passed as list
    ((2, 8, 40), [3], [2], [1], [1], False),
]


@pytest.mark.max_pool1d_with_indices
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode", MAXPOOL1D_CONFIGS
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_pool1d_with_indices(
    shape, kernel_size, stride, padding, dilation, ceil_mode, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, upcast=True)

    ref_out, ref_indices = torch.nn.functional.max_pool1d_with_indices(
        ref_inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    res_out, res_indices = flag_gems.max_pool1d_with_indices(
        inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_indices, ref_indices)
