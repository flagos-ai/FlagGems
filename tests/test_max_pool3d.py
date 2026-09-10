import pytest
import torch

import flag_gems
from flag_gems import max_pool3d

from . import accuracy_utils as utils

# Representative 5-D pooling configs: (shape, kernel_size, stride, padding,
# dilation, ceil_mode). Chosen to cover cubic/non-cubic kernels, strides,
# symmetric/asymmetric padding, dilation, ceil_mode and a typical 3D-CNN shape.
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
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode", MAXPOOL3D_CONFIGS
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_max_pool3d(shape, kernel_size, stride, padding, dilation, ceil_mode, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.max_pool3d(
        ref_inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    res_out = max_pool3d(
        inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
