import pytest
import torch
import torch.nn.functional as F

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference
from .conftest import QUICK_MODE

FLOAT_DTYPES = [torch.float32] if QUICK_MODE else FLOAT_DTYPES

AVGPOOL3D_CONFIGS = [
    # (shape, kernel_size, stride, padding, ceil_mode, count_include_pad)
    ((2, 4, 8, 8, 8), 2, 2, 0, False, True),
    ((2, 8, 16, 16, 16), 3, 1, 1, False, True),
    ((2, 16, 32, 32, 32), 2, 2, 0, False, True),
    ((1, 1, 4, 4, 4), 2, 1, 0, False, True),
    # ceil_mode
    ((1, 1, 5, 5, 5), 2, 2, 0, True, True),
    # count_include_pad=False
    ((2, 4, 8, 8, 8), 2, 1, 1, False, False),
    # Non-cubic kernel
    ((2, 4, 8, 8, 8), (2, 3, 3), (1, 2, 2), (0, 1, 1), False, True),
    # Large batch
    ((4, 8, 16, 16, 16), 3, 2, 1, False, True),
]


@pytest.mark.avg_pool3d
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, ceil_mode, count_include_pad",
    AVGPOOL3D_CONFIGS,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_avg_pool3d(
    shape, kernel_size, stride, padding, ceil_mode, count_include_pad, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = F.avg_pool3d(
        ref_inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )

    with flag_gems.use_gems():
        res_out = F.avg_pool3d(
            inp,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )

    gems_assert_close(res_out, ref_out, dtype)
