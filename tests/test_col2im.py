import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference
from .conftest import QUICK_MODE

COL2IM_CONFIGS = [
    # (batch, channels, kernel_size, output_size, stride, padding, dilation)
    (2, 3, (2, 2), (4, 5), (1, 1), (0, 0), (1, 1)),
    (1, 2, (3, 3), (8, 8), (2, 2), (0, 0), (1, 1)),
    (2, 4, (2, 2), (6, 6), (1, 1), (1, 1), (1, 1)),
    (1, 2, (2, 2), (8, 8), (1, 1), (0, 0), (2, 2)),
    (2, 3, (3, 3), (10, 12), (2, 2), (1, 1), (2, 2)),
]

if QUICK_MODE:
    COL2IM_CONFIGS = COL2IM_CONFIGS[:2]


@pytest.mark.col2im
@pytest.mark.parametrize("config", COL2IM_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_col2im(config, dtype):
    batch, channels, kernel_size, output_size, stride, padding, dilation = config
    kernel_h, kernel_w = kernel_size
    output_h, output_w = output_size
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation

    L_h = (output_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    L_w = (output_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    L = L_h * L_w

    inp = torch.randn(
        batch, channels * kernel_h * kernel_w, L, device=flag_gems.device, dtype=dtype
    )
    ref_inp = to_reference(inp, True)

    ref_out = torch.ops.aten.col2im(
        ref_inp, output_size, kernel_size, dilation, padding, stride
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.col2im(
            inp, output_size, kernel_size, dilation, padding, stride
        )

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=kernel_h * kernel_w)
