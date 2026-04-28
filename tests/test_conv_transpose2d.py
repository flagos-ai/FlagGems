import pytest
import torch
import torch.nn.functional as F

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference
from .conftest import QUICK_MODE

FLOAT_DTYPES = [torch.float32] if QUICK_MODE else FLOAT_DTYPES

CONV_TRANSPOSE2D_CONFIGS = [
    # (in_shape, weight_shape, stride, padding, output_padding, groups, dilation)
    ((2, 4, 8, 8), (4, 2, 3, 3), 1, 0, 0, 1, 1),
    ((2, 4, 8, 8), (4, 2, 3, 3), 2, 0, 0, 1, 1),
    ((2, 4, 8, 8), (4, 2, 3, 3), 2, 1, 1, 1, 1),
    ((2, 4, 8, 8), (4, 2, 3, 3), 1, 1, 0, 1, 1),
    # groups
    ((2, 4, 8, 8), (4, 1, 3, 3), 1, 0, 0, 4, 1),
    # dilation
    ((1, 2, 16, 16), (2, 2, 3, 3), 1, 0, 0, 1, 2),
    # larger kernel
    ((1, 8, 16, 16), (8, 4, 5, 5), 2, 2, 1, 1, 1),
]


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize(
    "in_shape, weight_shape, stride, padding, output_padding, groups, dilation",
    CONV_TRANSPOSE2D_CONFIGS,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_conv_transpose2d(
    in_shape, weight_shape, stride, padding, output_padding, groups, dilation, dtype
):
    inp = torch.randn(in_shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(weight_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(inp, True)
    ref_weight = to_reference(weight, True)

    ref_out = F.conv_transpose2d(
        ref_inp,
        ref_weight,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )

    with flag_gems.use_gems():
        res_out = F.conv_transpose2d(
            inp,
            weight,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        )

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_conv_transpose2d_with_bias(dtype):
    inp = torch.randn(2, 4, 8, 8, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(4, 2, 3, 3, dtype=dtype, device=flag_gems.device)
    bias = torch.randn(2, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(inp, True)
    ref_weight = to_reference(weight, True)
    ref_bias = to_reference(bias, True)

    ref_out = F.conv_transpose2d(ref_inp, ref_weight, bias=ref_bias)

    with flag_gems.use_gems():
        res_out = F.conv_transpose2d(inp, weight, bias=bias)

    gems_assert_close(res_out, ref_out, dtype)
