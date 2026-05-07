import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SHAPE_DEPTHWISE = [
    ((32, 4, 8, 8), (32, 1, 2, 2), (2, 2)),
    ((18, 16, 4, 4), (16, 1, 2, 2), (2, 2)),
    # ((9, 32, 4, 4), (128, 1, 2, 2), (2, 2)),
    # ((32, 16, 8, 8), (32, 1, 4, 4), (4, 4)),
    # ((18, 8, 4, 4), (16, 1, 2, 2), (2, 2)),
    # ((9, 4, 4, 4), (128, 1, 2, 2), (2, 2)),
    # ((32, 4, 8, 8), (32, 1, 3, 3), (3, 3)),
    # ((18, 16, 13, 13), (16, 1, 5, 5), (5, 5)),
    # ((9, 32, 8, 8), (128, 1, 3, 3), (3, 3)),
    # ((32, 16, 9, 9), (32, 1, 5, 5), (5, 5)),
    # ((18, 8, 7, 7), (16, 1, 3, 3), (3, 3)),
    # ((9, 4, 6, 6), (128, 1, 3, 3), (3, 3)),
]


@pytest.mark.skip(reason="This operator currently fails the test.")
@pytest.mark.conv_depthwise2d
@pytest.mark.parametrize("shape_input, shape_weight, kernel", SHAPE_DEPTHWISE)
@pytest.mark.parametrize("stride", [[1, 1], [2, 2]])
@pytest.mark.parametrize("padding", [[0, 0], [1, 1]])
@pytest.mark.parametrize("dilation", [[1, 1]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("bias", [True, False])
def test_conv_depthwise2d(
    shape_input, shape_weight, kernel, stride, padding, dilation, dtype, bias
):
    inp = torch.randn(shape_input, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(shape_weight, dtype=dtype, device=flag_gems.device)
    ref_weight = utils.to_reference(weight, False)

    if bias:
        bias_tensor = torch.randn(shape_weight[0], dtype=dtype, device=flag_gems.device)
        ref_bias = utils.to_reference(bias_tensor, False)
    else:
        bias_tensor = None
        ref_bias = None

    ref_out = torch.ops.aten._conv_depthwise2d(
        ref_inp,
        ref_weight,
        kernel,
        ref_bias,
        stride,
        padding,
        dilation,
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten._conv_depthwise2d(
            inp, weight, kernel, bias_tensor, stride, padding, dilation
        )
    utils.gems_assert_close(res_out, ref_out, dtype)
