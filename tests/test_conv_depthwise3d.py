import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SHAPE_DEPTHWISE = [
    ((2, 4, 8, 8, 8), (4, 1, 2, 2, 2), (2, 2, 2)),
    ((3, 16, 6, 6, 6), (16, 1, 2, 2, 2), (2, 2, 2)),
    ((2, 8, 5, 7, 9), (8, 1, 3, 3, 3), (3, 3, 3)),
]


@pytest.mark.conv_depthwise3d
@pytest.mark.parametrize("shape_input, shape_weight, kernel", SHAPE_DEPTHWISE)
@pytest.mark.parametrize("stride", [[1, 1, 1], [2, 2, 2]])
@pytest.mark.parametrize("padding", [[0, 0, 0], [1, 1, 1]])
@pytest.mark.parametrize("dilation", [[1, 1, 1]])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("bias", [True, False])
def test_conv_depthwise3d(
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

    # aten.conv_depthwise3d is CUDA-only, so it cannot serve as a CPU reference
    # (the --ref=cpu quick-mode CI run would fail). A grouped conv3d with
    # groups=channels is mathematically identical and shares the same weight
    # layout (C, 1, kd, kh, kw), so it works on both CPU and GPU references.
    channels = shape_input[1]
    ref_out = torch.ops.aten.conv_depthwise3d(
        ref_inp,
        ref_weight,
        ref_bias,
        stride,
        padding,
        dilation,
        groups=channels,
    )

    res_out = flag_gems.conv_depthwise3d(
        inp, weight, kernel, bias_tensor, stride, padding, dilation
    )
    utils.gems_assert_close(res_out, ref_out, dtype)
