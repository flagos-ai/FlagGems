import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference

SHAPE_CONV_TRANSPOSE2D = [
    # (input_shape, weight_shape [C_in, C_out/groups, kH, kW], groups)
    ((1, 2, 4, 4), (2, 1, 3, 3), 1),
    ((2, 3, 6, 6), (3, 1, 3, 3), 1),
    ((1, 4, 8, 8), (4, 1, 3, 5), 1),
    ((1, 4, 8, 8), (4, 1, 5, 3), 1),
    ((1, 3, 8, 16), (3, 2, 3, 3), 1),
    ((1, 4, 8, 8), (4, 2, 3, 3), 2),
]


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CONV_TRANSPOSE2D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("output_padding", [0, 1])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias", [True, False])
def test_accuracy_conv_transpose2d(
    shape, kernel, groups, stride, padding, output_padding, dilation, dtype, bias
):
    if output_padding >= stride:
        pytest.skip("output_padding must be < stride")
    kH, kW = kernel[2], kernel[3]
    if dilation * (kH - 1) < padding or dilation * (kW - 1) < padding:
        pytest.skip("effective padding would be negative")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, True)

    if bias:
        b = torch.randn(kernel[1] * groups, dtype=dtype, device=flag_gems.device)
        ref_b = to_reference(b, True)
    else:
        b = None
        ref_b = None

    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp,
        ref_weight,
        bias=ref_b,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    ).to(dtype)

    res_out = flag_gems.conv_transpose2d(
        inp,
        weight,
        bias=b,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )

    gems_assert_close(res_out, ref_out, dtype)
