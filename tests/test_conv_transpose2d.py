import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference
from .conftest import QUICK_MODE

FLOAT_DTYPES = [torch.float32] if QUICK_MODE else [torch.float16, torch.float32, torch.bfloat16]

SHAPE_CONV_TRANSPOSE2D = [
    # (input_shape, weight_shape, groups)
    ((1, 2, 5, 5), (2, 1, 3, 3), 1),  # small, groups=1
    ((2, 3, 9, 9), (3, 1, 3, 3), 1),  # medium, groups=1
    ((32, 8, 8, 8), (8, 32, 2, 2), 1),  # larger batch
    ((2, 4, 5, 5), (4, 2, 3, 3), 2),  # groups=2
    ((1, 8, 4, 4), (8, 2, 3, 3), 4),  # groups=4
    ((1, 1, 3, 3), (1, 1, 3, 3), 1),  # minimal
    ((4, 16, 16, 16), (16, 8, 5, 5), 1),  # larger kernel
    ((1, 2, 7, 7), (2, 3, 3, 3), 1),  # odd spatial dim
]


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CONV_TRANSPOSE2D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_accuracy_conv_transpose2d(
    shape, kernel, stride, padding, groups, dtype, dilation, bias
):
    torch.backends.cudnn.allow_tf32 = False

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, True)

    if bias:
        out_channels = kernel[1] * groups
        bias_tensor = torch.randn(out_channels, dtype=dtype, device=flag_gems.device)
        ref_bias = to_reference(bias_tensor, True)
    else:
        bias_tensor = None
        ref_bias = None

    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp,
        ref_weight,
        bias=ref_bias,
        stride=stride,
        padding=padding,
        output_padding=0,
        groups=groups,
        dilation=dilation,
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.conv_transpose2d(
            inp,
            weight,
            bias=bias_tensor,
            stride=stride,
            padding=padding,
            output_padding=0,
            groups=groups,
            dilation=dilation,
        )

    in_c_per_group = shape[1] // groups
    reduce_dim = in_c_per_group * kernel[2] * kernel[3]
    if dtype == torch.bfloat16:
        reduce_dim = max(reduce_dim, 128)
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize(
    "shape, kernel, groups",
    [
        ((2, 4, 8, 8), (4, 2, 3, 3), 1),
        ((1, 2, 5, 5), (2, 1, 3, 3), 1),
        ((2, 4, 5, 5), (4, 2, 3, 3), 2),
    ],
)
@pytest.mark.parametrize("stride", [2, 3])
@pytest.mark.parametrize("output_padding_val", [1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("bias", [True, False])
def test_accuracy_conv_transpose2d_output_padding(
    shape, kernel, groups, stride, output_padding_val, dtype, bias
):
    torch.backends.cudnn.allow_tf32 = False
    padding = 1

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, True)

    if bias:
        out_channels = kernel[1] * groups
        bias_tensor = torch.randn(out_channels, dtype=dtype, device=flag_gems.device)
        ref_bias = to_reference(bias_tensor, True)
    else:
        bias_tensor = None
        ref_bias = None

    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp,
        ref_weight,
        bias=ref_bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding_val,
        groups=groups,
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.conv_transpose2d(
            inp,
            weight,
            bias=bias_tensor,
            stride=stride,
            padding=padding,
            output_padding=output_padding_val,
            groups=groups,
        )

    in_c_per_group = shape[1] // groups
    reduce_dim = in_c_per_group * kernel[2] * kernel[3]
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.conv_transpose2d
def test_accuracy_conv_transpose2d_non_contiguous():
    torch.backends.cudnn.allow_tf32 = False
    dtype = torch.float32

    inp = torch.randn(2, 4, 8, 6, dtype=dtype, device=flag_gems.device)
    inp_nc = inp.transpose(2, 3)
    assert not inp_nc.is_contiguous()

    weight = torch.randn(4, 2, 3, 3, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp_nc, True)
    ref_weight = to_reference(weight, True)

    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp,
        ref_weight,
        stride=1,
        padding=1,
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.conv_transpose2d(
            inp_nc,
            weight,
            stride=1,
            padding=1,
        )

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=4 * 3 * 3)

