import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# conv_transpose3d test shapes: (input_shape, weight_shape)
# input: (N, in_channels, iD, iH, iW); weight: (in_channels, out_channels/groups, kD, kH, kW)
# A few small volumes keep the 3D transposed conv cheap while covering channel up/down mixes.
SHAPE_CONV_TRANSPOSE3D = [
    ((2, 4, 4, 5, 6), (4, 8, 3, 3, 3)),
    ((1, 8, 6, 6, 6), (8, 16, 3, 3, 3)),
    ((2, 16, 8, 8, 8), (16, 8, 2, 2, 2)),
]
# Cover both non-strided/strided and zero/nonzero padding paths.
STRIDES = [1, 2]
PADDINGS = [0, 1]


@pytest.mark.conv_transpose3d
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV_TRANSPOSE3D)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_conv_transpose3d(shape, kernel, stride, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = utils.to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.nn.functional.conv_transpose3d(
        ref_inp, ref_weight, bias=None, stride=stride, padding=padding, dilation=1
    )
    res_out = flag_gems.conv_transpose3d(
        inp, weight, bias=None, stride=stride, padding=padding, dilation=1
    )

    in_channels = kernel[0]
    out_channels = kernel[1]
    kernel_volume = kernel[2] * kernel[3] * kernel[4]
    utils.gems_assert_close(
        res_out,
        ref_out,
        dtype,
        reduce_dim=kernel_volume * max(in_channels, out_channels),
    )


@pytest.mark.conv_transpose3d
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV_TRANSPOSE3D)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_conv_transpose3d_bias(shape, kernel, stride, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = utils.to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = utils.to_reference(weight, True)
    out_channels = kernel[1]
    bias = torch.randn(out_channels, dtype=dtype, device=flag_gems.device)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.nn.functional.conv_transpose3d(
        ref_inp, ref_weight, bias=ref_bias, stride=stride, padding=padding, dilation=1
    )
    res_out = flag_gems.conv_transpose3d(
        inp, weight, bias=bias, stride=stride, padding=padding, dilation=1
    )

    in_channels = kernel[0]
    kernel_volume = kernel[2] * kernel[3] * kernel[4]
    utils.gems_assert_close(
        res_out,
        ref_out,
        dtype,
        reduce_dim=kernel_volume * max(in_channels, out_channels),
    )


@pytest.mark.conv_transpose3d
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_conv_transpose3d_dilation(stride, padding, dtype):
    # Small 5-dim shapes (N, C_in, D, H, W) / (C_in, C_out, kD, kH, kW) keep the
    # dilated 3D transposed conv cheap while still exercising all spatial dims.
    shape = (2, 4, 5, 5, 5)
    kernel = (4, 8, 3, 3, 3)
    dilation = 2
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = utils.to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.nn.functional.conv_transpose3d(
        ref_inp,
        ref_weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    res_out = flag_gems.conv_transpose3d(
        inp, weight, bias=None, stride=stride, padding=padding, dilation=dilation
    )

    in_channels = kernel[0]
    out_channels = kernel[1]
    kernel_volume = kernel[2] * kernel[3] * kernel[4]
    utils.gems_assert_close(
        res_out,
        ref_out,
        dtype,
        reduce_dim=kernel_volume * max(in_channels, out_channels),
    )


@pytest.mark.conv_transpose3d
@pytest.mark.parametrize(
    "shape, kernel, groups",
    [
        ((2, 8, 4, 4, 4), (8, 4, 3, 3, 3), 2),
        ((1, 12, 5, 5, 5), (12, 4, 2, 2, 2), 3),
    ],
)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_conv_transpose3d_groups(shape, kernel, groups, stride, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = utils.to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.nn.functional.conv_transpose3d(
        ref_inp,
        ref_weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=1,
        groups=groups,
    )
    res_out = flag_gems.conv_transpose3d(
        inp,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=1,
        groups=groups,
    )

    in_channels_per_group = kernel[0] // groups
    kernel_volume = kernel[2] * kernel[3] * kernel[4]
    utils.gems_assert_close(
        res_out, ref_out, dtype, reduce_dim=kernel_volume * in_channels_per_group
    )


@pytest.mark.conv_transpose3d
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_conv_transpose3d_output_padding(dtype):
    # Small 5-dim shapes (N, C_in, D, H, W) / (C_in, C_out, kD, kH, kW); output_padding
    # requires stride > 1, so a compact case is enough to validate the extra offset.
    shape = (2, 4, 5, 5, 5)
    kernel = (4, 8, 3, 3, 3)
    stride = 2
    padding = 1
    output_padding = 1
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = utils.to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.nn.functional.conv_transpose3d(
        ref_inp,
        ref_weight,
        bias=None,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=1,
    )
    res_out = flag_gems.conv_transpose3d(
        inp,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=1,
    )

    in_channels = kernel[0]
    out_channels = kernel[1]
    kernel_volume = kernel[2] * kernel[3] * kernel[4]
    utils.gems_assert_close(
        res_out,
        ref_out,
        dtype,
        reduce_dim=kernel_volume * max(in_channels, out_channels),
    )
