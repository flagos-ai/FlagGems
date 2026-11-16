# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


@pytest.fixture(autouse=True)
def _full_precision_musa_convolution():
    if flag_gems.vendor_name == "mthreads":
        from torch_musa.core import mudnn

        with mudnn.flags(allow_tf32=False):
            yield
    else:
        yield


@pytest.mark.conv_transpose3d
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV_TRANSPOSE3D)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_conv_transpose3d(shape, kernel, stride, padding, dtype):
    if (
        flag_gems.vendor_name == "mthreads"
        and stride == 2
        and shape == (2, 16, 8, 8, 8)
        and kernel == (16, 8, 2, 2, 2)
    ):
        pytest.skip("MUSA native stride-2 kernel-2 convolution is unsupported for this shape")
    torch.manual_seed(0)
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
    if (
        flag_gems.vendor_name == "mthreads"
        and stride == 2
        and shape == (2, 16, 8, 8, 8)
        and kernel == (16, 8, 2, 2, 2)
    ):
        pytest.skip("MUSA native stride-2 kernel-2 convolution is unsupported for this shape")
    torch.manual_seed(0)
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
    torch.manual_seed(0)
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
    if flag_gems.vendor_name == "mthreads" and stride == 2 and groups == 3:
        pytest.skip("MUSA native conv_transpose3d fails this stride-2 three-group geometry")
    torch.manual_seed(0)
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
    torch.manual_seed(0)
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


@pytest.mark.conv_transpose3d
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_conv_transpose3d_noncontiguous(dtype):
    torch.manual_seed(0)
    base = torch.randn((1, 8, 5, 6, 7), dtype=dtype, device=flag_gems.device)
    inp = base[:, ::2, :, :, :]
    weight = torch.randn((4, 6, 3, 3, 3), dtype=dtype, device=flag_gems.device)
    ref = torch.nn.functional.conv_transpose3d(
        utils.to_reference(inp, True), utils.to_reference(weight, True),
        stride=2, padding=1, output_padding=1
    )
    result = flag_gems.conv_transpose3d(
        inp, weight, stride=2, padding=1, output_padding=1
    )
    utils.gems_assert_close(result, ref, dtype, reduce_dim=108)
    assert result.shape == ref.shape
    assert result.dtype == dtype


@pytest.mark.conv_transpose3d
def test_conv_transpose3d_invalid_output_padding():
    if flag_gems.vendor_name == "mthreads":
        pytest.skip("MUSA native conv_transpose3d accepts this invalid output padding")
    torch.manual_seed(0)
    inp = torch.randn((1, 4, 4, 4, 4), device=flag_gems.device)
    weight = torch.randn((4, 6, 3, 3, 3), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.conv_transpose3d(inp, weight, stride=1, output_padding=1)
