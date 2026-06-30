import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

flag_gems.use_gems()

SHAPE_CONV_GENERIC_1D = [
    ((32, 2, 16), (4, 2, 3)),
    ((16, 8, 32), (16, 8, 5)),
]


@pytest.mark.convolution
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV_GENERIC_1D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1, 2])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_convolution_1d(shape, kernel, stride, padding, dtype):
    """Test 1D convolution using the general convolution interface."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.convolution(
        ref_inp,
        ref_weight,
        None,
        stride=[stride],
        padding=[padding],
        dilation=[1],
        transposed=False,
        output_padding=[0],
        groups=1,
    )

    res_out = torch.convolution(
        inp,
        weight,
        None,
        stride=[stride],
        padding=[padding],
        dilation=[1],
        transposed=False,
        output_padding=[0],
        groups=1,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


SHAPE_CONV_GENERIC_2D = [
    ((1, 2, 8, 8), (4, 2, 3, 3)),
    ((2, 4, 16, 16), (8, 4, 3, 3)),
    ((4, 8, 12, 12), (8, 8, 3, 3)),
]


@pytest.mark.convolution
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV_GENERIC_2D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_convolution_2d(shape, kernel, stride, padding, dtype):
    """Test 2D convolution using the general convolution interface."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.convolution(
        ref_inp,
        ref_weight,
        None,
        stride=[stride, stride],
        padding=[padding, padding],
        dilation=[1, 1],
        transposed=False,
        output_padding=[0, 0],
        groups=1,
    )

    res_out = torch.convolution(
        inp,
        weight,
        None,
        stride=[stride, stride],
        padding=[padding, padding],
        dilation=[1, 1],
        transposed=False,
        output_padding=[0, 0],
        groups=1,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


# shapes where in_channels and out_channels are divisible by groups
SHAPE_CONV_GENERIC_GROUPS_1 = [
    ((2, 4, 8, 8), (4, 2, 3, 3)),
]


@pytest.mark.convolution
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV_GENERIC_GROUPS_1)
@pytest.mark.parametrize("groups", [2])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_convolution_groups(shape, kernel, groups, dtype):
    """Test grouped convolution using the general convolution interface."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    # kernel[1] is already C_in/groups for the default groups value
    weight = torch.randn(
        kernel[0],
        kernel[1],
        kernel[2],
        kernel[3],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.convolution(
        ref_inp,
        ref_weight,
        None,
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        transposed=False,
        output_padding=[0, 0],
        groups=groups,
    )

    res_out = torch.convolution(
        inp,
        weight,
        None,
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        transposed=False,
        output_padding=[0, 0],
        groups=groups,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


SHAPE_CONV_GENERIC_GROUPS_2 = [
    ((2, 8, 8, 8), (8, 4, 3, 3)),
]


@pytest.mark.convolution
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV_GENERIC_GROUPS_2)
@pytest.mark.parametrize("groups", [2])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_convolution_groups_2(shape, kernel, groups, dtype):
    """Test grouped convolution using the general convolution interface."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    weight = torch.randn(
        kernel[0], kernel[1], kernel[2], kernel[3], dtype=dtype, device=flag_gems.device
    )
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.convolution(
        ref_inp,
        ref_weight,
        None,
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        transposed=False,
        output_padding=[0, 0],
        groups=groups,
    )

    res_out = torch.convolution(
        inp,
        weight,
        None,
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        transposed=False,
        output_padding=[0, 0],
        groups=groups,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
