import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

vendor_name = flag_gems.vendor_name

# Each entry: (input_shape, weight_shape, groups). weight is
# [out_channels, in_channels // groups, *kernel_size].
SHAPE_CONV_MODE_1D = [
    ((2, 3, 10), (6, 3, 5), 1),
    ((3, 4, 14), (8, 2, 3), 2),
]


# Same layout as 1D for the 2D convolution path (groups=1 and grouped=2).
SHAPE_CONV_MODE_2D = [
    ((1, 3, 8, 8), (6, 3, 3, 3), 1),
    ((2, 4, 9, 9), (8, 2, 3, 3), 2),
]


# Same layout as 1D/2D for the 3D convolution path.
SHAPE_CONV_MODE_3D = [
    ((1, 3, 7, 7, 7), (6, 3, 3, 3, 3), 1),
]


STR_PADDINGS = ["valid", "same"]


def _run_and_assert(
    input_shape, weight_shape, groups, stride, padding, dilation, dtype, bias_flag
):
    torch.backends.cudnn.allow_tf32 = False
    if padding == "same" and any(s != 1 for s in stride):
        pytest.skip("padding='same' requires unit strides")
    inp, weight, bias, ref_inp, ref_weight, ref_bias = _make_inputs(
        input_shape, weight_shape, groups, dtype, bias_flag
    )
    dil = tuple(dilation for _ in range(len(stride)))
    ref_out = torch.ops.aten._convolution_mode(
        ref_inp, ref_weight, ref_bias, stride, padding, dil, groups
    ).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._convolution_mode(
            inp, weight, bias, stride, padding, dil, groups
        )
    utils.gems_assert_close(res_out, ref_out, dtype)


def _make_inputs(input_shape, weight_shape, groups, dtype, bias_flag):
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(weight_shape, dtype=dtype, device=flag_gems.device)
    bias = (
        torch.randn(weight_shape[0], dtype=dtype, device=flag_gems.device)
        if bias_flag
        else None
    )
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True) if bias is not None else None
    return inp, weight, bias, ref_inp, ref_weight, ref_bias


@pytest.mark.convolution_mode
@pytest.mark.parametrize("input_shape, weight_shape, groups", SHAPE_CONV_MODE_2D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", STR_PADDINGS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.skipif(
    vendor_name == "tsingmicro", reason="Issue #4131: conv kernels not working"
)
def test_convolution_mode_2d(
    input_shape, weight_shape, groups, stride, padding, dtype, bias
):
    _run_and_assert(
        input_shape, weight_shape, groups, (stride, stride), padding, 1, dtype, bias
    )


# 'same' with dilation > 1 exercises the asymmetric-pad path; kept separate and
# small so the autotune keys stay limited.


@pytest.mark.convolution_mode
@pytest.mark.parametrize("input_shape, weight_shape, groups", SHAPE_CONV_MODE_2D)
@pytest.mark.parametrize("padding", STR_PADDINGS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.skipif(
    vendor_name == "tsingmicro", reason="Issue #4131: conv kernels not working"
)
def test_convolution_mode_2d_dilation(
    input_shape, weight_shape, groups, padding, dtype, bias
):
    _run_and_assert(input_shape, weight_shape, groups, (1, 1), padding, 2, dtype, bias)


@pytest.mark.convolution_mode
@pytest.mark.parametrize("input_shape, weight_shape, groups", SHAPE_CONV_MODE_1D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", STR_PADDINGS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.skipif(
    vendor_name == "tsingmicro", reason="Issue #4131: conv kernels not working"
)
def test_convolution_mode_1d(
    input_shape, weight_shape, groups, stride, padding, dtype, bias
):
    _run_and_assert(
        input_shape, weight_shape, groups, (stride,), padding, 1, dtype, bias
    )


@pytest.mark.convolution_mode
@pytest.mark.parametrize("input_shape, weight_shape, groups", SHAPE_CONV_MODE_3D)
@pytest.mark.parametrize("padding", STR_PADDINGS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.skipif(
    vendor_name == "tsingmicro", reason="Issue #4131: conv kernels not working"
)
def test_convolution_mode_3d(input_shape, weight_shape, groups, padding, dtype, bias):
    _run_and_assert(
        input_shape, weight_shape, groups, (1, 1, 1), padding, 1, dtype, bias
    )


@pytest.mark.convolution_mode
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_convolution_mode_invalid_padding(dtype):
    inp = torch.randn(1, 3, 8, 8, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(6, 3, 3, 3, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        with pytest.raises(ValueError):
            torch.ops.aten._convolution_mode(
                inp, weight, None, (1, 1), "reflect", (1, 1), 1
            )


@pytest.mark.convolution_mode
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_convolution_mode_same_strided_raises(dtype):
    inp = torch.randn(1, 3, 8, 8, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(6, 3, 3, 3, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        with pytest.raises(ValueError):
            torch.ops.aten._convolution_mode(
                inp, weight, None, (2, 2), "same", (1, 1), 1
            )
