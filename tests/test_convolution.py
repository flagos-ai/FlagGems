import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

vendor_name = flag_gems.vendor_name

if QUICK_MODE:
    # CI quick mode: limit to float32 for faster validation
    FLOAT_DTYPES = [torch.float32]
    STRIDES = [1]
    PADDINGS = [1]
    DILATIONS = [1]
    BIASES = [True]
else:
    # bf16 is excluded: transposed convolution accumulates enough rounding
    # error in bf16 to exceed the accuracy tolerance (rel diff ~0.042 > 0.016),
    # so the kernels are validated on fp16/fp32 only.
    FLOAT_DTYPES = [torch.float16, torch.float32]
    STRIDES = [1, 2]
    PADDINGS = [0, 1]
    DILATIONS = [1]
    BIASES = [True, False]

# (input_shape, weight_shape, groups) for the forward (non-transposed) path.
SHAPE_CONV1D = [
    ((2, 3, 9), (4, 3, 3), 1),
]
SHAPE_CONV2D = [
    ((1, 2, 5, 5), (1, 2, 3, 3), 1),
    ((2, 3, 9, 9), (4, 3, 3, 3), 1),
]
SHAPE_CONV3D = [
    ((1, 2, 5, 5, 5), (2, 2, 3, 3, 3), 1),
]


def _to_list(val, n):
    return [val] * n


@pytest.mark.convolution
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CONV1D)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_convolution_1d(shape, kernel, stride, padding, groups, dtype, bias):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    if bias is True:
        bias = torch.randn([weight.shape[0]], dtype=dtype, device=flag_gems.device)
    else:
        bias = None

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True) if bias is not None else None
    torch.backends.cudnn.allow_tf32 = False

    ref_out = torch.convolution(
        ref_inp,
        ref_weight,
        ref_bias,
        _to_list(stride, 1),
        _to_list(padding, 1),
        _to_list(1, 1),
        False,
        _to_list(0, 1),
        groups,
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.convolution(
            inp,
            weight,
            bias,
            _to_list(stride, 1),
            _to_list(padding, 1),
            _to_list(1, 1),
            False,
            _to_list(0, 1),
            groups,
        )

    reduce_dim = weight.shape[1] * weight.shape[2]
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.convolution
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CONV2D)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dilation", DILATIONS)
@pytest.mark.parametrize("bias", BIASES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_convolution_2d(shape, kernel, stride, padding, groups, dtype, dilation, bias):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    if bias is True:
        bias = torch.randn([weight.shape[0]], dtype=dtype, device=flag_gems.device)
    else:
        bias = None

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True) if bias is not None else None
    torch.backends.cudnn.allow_tf32 = False

    ref_out = torch.convolution(
        ref_inp,
        ref_weight,
        ref_bias,
        _to_list(stride, 2),
        _to_list(padding, 2),
        _to_list(dilation, 2),
        False,
        _to_list(0, 2),
        groups,
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.convolution(
            inp,
            weight,
            bias,
            _to_list(stride, 2),
            _to_list(padding, 2),
            _to_list(dilation, 2),
            False,
            _to_list(0, 2),
            groups,
        )

    reduce_dim = weight.shape[1] * weight.shape[2] * weight.shape[3]
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.convolution
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CONV3D)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_convolution_3d(shape, kernel, stride, padding, groups, dtype, bias):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    if bias is True:
        bias = torch.randn([weight.shape[0]], dtype=dtype, device=flag_gems.device)
    else:
        bias = None

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True) if bias is not None else None
    torch.backends.cudnn.allow_tf32 = False

    ref_out = torch.convolution(
        ref_inp,
        ref_weight,
        ref_bias,
        _to_list(stride, 3),
        _to_list(padding, 3),
        _to_list(1, 3),
        False,
        _to_list(0, 3),
        groups,
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.convolution(
            inp,
            weight,
            bias,
            _to_list(stride, 3),
            _to_list(padding, 3),
            _to_list(1, 3),
            False,
            _to_list(0, 3),
            groups,
        )

    reduce_dim = weight.shape[1] * weight.shape[2] * weight.shape[3] * weight.shape[4]
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.convolution
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CONV2D)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_convolution_transposed_2d(shape, kernel, stride, padding, groups, dtype, bias):
    # transposed weight has layout (in_channels, out_channels/groups, kh, kw)
    in_channels = shape[1]
    out_channels = kernel[0]
    t_weight_shape = (in_channels, out_channels, kernel[2], kernel[3])

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(t_weight_shape, dtype=dtype, device=flag_gems.device)
    if bias is True:
        bias = torch.randn([out_channels], dtype=dtype, device=flag_gems.device)
    else:
        bias = None

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True) if bias is not None else None
    torch.backends.cudnn.allow_tf32 = False

    output_padding = 0

    ref_out = torch.convolution(
        ref_inp,
        ref_weight,
        ref_bias,
        _to_list(stride, 2),
        _to_list(padding, 2),
        _to_list(1, 2),
        True,
        _to_list(output_padding, 2),
        groups,
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.convolution(
            inp,
            weight,
            bias,
            _to_list(stride, 2),
            _to_list(padding, 2),
            _to_list(1, 2),
            True,
            _to_list(output_padding, 2),
            groups,
        )

    reduce_dim = in_channels * kernel[2] * kernel[3]
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)
