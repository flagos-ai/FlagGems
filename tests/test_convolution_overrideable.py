import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# (input_shape, weight_shape, groups) — channels chosen >= 16 to satisfy the
# tl.dot K >= 16 constraint of the underlying im2col kernel.
CONV_SHAPES = [
    ((1, 16, 8, 8), (6, 16, 3, 3), 1),
    ((2, 16, 9, 9), (4, 16, 3, 3), 1),
    ((4, 32, 8, 8), (16, 32, 2, 2), 1),
    ((2, 32, 10, 10), (8, 16, 3, 3), 2),
]


PADDINGS = [0, 1]


STRIDES = [1, 2]


def _reference_convolution(
    inp, weight, bias, stride, padding, dilation, transposed, output_padding, groups
):
    # convolution_overrideable is a dispatch-only stub with no native CUDA
    # kernel, so the reference uses the fully-general aten::convolution.
    return torch.ops.aten.convolution(
        inp,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )


@pytest.mark.convolution_overrideable
@pytest.mark.parametrize("shape, kernel, groups", CONV_SHAPES)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_convolution_overrideable(shape, kernel, groups, stride, padding, bias, dtype):
    torch.backends.cudnn.allow_tf32 = False
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    if bias:
        bias_tensor = torch.randn([kernel[0]], dtype=dtype, device=flag_gems.device)
    else:
        bias_tensor = None

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias_tensor, True) if bias else None

    dilation = [1, 1]
    output_padding = [0, 0]
    transposed = False

    ref_out = _reference_convolution(
        ref_inp,
        ref_weight,
        ref_bias,
        [stride, stride],
        [padding, padding],
        dilation,
        transposed,
        output_padding,
        groups,
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.convolution_overrideable(
            inp,
            weight,
            bias_tensor,
            [stride, stride],
            [padding, padding],
            dilation,
            transposed,
            output_padding,
            groups,
        )

    # Scale tolerance by the contraction size (in_channels/group * kH * kW).
    reduce_dim = max((kernel[1]) * kernel[2] * kernel[3], 1)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.convolution_overrideable
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_convolution_overrideable_transposed(dtype):
    torch.backends.cudnn.allow_tf32 = False
    inp = torch.randn((2, 16, 8, 8), dtype=dtype, device=flag_gems.device)
    # transposed weight layout: (in_channels, out_channels/groups, kH, kW)
    weight = torch.randn((16, 6, 3, 3), dtype=dtype, device=flag_gems.device)
    bias_tensor = torch.randn([6], dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias_tensor, True)

    ref_out = _reference_convolution(
        ref_inp,
        ref_weight,
        ref_bias,
        [1, 1],
        [1, 1],
        [1, 1],
        True,
        [0, 0],
        1,
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.convolution_overrideable(
            inp,
            weight,
            bias_tensor,
            [1, 1],
            [1, 1],
            [1, 1],
            True,
            [0, 0],
            1,
        )

    # Transposed weight layout is (in_channels, out_channels/groups, kH, kW), so
    # each output element accumulates over in_channels * kH * kW terms (groups=1
    # here). Size the tolerance off that reduction extent, matching the
    # conv_transpose2d accuracy test; the previous value used out_channels and
    # under-counted the reduction, making the bf16 case flaky by ~1 ULP.
    reduce_dim = max(weight.shape[0] * weight.shape[2] * weight.shape[3], 1)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.convolution_overrideable_out
@pytest.mark.parametrize("shape, kernel, groups", CONV_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_convolution_overrideable_out(shape, kernel, groups, dtype):
    torch.backends.cudnn.allow_tf32 = False
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    bias_tensor = torch.randn([kernel[0]], dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias_tensor, True)

    ref_out = _reference_convolution(
        ref_inp,
        ref_weight,
        ref_bias,
        [1, 1],
        [1, 1],
        [1, 1],
        False,
        [0, 0],
        groups,
    ).to(dtype)

    out = torch.empty_like(ref_out.to(flag_gems.device))
    with flag_gems.use_gems():
        res_out = torch.ops.aten.convolution_overrideable.out(
            inp,
            weight,
            bias_tensor,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            groups,
            out=out,
        )

    reduce_dim = max((kernel[1]) * kernel[2] * kernel[3], 1)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)
    assert res_out.data_ptr() == out.data_ptr()
