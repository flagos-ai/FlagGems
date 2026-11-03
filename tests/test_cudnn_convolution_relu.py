import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

FLOAT_DTYPES = utils.FLOAT_DTYPES


# (input, weight, groups, stride, padding, dilation). Stride/padding/
# dilation are folded into each case rather than a full cross-product:
# each pairing is exercised once across the set, as in test_cudnn_convolution.
CASES_2D = [
    ((2, 8, 16, 16), (6, 8, 3, 3), 1, 1, 1, 1),  # basic 2D with bias
    ((1, 16, 28, 28), (32, 16, 3, 3), 1, 1, 1, 1),  # C_in >= 16, larger tile
    ((2, 3, 9, 9), (6, 3, 3, 3), 1, 2, 0, 1),  # downsample, no padding
    ((4, 4, 8, 8), (4, 2, 3, 3), 2, 1, 1, 1),  # grouped
    ((2, 4, 9, 9), (4, 1, 3, 3), 4, 1, 1, 1),  # depthwise
    ((2, 4, 9, 9), (6, 4, 1, 1), 1, 1, 0, 1),  # 1x1 kernel, C_in < 16
    ((2, 3, 7, 7), (4, 3, 3, 3), 1, 1, 2, 1),  # padding > kernel/2
]


# (input, weight, groups, stride, padding, dilation), stride held at 1.
# Dilation rewrites the effective kernel, so it is the only thing these
# cases add over the base set (which runs dilation == 1).
CASES_2D_DILATION = [
    ((1, 2, 9, 9), (3, 2, 3, 3), 1, 1, 0, 2),
    ((2, 3, 11, 11), (6, 3, 3, 3), 1, 1, 1, 2),
]


# (input, weight, groups, stride(h, w), padding(h, w), dilation(h, w))
CASES_2D_ASYMMETRIC = [
    ((2, 4, 15, 15), (4, 4, 3, 5), 1, (2, 1), (1, 2), (1, 1)),
    ((1, 4, 12, 18), (4, 4, 5, 3), 1, (1, 2), (0, 1), (2, 1)),
]


CASES_3D = [
    ((1, 4, 8, 8, 8), (4, 4, 3, 3, 3), 1, 1, 1, 1),  # basic 3D with bias
    ((1, 4, 6, 6, 6), (4, 1, 3, 3, 3), 4, 1, 0, 1),  # 3D depthwise
    ((2, 4, 5, 5, 5), (4, 4, 3, 3, 3), 1, 2, 1, 1),  # downsample with padding
]


CASES_1D = [
    ((2, 4, 16), (6, 4, 3), 1, 1, 1, 1),  # 1D with bias
    ((2, 4, 24), (4, 1, 3), 4, 2, 0, 1),  # 1D depthwise, downsample
]


# Bias-free runs keep the dedicated depthwise/pointwise/direct conv paths.
NO_BIAS_CASES = [
    ((2, 8, 16, 16), (6, 8, 3, 3), 1, 1, 1, 1),  # generic 2D
    ((1, 16, 8, 8), (32, 16, 1, 1), 1, 1, 0, 1),  # pointwise 1x1
    ((1, 8, 6, 6), (8, 1, 3, 3), 8, 1, 0, 1),  # depthwise
    ((1, 2, 9, 9, 9), (3, 2, 1, 1, 1), 1, 1, 0, 1),  # pointwise 3D
]


# Bias forms the cuDNN engine accepts: the flat (C_out,) vector (the
# canonical aten shape), (1, C_out, 1, ...), and -- only for C_out == 1
# (measured: a scalar bias with C_out > 1 finds no cuDNN engine) -- a
# scalar.
BIAS_SHAPE_CASES = [
    ((2, 4, 8, 8), (6, 4, 3, 3), (6,)),
    ((2, 4, 8, 8), (6, 4, 3, 3), (1, 6, 1, 1)),
    ((2, 4, 8, 8), (1, 4, 3, 3), ()),
]


aten = torch.ops.aten.cudnn_convolution_relu


_CONV_REF_FN = {
    1: torch.nn.functional.conv1d,
    2: torch.nn.functional.conv2d,
    3: torch.nn.functional.conv3d,
}


def _check_out(inp, weight, bias, stride, padding, dilation, groups, dtype):
    ndim = inp.ndim - 2
    ref_out = _reference_output(inp, weight, bias, stride, padding, dilation, groups)
    # The reference (under --ref cpu the fp64 CPU copy) only sizes the buffer;
    # the gems side writes into its own GPU tensor.
    out = torch.empty(ref_out.shape, dtype=dtype, device=flag_gems.device)
    flag_gems.cudnn_convolution_relu_out(
        inp,
        weight,
        bias,
        _as_list(stride, ndim),
        _as_list(padding, ndim),
        _as_list(dilation, ndim),
        groups,
        out=out,
    )
    utils.gems_assert_close(out, ref_out, dtype)


def _check(inp, weight, bias, stride, padding, dilation, groups, dtype):
    ref_out = _reference_output(inp, weight, bias, stride, padding, dilation, groups)
    res_out = flag_gems.cudnn_convolution_relu(
        inp,
        weight,
        bias,
        _as_list(stride, inp.ndim - 2),
        _as_list(padding, inp.ndim - 2),
        _as_list(dilation, inp.ndim - 2),
        groups,
    )
    utils.gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# 2D convolution + ReLU
# ---------------------------------------------------------------------------


def _reference_output(inp, weight, bias, stride, padding, dilation, groups):
    """Reference for one case.

    The aten operator is CUDA-only, so the device reference is the aten op
    itself at the operator's dtype -- except for 1D/3D inputs, which need a
    rebuilt reference: cuDNN's fused engine rejects 1D outright (measured:
    "xDesc.getNbDims() != bDesc.getNbDims()") and its 3D fp32 engine may run
    with TF32 tensor cores, whose extra rounding neither the fp32 conv
    accumulator nor the Triton kernel (both closer to the fp64 truth) can
    match at the 1e-4 tolerance. Under --ref cpu the reference is the same
    conv*d + relu built on the fp64-upcast CPU copy.
    """
    ndim = inp.ndim - 2
    if utils.TO_CPU or ndim == 1:
        if utils.TO_CPU:
            ref_inp = utils.to_reference(inp, True)
            ref_weight = utils.to_reference(weight, True)
            ref_bias = None if bias is None else utils.to_reference(bias, True)
        else:
            # cuDNN's 1D fused engine does not exist, so the on-device 1D
            # reference is rebuilt from F.conv1d + relu in fp32 (the conv
            # kernels' accumulator precision): keeping the reference at the
            # operator dtype would bake its own fp16 conv rounding into the
            # expected values, which no accumulator-order change can match.
            ref_inp = inp.to(torch.float32)
            ref_weight = weight.to(torch.float32)
            ref_bias = None if bias is None else bias.to(torch.float32)
        if ref_bias is not None:
            # F.conv*d requires a flat (C_out,) bias; the aten operator also
            # accepts (1, C_out, 1, ...) and scalar forms, so normalize before
            # handing it to the reference.
            ref_bias = ref_bias.reshape(weight.shape[0])
        ref_out = _CONV_REF_FN[ndim](
            ref_inp, ref_weight, ref_bias, stride, padding, dilation, groups
        )
        return torch.relu(ref_out)
    if ndim == 3 and inp.dtype == torch.float32:
        # The fused 3D fp32 engine may dispatch to TF32 tensor cores; the
        # reference is rebuilt from F.conv3d with TF32 disabled so it tracks
        # the fp32 conv accumulator, matching both the aten fp64 error and
        # the Triton kernel.
        prev_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        try:
            ref_out = torch.nn.functional.conv3d(
                inp,
                weight,
                bias.reshape(-1) if bias is not None else None,
                stride=_as_list(stride, ndim),
                padding=_as_list(padding, ndim),
                dilation=_as_list(dilation, ndim),
                groups=groups,
            )
        finally:
            torch.backends.cudnn.allow_tf32 = prev_tf32
        return torch.relu(ref_out)
    return aten(
        inp,
        weight,
        bias,
        _as_list(stride, ndim),
        _as_list(padding, ndim),
        _as_list(dilation, ndim),
        groups,
    )


def _as_list(param, ndim):
    if isinstance(param, (list, tuple)):
        return list(param)
    return [param] * ndim


_CONV_REF_FN = {
    1: torch.nn.functional.conv1d,
    2: torch.nn.functional.conv2d,
    3: torch.nn.functional.conv3d,
}


@pytest.mark.cudnn_convolution_relu
@pytest.mark.parametrize("shape, kernel, groups, stride, padding, dilation", CASES_2D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_relu(
    shape, kernel, groups, stride, padding, dilation, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    bias = torch.randn(kernel[0], dtype=dtype, device=flag_gems.device)
    _check(inp, weight, bias, stride, padding, dilation, groups, dtype)


@pytest.mark.cudnn_convolution_relu
@pytest.mark.parametrize(
    "shape, kernel, groups, stride, padding, dilation", CASES_2D_DILATION
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_relu_dilation(
    shape, kernel, groups, stride, padding, dilation, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    bias = torch.randn(kernel[0], dtype=dtype, device=flag_gems.device)
    _check(inp, weight, bias, stride, padding, dilation, groups, dtype)


@pytest.mark.cudnn_convolution_relu
@pytest.mark.parametrize(
    "shape, kernel, groups, stride, padding, dilation", CASES_2D_ASYMMETRIC
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_relu_asymmetric(
    shape, kernel, groups, stride, padding, dilation, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    bias = torch.randn(kernel[0], dtype=dtype, device=flag_gems.device)
    _check(inp, weight, bias, stride, padding, dilation, groups, dtype)


# ---------------------------------------------------------------------------
# 3D / 1D convolution + ReLU
# ---------------------------------------------------------------------------


@pytest.mark.cudnn_convolution_relu
@pytest.mark.parametrize("shape, kernel, groups, stride, padding, dilation", CASES_3D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_relu_3d(
    shape, kernel, groups, stride, padding, dilation, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    bias = torch.randn(kernel[0], dtype=dtype, device=flag_gems.device)
    _check(inp, weight, bias, stride, padding, dilation, groups, dtype)


@pytest.mark.cudnn_convolution_relu
@pytest.mark.parametrize("shape, kernel, groups, stride, padding, dilation", CASES_1D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_relu_1d(
    shape, kernel, groups, stride, padding, dilation, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    bias = torch.randn(kernel[0], dtype=dtype, device=flag_gems.device)
    _check(inp, weight, bias, stride, padding, dilation, groups, dtype)


# ---------------------------------------------------------------------------
# Bias-free path (the dedicated depthwise/pointwise/direct conv kernels)
# ---------------------------------------------------------------------------


@pytest.mark.cudnn_convolution_relu
@pytest.mark.parametrize(
    "shape, kernel, groups, stride, padding, dilation", NO_BIAS_CASES
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_relu_no_bias(
    shape, kernel, groups, stride, padding, dilation, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    _check(inp, weight, None, stride, padding, dilation, groups, dtype)


# ---------------------------------------------------------------------------
# Bias shapes
# ---------------------------------------------------------------------------


@pytest.mark.cudnn_convolution_relu
@pytest.mark.parametrize("shape, kernel, bias_shape", BIAS_SHAPE_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_relu_bias_shapes(shape, kernel, bias_shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    bias = torch.randn(bias_shape, dtype=dtype, device=flag_gems.device)
    _check(inp, weight, bias, 1, 1, 1, 1, dtype)


# ---------------------------------------------------------------------------
# ReLU boundary: values that land on the clipping edge
#
# A constant input with an identity 1x1 kernel makes the conv output equal the
# input exactly, so the bias alone sets the clip boundary per channel and the
# negative-side saturation is exercised without any convolution rounding.
# ---------------------------------------------------------------------------


@pytest.mark.cudnn_convolution_relu
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_relu_boundary(dtype):
    inp = torch.full((2, 4, 8, 8), -2.0, dtype=dtype, device=flag_gems.device)
    weight = torch.eye(4, dtype=dtype, device=flag_gems.device).view(4, 4, 1, 1)
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype, device=flag_gems.device)
    _check(inp, weight, bias, 1, 0, 1, 1, dtype)


# ---------------------------------------------------------------------------
# .out variant
# ---------------------------------------------------------------------------


@pytest.mark.cudnn_convolution_relu_out
@pytest.mark.parametrize("shape, kernel, groups, stride, padding, dilation", CASES_2D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_relu_out(
    shape, kernel, groups, stride, padding, dilation, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    bias = torch.randn(kernel[0], dtype=dtype, device=flag_gems.device)
    _check_out(inp, weight, bias, stride, padding, dilation, groups, dtype)


@pytest.mark.cudnn_convolution_relu_out
@pytest.mark.parametrize(
    "shape, kernel, groups, stride, padding, dilation", NO_BIAS_CASES
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_relu_out_no_bias(
    shape, kernel, groups, stride, padding, dilation, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    _check_out(inp, weight, None, stride, padding, dilation, groups, dtype)


# ---------------------------------------------------------------------------
# 2D convolution + ReLU
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3D / 1D convolution + ReLU
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Bias-free path (the dedicated depthwise/pointwise/direct conv kernels)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Bias shapes
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ReLU boundary: values that land on the clipping edge
#
# A constant input with an identity 1x1 kernel makes the conv output equal the
# input exactly, so the bias alone sets the clip boundary per channel and the
# negative-side saturation is exercised without any convolution rounding.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# .out variant
# ---------------------------------------------------------------------------
