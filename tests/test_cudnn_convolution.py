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
from .conftest import QUICK_MODE

vendor_name = flag_gems.vendor_name

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)

# The reference below is the device's own convolution, so it has to be fp32-exact
# for an fp32 comparison to mean anything. Without --ref cpu, to_reference()
# cannot upcast (the device has no fp64) and the reference runs in fp32 on the
# NPU -- where the cube rounds both operands to an ~11-bit mantissa. That is the
# same reduced-precision trade-off the CUDA conv tests already opt out of with
# torch.backends.cudnn.allow_tf32 = False (see test_conv2d.py); allow_hf32 is
# its Ascend equivalent.
#
# Measured against an fp64 CPU reference on (1,4,16)/(4,1,3): the vendor conv is
# off by 4.5e-4 with HF32 on and 2.4e-7 with it off, while this operator's kernel
# is 1.2e-7 either way. So leaving it on fails the fp32 column at atol=1e-4 over
# a difference that lives entirely in the reference.
if vendor_name == "ascend":
    try:
        torch.npu.conv.allow_hf32 = False
    except AttributeError:
        pass

# ---------------------------------------------------------------------------
# Parameter sets
#
# aten::cudnn_convolution is a lower-level, bias-free convolution. It
# dispatches to conv1d/conv2d/conv3d based on ``input.ndim - 2`` and receives
# padding/stride/dilation as lists (SymInt[]). The reference below uses
# torch.nn.functional.conv{1,2,3}d so it stays portable across devices and
# --ref cpu, unlike torch.cudnn_convolution which is CUDA-only.
# ---------------------------------------------------------------------------

if QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    SHAPES_1D = [((2, 2, 8), (3, 2, 3), 1)]
    SHAPES_1D_DILATION = [((1, 2, 12), (3, 2, 3), 1)]
    SHAPES_2D = [
        ((1, 2, 5, 5), (3, 2, 3, 3), 1),
        ((1, 16, 8, 8), (32, 16, 3, 3), 1),  # C_in >= 16 -> no channel padding
    ]
    SHAPES_2D_DILATION = [((1, 2, 9, 9), (3, 2, 3, 3), 1)]
    SHAPES_2D_ASYMMETRIC = [((1, 3, 16, 16), (4, 3, 3, 5), 1, (1, 2), (1, 2), (1, 1))]
    SHAPES_3D = [
        ((1, 2, 5, 5, 5), (3, 2, 3, 3, 3), 1),
        ((1, 16, 4, 4, 4), (16, 16, 3, 3, 3), 1),  # C_in >= 16 -> no channel padding
    ]
    SHAPES_3D_DILATION = [((1, 2, 7, 7, 7), (3, 2, 3, 3, 3), 1)]
    STRIDES = [1]
    PADDINGS = [1]
    DILATIONS = [1]
else:
    FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
    SHAPES_1D = [
        ((2, 2, 8), (3, 2, 3), 1),  # basic
        ((2, 3, 12), (5, 3, 3), 1),  # multi-channel, C_in and C_out both < 16
        ((1, 4, 16), (4, 1, 3), 4),  # depthwise (groups == C_in == C_out)
        ((2, 3, 7), (4, 3, 1), 1),  # 1x1 kernel, lifted to a 2D conv internally
    ]
    SHAPES_1D_DILATION = [
        ((1, 2, 12), (3, 2, 3), 1),
        ((2, 3, 20), (5, 3, 3), 1),
        ((2, 4, 24), (4, 4, 7), 1),  # 7 taps dilated by 3 -> 19 wide
    ]
    SHAPES_2D = [
        ((1, 2, 5, 5), (3, 2, 3, 3), 1),  # basic, C_in < 16 (padding path)
        ((1, 16, 8, 8), (32, 16, 3, 3), 1),  # C_in >= 16 -> no channel padding
        ((2, 3, 9, 9), (6, 3, 3, 3), 1),  # multi-channel, C_in and C_out both < 16
        ((2, 3, 8, 8), (6, 3, 3, 3), 1),  # even spatial
        ((4, 4, 8, 8), (4, 2, 3, 3), 2),  # grouped (groups = 2)
        ((1, 8, 6, 6), (8, 1, 3, 3), 8),  # depthwise
        ((2, 3, 7, 7), (4, 3, 1, 1), 1),  # 1x1 kernel - pointwise until p or s moves
        ((1, 2, 4, 4), (5, 2, 3, 3), 1),  # small spatial, C_in < 16
    ]
    SHAPES_2D_DILATION = [
        ((1, 2, 9, 9), (3, 2, 3, 3), 1),
        ((2, 3, 11, 11), (6, 3, 3, 3), 1),
    ]
    # (input, weight, groups, stride(h, w), padding(h, w), dilation(h, w))
    SHAPES_2D_ASYMMETRIC = [
        ((1, 3, 16, 16), (4, 3, 3, 5), 1, (1, 2), (1, 2), (1, 1)),
        ((2, 2, 12, 18), (6, 2, 5, 3), 1, (2, 1), (0, 1), (1, 1)),
        ((1, 4, 15, 15), (4, 2, 3, 3), 2, (1, 2), (2, 0), (2, 1)),
    ]
    SHAPES_3D = [
        ((1, 2, 5, 5, 5), (3, 2, 3, 3, 3), 1),  # basic, C_in < 16
        ((1, 16, 4, 4, 4), (16, 16, 3, 3, 3), 1),  # C_in >= 16 -> no channel padding
        ((2, 3, 7, 7, 7), (6, 3, 3, 3, 3), 1),  # multi-channel
        ((2, 4, 5, 5, 5), (4, 4, 3, 3, 3), 1),  # multi-channel, odd spatial
        ((1, 4, 6, 6, 6), (4, 1, 3, 3, 3), 4),  # depthwise
        ((1, 2, 9, 9, 9), (3, 2, 1, 1, 1), 1),  # 1x1x1 kernel, collapsed to a GEMM
    ]
    SHAPES_3D_DILATION = [
        ((1, 2, 7, 7, 7), (3, 2, 3, 3, 3), 1),
    ]
    STRIDES = [1, 2]
    PADDINGS = [0, 1]
    DILATIONS = [1, 2]


# (input, weight, groups, stride, padding, dilation)
SHAPES_1D_ASYMMETRIC = [
    ((2, 4, 15), (6, 2, 3), 2, (2,), (1,), (2,)),
]

SHAPES_3D_ASYMMETRIC = [
    ((1, 4, 7, 8, 9), (4, 4, 3, 2, 3), 1, (1, 2, 1), (1, 0, 1), (1, 1, 2)),
]


LARGE_KERNEL_CASES = [
    ((2, 16, 12, 12), (16, 16, 5, 5), 1, 1, 0, 1),
    ((2, 16, 12, 12), (16, 16, 5, 5), 1, 1, 1, 1),
    ((1, 4, 11, 11), (4, 4, 7, 7), 1, 1, 3, 1),  # 49 taps, dilation 3 -> 19 wide
    ((1, 4, 9, 13), (4, 4, 3, 7), 1, 1, 1, 1),  # non-square kernel
]

CHANNEL_SHAPES_2D = [
    ((2, 1, 8, 8), (4, 1, 3, 3), 1),  # C_in == 1
    ((2, 9, 8, 8), (8, 9, 3, 3), 1),  # just past _DIRECT_MAX_C
    ((2, 15, 8, 8), (8, 15, 3, 3), 1),  # just below _MIN_DOT_K
    ((2, 17, 8, 8), (8, 17, 3, 3), 1),  # just past _MIN_DOT_K
    ((2, 8, 8, 8), (3, 8, 3, 3), 1),  # C_out < C_in
    ((2, 8, 8, 8), (1, 8, 3, 3), 1),  # C_out == 1
    ((2, 4, 8, 8), (8, 1, 3, 3), 4),  # depthwise, channel multiplier 2
]

# The same boundaries in 1D and 3D, where the channel count is the reduction
# length of a differently-shaped kernel rather than of the 2D one.
CHANNEL_SHAPES_1D_3D = [
    ((2, 1, 16), (4, 1, 3), 1),
    ((2, 4, 16), (8, 1, 3), 4),  # 1D depthwise, channel multiplier 2
    ((2, 15, 16), (8, 15, 3), 1),
    ((1, 1, 5, 5, 5), (2, 1, 3, 3, 3), 1),
    ((1, 4, 5, 5, 5), (8, 1, 3, 3, 3), 4),  # 3D depthwise, channel multiplier 2
    ((1, 15, 5, 5, 5), (4, 15, 3, 3, 3), 1),
]

POINTWISE_ESCAPE_CASES = [
    ((2, 3, 9, 9), (6, 3, 1, 1), 1, 1, 0, 2),  # dilation != 1
    ((2, 4, 9, 9), (4, 2, 1, 1), 2, 1, 0, 1),  # grouped
    ((2, 3, 1, 1), (4, 3, 1, 1), 1, 1, 0, 1),  # single pixel, still pointwise
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONV_REF_FN = {
    1: torch.nn.functional.conv1d,
    2: torch.nn.functional.conv2d,
    3: torch.nn.functional.conv3d,
}


def _to_list(param, ndim):
    """aten::cudnn_convolution takes padding/stride/dilation as SymInt[].

    A scalar parameter is broadcast to every spatial dimension (length ndim),
    matching how the aten operator materializes the per-dimension list.
    """
    if isinstance(param, (list, tuple)):
        return list(param)
    return [param] * ndim


def _reference_output(inp, weight, padding, stride, dilation, groups, dtype):
    ndim = inp.ndim - 2
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    out = _CONV_REF_FN[ndim](
        ref_inp,
        ref_weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    return out.to(dtype)


def _gem_output(inp, weight, padding, stride, dilation, groups):
    ndim = inp.ndim - 2
    return flag_gems.cudnn_convolution(
        inp,
        weight,
        padding=_to_list(padding, ndim),
        stride=_to_list(stride, ndim),
        dilation=_to_list(dilation, ndim),
        groups=groups,
        benchmark=False,
        deterministic=False,
        allow_tf32=False,
    )


def _check(inp, weight, padding, stride, dilation, groups, dtype):
    ref_out = _reference_output(inp, weight, padding, stride, dilation, groups, dtype)
    res_out = _gem_output(inp, weight, padding, stride, dilation, groups)
    utils.gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# 1D convolution
# ---------------------------------------------------------------------------


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize("shape, kernel, groups", SHAPES_1D)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_1d(shape, kernel, groups, stride, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    _check(inp, weight, padding, stride, 1, groups, dtype)


# 1D dilation goes through conv1d -> conv2d internally, so it is also exercised
# by the 2D dilation test, but the parameter rewrite is 1D-specific and worth
# covering on its own. Stride is held at 1 to keep the cross-product affordable.
@pytest.mark.cudnn_convolution
@pytest.mark.parametrize("shape, kernel, groups", SHAPES_1D_DILATION)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dilation", DILATIONS)
def test_cudnn_convolution_1d_dilation(
    shape, kernel, groups, stride, padding, dtype, dilation
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    _check(inp, weight, padding, stride, dilation, groups, dtype)


# ---------------------------------------------------------------------------
# 2D convolution
# ---------------------------------------------------------------------------


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize("shape, kernel, groups", SHAPES_2D)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_2d(shape, kernel, groups, stride, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    _check(inp, weight, padding, stride, 1, groups, dtype)


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize(
    "shape, kernel, groups, stride, padding, dilation", LARGE_KERNEL_CASES
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_large_kernel(
    shape, kernel, groups, stride, padding, dilation, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    _check(inp, weight, padding, stride, dilation, groups, dtype)


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize("shape, kernel, groups", SHAPES_2D_DILATION)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dilation", DILATIONS)
def test_cudnn_convolution_2d_dilation(
    shape, kernel, groups, stride, padding, dtype, dilation
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    _check(inp, weight, padding, stride, dilation, groups, dtype)


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize(
    "shape, kernel, groups, stride, padding, dilation", SHAPES_2D_ASYMMETRIC
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_2d_asymmetric(
    shape, kernel, groups, stride, padding, dilation, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    _check(inp, weight, padding, stride, dilation, groups, dtype)


# ---------------------------------------------------------------------------
# 3D convolution
# ---------------------------------------------------------------------------


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize("shape, kernel, groups", SHAPES_3D)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_3d(shape, kernel, groups, stride, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    _check(inp, weight, padding, stride, 1, groups, dtype)


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize("shape, kernel, groups", SHAPES_3D_DILATION)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dilation", DILATIONS)
def test_cudnn_convolution_3d_dilation(
    shape, kernel, groups, stride, padding, dtype, dilation
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    _check(inp, weight, padding, stride, dilation, groups, dtype)


# ---------------------------------------------------------------------------
# 1D / 3D asymmetric parameters
#
# SHAPES_2D_ASYMMETRIC already varies the parameters per dimension in 2D. The
# 1D and 3D rewrites (unit height in 1D, three independent axes in 3D) are
# separate code paths, so they get a case each rather than inheriting the 2D
# one's confidence.
# ---------------------------------------------------------------------------


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize(
    "shape, kernel, groups, stride, padding, dilation", SHAPES_1D_ASYMMETRIC
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_1d_asymmetric(
    shape, kernel, groups, stride, padding, dilation, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    _check(inp, weight, padding, stride, dilation, groups, dtype)


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize(
    "shape, kernel, groups, stride, padding, dilation", SHAPES_3D_ASYMMETRIC
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_3d_asymmetric(
    shape, kernel, groups, stride, padding, dilation, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    _check(inp, weight, padding, stride, dilation, groups, dtype)


# ---------------------------------------------------------------------------
# Channel-count boundaries
# ---------------------------------------------------------------------------


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize("shape, kernel, groups", CHANNEL_SHAPES_2D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_channel_counts_2d(shape, kernel, groups, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    _check(inp, weight, 1, 1, 1, groups, dtype)


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize("shape, kernel, groups", CHANNEL_SHAPES_1D_3D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_channel_counts_1d_3d(shape, kernel, groups, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    _check(inp, weight, 1, 1, 1, groups, dtype)


# ---------------------------------------------------------------------------
# 1x1 kernels that miss the pointwise fast path
# ---------------------------------------------------------------------------


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize(
    "shape, kernel, groups, stride, padding, dilation", POINTWISE_ESCAPE_CASES
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_pointwise_escape(
    shape, kernel, groups, stride, padding, dilation, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    _check(inp, weight, padding, stride, dilation, groups, dtype)


# ---------------------------------------------------------------------------
# Scalar parameter form
#
# The aten operator always passes padding/stride/dilation as lists, but the
# flag_gems.cudnn_convolution wrapper (and conv1d/conv2d/conv3d beneath it)
# also accepts plain ints. One shape per rank, rather than every shape above,
# keeps the duplicate coverage from doubling the CI budget.
# ---------------------------------------------------------------------------

SCALAR_RANK_CASES = [
    ((2, 3, 9), (4, 3, 3), 1),
    ((2, 3, 9, 9), (4, 3, 3, 3), 1),
    ((1, 3, 5, 5, 5), (4, 3, 3, 3, 3), 1),
]


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize("shape, kernel, groups", SCALAR_RANK_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_scalar_params(shape, kernel, groups, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)

    ref_out = _reference_output(inp, weight, 1, 1, 1, groups, dtype)
    res_out = flag_gems.cudnn_convolution(
        inp,
        weight,
        padding=1,
        stride=1,
        dilation=1,
        groups=groups,
        benchmark=False,
        deterministic=False,
        allow_tf32=False,
    )
    utils.gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# Output shape
#
# The reference check above would already catch a wrong extent indirectly, but
# not distinguish it from a wrong value, and the two are worth telling apart:
# a wrong extent is a tiling/parameter bug, a wrong value is arithmetic. This
# exercises the formula over a grid of parameter combinations instead.
# ---------------------------------------------------------------------------


def _expected_out_shape(inp, weight, padding, stride, dilation):
    ndim = inp.ndim - 2
    padding, stride, dilation = (
        _to_list(padding, ndim),
        _to_list(stride, ndim),
        _to_list(dilation, ndim),
    )
    shape = [inp.shape[0], weight.shape[0]]
    for i in range(ndim):
        k = weight.shape[2 + i]
        shape.append(
            (inp.shape[2 + i] + 2 * padding[i] - dilation[i] * (k - 1) - 1) // stride[i]
            + 1
        )
    return shape


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize(
    "shape, kernel, groups, stride, padding, dilation",
    SHAPES_2D_ASYMMETRIC + POINTWISE_ESCAPE_CASES[1:] + SHAPES_3D_ASYMMETRIC,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_output_shape(
    shape, kernel, groups, stride, padding, dilation, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    res_out = _gem_output(inp, weight, padding, stride, dilation, groups)
    assert list(res_out.shape) == _expected_out_shape(
        inp, weight, padding, stride, dilation
    )


# Every shape here is also value-checked elsewhere (the first two via
# SCALAR_RANK_CASES' scalar-params test, the last two via SHAPES_2D/SHAPES_3D),
# so this test only needs to assert the extent, not the values.
SHAPE_GRID_CASES = [
    ((2, 3, 9), (4, 3, 3), 1),
    ((1, 3, 9, 9), (4, 3, 3, 3), 1),  # one output channel row
    ((2, 3, 8, 8), (6, 3, 3, 3), 1),
    ((2, 4, 5, 5, 5), (4, 4, 3, 3, 3), 1),
]


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_GRID_CASES)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
def test_cudnn_convolution_output_shape_grid(shape, kernel, groups, stride, padding):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=torch.float32, device=flag_gems.device)
    res_out = _gem_output(inp, weight, padding, stride, 1, groups)
    assert list(res_out.shape) == _expected_out_shape(inp, weight, padding, stride, 1)


# ---------------------------------------------------------------------------
# Non-contiguous operands
#
# The aten operator receives whatever layout the caller had. A strided input or
# a transposed weight must give the same answer as its contiguous copy, not a
# silently re-interpreted one.
# ---------------------------------------------------------------------------

NON_CONTIGUOUS_CASES = [
    # (shape, out_c, kernel, groups, how to make the input non-contiguous)
    ((2, 4, 12, 12), 4, (3, 3), 1, "strided_spatial"),
    ((2, 4, 12, 12), 4, (3, 3), 1, "strided_channel"),
    ((2, 8, 12, 12), 4, (3, 3), 2, "strided_channel"),
    ((4, 3, 16), 6, (3,), 1, "strided_spatial"),
    # strided_channel halves C to 4, so groups=4 multiplies each of them 2x
    ((2, 8, 12, 12), 8, (3, 3), 4, "strided_channel"),
]


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize("shape, out_c, kernel, groups, layout", NON_CONTIGUOUS_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_non_contiguous_input(
    shape, out_c, kernel, groups, layout, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if layout == "strided_channel":
        inp = inp[:, ::2]
    elif inp.ndim == 4:
        inp = inp[:, :, ::2, ::2]
    else:
        inp = inp[:, :, ::2]
    assert not inp.is_contiguous()

    # Built after the view, so C_in matches what the view actually has.
    weight = torch.randn(
        (out_c, inp.shape[1] // groups, *kernel), dtype=dtype, device=flag_gems.device
    )
    ref_out = _reference_output(inp, weight, 0, 1, 1, groups, dtype)
    res_out = _gem_output(inp, weight, 0, 1, 1, groups)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_non_contiguous_weight(dtype):
    inp = torch.randn((2, 4, 8, 8), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((4, 4, 3, 5), dtype=dtype, device=flag_gems.device)
    transposed = weight.transpose(2, 3)
    assert not transposed.is_contiguous()

    ref_out = _reference_output(inp, transposed, 1, 1, 1, 1, dtype)
    res_out = _gem_output(inp, transposed, 1, 1, 1, 1)
    utils.gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------

INTERFACE_SHAPES = [
    ((2, 3, 9), (4, 3, 3), 1),
    ((2, 3, 9, 9), (4, 3, 3, 3), 1),
    ((1, 3, 5, 5, 5), (4, 3, 3, 3, 3), 1),
]


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize("shape, kernel, groups", INTERFACE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_interface(shape, kernel, groups, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    inp_copy, weight_copy = inp.clone(), weight.clone()
    ndim = inp.ndim - 2
    pad, strd, dil = [1] * ndim, [1] * ndim, [1] * ndim

    with flag_gems.use_gems():
        base = torch.cudnn_convolution(
            inp, weight, pad, strd, dil, groups, False, False, False
        )
        flagged = torch.cudnn_convolution(
            inp, weight, pad, strd, dil, groups, True, True, True
        )
        again = torch.cudnn_convolution(
            inp, weight, pad, strd, dil, groups, False, False, False
        )

    # benchmark / deterministic / allow_tf32 are accepted for interface
    # compatibility only; neither implementation selects an algorithm on them.
    assert torch.equal(base, flagged)
    # Same inputs, same result - the kernels are deterministic.
    assert torch.equal(base, again)
    # The operands are read, not modified.
    assert torch.equal(inp, inp_copy) and torch.equal(weight, weight_copy)

    assert base.dtype == dtype
    assert base.device == inp.device
    assert base.is_contiguous()
    assert base.shape[0] == inp.shape[0] and base.shape[1] == weight.shape[0]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize(
    "shape, kernel, groups, stride, padding, dilation",
    [
        # 1x1 kernel with C_in == 16 (no channel padding, pointwise)
        ((1, 16, 4, 4), (32, 16, 1, 1), 1, 1, 0, 1),
        # padding larger than the kernel -> output larger than input
        ((1, 2, 4, 4), (3, 2, 1, 1), 1, 1, 2, 1),
        # single-pixel spatial dims
        ((1, 2, 1, 1), (3, 2, 1, 1), 1, 1, 0, 1),
        # small C_in (< 16) triggers the grouped channel-padding path
        ((2, 3, 5, 5), (8, 3, 3, 3), 1, 2, 1, 1),
    ],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_edge_cases(
    shape, kernel, groups, stride, padding, dilation, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    _check(inp, weight, padding, stride, dilation, groups, dtype)


# ---------------------------------------------------------------------------
# Invalid dimensionality
# ---------------------------------------------------------------------------


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize(
    "shape, kernel",
    [
        # 0 spatial dims
        ((2, 2), (3, 2)),
        # 4 spatial dims
        ((1, 2, 3, 3, 3, 3), (3, 2, 2, 2, 2, 2)),
    ],
)
def test_cudnn_convolution_invalid_ndim(shape, kernel):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(ValueError):
        flag_gems.cudnn_convolution(
            inp,
            weight,
            padding=[0],
            stride=[1],
            dilation=[1],
            groups=1,
            benchmark=False,
            deterministic=False,
            allow_tf32=False,
        )
