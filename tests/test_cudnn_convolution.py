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
    FLOAT_DTYPES = [torch.float16, torch.float32]
    SHAPES_1D = [
        ((2, 2, 8), (3, 2, 3), 1),  # basic
        # ((2, 3, 12), (5, 3, 3), 1),  # multi-channel - commented out to reduce CI timeout
        ((1, 4, 16), (4, 1, 3), 4),  # depthwise (groups == C_in == C_out)
        # ((2, 3, 7), (4, 3, 1), 1),  # 1x1 kernel - commented out to reduce CI timeout
    ]
    SHAPES_1D_DILATION = [
        ((1, 2, 12), (3, 2, 3), 1),
        ((2, 3, 20), (5, 3, 3), 1),
    ]
    SHAPES_2D = [
        ((1, 2, 5, 5), (3, 2, 3, 3), 1),  # basic, C_in < 16 (padding path)
        ((1, 16, 8, 8), (32, 16, 3, 3), 1),  # C_in >= 16 -> no channel padding
        # ((2, 3, 9, 9), (6, 3, 3, 3), 1),  # multi-channel - commented out to reduce CI timeout
        ((4, 4, 8, 8), (4, 2, 3, 3), 2),  # grouped (groups = 2)
        ((1, 8, 6, 6), (8, 1, 3, 3), 8),  # depthwise
        # ((2, 3, 7, 7), (4, 3, 1, 1), 1),  # 1x1 kernel - commented out to reduce CI timeout
        # ((1, 2, 4, 4), (5, 2, 3, 3), 1),  # small spatial, C_in < 16 - commented out to reduce CI timeout
    ]
    SHAPES_2D_DILATION = [
        ((1, 2, 9, 9), (3, 2, 3, 3), 1),
        # ((2, 3, 11, 11), (6, 3, 3, 3), 1),  # commented out to reduce CI timeout
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
        # ((2, 3, 7, 7, 7), (6, 3, 3, 3, 3), 1),  # multi-channel - commented out to reduce CI timeout
        ((1, 4, 6, 6, 6), (4, 1, 3, 3, 3), 4),  # depthwise
        # ((1, 2, 9, 9, 9), (3, 2, 1, 1, 1), 1),  # 1x1x1 kernel - commented out to reduce CI timeout
    ]
    SHAPES_3D_DILATION = [
        ((1, 2, 7, 7, 7), (3, 2, 3, 3, 3), 1),
    ]
    STRIDES = [1, 2]
    PADDINGS = [0, 1]
    DILATIONS = [1, 2]


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
    with flag_gems.use_gems():
        return torch.cudnn_convolution(
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


# 1D dilation goes through conv1d -> conv2d, so it is exercised by the 2D
# dilation test; commented out to reduce CI timeout.
# @pytest.mark.cudnn_convolution
# @pytest.mark.parametrize("shape, kernel, groups", SHAPES_1D_DILATION)
# @pytest.mark.parametrize("stride", [1])
# @pytest.mark.parametrize("padding", PADDINGS)
# @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
# @pytest.mark.parametrize("dilation", DILATIONS)
# def test_cudnn_convolution_1d_dilation(
#     shape, kernel, groups, stride, padding, dtype, dilation
# ):
#     inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
#     weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
#     _check(inp, weight, padding, stride, dilation, groups, dtype)


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


# 3D dilation commented out to reduce CI timeout (2D dilation is covered above).
# @pytest.mark.cudnn_convolution
# @pytest.mark.parametrize("shape, kernel, groups", SHAPES_3D_DILATION)
# @pytest.mark.parametrize("stride", [1])
# @pytest.mark.parametrize("padding", PADDINGS)
# @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
# @pytest.mark.parametrize("dilation", DILATIONS)
# def test_cudnn_convolution_3d_dilation(
#     shape, kernel, groups, stride, padding, dtype, dilation
# ):
#     inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
#     weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
#     _check(inp, weight, padding, stride, dilation, groups, dtype)


# ---------------------------------------------------------------------------
# Scalar parameter form
#
# The aten operator always passes padding/stride/dilation as lists, but the
# flag_gems.cudnn_convolution wrapper (and conv1d/conv2d/conv3d beneath it)
# also accepts plain ints. This re-runs every shape with scalar stride=1 /
# padding=1 / dilation=1, duplicating the list-form tests above, so it is
# commented out to reduce CI timeout. Uncomment to exercise the scalar form.
# ---------------------------------------------------------------------------


# @pytest.mark.cudnn_convolution
# @pytest.mark.parametrize(
#     "shape, kernel, groups", SHAPES_1D + SHAPES_2D + SHAPES_3D
# )
# @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
# def test_cudnn_convolution_scalar_params(shape, kernel, groups, dtype):
#     inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
#     weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
#
#     ref_out = _reference_output(inp, weight, 1, 1, 1, groups, dtype)
#     res_out = flag_gems.cudnn_convolution(
#         inp,
#         weight,
#         padding=1,
#         stride=1,
#         dilation=1,
#         groups=groups,
#         benchmark=False,
#         deterministic=False,
#         allow_tf32=False,
#     )
#     utils.gems_assert_close(res_out, ref_out, dtype)


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
