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

"""Correctness tests for ``aten::thnn_conv2d``.

``thnn_conv2d`` is the legacy THNN 2-D convolution entry point: it consumes an
NCHW input and an ``(out_channels, in_channels, kH, kW)`` weight with an
optional bias, and returns the NCHW output plane.

The comparison is against the native ``torch.ops.aten.thnn_conv2d`` result on
device. The native operator accumulates through an im2col + GEMM path with TF32
tensor cores for float32 inputs, so the tolerance is the convolution tolerance
rather than bit-exactness; float64 and the half precisions use their own
accumulation modes.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

gems_device = flag_gems.device

# The native thnn_conv2d routes float32 through an im2col + GEMM path whose
# TF32 tensor-core accumulation introduces ~1e-2 absolute error on these
# shapes, while the candidate's own accumulation is precise to ~1e-5. Comparing
# the two therefore requires the reference to use the same precision class, so
# TF32 is disabled for this module: the native reference then matches the exact
# fp64 result to ~1e-6 and the comparison tests the candidate rather than the
# reference's reduced precision.
_CONV_ATOL = {
    torch.float16: 5e-2,
    torch.bfloat16: 5e-2,
    torch.float32: 1e-4,
    torch.float64: 1e-6,
}


def _conv_atol(dtype):
    return _CONV_ATOL.get(dtype, 1e-4)


@pytest.fixture(autouse=True)
def _disable_tf32():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


_CAPABILITY_FLAGS = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.float64: utils.fp64_is_supported,
}

DTYPES = [
    dtype
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    if _CAPABILITY_FLAGS.get(dtype, True)
]

# NCHW shape grid: the first extent is the batch. The 1x1 and 3x3 rows cover
# the pure-GEMM and the deep-K im2col paths; the larger rows give the grid
# enough output pixels to be launch-relevant.
_CONV_SHAPES = [
    ((2, 3, 16, 16), 4),
    ((1, 1, 8, 8), 2),
    ((4, 8, 9, 7), 8),
    ((1, 16, 32, 32), 8),
    ((8, 4, 28, 28), 4),
]

# (kernel, padding, stride) variants, including a non-square kernel, asymmetric
# padding and stride 2.
_KERNEL_PAD_STRIDE = [
    (3, [1, 1], [1, 1]),
    (1, [0, 0], [1, 1]),
    (5, [2, 2], [1, 1]),
    ((3, 5), [1, 2], [1, 1]),
    (3, [1, 1], [2, 2]),
    (4, [0, 0], [3, 1]),
]

# Zero-extent rows. The batch-0 row is accepted by the native operator and the
# candidate returns the matching empty plane; the zero-spatial-extent row is
# rejected by the native kernel ("non-empty input tensor expected") and only the
# candidate's own empty result is checked.
_EMPTY_SHAPES = [
    ((0, 3, 16, 16), 4),
    ((2, 3, 0, 0), 4),
]


def _conv_operands(shape, out_channels, dtype, kernel):
    kH, kW = (kernel, kernel) if isinstance(kernel, int) else kernel
    inp = torch.randn(shape, dtype=dtype, device=gems_device)
    weight = torch.randn(
        (out_channels, shape[1], kH, kW), dtype=dtype, device=gems_device
    )
    bias = torch.randn(out_channels, dtype=dtype, device=gems_device)
    return inp, weight, bias


def _native(inp, weight, kernel, bias, stride, padding):
    # The native schema types kernel_size / stride / padding as SymInt[2], so
    # an int must be normalised to a 2-element list before the reference call.
    def _pair(value):
        if isinstance(value, (tuple, list)):
            return list(value)
        return [int(value), int(value)]

    return torch.ops.aten.thnn_conv2d(
        inp, weight, _pair(kernel), bias, _pair(stride), _pair(padding)
    )


def _reference(inp, weight, kernel, bias, stride, padding):
    return _native(
        utils.to_reference(inp),
        utils.to_reference(weight),
        kernel,
        utils.to_reference(bias),
        stride,
        padding,
    )


@pytest.mark.thnn_conv2d
@pytest.mark.parametrize("shape,out_channels", _CONV_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_thnn_conv2d(shape, out_channels, dtype):
    inp, weight, bias = _conv_operands(shape, out_channels, dtype, 3)

    ref_out = _native(
        utils.to_reference(inp),
        utils.to_reference(weight),
        (3, 3),
        utils.to_reference(bias),
        [1, 1],
        [1, 1],
    )
    res_out = flag_gems.thnn_conv2d(inp, weight, (3, 3), bias, [1, 1], [1, 1])

    assert res_out.shape == ref_out.shape
    utils.gems_assert_close(res_out, ref_out, dtype, atol=_conv_atol(dtype))


@pytest.mark.thnn_conv2d
@pytest.mark.parametrize("kernel,padding,stride", _KERNEL_PAD_STRIDE)
@pytest.mark.parametrize("dtype", DTYPES)
def test_thnn_conv2d_kernel_padding_stride(kernel, padding, stride, dtype):
    kH, kW = (kernel, kernel) if isinstance(kernel, int) else kernel
    # One asymmetric plane per kernel/padding/stride variant keeps every row
    # allocation-friendly while still exercising the mask boundaries.
    shape = (2, 3, 19, 17)
    inp = torch.randn(shape, dtype=dtype, device=gems_device)
    weight = torch.randn((5, 3, kH, kW), dtype=dtype, device=gems_device)
    bias = torch.randn(5, dtype=dtype, device=gems_device)

    ref_out = _native(
        utils.to_reference(inp),
        utils.to_reference(weight),
        kernel,
        utils.to_reference(bias),
        stride,
        padding,
    )
    res_out = flag_gems.thnn_conv2d(inp, weight, kernel, bias, stride, padding)

    assert res_out.shape == ref_out.shape
    utils.gems_assert_close(res_out, ref_out, dtype, atol=_conv_atol(dtype))


@pytest.mark.thnn_conv2d
@pytest.mark.parametrize("dtype", DTYPES)
def test_thnn_conv2d_without_bias(dtype):
    inp, weight, _ = _conv_operands((2, 3, 16, 16), 4, dtype, 3)

    ref_out = _native(
        utils.to_reference(inp),
        utils.to_reference(weight),
        (3, 3),
        None,
        [1, 1],
        [1, 1],
    )
    res_out = flag_gems.thnn_conv2d(inp, weight, (3, 3), None, [1, 1], [1, 1])

    utils.gems_assert_close(res_out, ref_out, dtype, atol=_conv_atol(dtype))


@pytest.mark.thnn_conv2d
@pytest.mark.parametrize("dtype", DTYPES)
def test_thnn_conv2d_default_bias(dtype):
    """Omitting bias uses the schema default (None), not a required argument."""
    inp, weight, _ = _conv_operands((2, 3, 16, 16), 4, dtype, 3)

    # Both sides use the schema defaults: bias None, stride [1, 1],
    # padding [0, 0].
    ref_out = _native(
        utils.to_reference(inp),
        utils.to_reference(weight),
        (3, 3),
        None,
        [1, 1],
        [0, 0],
    )
    res_out = flag_gems.thnn_conv2d(inp, weight, (3, 3))

    utils.gems_assert_close(res_out, ref_out, dtype, atol=_conv_atol(dtype))


@pytest.mark.thnn_conv2d
@pytest.mark.parametrize("dtype", DTYPES)
def test_thnn_conv2d_int_kernel_and_defaults(dtype):
    """An int kernel_size and the default stride/padding follow the schema."""
    inp, weight, bias = _conv_operands((2, 3, 16, 16), 4, dtype, 3)

    ref_out = _native(
        utils.to_reference(inp),
        utils.to_reference(weight),
        3,
        utils.to_reference(bias),
        [1, 1],
        [0, 0],
    )
    # The candidate accepts the bare int kernel_size with the schema defaults
    # (stride 1, padding 0); the reference call above mirrors those defaults.
    res_out = flag_gems.thnn_conv2d(inp, weight, 3, bias)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=_conv_atol(dtype))


@pytest.mark.thnn_conv2d
@pytest.mark.parametrize("shape,out_channels", _EMPTY_SHAPES)
# The zero-extent rows only need one float dtype: the result is empty and the
# tolerance never enters the comparison.
@pytest.mark.parametrize("dtype", [torch.float32])
def test_thnn_conv2d_empty(shape, out_channels, dtype):
    """A zero-extent input or output returns an empty plane without launching."""
    inp, weight, bias = _conv_operands(shape, out_channels, dtype, 3)

    res_out = flag_gems.thnn_conv2d(inp, weight, (3, 3), bias, [1, 1], [1, 1])

    if shape[0] == 0:
        # The native operator accepts an empty batch and returns the same plane.
        ref_out = _native(
            utils.to_reference(inp),
            utils.to_reference(weight),
            (3, 3),
            utils.to_reference(bias),
            [1, 1],
            [1, 1],
        )
        assert res_out.shape == ref_out.shape
        assert res_out.numel() == ref_out.numel()
        return
    # A zero spatial extent is rejected by the native kernel; the candidate must
    # still return the matching empty plane without launching.
    assert res_out.numel() == 0
    assert res_out.dtype == dtype


@pytest.mark.thnn_conv2d
# The non-contiguous row only needs one float dtype: the materialisation path
# is dtype-agnostic and the values are compared with the float32 tolerance.
@pytest.mark.parametrize("dtype", [torch.float32])
def test_thnn_conv2d_noncontiguous_input(dtype):
    """A non-contiguous input is read through its logical indices."""
    base = torch.randn((2, 16, 16, 3), dtype=dtype, device=gems_device)
    inp = base.permute(0, 3, 1, 2)  # NCHW view with transposed H/W strides
    assert not inp.is_contiguous()
    weight = torch.randn((4, 3, 3, 3), dtype=dtype, device=gems_device)
    bias = torch.randn(4, dtype=dtype, device=gems_device)

    ref_out = _native(
        utils.to_reference(inp),
        utils.to_reference(weight),
        (3, 3),
        utils.to_reference(bias),
        [1, 1],
        [1, 1],
    )
    res_out = flag_gems.thnn_conv2d(inp, weight, (3, 3), bias, [1, 1], [1, 1])

    utils.gems_assert_close(res_out, ref_out, dtype, atol=_conv_atol(dtype))


@pytest.mark.thnn_conv2d
@pytest.mark.parametrize("dtype", DTYPES)
def test_thnn_conv2d_does_not_mutate_operands(dtype):
    """The inputs are read-only; the result is a fresh buffer."""
    inp, weight, bias = _conv_operands((2, 3, 16, 16), 4, dtype, 3)
    # The snapshots must live on the oracle device: gems_assert_equal moves the
    # candidate result to the reference side and asserts the reference is
    # already there, which a CUDA-side clone would violate in quick-cpu mode.
    before = (
        utils.to_reference(inp.clone()),
        utils.to_reference(weight.clone()),
        utils.to_reference(bias.clone()),
    )

    res_out = flag_gems.thnn_conv2d(inp, weight, (3, 3), bias, [1, 1], [1, 1])

    utils.gems_assert_equal(inp, before[0], equal_nan=True)
    utils.gems_assert_equal(weight, before[1], equal_nan=True)
    utils.gems_assert_equal(bias, before[2], equal_nan=True)
    assert not res_out._is_view()


@pytest.mark.thnn_conv2d
@pytest.mark.parametrize(
    "bad_dtype", [torch.int32, torch.int8, torch.uint8, torch.bool]
)
def test_thnn_conv2d_rejects_unsupported_dtype(bad_dtype):
    """Integer and bool inputs have no convolution kernel and must raise."""
    inp = torch.ones((2, 3, 8, 8), dtype=bad_dtype, device=gems_device)
    weight = torch.ones((4, 3, 3, 3), dtype=bad_dtype, device=gems_device)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.thnn_conv2d(inp, weight, (3, 3), None, [1, 1], [1, 1])


@pytest.mark.thnn_conv2d
def test_thnn_conv2d_rejects_non_tensor_weight():
    inp = torch.randn((2, 3, 8, 8), dtype=torch.float32, device=gems_device)

    with pytest.raises((RuntimeError, TypeError, AttributeError)):
        flag_gems.thnn_conv2d(inp, [[[0.0]]], (3, 3), None, [1, 1], [1, 1])


@pytest.mark.thnn_conv2d
def test_thnn_conv2d_rejects_rank3_weight():
    """A rank-3 weight has no native kernel ("non-empty 2D or 4D weight tensor
    expected") and the candidate's packed addressing cannot accept it either."""
    inp = torch.randn((2, 3, 8, 8), dtype=torch.float32, device=gems_device)
    weight = torch.randn((4, 3, 3), dtype=torch.float32, device=gems_device)

    with pytest.raises(RuntimeError):
        flag_gems.thnn_conv2d(inp, weight, (3, 3), None, [1, 1], [1, 1])


@pytest.mark.thnn_conv2d
def test_thnn_conv2d_rejects_cin_mismatch():
    """A weight whose C_in differs from the input's is rejected, matching the
    native "Expected input tensor to have channel dimension N at index 1"."""
    inp = torch.randn((2, 3, 8, 8), dtype=torch.float32, device=gems_device)
    weight = torch.randn((4, 5, 3, 3), dtype=torch.float32, device=gems_device)

    with pytest.raises(RuntimeError):
        flag_gems.thnn_conv2d(inp, weight, (3, 3), None, [1, 1], [1, 1])
