# Copyright 2026, The FlagGems Authors.
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

"""Correctness tests for ``aten::mkldnn_max_pool2d``.

oneDNN-only operator: the real operand is a CPU oneDNN tensor, so the reference
and the candidate both receive ``to_mkldnn()`` operands built on the CPU.
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu

pytestmark = pytest.mark.mkldnn_max_pool2d

# (kernel_size, stride, padding, dilation, ceil_mode)
_KERNEL = ([2, 2], [2, 2], [0, 0], [1, 1], False)
_ASYMMETRIC = ([3, 5], [2, 1], [1, 2], [1, 1], False)
_CEIL = ([3, 3], [2, 2], [1, 1], [1, 1], True)
# An empty stride takes the schema default ``stride = kernel_size``.
_EMPTY_STRIDE = ([3, 3], [], [0, 0], [1, 1], False)
# Scalar ints are accepted for the int[2] parameters.
_SCALAR = (2, 2, 1, 1, True)
_KERNEL_ONE = ([1, 1], [1, 1], [0, 0], [1, 1], False)
_KERNEL_LARGER = ([9, 9], [3, 3], [2, 2], [1, 1], False)
_DILATION = ([2, 2], [1, 1], [0, 0], [2, 2], False)

_FORMS = [_KERNEL, _ASYMMETRIC]
_PARAM_FORMS = [_CEIL, _EMPTY_STRIDE, _SCALAR, _KERNEL_ONE, _KERNEL_LARGER]

_FULL_SHAPES = [(16, 128, 64, 60), (2, 3, 19, 7), (1, 2, 9, 13)]
# Empty extents are legal oneDNN operands. (2, 3, 0, 0) has no creatable
# descriptor for the asymmetric or large-kernel rows (oneDNN reports "could not
# create a descriptor for a pooling forward propagation primitive" /
# "could not construct a memory descriptor using a format tag"), so the empty
# rows carry only the two forms that are valid for them.
_EMPTY_SHAPES = [(2, 3, 0, 0), (0, 3, 8, 8), (2, 0, 8, 8)]
_EMPTY_FORMS = [_KERNEL, _CEIL]

_VALUE_ROWS = tu.selected_cases(
    [(shape, params) for shape in _FULL_SHAPES for params in _FORMS]
    + [(shape, params) for shape in _EMPTY_SHAPES for params in _EMPTY_FORMS],
    quick=[(shape, params) for shape in _FULL_SHAPES[1:] for params in _FORMS]
    + [(shape, params) for shape in _EMPTY_SHAPES for params in _EMPTY_FORMS],
)

# oneDNN holds only these element types: int32/int64/float64/bool/fp8 operands
# cannot be built ("dense_to_mkldnn expects float, bfloat16, half, uint8, int8").
_DTYPES = [torch.float32, torch.bfloat16, torch.float16, torch.int8, torch.uint8]
_FLOAT_DTYPES = [torch.float32, torch.bfloat16, torch.float16]

# Quick keeps every supported dtype and call form and only trims the range grid.
_RANGES = tu.selected_cases(tu.selected_ranges(), quick=[["-1", "1"]])

_INVALID_CASES = [
    # pooling requires rank-4 (N, C, H, W) operands
    ((2, 3, 8), _KERNEL),
    # dilation must stay 1
    ((2, 3, 8, 8), _DILATION),
    # negative padding
    ((2, 3, 8, 8), ([2, 2], [1, 1], [-1, -1], [1, 1], False)),
]


def _mkldnn(tensor):
    """Real operand type: oneDNN pooling accepts nothing else."""
    return tensor.detach().cpu().contiguous().to_mkldnn()


def _assert_operand_metadata(res_out, ref_out, operand):
    # Checked before any to_dense() observation, so a dense stand-in cannot pass
    # for the real oneDNN result.
    assert res_out.layout == ref_out.layout == operand.layout == torch._mkldnn
    assert res_out.dtype == ref_out.dtype
    assert res_out.device == operand.device


@pytest.mark.parametrize("shape,params", _VALUE_ROWS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("value_range", _RANGES)
def test_mkldnn_max_pool2d(shape, dtype, value_range, params):
    kernel_size, stride, padding, dilation, ceil_mode = params
    inp = tu.make_input(dtype, shape, value_range)

    ref_out = torch.ops.aten.mkldnn_max_pool2d(
        _mkldnn(inp), kernel_size, stride, padding, dilation, ceil_mode
    )
    # Independent candidate operand plus a read-only snapshot of its values.
    cand_in = _mkldnn(inp)
    operand_before = cand_in.to_dense().clone()
    res_out = flag_gems.mkldnn_max_pool2d(
        cand_in, kernel_size, stride, padding, dilation, ceil_mode
    )

    _assert_operand_metadata(res_out, ref_out, cand_in)
    tu.assert_result_equal(res_out.to_dense(), ref_out.to_dense())
    # Not in place: the operand keeps its values.
    tu.assert_result_equal(cand_in.to_dense(), operand_before)


@pytest.mark.parametrize("params", _PARAM_FORMS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_mkldnn_max_pool2d_param_forms(dtype, params):
    kernel_size, stride, padding, dilation, ceil_mode = params
    inp = tu.make_input(dtype, (1, 2, 13, 11), ["-1", "1"])

    ref_out = torch.ops.aten.mkldnn_max_pool2d(
        _mkldnn(inp), kernel_size, stride, padding, dilation, ceil_mode
    )
    cand_in = _mkldnn(inp)
    res_out = flag_gems.mkldnn_max_pool2d(
        cand_in, kernel_size, stride, padding, dilation, ceil_mode
    )

    _assert_operand_metadata(res_out, ref_out, cand_in)
    tu.assert_result_equal(res_out.to_dense(), ref_out.to_dense())


@pytest.mark.parametrize("kernel_size", [[2, 2], [3, 3]])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_mkldnn_max_pool2d_default_args(dtype, kernel_size):
    # stride/padding/dilation/ceil_mode omitted: stride defaults to kernel_size.
    inp = tu.make_input(dtype, (1, 2, 13, 11), ["-1", "1"])

    ref_out = torch.ops.aten.mkldnn_max_pool2d(_mkldnn(inp), kernel_size)
    cand_in = _mkldnn(inp)
    res_out = flag_gems.mkldnn_max_pool2d(cand_in, kernel_size)

    _assert_operand_metadata(res_out, ref_out, cand_in)
    tu.assert_result_equal(res_out.to_dense(), ref_out.to_dense())


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("value_range", _RANGES)
def test_mkldnn_max_pool2d_out(dtype, value_range):
    kernel_size, stride, padding, dilation, ceil_mode = _ASYMMETRIC
    inp = tu.make_input(dtype, (2, 3, 19, 7), value_range)

    ref_in = _mkldnn(inp)
    out_shape = list(
        torch.ops.aten.mkldnn_max_pool2d(
            ref_in, kernel_size, stride, padding, dilation, ceil_mode
        ).shape
    )
    ref_buf = torch.zeros(out_shape, dtype=dtype).to_mkldnn()
    ref_res = torch.ops.aten.mkldnn_max_pool2d.out(
        ref_in, kernel_size, stride, padding, dilation, ceil_mode, out=ref_buf
    )

    cand_in = _mkldnn(inp)
    cand_buf = torch.zeros(out_shape, dtype=dtype).to_mkldnn()
    res_out = flag_gems.mkldnn_max_pool2d(
        cand_in, kernel_size, stride, padding, dilation, ceil_mode, out=cand_buf
    )
    # Identity, not merely equal pointers: the out overload reuses the buffer.
    assert res_out is cand_buf
    _assert_operand_metadata(res_out, ref_res, cand_in)
    tu.assert_result_equal(res_out.to_dense(), ref_res.to_dense())


_SPECIAL_ROWS = tu.selected_cases(tu.special_value_cases(_FLOAT_DTYPES), quick=[])


@pytest.mark.parametrize("dtype,scenario", _SPECIAL_ROWS)
def test_mkldnn_max_pool2d_special_values(dtype, scenario):
    # oneDNN propagates +/-inf (inf wins over any finite value) and ignores NaN
    # inside a window, so the shared payloads compare with zero tolerance.
    inp = tu.make_special_input(dtype, scenario).repeat(4).reshape(1, 1, 4, 5)

    ref_out = torch.ops.aten.mkldnn_max_pool2d(_mkldnn(inp), *_KERNEL)
    cand_in = _mkldnn(inp)
    res_out = flag_gems.mkldnn_max_pool2d(cand_in, *_KERNEL)

    _assert_operand_metadata(res_out, ref_out, cand_in)
    tu.assert_result_equal(res_out.to_dense(), ref_out.to_dense())


_BACKWARD_SHAPES = tu.selected_cases([(2, 3, 19, 7), (1, 2, 13, 11)], quick=[])


@pytest.mark.parametrize("shape", _BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_mkldnn_max_pool2d_backward(dtype, shape):
    # to_mkldnn() exists for CPU operands only, so the leaves and the resulting
    # gradient live on the CPU. Backward is supported for every floated dtype
    # the operator accepts, so no missing-derivative negative is needed here.
    inp = tu.make_input(dtype, shape, ["-1", "1"])

    ref_leaf = tu.to_reference(inp).cpu().requires_grad_(True)
    ref_out = torch.ops.aten.mkldnn_max_pool2d(ref_leaf.to_mkldnn(), *_KERNEL)
    # Genuine nonuniform upstream gradient, shaped like the pooled output.
    # Pooling backward requires the incoming gradient in the operator's own
    # layout (Mkldnn, not Strided), so the upstream is built as an operand.
    upstream = tu.make_input(dtype, tuple(ref_out.shape), ["-1", "1"]).cpu().to_mkldnn()
    (ref_grad,) = torch.autograd.grad(ref_out, ref_leaf, grad_outputs=upstream)

    cand_leaf = tu.to_reference(inp).cpu().requires_grad_(True)
    res_out = flag_gems.mkldnn_max_pool2d(cand_leaf.to_mkldnn(), *_KERNEL)
    assert res_out.layout == torch._mkldnn
    tu.assert_result_equal(res_out.to_dense(), ref_out.to_dense())
    (res_grad,) = torch.autograd.grad(res_out, cand_leaf, grad_outputs=upstream)

    assert res_grad.device == cand_leaf.device
    # These stride-two windows do not overlap; gradients are copied exactly.
    tu.assert_result_equal(res_grad, ref_grad)


@pytest.mark.parametrize("shape,params", _INVALID_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_mkldnn_max_pool2d_invalid_args(shape, dtype, params):
    kernel_size, stride, padding, dilation, ceil_mode = params
    inp = torch.zeros(shape, dtype=dtype).to_mkldnn()

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.mkldnn_max_pool2d(
            inp, kernel_size, stride, padding, dilation, ceil_mode
        )


def test_mkldnn_max_pool2d_out_rejects_dense_buffer():
    kernel_size, stride, padding, dilation, ceil_mode = _ASYMMETRIC
    inp = tu.make_input(torch.float32, (2, 3, 19, 7), ["-1", "1"])
    out_shape = list(
        torch.ops.aten.mkldnn_max_pool2d(
            _mkldnn(inp), kernel_size, stride, padding, dilation, ceil_mode
        ).shape
    )

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.mkldnn_max_pool2d(
            _mkldnn(inp),
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            out=torch.zeros(out_shape, dtype=torch.float32),
        )
