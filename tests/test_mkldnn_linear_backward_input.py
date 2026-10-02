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

import math

import pytest
import torch

import flag_gems

from . import test_utils as tu

# The native kernel (ATen/native/mkldnn/Linear.cpp) takes an mkldnn grad_output
# and a dense CPU float32 weight, so the reference and the injected candidate
# receive exactly those CPU argument types. mkldnn storage only carries
# float32/float16/bfloat16 here: torch.to_mkldnn rejects
# float8_e4m3fn/float8_e5m2/int32/int64/float64/bool ("dense_to_mkldnn expects
# float, bfloat16, half, uint8, int8 tensor input"), and the int8/uint8 tensors
# it does accept fail the ideep primitive descriptor -- those two are negative
# cases below instead of supported operand dtypes.
GRAD_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# TORCH_CHECK rejections are RuntimeError; a form with no registered kernel
# (dense grad_output, dense out buffer) raises NotImplementedError.
_INVALID_INPUT_ERRORS = (RuntimeError, NotImplementedError)


# No gradient workload: the native op registers no derivative ("derivative for
# aten::mkldnn_linear_backward_input is not implemented"), and the operator is
# itself the backward pass of a linear layer.


def _shape_param(label, grad_shape, weight_shape, input_size):
    return pytest.param((grad_shape, weight_shape, input_size), id=label)


# (grad_output shape, weight shape, input_size). Two spec shapes are not
# representable here: torch.randn(()).to_mkldnn() already fails with "could not
# create a primitive descriptor for the reorder primitive", and a 1-D
# grad_output fails the ideep primitive descriptor (kept as a negative case).
_SHAPE_CASES = [
    _shape_param("quick-2x19x7", (38, 8), (8, 7), [2, 19, 7]),
    _shape_param("spec-256", (256, 4), (4, 8), [256, 8]),
    _shape_param("spec-1024x1024", (1024, 64), (64, 1024), [1024, 1024]),
    _shape_param("spec-20x320x15", (6400, 32), (32, 15), [20, 320, 15]),
    _shape_param("spec-16x128x64x60", (131072, 16), (16, 60), [16, 128, 64, 60]),
    _shape_param("spec-16x7x57x32x29", (204288, 16), (16, 29), [16, 7, 57, 32, 29]),
    # a grad_output with more than two dims is flattened to (M, N) first
    _shape_param("grad-rank4", (4, 64, 8), (8, 16), [256, 16]),
    # M == 1 and K == 1 boundaries
    _shape_param("single-row", (1, 8), (8, 4), [1, 4]),
    _shape_param("single-feature", (64, 16), (16, 1), [64, 1]),
    # an input_size of length <= 2 is not used by the native kernel, so the
    # result keeps its (M, K) shape even when input_size disagrees
    _shape_param("short-input-size-ignored", (256, 4), (4, 8), [999, 7]),
    _shape_param("empty-input-size", (256, 4), (4, 8), []),
    _shape_param("one-element-input-size", (256, 4), (4, 8), [7]),
]

# Quick keeps every supported dtype and every cheap boundary workload (rank-4 and
# single-row/single-feature grads, all small input_size forms); it only drops the
# large spec shapes. The default case list is the full set above, so it always
# contains the quick rows.
_QUICK_SHAPE_IDS = {
    "quick-2x19x7",
    "grad-rank4",
    "single-row",
    "single-feature",
    "short-input-size-ignored",
    "empty-input-size",
    "one-element-input-size",
}

SHAPE_CASES = tu.selected_cases(
    _SHAPE_CASES,
    quick=[case for case in _SHAPE_CASES if case.id in _QUICK_SHAPE_IDS],
)

# Both out rows are small enough to stay in quick as well.
_OUT_SHAPE_CASES = [
    _shape_param("out-2d", (256, 4), (4, 8), [256, 8]),
    _shape_param("out-rank2", (16, 32), (32, 64), [16, 64]),
]

# Positive nan/inf scenarios are default-only.
SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(GRAD_DTYPES), quick=[])

_NEGATIVE_GRAD_DTYPES = [torch.int8, torch.uint8]


def _mkldnn_grad(dtype, shape, value_range):
    # mkldnn storage is CPU-only, so the operand is generated on
    # flag_gems.device and moved to CPU before the layout conversion.
    return tu.make_input(dtype, shape, value_range).cpu().to_mkldnn()


def _dense_weight(shape, value_range):
    # The native kernel requires float32 weights, independently of grad dtype.
    return tu.make_input(torch.float32, shape, value_range).cpu()


def _result_shape(grad_shape, weight_shape, input_size):
    # The mkldnn result is reshaped to input_size only when it holds more than
    # two entries; otherwise it stays (M, K).
    if len(input_size) > 2:
        return tuple(input_size)
    return (math.prod(grad_shape[:-1]), weight_shape[1])


def _assert_backward_input(res, ref, grad, dtype, out_shape):
    # Compare the materialised dense values of the tested dtype, so the tested
    # dtype's own tolerance applies, and check the mkldnn layout explicitly
    # because the shared value assertions cannot see it.
    assert res.layout == torch._mkldnn
    assert res.dtype == dtype
    assert res.shape == out_shape
    assert res.device == grad.device
    tu.assert_result_close(res.to_dense(), ref.to_dense())


@pytest.mark.mkldnn_linear_backward_input
@pytest.mark.parametrize("shape_case", SHAPE_CASES)
@pytest.mark.parametrize("dtype", GRAD_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_mkldnn_linear_backward_input(shape_case, dtype, value_range):
    grad_shape, weight_shape, input_size = shape_case
    grad = _mkldnn_grad(dtype, grad_shape, value_range)
    weight = _dense_weight(weight_shape, value_range)

    ref_grad, ref_weight = tu.to_reference(grad), tu.to_reference(weight)
    ref_out = torch.ops.aten.mkldnn_linear_backward_input(
        input_size, ref_grad, ref_weight
    )
    res_out = flag_gems.mkldnn_linear_backward_input(input_size, grad, weight)

    tu.assert_result_equal(grad.to_dense(), ref_grad.to_dense())
    tu.assert_result_equal(weight, ref_weight)
    _assert_backward_input(
        res_out,
        ref_out,
        grad,
        dtype,
        _result_shape(grad_shape, weight_shape, input_size),
    )


@pytest.mark.mkldnn_linear_backward_input
@pytest.mark.parametrize("shape_case", _OUT_SHAPE_CASES)
@pytest.mark.parametrize("dtype", GRAD_DTYPES)
def test_mkldnn_linear_backward_input_out(shape_case, dtype):
    grad_shape, weight_shape, input_size = shape_case
    # the shared range framework takes bound symbols, not raw numbers
    grad = _mkldnn_grad(dtype, grad_shape, ("-1", "1"))
    weight = _dense_weight(weight_shape, ("-1", "1"))
    out_shape = _result_shape(grad_shape, weight_shape, input_size)

    ref_buf = torch.empty(out_shape, dtype=dtype).to_mkldnn()
    ref_out = torch.ops.aten.mkldnn_linear_backward_input.out(
        input_size, tu.to_reference(grad), tu.to_reference(weight), out=ref_buf
    )
    res_buf = torch.empty(out_shape, dtype=dtype).to_mkldnn()
    res_out = flag_gems.mkldnn_linear_backward_input(
        input_size, grad, weight, out=res_buf
    )

    assert res_out is res_buf
    _assert_backward_input(res_out, ref_out, grad, dtype, out_shape)


@pytest.mark.mkldnn_linear_backward_input
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test_mkldnn_linear_backward_input_special_values(dtype, scenario):
    # one payload value per grad_output row, so every row keeps its own value
    grad = tu.make_special_input(dtype, scenario).cpu().reshape(-1, 1).to_mkldnn()
    # an all-ones dense float32 weight reproduces the row values unchanged
    weight = torch.ones(1, 4, dtype=torch.float32)

    ref_out = torch.ops.aten.mkldnn_linear_backward_input(
        [5, 4], tu.to_reference(grad), tu.to_reference(weight)
    )
    res_out = flag_gems.mkldnn_linear_backward_input([5, 4], grad, weight)

    _assert_backward_input(res_out, ref_out, grad, dtype, (5, 4))


@pytest.mark.mkldnn_linear_backward_input
def test_mkldnn_linear_backward_input_rejects_dense_grad_output():
    # only a MkldnnCPU kernel exists, so a dense grad_output finds no kernel
    grad = torch.randn(256, 4)
    weight = torch.randn(4, 8, dtype=torch.float32)
    with pytest.raises(_INVALID_INPUT_ERRORS):
        flag_gems.mkldnn_linear_backward_input([256, 8], grad, weight)


@pytest.mark.mkldnn_linear_backward_input
def test_mkldnn_linear_backward_input_rejects_mkldnn_weight():
    grad = torch.randn(256, 4).to_mkldnn()
    weight = torch.randn(4, 8).to_mkldnn()
    with pytest.raises(_INVALID_INPUT_ERRORS):
        flag_gems.mkldnn_linear_backward_input([256, 8], grad, weight)


@pytest.mark.mkldnn_linear_backward_input
def test_mkldnn_linear_backward_input_rejects_non_float32_weight():
    grad = torch.randn(256, 4).to_mkldnn()
    weight = torch.randn(4, 8, dtype=torch.float16)
    with pytest.raises(_INVALID_INPUT_ERRORS):
        flag_gems.mkldnn_linear_backward_input([256, 8], grad, weight)


@pytest.mark.mkldnn_linear_backward_input
@pytest.mark.parametrize("dtype", _NEGATIVE_GRAD_DTYPES)
def test_mkldnn_linear_backward_input_rejects_unsupported_grad_dtype(dtype):
    # int8/uint8 are the remaining required dtypes torch.to_mkldnn accepts; the
    # ideep inner-product primitive descriptor rejects both of them
    grad = torch.randn(256, 4).to(dtype).to_mkldnn()
    weight = torch.randn(4, 8, dtype=torch.float32)
    with pytest.raises(_INVALID_INPUT_ERRORS):
        flag_gems.mkldnn_linear_backward_input([256, 8], grad, weight)


@pytest.mark.mkldnn_linear_backward_input
def test_mkldnn_linear_backward_input_rejects_1d_grad_output():
    grad = torch.randn(4).to_mkldnn()
    weight = torch.randn(4, 8, dtype=torch.float32)
    with pytest.raises(_INVALID_INPUT_ERRORS):
        flag_gems.mkldnn_linear_backward_input([256, 8], grad, weight)


def _bad_input_size(kind):
    if kind == "float":
        return 1.5
    if kind == "tensor":
        return torch.tensor([256, 8])
    # a length >= 3 target is validated by the result reshape
    return [1, 2, 4]


@pytest.mark.mkldnn_linear_backward_input
@pytest.mark.parametrize("kind", ["float", "tensor", "numel-mismatch"])
def test_mkldnn_linear_backward_input_rejects_invalid_input_size(kind):
    grad = torch.randn(256, 4).to_mkldnn()
    weight = torch.randn(4, 8, dtype=torch.float32)
    with pytest.raises(_INVALID_INPUT_ERRORS):
        flag_gems.mkldnn_linear_backward_input(_bad_input_size(kind), grad, weight)


@pytest.mark.mkldnn_linear_backward_input
def test_mkldnn_linear_backward_input_out_rejects_dense_buffer():
    grad = torch.randn(256, 4).to_mkldnn()
    weight = torch.randn(4, 8, dtype=torch.float32)
    out = torch.empty(256, 8)
    with pytest.raises(_INVALID_INPUT_ERRORS):
        flag_gems.mkldnn_linear_backward_input([256, 8], grad, weight, out=out)


@pytest.mark.mkldnn_linear_backward_input
def test_mkldnn_linear_backward_input_out_rejects_wrong_shape_buffer():
    grad = torch.randn(256, 4).to_mkldnn()
    weight = torch.randn(4, 8, dtype=torch.float32)
    out = torch.empty(256, 9).to_mkldnn()
    with pytest.raises(_INVALID_INPUT_ERRORS):
        flag_gems.mkldnn_linear_backward_input([256, 8], grad, weight, out=out)
