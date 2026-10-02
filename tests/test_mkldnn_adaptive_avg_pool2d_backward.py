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
#
# Correctness tests for aten::mkldnn_adaptive_avg_pool2d_backward.
#
# The operator takes no parameters: kernel_size[k - 2] is derived as
# input.size(k) // grad_output.size(k) and every grad_output value is spread
# uniformly over that window. It is dispatched only by the MkldnnCPU key, so
# both operands are host oneDNN tensors (no accelerator device is involved)
# and only the dense materialization of a result is used for value comparison.

import pytest
import torch

import flag_gems

from . import test_utils as tu

_EXPECTED_ERRORS = (RuntimeError, TypeError, ValueError)

# (input shape, grad_output shape). Every row was probed against the native
# MkldnnCPU kernel; the extents fix the average window: 1x1 is the identity,
# 2x2 and 3x3 are square windows, (2, 2, 2, 1)/(2, 2, 1, 1) is the asymmetric
# 2x1 window and (16, 128, 64, 60) is the spec's 4-dim shape.
SMALL_ROWS = [
    ((1, 1, 1, 1), (1, 1, 1, 1)),
    ((1, 1, 4, 4), (1, 1, 2, 2)),
    ((2, 2, 2, 1), (2, 2, 1, 1)),
    ((2, 3, 8, 6), (2, 3, 4, 3)),
    ((1, 2, 9, 9), (1, 2, 3, 3)),
    ((1, 1, 10, 10), (1, 1, 10, 10)),
]

BIG_ROWS = [
    ((4, 8, 32, 32), (4, 8, 16, 16)),
    ((16, 128, 64, 60), (16, 128, 32, 30)),
]

SIZE_CASES = tu.selected_cases(SMALL_ROWS + BIG_ROWS, quick=SMALL_ROWS)

# Native-valid dense families: to_mkldnn() packs float, half and bfloat16 for
# this kernel. See the unsupported-dtype negative for the probed rejections of
# the remaining required dtypes.
SUPPORTED_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# int8/uint8 pack into oneDNN but have no pooling-backward primitive; float64,
# the wide integers, bool and fp8 cannot be packed at all.
UNSUPPORTED_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float64,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]

OUT_ROWS = [((1, 1, 4, 4), (1, 1, 2, 2)), ((2, 3, 8, 6), (2, 3, 4, 3))]
OUT_CASES = OUT_ROWS

# Both rows give grad_output five elements, so the shared special-value payload
# fits them exactly; the second row averages a real 2x1 window instead of
# passing the values through unchanged.
_SPECIAL_ROWS = [((1, 1, 5, 1), (1, 1, 5, 1)), ((1, 1, 10, 1), (1, 1, 5, 1))]
SPECIAL_CASES = tu.selected_cases(
    [
        (input_shape, grad_shape, dtype, scenario)
        for input_shape, grad_shape in _SPECIAL_ROWS
        for dtype, scenario in tu.special_value_cases(SUPPORTED_DTYPES)
    ],
    quick=[],
)

RANK_ROWS = [((1, 2, 4, 4, 3), (1, 2, 2, 2, 2)), ((4, 4), (2, 2))]
INDIVISIBLE_ROWS = [((1, 2, 5, 5), (1, 2, 2, 2)), ((1, 1, 4, 6), (1, 1, 4, 4))]
NEGATIVE_DTYPES = [torch.float32, torch.bfloat16]


def _operands(dtype, input_shape, grad_shape, value_range):
    """Host operand pairs: equal values, independent candidate/reference storage.

    MkldnnCPU tensors exist only on the host, so the values are built on the
    host and packed twice; tu.to_reference gives the reference its own buffer.
    """
    dense_in = tu.make_input(dtype, input_shape, value_range).cpu()
    dense_grad = tu.make_input(dtype, grad_shape, value_range).cpu()
    return (
        dense_in.to_mkldnn(),
        dense_grad.to_mkldnn(),
        tu.to_reference(dense_in).to_mkldnn(),
        tu.to_reference(dense_grad).to_mkldnn(),
    )


def _special_dense(dtype, scenario, shape):
    """Tile the shared special-value payload to the workload's element count."""
    payload = tu.make_special_input(dtype, scenario).cpu()
    numel = 1
    for extent in shape:
        numel *= extent
    repeats = -(-numel // payload.numel())
    return payload.repeat(repeats)[:numel].reshape(shape)


def _assert_mkldnn_result(res_out, ref_out):
    # torch.testing.assert_close rejects the torch._mkldnn layout, so the real
    # outputs are checked for layout/shape/dtype first and only their dense
    # materialization is handed to the shared comparison helper.
    assert res_out.layout == torch._mkldnn
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    tu.assert_result_close(res_out.to_dense(), ref_out.to_dense())


@pytest.mark.mkldnn_adaptive_avg_pool2d_backward
@pytest.mark.parametrize("input_shape,grad_shape", SIZE_CASES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_mkldnn_adaptive_avg_pool2d_backward(
    input_shape, grad_shape, dtype, value_range
):
    inp, grad, ref_inp, ref_grad = _operands(
        dtype, input_shape, grad_shape, value_range
    )
    grad_snapshot, inp_snapshot = grad.to_dense(), inp.to_dense()

    ref_out = torch.ops.aten.mkldnn_adaptive_avg_pool2d_backward(ref_grad, ref_inp)
    res_out = flag_gems.mkldnn_adaptive_avg_pool2d_backward(grad, inp)

    _assert_mkldnn_result(res_out, ref_out)
    # The kernel must not clobber its operands.
    tu.assert_result_equal(grad.to_dense(), grad_snapshot)
    tu.assert_result_equal(inp.to_dense(), inp_snapshot)


@pytest.mark.mkldnn_adaptive_avg_pool2d_backward
@pytest.mark.parametrize("input_shape,grad_shape", OUT_CASES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_mkldnn_adaptive_avg_pool2d_backward_out(input_shape, grad_shape, dtype):
    inp, grad, ref_inp, ref_grad = _operands(
        dtype, input_shape, grad_shape, ["-1", "1"]
    )
    ref_out = torch.ops.aten.mkldnn_adaptive_avg_pool2d_backward.out(
        ref_grad,
        ref_inp,
        out=torch.zeros(input_shape, dtype=dtype, device="cpu").to_mkldnn(),
    )
    # The oneDNN overload returns the very buffer passed as out=, so the buffer
    # is pre-filled with defined values before the call.
    res_buf = torch.zeros(input_shape, dtype=dtype, device="cpu").to_mkldnn()
    res_out = flag_gems.mkldnn_adaptive_avg_pool2d_backward(grad, inp, out=res_buf)

    _assert_mkldnn_result(res_out, ref_out)
    assert res_out is res_buf


@pytest.mark.mkldnn_adaptive_avg_pool2d_backward
@pytest.mark.parametrize("input_shape,grad_shape,dtype,scenario", SPECIAL_CASES)
def test_mkldnn_adaptive_avg_pool2d_backward_special_values(
    input_shape, grad_shape, dtype, scenario
):
    dense_in = _special_dense(dtype, scenario, input_shape)
    dense_grad = _special_dense(dtype, scenario, grad_shape)

    ref_out = torch.ops.aten.mkldnn_adaptive_avg_pool2d_backward(
        tu.to_reference(dense_grad).to_mkldnn(), tu.to_reference(dense_in).to_mkldnn()
    )
    res_out = flag_gems.mkldnn_adaptive_avg_pool2d_backward(
        dense_grad.to_mkldnn(), dense_in.to_mkldnn()
    )

    _assert_mkldnn_result(res_out, ref_out)


@pytest.mark.mkldnn_adaptive_avg_pool2d_backward
@pytest.mark.parametrize("input_shape,grad_shape", RANK_ROWS)
@pytest.mark.parametrize("dtype", NEGATIVE_DTYPES)
def test_mkldnn_adaptive_avg_pool2d_backward_rejects_non_4d(
    input_shape, grad_shape, dtype
):
    inp = tu.make_input(dtype, input_shape, ["-1", "1"]).cpu().to_mkldnn()
    grad = tu.make_input(dtype, grad_shape, ["-1", "1"]).cpu().to_mkldnn()

    # Native: "mkldnn_adaptive_avg_pool2d: Input is expected a 4D tensor".
    with pytest.raises(_EXPECTED_ERRORS):
        flag_gems.mkldnn_adaptive_avg_pool2d_backward(grad, inp)


@pytest.mark.mkldnn_adaptive_avg_pool2d_backward
@pytest.mark.parametrize("input_shape,grad_shape", INDIVISIBLE_ROWS)
@pytest.mark.parametrize("dtype", NEGATIVE_DTYPES)
def test_mkldnn_adaptive_avg_pool2d_backward_rejects_indivisible_extent(
    input_shape, grad_shape, dtype
):
    inp, grad, _, _ = _operands(dtype, input_shape, grad_shape, ["-1", "1"])

    # Native: "input size is not divisible by the output size is not supported
    # yet".
    with pytest.raises(_EXPECTED_ERRORS):
        flag_gems.mkldnn_adaptive_avg_pool2d_backward(grad, inp)


@pytest.mark.mkldnn_adaptive_avg_pool2d_backward
@pytest.mark.parametrize("dtype", NEGATIVE_DTYPES)
def test_mkldnn_adaptive_avg_pool2d_backward_rejects_zero_output_size(dtype):
    # A zero grad_output extent is rejected before the divisibility test, so
    # empty operands are enough; their contents are never read.
    inp = torch.empty((1, 2, 4, 4), dtype=dtype, device="cpu").to_mkldnn()
    grad = torch.empty((1, 2, 0, 2), dtype=dtype, device="cpu").to_mkldnn()

    # Native: "output size can not be zero".
    with pytest.raises(_EXPECTED_ERRORS):
        flag_gems.mkldnn_adaptive_avg_pool2d_backward(grad, inp)


@pytest.mark.mkldnn_adaptive_avg_pool2d_backward
@pytest.mark.parametrize("dtype", NEGATIVE_DTYPES)
def test_mkldnn_adaptive_avg_pool2d_backward_requires_mkldnn_operands(dtype):
    grad = tu.make_input(dtype, (1, 1, 2, 2), ["-1", "1"]).cpu()
    inp = tu.make_input(dtype, (1, 1, 4, 4), ["-1", "1"]).cpu()

    # Strided CPU operands have no kernel: dispatch is MkldnnCPU only and
    # raises NotImplementedError naming the available backends.
    with pytest.raises(_EXPECTED_ERRORS):
        flag_gems.mkldnn_adaptive_avg_pool2d_backward(grad, inp)


@pytest.mark.mkldnn_adaptive_avg_pool2d_backward
@pytest.mark.parametrize("dtype", NEGATIVE_DTYPES)
def test_mkldnn_adaptive_avg_pool2d_backward_rejects_dense_out(dtype):
    inp, grad, _, _ = _operands(dtype, (1, 1, 4, 4), (1, 1, 2, 2), ["-1", "1"])
    dense_out = torch.zeros((1, 1, 4, 4), dtype=dtype, device="cpu")

    # Native: "copy_mkldnn_: between mkldnn layout and dense Tensors is not
    # implemented".
    with pytest.raises(_EXPECTED_ERRORS):
        flag_gems.mkldnn_adaptive_avg_pool2d_backward(grad, inp, out=dense_out)


@pytest.mark.mkldnn_adaptive_avg_pool2d_backward
@pytest.mark.parametrize("dtype", UNSUPPORTED_DTYPES)
def test_mkldnn_adaptive_avg_pool2d_backward_rejects_unsupported_dtype(dtype):
    # int8/uint8 pack but have no pooling-backward primitive ("could not create
    # a descriptor for a pooling backward propagation primitive"); float64, the
    # wide integers, bool and fp8 cannot be packed by to_mkldnn() at all.
    dense_grad = torch.randn((1, 2, 2, 2)).to(dtype)
    dense_in = torch.randn((1, 2, 4, 4)).to(dtype)

    # Build native-representable integer operands before the exception context.
    # Other dtypes cannot form an opaque operand; their dense call is rejected.
    grad = dense_grad.to_mkldnn() if dtype in (torch.int8, torch.uint8) else dense_grad
    inp = dense_in.to_mkldnn() if dtype in (torch.int8, torch.uint8) else dense_in
    with pytest.raises(_EXPECTED_ERRORS):
        flag_gems.mkldnn_adaptive_avg_pool2d_backward(grad, inp)


@pytest.mark.mkldnn_adaptive_avg_pool2d_backward
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_mkldnn_adaptive_avg_pool2d_backward_has_no_gradient(dtype):
    dense_leaf = torch.randn((1, 2, 4, 4), dtype=dtype).requires_grad_()
    ref_leaf = dense_leaf.detach().clone().requires_grad_()
    grad = torch.randn((1, 2, 2, 2), dtype=dtype).to_mkldnn()
    ref_out = torch.ops.aten.mkldnn_adaptive_avg_pool2d_backward(
        grad.clone(), ref_leaf.to_mkldnn()
    )
    out = flag_gems.mkldnn_adaptive_avg_pool2d_backward(grad, dense_leaf.to_mkldnn())
    _assert_mkldnn_result(out, ref_out)
    upstream = torch.randn_like(dense_leaf)
    error = (
        "derivative for aten::mkldnn_adaptive_avg_pool2d_backward is not implemented"
    )
    with pytest.raises(RuntimeError, match=error):
        torch.autograd.grad(ref_out.to_dense(), ref_leaf, grad_outputs=upstream)
    with pytest.raises(RuntimeError, match=error):
        torch.autograd.grad(out.to_dense(), dense_leaf, grad_outputs=upstream)
