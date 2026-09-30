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

from . import test_utils as tu

# aten::_mkldnn_transpose swaps two dimensions of an opaque oneDNN (mkldnn)
# operand. That layout is CPU-only, and dense_to_mkldnn builds it only from
# float32/float16/bfloat16/uint8/int8 ("dense_to_mkldnn expects float, bfloat16,
# half, uint8, int8 tensor input"), so fp8/int32/int64/fp64/bool can never be an
# operand of this operator. The candidate receives the same opaque CPU operand
# as the reference and must answer with that layout, not with a dense equivalent.
_MKLDNN_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float32,
    torch.bfloat16,
    torch.float16,
]

_QUICK_ROW = ((2, 19, 7), 0, 1)

# The spec's `()` shape is absent: dense_to_mkldnn raises "could not create a
# primitive descriptor for the reorder primitive" for a 0-dim operand. A rank-1
# operand has no second dimension, so it only takes the dim0 == dim1 relayout.
_VALUE_ROWS = tu.selected_cases(
    [
        ((1,), 0, 0),
        ((256,), 0, 0),
        ((1024, 1024), 0, 1),
        ((20, 320, 15), 0, 2),
        ((16, 128, 64, 60), 0, 3),
        ((16, 7, 57, 32, 29), 0, 4),
        ((2, 19, 7), 0, 1),
    ],
    quick=[((2, 19, 7), 0, 1), ((1,), 0, 0), ((256,), 0, 0)],
)

# Every dimension form the schema accepts on a valid operand: positive,
# reversed, negative, the dim0 == dim1 relayout, rank 1 and empty operands.
# An empty operand keeps its zero in a non-final dimension: converting a 2-D
# dense tensor whose last dim is 0 (e.g. (3, 0)) aborts the process inside the
# oneDNN dense->mkldnn reorder.
_DIM_ROWS = tu.selected_cases(
    [
        ((1024, 1024), 0, 1),
        ((1024, 1024), 1, 0),
        ((1024, 1024), -1, -2),
        ((1024, 1024), 0, 0),
        ((20, 320, 15), 0, 2),
        ((20, 320, 15), -1, -3),
        ((20, 320, 15), 0, -1),
        ((20, 320, 15), 1, 1),
        ((256,), 0, 0),
        ((0, 3, 4), 0, 2),
        ((0, 3), 0, 1),
        ((0,), 0, 0),
        ((2, 19, 7), 0, 1),
        ((2, 19, 7), 2, 0),
        ((2, 19, 7), -1, -3),
        ((2, 19, 7), 1, 1),
        ((19,), 0, 0),
    ],
    quick=[
        ((2, 19, 7), 0, 1),
        ((2, 19, 7), 2, 0),
        ((2, 19, 7), -1, -3),
        ((2, 19, 7), 1, 1),
        ((19,), 0, 0),
        ((0, 3, 4), 0, 2),
        ((256,), 0, 0),
        ((0, 3), 0, 1),
        ((0,), 0, 0),
    ],
)

_OUT_ROWS = tu.selected_cases(
    [
        ((1024, 1024), 0, 1),
        ((20, 320, 15), 0, 2),
        ((16, 7, 57, 32, 29), -1, -3),
        ((256,), 0, 0),
        ((2, 19, 7), 0, 1),
    ],
    quick=[((2, 19, 7), 0, 1), ((256,), 0, 0)],
)

_OOB_DIM_ROWS = [
    ((2, 3, 4), 0, 3),
    ((2, 3, 4), -4, 0),
    ((20, 320, 15), 3, 0),
    ((1024, 1024), 2, 1),
    ((256,), 0, 1),
    ((256,), -2, 0),
]

_NEG_DTYPES = [torch.float32, torch.bfloat16]


def _mkldnn_operand(dense):
    """Move a generated operand onto the CPU-only layout of this operator."""
    return dense.cpu().to_mkldnn()


def _transposed_shape(shape, dim0, dim1):
    out_shape = list(shape)
    out_shape[dim0], out_shape[dim1] = out_shape[dim1], out_shape[dim0]
    return out_shape


def _assert_opaque(result, shape, dtype, dim0, dim1):
    """The native result is an opaque mkldnn tensor of the transposed shape."""
    assert result.layout == torch._mkldnn
    assert result.dtype == dtype
    assert list(result.size()) == _transposed_shape(shape, dim0, dim1)


@pytest.mark.mkldnn_transpose
@pytest.mark.parametrize("dtype", _MKLDNN_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape,dim0,dim1", _VALUE_ROWS)
def test__mkldnn_transpose(shape, dim0, dim1, value_range, dtype):
    inp = _mkldnn_operand(tu.make_input(dtype, shape, value_range))
    snapshot = inp.to_dense().clone()
    meta = (inp.size(), inp.stride(), inp.storage_offset())
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._mkldnn_transpose(ref_inp, dim0, dim1)
    res_out = flag_gems._mkldnn_transpose(inp, dim0, dim1)

    _assert_opaque(res_out, shape, dtype, dim0, dim1)
    assert res_out is not inp
    # Relayout copies stored values without arithmetic: exact comparison of the
    # values behind both opaque results.
    tu.assert_result_equal(res_out.to_dense(), ref_out.to_dense())
    # The operand is read-only: its metadata and stored values stay untouched.
    assert (inp.size(), inp.stride(), inp.storage_offset()) == meta
    tu.assert_result_equal(inp.to_dense(), snapshot)


@pytest.mark.mkldnn_transpose
@pytest.mark.parametrize("dtype", _MKLDNN_DTYPES)
@pytest.mark.parametrize("shape,dim0,dim1", _DIM_ROWS)
def test__mkldnn_transpose_dims(shape, dim0, dim1, dtype):
    inp = _mkldnn_operand(tu.make_input(dtype, shape, ["-1", "1"]))
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._mkldnn_transpose(ref_inp, dim0, dim1)
    res_out = flag_gems._mkldnn_transpose(inp, dim0, dim1)

    _assert_opaque(res_out, shape, dtype, dim0, dim1)
    tu.assert_result_equal(res_out.to_dense(), ref_out.to_dense())


@pytest.mark.mkldnn_transpose
@pytest.mark.parametrize("dtype", _MKLDNN_DTYPES)
@pytest.mark.parametrize("shape,dim0,dim1", _OUT_ROWS)
def test__mkldnn_transpose_out(shape, dim0, dim1, dtype):
    inp = _mkldnn_operand(tu.make_input(dtype, shape, ["-1", "1"]))
    ref_inp = tu.to_reference(inp)
    # The destination must itself be an mkldnn tensor of the transposed shape
    # and dtype; its contents stay undefined until the operator writes them.
    out_shape = _transposed_shape(shape, dim0, dim1)
    ref_buf = torch.empty(out_shape, dtype=dtype).to_mkldnn()
    res_buf = torch.empty(out_shape, dtype=dtype).to_mkldnn()

    ref_out = torch.ops.aten._mkldnn_transpose.out(ref_inp, dim0, dim1, out=ref_buf)
    res_out = flag_gems._mkldnn_transpose(inp, dim0, dim1, out=res_buf)

    assert res_out is res_buf  # writes the caller's destination, not a new tensor
    tu.assert_result_equal(res_out.to_dense(), ref_out.to_dense())


@pytest.mark.mkldnn_transpose
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_MKLDNN_DTYPES), quick=[]),
)
def test__mkldnn_transpose_special_values(dtype, scenario):
    payload = tu.make_special_input(dtype, scenario).reshape(1, -1)
    inp = _mkldnn_operand(payload)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._mkldnn_transpose(ref_inp, 0, 1)
    res_out = flag_gems._mkldnn_transpose(inp, 0, 1)

    _assert_opaque(res_out, inp.shape, dtype, 0, 1)
    # nan / inf / -inf / +-0 pass through the relayout unchanged.
    tu.assert_result_equal(res_out.to_dense(), ref_out.to_dense())


@pytest.mark.mkldnn_transpose
def test__mkldnn_transpose_backward():
    # The candidate is exercised through the original leaf: the mkldnn operand
    # is that leaf's conversion, and the upstream gradient is non-uniform so a
    # fabricated gradient could not pass. Native aten::_mkldnn_transpose
    # registers no derivative ("derivative for aten::_mkldnn_transpose is not
    # implemented"), so differentiating must fail instead of returning a
    # gradient the reference cannot produce.
    leaf = torch.randn(3, 4, requires_grad=True)
    inp = leaf.to_mkldnn()
    upstream = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3).to_mkldnn()

    res_out = flag_gems._mkldnn_transpose(inp, 0, 1)

    with pytest.raises(RuntimeError):
        torch.autograd.grad(res_out, leaf, grad_outputs=upstream)


@pytest.mark.mkldnn_transpose
@pytest.mark.parametrize("dtype", _NEG_DTYPES)
@pytest.mark.parametrize("shape,dim0,dim1", _OOB_DIM_ROWS)
def test__mkldnn_transpose_rejects_out_of_range_dims(shape, dim0, dim1, dtype):
    inp = _mkldnn_operand(tu.make_input(dtype, shape, ["-1", "1"]))
    with pytest.raises((IndexError, RuntimeError, ValueError)):
        flag_gems._mkldnn_transpose(inp, dim0, dim1)


@pytest.mark.mkldnn_transpose
@pytest.mark.parametrize("dtype", _NEG_DTYPES)
@pytest.mark.parametrize("dim0,dim1", [(0.5, 1), (0, "1")])
def test__mkldnn_transpose_rejects_non_int_dims(dim0, dim1, dtype):
    inp = _mkldnn_operand(tu.make_input(dtype, (2, 3, 4), ["-1", "1"]))
    with pytest.raises((TypeError, RuntimeError, ValueError)):
        flag_gems._mkldnn_transpose(inp, dim0, dim1)


@pytest.mark.mkldnn_transpose
def test__mkldnn_transpose_rejects_non_tensor_self():
    # The first schema argument is a Tensor; native dispatch rejects a
    # non-tensor with "failed to match any schema".
    with pytest.raises((IndexError, RuntimeError, TypeError, ValueError)):
        flag_gems._mkldnn_transpose(3, 0, 1)


@pytest.mark.mkldnn_transpose
@pytest.mark.parametrize("dtype", tu.REQUIRED_DTYPES)
def test__mkldnn_transpose_rejects_strided_operand(dtype):
    # The operand must already be opaque: a plain strided tensor is outside this
    # operator's domain (native dispatch raises NotImplementedError), which is
    # why the spec's wide dtype grid cannot apply here - only the five dtypes
    # that dense_to_mkldnn accepts can be operands at all.
    dense = torch.empty((2, 3), dtype=dtype, device="cpu")
    with pytest.raises((IndexError, RuntimeError, TypeError, ValueError)):
        flag_gems._mkldnn_transpose(dense, 0, 1)


@pytest.mark.mkldnn_transpose
def test__mkldnn_transpose_out_rejects_wrong_shape():
    inp = _mkldnn_operand(tu.make_input(torch.float32, (2, 3, 4), ["-1", "1"]))
    out = torch.empty(3, 2, 5, dtype=torch.float32).to_mkldnn()
    with pytest.raises((IndexError, RuntimeError, ValueError)):
        flag_gems._mkldnn_transpose(inp, 0, 1, out=out)


@pytest.mark.mkldnn_transpose
def test__mkldnn_transpose_out_rejects_dtype_mismatch():
    inp = _mkldnn_operand(tu.make_input(torch.float32, (2, 3, 4), ["-1", "1"]))
    out = torch.empty(3, 2, 4, dtype=torch.bfloat16).to_mkldnn()
    with pytest.raises((IndexError, RuntimeError, TypeError, ValueError)):
        flag_gems._mkldnn_transpose(inp, 0, 1, out=out)
