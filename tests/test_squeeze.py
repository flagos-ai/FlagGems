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
from . import test_utils as tu

_DTYPE_FLAGS = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.float64: utils.fp64_is_supported,
    torch.complex128: utils.fp64_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
}

# squeeze is a metadata-only view (aten::squeeze(Tensor(a) self) -> Tensor(a)):
# it drops size-1 dims and aliases its input. Call forms used here are
# squeeze(x), squeeze(x, dim) and squeeze(x, [dim, ...]). There is no .out
# overload, and .dimname is uncallable in this build (named tensors raise
# 'NYI: Named tensors are currently unsupported in TorchScript').
_SUPPORTED_DTYPES = list(tu.REQUIRED_DTYPES) + [torch.bool, torch.complex64]
if utils.fp64_is_supported:
    _SUPPORTED_DTYPES.append(torch.float64)
_SUPPORTED_DTYPES = [
    dtype for dtype in _SUPPORTED_DTYPES if _DTYPE_FLAGS.get(dtype, True)
]

_RANGE = ["-1", "1"]

# The spec shapes carry no size-1 extent, and a size-1 or empty dim is the only
# thing squeeze reacts to, so singleton/empty boundaries are added. All of them
# are cheap and stay in quick; only the large rank-5 extension is default-only.
_SMALL_SINGLETON_SHAPES = [
    (1, 1),
    (1, 256),
    (256, 1),
    (2, 1, 3),
    (1, 7, 1, 32, 1),
    (0,),
    (0, 1, 3),
]
_LARGE_SINGLETON_SHAPES = [(16, 1, 128, 64, 60)]
_SHAPES = tu.selected_shapes() + tu.selected_cases(
    _SMALL_SINGLETON_SHAPES + _LARGE_SINGLETON_SHAPES,
    quick=[()] + _SMALL_SINGLETON_SHAPES,
)

# (shape, dim): singleton removal, negative dims, a non-singleton dim (a no-op
# that still returns a view) and a size-3 last dim. Cheap, so both modes.
_DIM_CASES = [
    ((1,), 0),
    ((1, 256), 0),
    ((256, 1), 1),
    ((2, 1, 3), 1),
    ((2, 1, 3), -2),
    ((2, 1, 3), -1),
    ((2, 1, 3), 2),
    ((1, 7, 1, 32, 1), 0),
]

# (shape, dims): explicit lists, unsorted order (order does not matter), the
# empty list (a no-op) and a list holding non-singleton entries (also a no-op).
_DIMS_CASES = [
    ((3, 1, 5, 1, 1), [1]),
    ((3, 1, 5, 1, 1), [1, 3, 4]),
    ((3, 1, 5, 1, 1), [-1, -3]),
    ((3, 1, 5, 1, 1), [4, 2]),
    ((3, 1, 5, 1, 1), []),
    ((3, 1, 5, 1, 1), [0, 2]),
]

# Layouts a view has to compose with: a storage offset, an expanded stride-0
# dim, a permutation, a transpose and a lazily conj'd non-contiguous input.
_LAYOUTS = [
    "slice_with_offset",
    "expanded",
    "permuted",
    "transposed",
    "transposed_conj",
]

# squeeze re-labels dims, so the upstream gradient reaches the leaf unchanged.
_BACKWARD_CASES = tu.selected_cases(
    [((1, 4, 1, 3), None), ((1, 4, 1, 3), 2), ((2, 1, 5, 1, 1), [1, 4])],
    quick=[],
)

_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_SUPPORTED_DTYPES), quick=[])

# Invalid dims are rejected in both modes.
_NEGATIVE_DIM_CASES = [((6, 1, 8), 3), ((6, 1, 8), -4), ((6,), 1)]
_NEGATIVE_DIMS_CASES = [((2, 1, 3), [0, 5]), ((3, 1, 5, 1, 1), [1, 7])]


def _assert_view_metadata(res_out, ref_out, inp, ref_inp):
    """squeeze returns a new view object that aliases inp and keeps the native
    shape/stride/offset/conjunction state, leaving inp metadata untouched."""
    assert res_out is not inp
    assert res_out._is_view()
    assert res_out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
    assert tuple(res_out.shape) == tuple(ref_out.shape)
    assert tuple(res_out.stride()) == tuple(ref_out.stride())
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.is_conj() == ref_out.is_conj()
    assert tuple(inp.shape) == tuple(ref_inp.shape)
    assert tuple(inp.stride()) == tuple(ref_inp.stride())
    assert inp.storage_offset() == ref_inp.storage_offset()


def _make_layout_input(kind, dtype):
    if kind == "slice_with_offset":
        return tu.make_input(dtype, (5, 1, 6), _RANGE)[2:4]
    if kind == "expanded":
        return tu.make_input(dtype, (3, 1, 1), _RANGE).expand(3, 1, 5)
    if kind == "permuted":
        return tu.make_input(dtype, (2, 1, 3, 4), _RANGE).permute(2, 0, 1, 3)
    if kind == "transposed":
        return tu.make_input(dtype, (1, 3, 4), _RANGE).transpose(0, 1)
    if kind == "transposed_conj":
        return tu.make_input(dtype, (1, 3, 4), _RANGE).transpose(0, 1).conj()
    raise AssertionError("unknown layout kind: {}".format(kind))


@pytest.mark.squeeze
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_squeeze_all_size_one_dims(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.squeeze(ref_inp)
    res_out = flag_gems.squeeze(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_metadata(res_out, ref_out, inp, ref_inp)


@pytest.mark.squeeze
@pytest.mark.parametrize("shape,dim", _DIM_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_squeeze_dim(shape, dim, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.squeeze(ref_inp, dim)
    res_out = flag_gems.squeeze(inp, dim)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_metadata(res_out, ref_out, inp, ref_inp)


@pytest.mark.squeeze
@pytest.mark.parametrize("shape,dims", _DIMS_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_squeeze_dims(shape, dims, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.squeeze(ref_inp, dims)
    res_out = flag_gems.squeeze(inp, dims)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_metadata(res_out, ref_out, inp, ref_inp)


@pytest.mark.squeeze
@pytest.mark.parametrize("kind", _LAYOUTS)
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_squeeze_noncontiguous_layout(kind, dtype):
    inp = _make_layout_input(kind, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.squeeze(ref_inp)
    res_out = flag_gems.squeeze(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_metadata(res_out, ref_out, inp, ref_inp)


@pytest.mark.squeeze
def test_squeeze_keeps_conjugate_bit():
    inp = tu.make_input(torch.complex64, (2, 1, 3), _RANGE).conj()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.squeeze(ref_inp)
    res_out = flag_gems.squeeze(inp)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.is_conj()
    _assert_view_metadata(res_out, ref_out, inp, ref_inp)


@pytest.mark.squeeze
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_squeeze_result_writes_through_to_input(dtype):
    inp = tu.make_input(dtype, (2, 1, 3), _RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.squeeze(ref_inp)
    res_out = flag_gems.squeeze(inp)

    _assert_view_metadata(res_out, ref_out, inp, ref_inp)
    tu.assert_result_equal(res_out, ref_out)
    res_out.fill_(1)
    ref_out.fill_(1)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.squeeze
@pytest.mark.parametrize("shape,dim", _BACKWARD_CASES)
@pytest.mark.parametrize(
    "dtype",
    [
        dtype
        for dtype in _SUPPORTED_DTYPES
        if dtype.is_floating_point or dtype.is_complex
    ],
)
def test_squeeze_backward(shape, dim, dtype):
    inp = tu.make_input(dtype, shape, _RANGE).requires_grad_(True)
    ref_inp = tu.to_reference(inp)

    if dim is None:
        res_out = flag_gems.squeeze(inp)
        ref_out = torch.ops.aten.squeeze(ref_inp)
    else:
        res_out = flag_gems.squeeze(inp, dim)
        ref_out = torch.ops.aten.squeeze(ref_inp, dim)

    tu.assert_result_equal(res_out, ref_out)

    # grad_outputs must match the squeezed shape, not the input shape.
    upstream = tu.make_input(dtype, tuple(res_out.shape), _RANGE)
    (res_grad,) = torch.autograd.grad(res_out, inp, grad_outputs=upstream)
    (ref_grad,) = torch.autograd.grad(
        ref_out, ref_inp, grad_outputs=tu.to_reference(upstream)
    )

    assert tuple(res_grad.shape) == tuple(inp.shape)
    tu.assert_result_equal(res_grad, ref_grad)


@pytest.mark.squeeze
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_squeeze_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario).reshape(5, 1, 1)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.squeeze(ref_inp)
    res_out = flag_gems.squeeze(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_metadata(res_out, ref_out, inp, ref_inp)


@pytest.mark.squeeze
@pytest.mark.parametrize("shape,dim", _NEGATIVE_DIM_CASES)
def test_squeeze_out_of_range_dim_raises(shape, dim):
    inp = tu.make_input(torch.float32, shape, _RANGE)

    with pytest.raises(IndexError):
        flag_gems.squeeze(inp, dim)


@pytest.mark.squeeze
@pytest.mark.parametrize("shape,dims", _NEGATIVE_DIMS_CASES)
def test_squeeze_out_of_range_dims_raises(shape, dims):
    inp = tu.make_input(torch.float32, shape, _RANGE)

    with pytest.raises(IndexError):
        flag_gems.squeeze(inp, dims)


@pytest.mark.squeeze
@pytest.mark.parametrize("dims", [[1, 1], [1, -2], [0, 0, 1]])
def test_squeeze_duplicate_dims_raises(dims):
    inp = tu.make_input(torch.float32, (6, 1, 8), _RANGE)

    with pytest.raises(RuntimeError):
        flag_gems.squeeze(inp, dims)


@pytest.mark.squeeze
def test_squeeze_rejects_non_tensor():
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.squeeze([1, 2, 3])
