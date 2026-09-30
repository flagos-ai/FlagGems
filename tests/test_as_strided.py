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

"""Correctness tests for ``aten::as_strided``.

``as_strided(self, size, stride, storage_offset=None) -> Tensor(a)`` is a pure
view/metadata operator: it returns a new tensor over ``self``'s storage without
reading or moving element data, so values are compared exactly (zero tolerance)
and the operator-specific checks are the requested metadata, the storage alias,
the fresh result object, the untouched input, the lazy conj/neg bits and the
gradient placement. An explicit ``storage_offset`` is absolute inside the
underlying storage while omitting it (or passing ``None``) keeps the input's own
offset; negative size/stride/offset, rank mismatch and out-of-range views raise
RuntimeError.

``as_strided`` never broadcasts its argument, so the spec's broadcast dimension
does not apply; the stride-0 and expanded-layout rows cover the layout such an
operand would produce.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# A metadata rewrite can address any dtype whose storage the backend holds, so
# the required dtype list is used as-is plus the wider float/complex and bool
# types. int8/uint8/fp8 are required coverage, not optional extras.
_AS_STRIDED_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64, torch.complex128] if utils.fp64_is_supported else [])
    + [torch.complex64, torch.bool]
)


def _contiguous_strides(size):
    strides = [1] * len(size)
    for dim in range(len(size) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * size[dim + 1]
    return strides


def _reversed_layout(shape):
    """Reverse the sizes and recompute their contiguous strides.

    This reinterprets the same buffer without moving or permuting data: the
    native view and the candidate view read the same storage positions, so the
    comparison stays exact even though the reinterpreted read order is not a
    flip along any axis.
    """
    size = list(reversed(shape))
    return size, _contiguous_strides(size), 0


def _apply_layout(base, kind, expand_shape=None):
    if kind == "asis":
        return base
    if kind == "window":
        return base[2:6, 1:5]
    if kind == "expanded":
        return base.expand(tuple(expand_shape))
    raise ValueError("unsupported layout " + repr(kind))


def _input_metadata(inp):
    return (tuple(inp.size()), inp.stride(), inp.storage_offset())


def _assert_view_semantics(res_out, ref_out, inp, inp_meta):
    # Equal values alone cannot show that the candidate returned a view with the
    # requested metadata instead of a materialized copy or an in-place rewrite,
    # so the metadata is matched against the native result and the input is
    # checked to be untouched. An in-place candidate returns the input itself,
    # which is why the result object is also compared by identity.
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out._is_view() == ref_out._is_view()
    assert res_out is not inp
    assert res_out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
    assert _input_metadata(inp) == inp_meta


@pytest.mark.as_strided
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _AS_STRIDED_DTYPES)
def test_as_strided(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    size, stride, offset = _reversed_layout(shape)
    inp_meta = _input_metadata(inp)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.as_strided(ref_inp, size, stride, offset)
    res_out = flag_gems.as_strided(inp, size, stride, offset)

    _assert_view_semantics(res_out, ref_out, inp, inp_meta)
    tu.assert_result_equal(res_out, ref_out)


# (storage shape, layout kind, expand shape, requested size, stride, offset).
# Covers the identity layout, a stride-differing relayout, a non-contiguous
# windowed input with an explicit offset, a stride-0 input view (as_strided
# never broadcasts operands, so a stride-0 view is how a broadcast layout
# reaches it), 0-dim, single-element, rank change and zero-extent/empty views.
_CONTRACT_ROWS = [
    ((2, 3), "asis", None, [2, 3], [3, 1], 0),
    ((3, 4), "asis", None, [4, 3], [1, 3], 0),
    ((8, 8), "window", None, [2, 2], [8, 1], 3),
    ((4,), "expanded", (3, 4), [3, 4], [0, 1], 0),
    ((0,), "asis", None, [0], [1], 0),
    ((2, 0), "asis", None, [0, 2], [1, 1], 0),
    ((0, 3), "asis", None, [0], [3], 0),
    ((0,), "asis", None, [0], [0], 0),
    ((), "asis", None, [], [], 0),
    ((1,), "asis", None, [1], [1], 0),
    ((256,), "asis", None, [16, 16], [16, 1], 0),
    ((10, 10, 10, 10, 10), "asis", None, [4, 4, 4, 4, 4], [1, 100, 10, 1000, 10000], 0),
]

# Quick keeps every small row (identity, relayout, stride-0, 0-dim,
# single-element, rank change and the zero-extent views) so the view-versus-
# in-place and boundary discriminators run in both modes; only the trailing
# 100k-element rank-5 row is default-only.
_QUICK_CONTRACT_ROWS = _CONTRACT_ROWS[:-1]
CONTRACT_CASES = tu.selected_cases(_CONTRACT_ROWS, quick=_QUICK_CONTRACT_ROWS)


@pytest.mark.as_strided
@pytest.mark.parametrize("storage,kind,expand,size,stride,offset", CONTRACT_CASES)
def test_as_strided_layout_contract(storage, kind, expand, size, stride, offset):
    base = tu.make_input(torch.float32, storage, ["-1", "1"])
    inp = _apply_layout(base, kind, expand)
    ref_inp = _apply_layout(tu.to_reference(base.detach()), kind, expand)
    inp_meta = _input_metadata(inp)

    ref_out = torch.ops.aten.as_strided(ref_inp, size, stride, offset)
    res_out = flag_gems.as_strided(inp, size, stride, offset)

    _assert_view_semantics(res_out, ref_out, inp, inp_meta)
    tu.assert_result_equal(res_out, ref_out)


# Both call forms of an inherited offset: the argument omitted entirely and the
# argument passed as ``None``.
_INHERIT_FORMS = [([2, 2], [8, 1]), ([2, 2], [8, 1], None)]
INHERIT_CASES = _INHERIT_FORMS


@pytest.mark.as_strided
@pytest.mark.parametrize("args", INHERIT_CASES)
def test_as_strided_inherited_storage_offset(args):
    # Omitting storage_offset or passing None keeps the input tensor's own
    # offset (17 for this window) instead of starting at storage element 0; the
    # helper's result-offset and value checks against the native view are what
    # pin that down.
    base = tu.make_input(torch.float32, (8, 8), ["-1", "1"])
    inp = _apply_layout(base, "window")
    ref_inp = _apply_layout(tu.to_reference(base.detach()), "window")
    inp_meta = _input_metadata(inp)

    ref_out = torch.ops.aten.as_strided(ref_inp, *args)
    res_out = flag_gems.as_strided(inp, *args)

    _assert_view_semantics(res_out, ref_out, inp, inp_meta)
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.as_strided
@pytest.mark.parametrize("offset", [0, 1, 3])
def test_as_strided_absolute_storage_offset(offset):
    # An explicit offset is absolute inside the storage, so offset 0 selects
    # storage element 0 rather than the input window's own offset 17; the native
    # result's storage_offset assertion in the helper is what pins that down.
    base = tu.make_input(torch.float32, (8, 8), ["-1", "1"])
    inp = _apply_layout(base, "window")
    ref_inp = _apply_layout(tu.to_reference(base.detach()), "window")
    inp_meta = _input_metadata(inp)

    ref_out = torch.ops.aten.as_strided(ref_inp, [2, 2], [8, 1], offset)
    res_out = flag_gems.as_strided(inp, [2, 2], [8, 1], offset)

    _assert_view_semantics(res_out, ref_out, inp, inp_meta)
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.as_strided
@pytest.mark.parametrize(
    "kind,dtype", [("conj", torch.complex64), ("neg", torch.float32)]
)
def test_as_strided_lazy_flag(kind, dtype):
    # A metadata rewrite must keep the input's lazy conjugate / negative bit.
    base = tu.make_input(dtype, (3, 4), ["-1", "1"])
    ref_base = tu.to_reference(base.detach())
    inp = base.conj() if kind == "conj" else torch._neg_view(base)
    ref_inp = ref_base.conj() if kind == "conj" else torch._neg_view(ref_base)
    inp_meta = _input_metadata(inp)

    ref_out = torch.ops.aten.as_strided(ref_inp, [4, 3], [1, 3], 0)
    res_out = flag_gems.as_strided(inp, [4, 3], [1, 3], 0)

    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.is_neg() == ref_out.is_neg()
    _assert_view_semantics(res_out, ref_out, inp, inp_meta)
    tu.assert_result_equal(res_out, ref_out)


# Both rows cover the whole aliasing contract (a write through the result must
# be visible through the input) in quick as well as default mode.
_MUTATION_ROWS = [
    ((3, 4), [4, 3], [1, 3], 0),
    ((8, 8), [2, 2], [8, 1], 3),
]
MUTATION_CASES = _MUTATION_ROWS


@pytest.mark.as_strided
@pytest.mark.parametrize("shape,size,stride,offset", MUTATION_CASES)
def test_as_strided_view_writes_through(shape, size, stride, offset):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp.detach())
    inp_meta = _input_metadata(inp)
    inp_before = tu.to_reference(inp.detach())

    ref_out = torch.ops.aten.as_strided(ref_inp, size, stride, offset)
    res_out = flag_gems.as_strided(inp, size, stride, offset)

    _assert_view_semantics(res_out, ref_out, inp, inp_meta)
    tu.assert_result_equal(res_out, ref_out)
    # Building the view must not write to the storage.
    tu.assert_result_equal(inp, inp_before)

    # The result aliases the input storage, so a write through the candidate view
    # has to be observable through the input at the same positions.
    res_out.fill_(3.0)
    ref_out.fill_(3.0)
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


# Backward. The placement rows map every storage element to at most one view
# element, so the gradient is a plain placement (exact comparison); the
# overlapping window and the stride-0 row accumulate and are compared as
# arithmetic results. Every real (fp16/fp32/bf16/fp64) and complex
# (complex64/complex128) combination was probed per row with explicit
# grad_outputs and produced a gradient. float8 is excluded because the autograd
# path needs sum/index_add kernels that are unimplemented for float8 in this
# environment ('sum_cpu' for the placement rows, 'index_add_' for the
# overlapping ones), so it cannot produce a gradient at all.
_BACKWARD_ROWS = [
    ((4, 6), [3, 4], [6, 1], 0, False),
    ((8, 8), [2, 2], [8, 1], 3, False),
    ((4, 6), [2, 3], [1, 1], 2, True),
    ((12,), [8], [1], 2, True),
    ((6, 4), [3, 4], [0, 1], 0, True),
]
_BACKWARD_REAL_DTYPES = [torch.float16, torch.float32, torch.bfloat16] + (
    [torch.float64] if utils.fp64_is_supported else []
)
_BACKWARD_COMPLEX_DTYPES = [torch.complex64] + (
    [torch.complex128] if utils.fp64_is_supported else []
)
_BACKWARD_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]


def _backward_dtypes(accumulates):
    dtypes = _BACKWARD_REAL_DTYPES + _BACKWARD_COMPLEX_DTYPES
    return dtypes if accumulates else dtypes + _BACKWARD_FP8_DTYPES


BACKWARD_CASES = tu.selected_cases(
    [
        (shape, size, stride, offset, accumulates, dtype)
        for shape, size, stride, offset, accumulates in _BACKWARD_ROWS
        for dtype in _backward_dtypes(accumulates)
    ],
    quick=[],
)


@pytest.mark.as_strided
@pytest.mark.parametrize("shape,size,stride,offset,accumulates,dtype", BACKWARD_CASES)
def test_as_strided_backward(shape, size, stride, offset, accumulates, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_inp = tu.to_reference(inp)
    upstream = tu.make_input(dtype, tuple(size), ["-1", "1"])
    ref_upstream = tu.to_reference(upstream.detach())
    inp_meta = _input_metadata(inp)

    ref_out = torch.ops.aten.as_strided(ref_inp, size, stride, offset)
    res_out = flag_gems.as_strided(inp, size, stride, offset)

    assert res_out.requires_grad
    _assert_view_semantics(res_out, ref_out, inp, inp_meta)
    tu.assert_result_equal(res_out, ref_out)

    # Differentiated through the original leaf: the view aliases the input, so the
    # gradient is scattered (and summed where the view overlaps itself) back into
    # the input's storage.
    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]
    ref_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)[0]

    if accumulates:
        tu.assert_result_close(res_grad, ref_grad)
    else:
        tu.assert_result_equal(res_grad, ref_grad)


# Positive special values are default-only; the matrix comes from the supported
# dtype list, so e4m3fn keeps its nan-only case and e5m2 covers nan/inf/mixed.
SPECIAL_VALUE_CASES = tu.selected_cases(
    tu.special_value_cases(_AS_STRIDED_DTYPES), quick=[]
)


@pytest.mark.as_strided
@pytest.mark.parametrize("dtype,scenario", SPECIAL_VALUE_CASES)
def test_as_strided_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)
    size = [1, inp.numel()]
    stride = [inp.numel(), 1]
    inp_meta = _input_metadata(inp)

    ref_out = torch.ops.aten.as_strided(ref_inp, size, stride, 0)
    res_out = flag_gems.as_strided(inp, size, stride, 0)

    _assert_view_semantics(res_out, ref_out, inp, inp_meta)
    tu.assert_result_equal(res_out, ref_out)


_INVALID_LAYOUT_ROWS = [
    ([2, 2], [2], None),  # rank mismatch between size and stride
    ([-2, 2], [2, 1], None),  # negative size
    ([2, 2], [-2, 1], None),  # negative stride
    ([2, 2], [2, 1], -1),  # negative storage offset
    ([3, 4], [4, 1], 0),  # index range beyond the 6-element storage
]


@pytest.mark.as_strided
@pytest.mark.parametrize("size,stride,offset", _INVALID_LAYOUT_ROWS)
def test_as_strided_invalid_layout(size, stride, offset):
    inp = tu.make_input(torch.float32, (2, 3), ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems.as_strided(inp, size, stride, offset)


@pytest.mark.as_strided
def test_as_strided_rejects_non_tensor_input():
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.as_strided([[1.0, 2.0, 3.0]], [3], [1])


@pytest.mark.as_strided
def test_as_strided_rejects_non_list_size():
    inp = tu.make_input(torch.float32, (2, 3), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.as_strided(inp, 3, [1])
