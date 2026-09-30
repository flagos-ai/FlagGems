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

"""Correctness tests for ``aten::as_strided_``.

``as_strided_`` reinterprets the storage of an existing tensor with new sizes,
strides and storage offset: it moves no element, returns that very tensor and
rewrites metadata only. The tests therefore assert the installed view metadata
(returned identity, strides, storage offset, reused buffer) next to the
materialised values, and prove that the stored bytes stay where they were.

Coverage: the spec's 5 ranges x 7 shapes grid runs with the identity view of
each shape, because the operand can only express a view over the buffer it
already owns. Non-trivial layouts (dilated, transposed, stride-0, rank
reducing, empty, 0-dim), operands that already carry a storage offset, the lazy
conjugate bit, the special-value matrix, backward and the invalid-argument
negatives have their own groups.

The storage_offset argument has three call forms: omitted, explicit None and an
explicit value. Probed natively on pre-offset operands, omission and None keep
the operand's current storage offset; only an explicit value replaces it.

Broadcast does not apply (a single tensor operand, no broadcasting) and there
is no scalar-operand overload: ``size`` and ``stride`` must be integer
sequences, which the negative cases pin down.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

_AS_STRIDED_DTYPES = (
    tu.REQUIRED_DTYPES
    + [torch.bool, torch.complex64]
    + ([torch.float64, torch.complex128] if utils.fp64_is_supported else [])
)
_FP8_GRAD_DTYPES = (
    [torch.float8_e4m3fn, torch.float8_e5m2] if utils.fp8_is_supported else []
)
_GRAD_DTYPES = (
    [torch.float16, torch.float32, torch.complex64]
    + ([torch.bfloat16] if utils.bf16_is_supported else [])
    + ([torch.float64, torch.complex128] if utils.fp64_is_supported else [])
)
_CONJ_DTYPES = [torch.complex64] + (
    [torch.complex128] if utils.fp64_is_supported else []
)


def _contiguous_view(shape):
    """(size, stride, storage_offset) of the identity view of ``shape``."""
    size = [int(extent) for extent in shape]
    stride = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        stride[i] = stride[i + 1] * size[i + 1]
    return size, stride, 0


def _storage_witness(tensor):
    """Flat view over the whole storage, independent of the tensor metadata."""
    elements = tensor.untyped_storage().size() // tensor.element_size()
    return torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
        tensor.untyped_storage(), 0, (elements,), (1,)
    )


def _assert_view_matches(res_out, ref_out, inp, storage_before):
    # Materialised values cannot show that the requested view was installed on
    # the operand itself, so assert identity, strides, offset and buffer reuse.
    assert res_out is inp
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.untyped_storage().data_ptr() == storage_before


@pytest.mark.as_strided_
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _AS_STRIDED_DTYPES)
def test_as_strided__value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    size, stride, offset = _contiguous_view(shape)
    ref_inp = tu.to_reference(inp)
    storage_before = inp.untyped_storage().data_ptr()
    storage_values = tu.to_reference(_storage_witness(inp))

    ref_out = torch.ops.aten.as_strided_(ref_inp, size, stride, offset)
    res_out = flag_gems.as_strided_(inp, size, stride, offset)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_matches(res_out, ref_out, inp, storage_before)
    # The mutation lands on the operand, not on a copy.
    tu.assert_result_equal(inp, ref_inp)
    # Reinterpreting the buffer must not rewrite any stored element.
    tu.assert_result_equal(_storage_witness(inp), storage_values)


# Layouts a contiguous (4, 6) buffer can express; each row was accepted by the
# real overload. All small offset, stride-0, transpose and empty boundaries
# stay in both levels.
_LAYOUT_ROWS = [
    ([4, 6], [2, 3], 0),
    ([4, 6], [2, 3], 1),
    ([4, 4], [0, 1], 0),
    ([6, 4], [1, 6], 0),
    ([2, 3], [12, 2], 1),
    ([24], [1], 0),
    ([0, 3], [6, 1], 0),
    ([], [], 2),
    ([4, 6], [6, 1], None),
]


def _layout_cases(rows):
    return [
        (size, stride, offset, dtype)
        for size, stride, offset in rows
        for dtype in _AS_STRIDED_DTYPES
    ]


LAYOUT_CASES = _layout_cases(_LAYOUT_ROWS)


@pytest.mark.as_strided_
@pytest.mark.parametrize("size,stride,offset,dtype", LAYOUT_CASES)
def test_as_strided__layouts(size, stride, offset, dtype):
    inp = tu.make_input(dtype, (4, 6), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    storage_before = inp.untyped_storage().data_ptr()
    storage_values = tu.to_reference(_storage_witness(inp))

    ref_out = torch.ops.aten.as_strided_(ref_inp, size, stride, offset)
    res_out = flag_gems.as_strided_(inp, size, stride, offset)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_matches(res_out, ref_out, inp, storage_before)
    tu.assert_result_equal(inp, ref_inp)
    # These views skip elements of the buffer; a candidate that materialises the
    # view inside the same storage would move those skipped bytes.
    tu.assert_result_equal(_storage_witness(inp), storage_values)


def _derive_operand(tensor, layout):
    if layout == "slice":
        return tensor[2:6, 1:7]
    if layout == "full":
        return tensor[:]
    if layout == "tail":
        return tensor[5:]
    return tensor[:, :6]


# Operands that already carry a storage offset, or that alias a larger buffer.
_STRIDED_INPUT_ROWS = [
    ((8, 8), "slice", [6, 4], [1, 8], 17),
    ((8, 8), "full", [4, 6], [1, 8], 0),
    ((6, 10), "columns", [6, 6], [1, 10], 0),
]
STRIDED_INPUT_CASES = [
    (shape, layout, size, stride, offset, dtype)
    for shape, layout, size, stride, offset in _STRIDED_INPUT_ROWS
    for dtype in _AS_STRIDED_DTYPES
]


@pytest.mark.as_strided_
@pytest.mark.parametrize("shape,layout,size,stride,offset,dtype", STRIDED_INPUT_CASES)
def test_as_strided__strided_input(shape, layout, size, stride, offset, dtype):
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = _derive_operand(base, layout)
    ref_inp = _derive_operand(ref_base, layout)
    storage_before = inp.untyped_storage().data_ptr()
    storage_values = tu.to_reference(_storage_witness(base))

    ref_out = torch.ops.aten.as_strided_(ref_inp, size, stride, offset)
    res_out = flag_gems.as_strided_(inp, size, stride, offset)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_matches(res_out, ref_out, inp, storage_before)
    assert res_out.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    # Only the operand's own metadata may change; its source buffer keeps both
    # its bytes and its own sizes/strides.
    tu.assert_result_equal(_storage_witness(base), storage_values)
    tu.assert_result_equal(base, ref_base)


# The three storage_offset call forms, each on an operand that already carries a
# non-zero offset: a ``[5:]`` slice (offset 5) and a ``[2:6, 1:7]`` slice
# (offset 17). Native probes show that omitting the argument and passing None
# both keep the operand's current offset, while an explicit value replaces it,
# so the explicit-zero rows read from the start of the buffer and are the
# comparison that makes the other two forms meaningful.
# (base_shape, layout, size, stride, offset_form, offset)
_OFFSET_ROWS = [
    ((20,), "tail", [2], [1], "omitted", None),
    ((20,), "tail", [2], [1], "none", None),
    ((20,), "tail", [2], [1], "explicit", 0),
    ((20,), "tail", [2], [1], "explicit", 3),
    ((8, 8), "slice", [6, 4], [1, 8], "omitted", None),
    ((8, 8), "slice", [6, 4], [1, 8], "none", None),
    ((8, 8), "slice", [6, 4], [1, 8], "explicit", 0),
]
OFFSET_CASES = [
    (shape, layout, size, stride, form, offset, dtype)
    for shape, layout, size, stride, form, offset in _OFFSET_ROWS
    for dtype in _AS_STRIDED_DTYPES
]


@pytest.mark.as_strided_
@pytest.mark.parametrize(
    "base_shape,layout,size,stride,offset_form,offset,dtype", OFFSET_CASES
)
def test_as_strided__storage_offset_forms(
    base_shape, layout, size, stride, offset_form, offset, dtype
):
    base = tu.make_input(dtype, base_shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = _derive_operand(base, layout)
    ref_inp = _derive_operand(ref_base, layout)
    original_offset = inp.storage_offset()
    storage_before = inp.untyped_storage().data_ptr()
    storage_values = tu.to_reference(_storage_witness(base))

    if offset_form == "omitted":
        ref_out = torch.ops.aten.as_strided_(ref_inp, size, stride)
        res_out = flag_gems.as_strided_(inp, size, stride)
    else:
        argument = None if offset_form == "none" else offset
        ref_out = torch.ops.aten.as_strided_(ref_inp, size, stride, argument)
        res_out = flag_gems.as_strided_(inp, size, stride, argument)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_matches(res_out, ref_out, inp, storage_before)
    # The native offset contract, asserted directly: omission and None keep the
    # operand's own offset, an explicit value replaces it.
    expected_offset = original_offset if offset_form in ("omitted", "none") else offset
    assert res_out.storage_offset() == expected_offset
    # Only the operand's metadata may change; its source buffer keeps its bytes.
    tu.assert_result_equal(_storage_witness(base), storage_values)


CONJ_CASES = _CONJ_DTYPES


@pytest.mark.as_strided_
@pytest.mark.parametrize("dtype", CONJ_CASES)
def test_as_strided__conjugate_view(dtype):
    # Rewriting the metadata of a lazy conjugate view must keep the <=> marker
    # that the view carries.
    base = tu.make_input(dtype, (4, 6), ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base.conj()
    ref_inp = ref_base.conj()
    storage_before = inp.untyped_storage().data_ptr()

    ref_out = torch.ops.aten.as_strided_(ref_inp, [4, 6], [6, 1], 0)
    res_out = flag_gems.as_strided_(inp, [4, 6], [6, 1], 0)

    assert res_out.is_conj() is True
    tu.assert_result_equal(res_out, ref_out)
    _assert_view_matches(res_out, ref_out, inp, storage_before)


SPECIAL_VALUE_CASES = tu.selected_cases(
    tu.special_value_cases(_AS_STRIDED_DTYPES), quick=[]
)


@pytest.mark.as_strided_
@pytest.mark.parametrize("dtype,scenario", SPECIAL_VALUE_CASES)
def test_as_strided__special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    size, stride, offset = _contiguous_view(inp.shape)
    ref_inp = tu.to_reference(inp)
    storage_before = inp.untyped_storage().data_ptr()

    ref_out = torch.ops.aten.as_strided_(ref_inp, size, stride, offset)
    res_out = flag_gems.as_strided_(inp, size, stride, offset)

    # NaN/Inf payloads must survive the metadata rewrite unchanged.
    tu.assert_result_equal(res_out, ref_out)
    _assert_view_matches(res_out, ref_out, inp, storage_before)


# Differentiable rows: identity, transposed metadata, rank-reducing flatten and
# two self-overlapping views (dilated with a non-zero offset, stride 0).
_BACKWARD_ROWS = [
    ((4, 6), [4, 6], [6, 1], 0, False),
    ((4, 6), [6, 4], [1, 6], 0, False),
    ((4, 6), [24], [1], 0, False),
    ((4, 6), [4, 6], [2, 3], 1, True),
    ((4, 6), [4, 4], [0, 1], 0, True),
]
BACKWARD_CASES = tu.selected_cases(
    [
        (shape, size, stride, offset, dtype)
        for shape, size, stride, offset, overlaps in _BACKWARD_ROWS
        for dtype in _GRAD_DTYPES + _FP8_GRAD_DTYPES
        # The accumulating gradient path of the overlapping rows dispatches to
        # index_add / sum, which have no FP8 kernel on this backend ("index_add
        # not implemented for 'Float8_e4m3fn'"), so FP8 keeps the
        # non-overlapping rows.
        if not (overlaps and dtype in _FP8_GRAD_DTYPES)
    ],
    quick=[],
)


@pytest.mark.as_strided_
@pytest.mark.parametrize("shape,size,stride,offset,dtype", BACKWARD_CASES)
def test_as_strided__backward(shape, size, stride, offset, dtype):
    # An in-place operator rejects a leaf requiring grad (and a view of one), so
    # the mutated operand is the non-leaf clone below while the differentiated
    # operand is the independent source leaf it was cloned from.
    src = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_src = tu.to_reference(src.detach()).requires_grad_(True)
    inp = src.clone()
    ref_inp = ref_src.clone()
    upstream = tu.make_input(dtype, tuple(size), ["-1", "1"])
    # The reference operands live on the configured reference device, which
    # --ref cpu moves to the host, so the reference needs its own upstream.
    ref_upstream = tu.to_reference(upstream)
    storage_before = inp.untyped_storage().data_ptr()

    ref_out = torch.ops.aten.as_strided_(ref_inp, size, stride, offset)
    res_out = flag_gems.as_strided_(inp, size, stride, offset)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_matches(res_out, ref_out, inp, storage_before)

    # The native operator re-installs the operand as a strided view of its
    # source, so the gradient reaching ``src`` scatters the upstream gradient
    # through the new strides and accumulates where the view overlaps itself.
    ref_grad = torch.autograd.grad(ref_out, ref_src, grad_outputs=ref_upstream)[0]
    res_grad = torch.autograd.grad(res_out, src, grad_outputs=upstream)[0]

    tu.assert_result_close(res_grad, ref_grad)


_OUT_OF_BOUNDS_ROWS = [
    ((4, 6), [4, 6], [7, 1], 0),
    ((4, 6), [4, 6], [1, 6], 0),
    ((4, 6), [4, 6], [6, 1], 4),
]


@pytest.mark.as_strided_
@pytest.mark.parametrize("shape,size,stride,offset", _OUT_OF_BOUNDS_ROWS)
def test_as_strided__rejects_out_of_bounds(shape, size, stride, offset):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.as_strided_(inp, size, stride, offset)
    # A rejected call must leave the operand untouched.
    assert tuple(inp.shape) == tuple(shape)
    assert inp.stride() == tuple(_contiguous_view(shape)[1])
    assert inp.storage_offset() == 0


@pytest.mark.as_strided_
@pytest.mark.parametrize("size,stride", [([4, 6], [-6, 1]), ([6, 4], [1, -6])])
def test_as_strided__rejects_negative_stride(size, stride):
    inp = tu.make_input(torch.float32, (4, 6), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.as_strided_(inp, size, stride, 0)


# Both rows really are rank mismatches: the length of ``size`` differs from the
# length of ``stride``. Equal-length triples such as ([4, 6, 1], [6, 1, 1]) are
# a valid in-bounds view and must not be used here.
@pytest.mark.as_strided_
@pytest.mark.parametrize("size,stride", [([4, 6], [6]), ([4], [6, 1])])
def test_as_strided__rejects_rank_mismatch(size, stride):
    inp = tu.make_input(torch.float32, (4, 6), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.as_strided_(inp, size, stride, 0)


@pytest.mark.as_strided_
def test_as_strided__rejects_negative_offset():
    inp = tu.make_input(torch.float32, (4, 6), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.as_strided_(inp, [4, 6], [6, 1], -1)


@pytest.mark.as_strided_
@pytest.mark.parametrize("bad_self", [3.14, [[0.0, 1.0]]])
def test_as_strided__rejects_non_tensor(bad_self):
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.as_strided_(bad_self, [1], [1], 0)


@pytest.mark.as_strided_
@pytest.mark.parametrize("size,stride", [(6, 1), ([4.0, 6.0], [6, 1])])
def test_as_strided__rejects_non_integer_sequence(size, stride):
    # The schema takes List[int] for size/stride: there is no scalar-size
    # overload, and float extents are rejected.
    inp = tu.make_input(torch.float32, (4, 6), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.as_strided_(inp, size, stride, 0)
