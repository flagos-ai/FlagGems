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

"""Correctness tests for both ``aten::view`` overloads.

A view moves no element, so every value comparison here is exact (zero
olerance) and the facts a value comparison cannot see -- strides, storage
offset, storage sharing, view identity and the lazy conjugate/negative bit --
are matched against a native reference built on an independent operand.

Both overloads are reached through the single public candidate
``flag_gems.view``:

* ``view(Tensor self, SymInt[] size)`` is a ``Tensor(a)`` view: the result
aliases ``self`` and ``_is_view()`` is True.
* ``view.dtype(Tensor self, ScalarType dtype)`` reinterprets the trailing
dimension in element units: the same storage, but no autograd view metadata
(``_is_view()`` is False, ``_base`` is None and ``requires_grad`` stays False
even for a grad-requiring input).

Probed native argument rules: a tuple, a list or ``size=`` names the target
shape, while a bare int is the dtype overload's ScalarType id (``view(x, 24)``
reinterprets and ``view(x, 100)`` aborts the native library), so the size
workloads pass sequences only and the single bare int exercised is the ``-1``
that the native call rejects. The dtype overload requires ``dim() >= 1``,
``stride(-1) == 1``, both ``shape[-1] * itemsize(self)`` and
``storage_offset()`` divisible by the element-size ratio, and no lazy
conjugate or negative bit.
"""

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Capability flags read at import time: collection allocates no tensor and runs
# no operator. A type outside this map is baseline storage every backend has.
_DTYPE_CAPABILITY = {
    torch.float64: utils.fp64_is_supported,
    torch.complex128: utils.fp64_is_supported,
    torch.bfloat16: utils.bf16_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
    torch.int64: utils.int64_is_supported,
}


def _dtype_supported(dtype):
    flag = _DTYPE_CAPABILITY.get(dtype)
    return True if flag is None else bool(flag)


# A size view rewrites metadata only, so it accepts every dtype the backend can
# store: the required nine plus bool/complex64 and the fp64-backed types.
_VIEW_DTYPES = [
    dtype
    for dtype in tu.REQUIRED_DTYPES
    + [torch.bool, torch.complex64, torch.float64, torch.complex128]
    if _dtype_supported(dtype)
]

# Only a floating source can carry a gradient or the lazy negative bit, and
# only a complex source can carry the lazy conjugate bit.
_FLOAT_VIEW_DTYPES = [
    dtype
    for dtype in (torch.float16, torch.float32, torch.bfloat16, torch.float64)
    if _dtype_supported(dtype)
]
_COMPLEX_VIEW_DTYPES = [torch.complex64]

# Element sizes are listed explicitly so that collection allocates no tensor.
_ITEM_SIZE = {
    torch.bool: 1,
    torch.uint8: 1,
    torch.int8: 1,
    torch.float8_e4m3fn: 1,
    torch.float8_e5m2: 1,
    torch.int16: 2,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int32: 4,
    torch.float32: 4,
    torch.int64: 8,
    torch.float64: 8,
    torch.complex64: 8,
    torch.complex128: 16,
}

# One row per element-size class transition: splits (float32 -> uint8 gives
# (4, 24)), identity-size reinterpretations and merges (float32 -> float64
# gives (4, 3)). Every row was validated against the native overload.
_DTYPE_PAIRS = [
    (torch.float32, torch.uint8),
    (torch.float32, torch.int8),
    (torch.float32, torch.float8_e4m3fn),
    (torch.float32, torch.float8_e5m2),
    (torch.float32, torch.float64),
    (torch.float32, torch.complex64),
    (torch.float16, torch.int8),
    (torch.float16, torch.float8_e4m3fn),
    (torch.float16, torch.bfloat16),
    (torch.bfloat16, torch.int8),
    (torch.bfloat16, torch.float8_e5m2),
    (torch.bool, torch.int8),
    (torch.uint8, torch.int8),
    (torch.uint8, torch.float8_e4m3fn),
    (torch.int8, torch.uint8),
    (torch.int8, torch.float32),
    (torch.int16, torch.int32),
    (torch.int16, torch.float32),
    (torch.int32, torch.float32),
    (torch.int32, torch.int64),
    (torch.int64, torch.float64),
    (torch.int64, torch.float32),
    (torch.float64, torch.float32),
    (torch.float64, torch.int32),
    (torch.complex64, torch.float32),
    (torch.complex64, torch.float64),
]


def _dtype_rows():
    # Rank-0 inputs are skipped: the dtype overload has no trailing dimension
    # to reinterpret. A pair is kept exactly when the element-size ratio
    # divides the trailing dimension, which is the native condition, so no
    # valid combination is dropped and no invalid one is collected.
    rows = []
    for shape in tu.selected_shapes():
        if not shape:
            continue
        for src, dst in _DTYPE_PAIRS:
            if not (_dtype_supported(src) and _dtype_supported(dst)):
                continue
            if shape[-1] * _ITEM_SIZE[src] % _ITEM_SIZE[dst] == 0:
                rows.append((shape, src, dst))
    return rows


_DTYPE_ROWS = _dtype_rows()

# Numel-preserving size targets. A contiguous input accepts any of them, so the
# result stays a view; the quick shape is listed as well so both modes resolve.
_TARGET_SHAPES = {
    (): (),
    (1,): (1, 1),
    (256,): (16, 16),
    (1024, 1024): (512, 2048),
    (20, 320, 15): (320, 300),
    (16, 128, 64, 60): (16, 128, 60, 64),
    (16, 7, 57, 32, 29): (16, 7, 57, 928),
    (2, 19, 7): (7, 38),
}


def _size_target(shape):
    # A listed row, otherwise the flat form, which a contiguous input accepts.
    return _TARGET_SHAPES.get(tuple(shape), (math.prod(shape),))


# Size targets on non-contiguous, offset and stride-0 operands. A target is
# accepted whenever the result is still stride-expressible, which can be
# reached by splitting the innermost dimension as well as by keeping the
# input's own shape; each row below is a probed native acceptance and the
# rejected targets live in _UNVIEWABLE_SIZE_ROWS.
_STRIDED_SIZE_ROWS = [
    ((4, 6), "transposed", None, (6, 4)),
    ((8, 8), "window", None, (2, 2, 4)),
    ((8, 8), "window", None, (4, 4, 1)),
    ((4, 12), "column_step", None, (2, 12)),
    ((4, 12), "column_step", None, (8, 3)),
    ((4, 12), "column_step", None, (24,)),
    ((12,), "step", None, (3, 2)),
    ((12,), "step", None, (2, 3)),
    ((12,), "step", None, (1, 6)),
    ((1, 4, 6), "expanded", (3, 4, 6), (3, 24)),
]

# Layouts whose target is not stride-expressible for the size form: the native
# operator raises "view size is not compatible with input tensor's size and
# stride", and so must the candidate.
_UNVIEWABLE_SIZE_ROWS = [
    ((4, 6), "transposed", None, (24,)),
    ((4, 6), "transposed", None, (2, 12)),
    ((8, 8), "window", None, (16,)),
    ((8, 8), "window", None, (2, 8)),
    ((1, 4, 6), "expanded", (3, 4, 6), (72,)),
]

# Concrete sequence structs accepted by the size overload, including the 0-d,
# empty, zero-extent and inferred-extent boundaries. A bare int is deliberately
# absent: it binds to the dtype overload instead of naming a size.
_SIZE_FORM_ROWS = [
    ((4, 6), (3, 8)),
    ((4, 6), [3, 8]),
    ((4, 6), (-1, 3, 4)),
    ((4, 6), (-1,)),
    ((), (1,)),
    ((), ()),
    ((2, 12), (24,)),
    ((3, 5, 2), (5, 3, 2)),
    ((4, 0, 6), (0,)),
    ((4, 6), (3, 2, 4)),
]

# Dtype-form workloads on non-contiguous layouts that the native overload
# accepts: a row window with a unit last stride and storage offset 17, and
# stride-0 expanded inputs.
_DTYPE_STRIDED_ROWS = [
    row
    for row in (
        ((8, 8), "window", None, torch.int8),
        ((1, 4, 6), "expanded", (3, 4, 6), torch.int8),
        ((1, 4, 6), "expanded", (3, 4, 6), torch.float64),
    )
    if _dtype_supported(row[3])
]

# Dtype-form inputs the native overload rejects on the active backend: an odd
# trailing dimension, a column slice whose last stride is 2, a row window whose
# storage offset is not divisible by the size ratio, and a rank-0 input.
_DTYPE_INVALID_ROWS = [
    row
    for row in (
        ((4, 5), "asis", None, torch.float64),
        ((4, 12), "column_step", None, torch.uint8),
        ((8, 8), "window", None, torch.float64),
        ((), "asis", None, torch.uint8),
    )
    if _dtype_supported(row[3])
]

_MUTATION_SIZE_ROWS = [
    ((4, 6), (2, 12)),
    ((3, 5, 2), (15, 2)),
    ((24,), (4, 6)),
]

_SHARED_STORAGE_PAIRS = [
    pair
    for pair in (
        (torch.float32, torch.uint8),
        (torch.int16, torch.int32),
        (torch.float32, torch.float64),
    )
    if _dtype_supported(pair[0]) and _dtype_supported(pair[1])
]

# Only a floating source can carry a gradient flag, so the non-autograd
# property of the dtype overload is asserted with grad-requiring float inputs.
_NO_GRAD_PAIRS = [
    pair
    for pair in (
        (torch.float32, torch.uint8),
        (torch.float32, torch.float64),
        (torch.float16, torch.uint8),
    )
    if _dtype_supported(pair[0]) and _dtype_supported(pair[1])
]

# Backward is default-only.
_BACKWARD_CASES = tu.selected_cases(
    [
        ((4, 6), (2, 12)),
        ((24,), (2, 3, 4)),
        ((7, 13, 29), (7, 377)),
        ((20, 320, 15), (320, 300)),
    ],
    quick=[],
)

_NUMEL_MISMATCH_ROWS = [
    ((4, 6), (49,)),
    ((3,), (2, 2)),
    ((24,), (4, 7)),
    ((0, 3), (4,)),
]

# Positive special values are default-only.
_SPECIAL_VALUE_CASES = tu.selected_cases(tu.special_value_cases(_VIEW_DTYPES), quick=[])


def _apply_layout(tensor, layout, expand_shape=None):
    if layout == "transposed":
        return tensor.t()
    if layout == "column_step":
        return tensor[:, ::2]
    if layout == "step":
        return tensor[::2]
    if layout == "window":
        return tensor[2:6, 1:5]
    if layout == "expanded":
        return tensor.expand(tuple(expand_shape))
    return tensor


def _layout_operands(storage_shape, layout, dtype, expand_shape=None):
    base = tu.make_input(dtype, storage_shape, ["-1", "1"])
    ref_base = tu.to_reference(base.detach())
    inp = _apply_layout(base, layout, expand_shape)
    ref_inp = _apply_layout(ref_base, layout, expand_shape)
    return inp, ref_inp, base, ref_base


def _assert_view_facts(res_out, ref_out, inp, ref_inp):
    # Layout, alias and autograd facts a value comparison cannot see. Both
    # overloads alias the input and return a fresh object exactly when the
    # native call does, so those relations are compared instead of assumed.
    assert tuple(res_out.stride()) == tuple(ref_out.stride())
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
    assert (res_out is inp) == (ref_out is ref_inp)
    # The size overload is a Tensor(a) view; the dtype overload shares storage
    # without view metadata.
    assert res_out._is_view() == ref_out._is_view()
    assert (res_out._base is None) == (ref_out._base is None)
    assert res_out.requires_grad == ref_out.requires_grad
    # The candidate must leave the operand's own metadata untouched.
    assert tuple(inp.size()) == tuple(ref_inp.size())
    assert tuple(inp.stride()) == tuple(ref_inp.stride())
    assert inp.storage_offset() == ref_inp.storage_offset()


def _assert_operands_unchanged(inp, ref_inp, base, ref_base):
    tu.assert_result_equal(inp, ref_inp)
    if base is not inp:
        tu.assert_result_equal(base, ref_base)


@pytest.mark.view
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VIEW_DTYPES)
def test_view_size(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    target = _size_target(shape)

    ref_out = torch.ops.aten.view(ref_inp, target)
    res_out = flag_gems.view(inp, target)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_facts(res_out, ref_out, inp, ref_inp)


@pytest.mark.view
@pytest.mark.parametrize("shape,target", _SIZE_FORM_ROWS)
def test_view_size_call_forms(shape, target):
    # Tuple, list, inferred extent, 0-d, empty and rank-changing targets all
    # reach the same public entry point.
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.view(ref_inp, target)
    res_out = flag_gems.view(inp, target)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_facts(res_out, ref_out, inp, ref_inp)


@pytest.mark.view
def test_view_size_keyword():
    inp = tu.make_input(torch.float32, (4, 6), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.view(ref_inp, size=(3, 8))
    res_out = flag_gems.view(inp, size=(3, 8))

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_facts(res_out, ref_out, inp, ref_inp)


@pytest.mark.view
@pytest.mark.parametrize("storage_shape,layout,expand_shape,target", _STRIDED_SIZE_ROWS)
def test_view_size_strided_input(storage_shape, layout, expand_shape, target):
    # Non-contiguous, offset and stride-0 operands: the result keeps the native
    # strides and storage offset instead of compacting them.
    inp, ref_inp, base, ref_base = _layout_operands(
        storage_shape, layout, torch.float32, expand_shape
    )

    ref_out = torch.ops.aten.view(ref_inp, target)
    res_out = flag_gems.view(inp, target)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_facts(res_out, ref_out, inp, ref_inp)
    _assert_operands_unchanged(inp, ref_inp, base, ref_base)


@pytest.mark.view
@pytest.mark.parametrize("dtype", _COMPLEX_VIEW_DTYPES)
def test_view_size_preserves_conjugate_bit(dtype):
    # A lazily conjugated input is still viewable by the size overload: the
    # flag has to survive on the result, which keeps aliasing the input. The
    # reference is flagged after its own transfer so the lazy bit is present on
    # both sides. The dtype overload rejects the same input instead.
    base = tu.make_input(dtype, (2, 3), ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp, ref_inp = base.conj(), ref_base.conj()

    ref_out = torch.ops.aten.view(ref_inp, (3, 2))
    res_out = flag_gems.view(inp, (3, 2))

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_facts(res_out, ref_out, inp, ref_inp)
    assert res_out.is_conj() is True
    assert res_out.is_conj() == ref_out.is_conj()


@pytest.mark.view
@pytest.mark.parametrize("dtype", _FLOAT_VIEW_DTYPES)
def test_view_size_preserves_negative_bit(dtype):
    # Same rule for the lazy negative bit: torch._neg_view only flips a flag and
    # the size overload must keep it on the aliasing result.
    base = tu.make_input(dtype, (2, 3), ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp, ref_inp = torch._neg_view(base), torch._neg_view(ref_base)

    ref_out = torch.ops.aten.view(ref_inp, (3, 2))
    res_out = flag_gems.view(inp, (3, 2))

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_facts(res_out, ref_out, inp, ref_inp)
    assert res_out.is_neg() is True
    assert res_out.is_neg() == ref_out.is_neg()


@pytest.mark.view
@pytest.mark.parametrize("shape,src,dst", _DTYPE_ROWS)
def test_view_dtype(shape, src, dst):
    inp = tu.make_input(src, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.view(ref_inp, dst)
    res_out = flag_gems.view(inp, dst)

    # The reinterpretation moves no element, so the stored bits are compared
    # exactly and the layout facts are matched separately.
    tu.assert_result_equal(res_out, ref_out)
    _assert_view_facts(res_out, ref_out, inp, ref_inp)
    _assert_operands_unchanged(inp, ref_inp, inp, ref_inp)


@pytest.mark.view
def test_view_dtype_keyword():
    inp = tu.make_input(torch.float32, (4, 6), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.view(ref_inp, dtype=torch.uint8)
    res_out = flag_gems.view(inp, dtype=torch.uint8)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_facts(res_out, ref_out, inp, ref_inp)


@pytest.mark.view
@pytest.mark.parametrize("storage_shape,layout,expand_shape,dst", _DTYPE_STRIDED_ROWS)
def test_view_dtype_strided_input(storage_shape, layout, expand_shape, dst):
    inp, ref_inp, base, ref_base = _layout_operands(
        storage_shape, layout, torch.float32, expand_shape
    )

    ref_out = torch.ops.aten.view(ref_inp, dst)
    res_out = flag_gems.view(inp, dst)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_facts(res_out, ref_out, inp, ref_inp)
    _assert_operands_unchanged(inp, ref_inp, base, ref_base)


@pytest.mark.view
@pytest.mark.parametrize("src,dst", _NO_GRAD_PAIRS)
def test_view_dtype_is_not_an_autograd_view(src, dst):
    # The dtype overload shares storage but stays out of the autograd graph, so
    # a grad-requiring floating input must not produce a requiring-grad result.
    inp = tu.make_input(src, (4, 8), ["-1", "1"]).requires_grad_(True)
    ref_inp = tu.to_reference(inp.detach()).requires_grad_(True)

    ref_out = torch.ops.aten.view(ref_inp, dst)
    res_out = flag_gems.view(inp, dst)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_facts(res_out, ref_out, inp, ref_inp)
    assert res_out.requires_grad is False
    assert res_out._is_view() is False
    assert res_out._base is None
    _assert_operands_unchanged(inp, ref_inp, inp, ref_inp)


@pytest.mark.view
@pytest.mark.parametrize("shape,target", _MUTATION_SIZE_ROWS)
def test_view_size_write_through(shape, target):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.view(ref_inp, target)
    res_out = flag_gems.view(inp, target)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_facts(res_out, ref_out, inp, ref_inp)

    # A Tensor(a) view writes through to its input.
    res_out.fill_(3.0)
    ref_out.fill_(3.0)
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.view
@pytest.mark.parametrize("src,dst", _SHARED_STORAGE_PAIRS)
def test_view_dtype_shares_storage(src, dst):
    inp = tu.make_input(src, (4, 8), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.view(ref_inp, dst)
    res_out = flag_gems.view(inp, dst)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_facts(res_out, ref_out, inp, ref_inp)

    # Sharing storage without view metadata still means a write through the
    # reinterpreted result reaches the input.
    res_out.fill_(7)
    ref_out.fill_(7)
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.view
@pytest.mark.parametrize("shape,target", _BACKWARD_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_VIEW_DTYPES)
def test_view_size_backward(shape, target, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_inp = tu.to_reference(inp.detach()).requires_grad_(True)
    upstream = tu.make_input(dtype, target, ["-1", "1"])
    ref_upstream = tu.to_reference(upstream.detach())

    ref_out = torch.ops.aten.view(ref_inp, target)
    res_out = flag_gems.view(inp, target)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_facts(res_out, ref_out, inp, ref_inp)
    assert res_out.requires_grad

    # A view gradient is a pure relayout of a non-uniform upstream, so it is
    # compared exactly rather than within a tolerance.
    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]
    ref_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)[0]
    tu.assert_result_equal(res_grad, ref_grad)
    _assert_operands_unchanged(inp, ref_inp, inp, ref_inp)


@pytest.mark.view
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_VALUE_CASES)
def test_view_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)
    target = (1, inp.numel())

    ref_out = torch.ops.aten.view(ref_inp, target)
    res_out = flag_gems.view(inp, target)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_facts(res_out, ref_out, inp, ref_inp)
    _assert_operands_unchanged(inp, ref_inp, inp, ref_inp)


@pytest.mark.view
@pytest.mark.parametrize("shape,target", _NUMEL_MISMATCH_ROWS)
def test_view_size_numel_mismatch(shape, target):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.view(inp, target)


@pytest.mark.view
def test_view_size_two_inferred_dims():
    inp = tu.make_input(torch.float32, (4, 6), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.view(inp, (-1, -1))


@pytest.mark.view
def test_view_size_rejects_bare_negative_extent():
    # A bare int binds to the dtype overload, not to a size: the native operator
    # reads it as a ScalarType id (view(x, 24) reinterprets as float8_e4m3fn,
    # view(x, 0) as a byte view, view(x, 100) aborts the library). Only a
    # negative bare int is rejected, and that is the workload asserted here.
    inp = tu.make_input(torch.float32, (4, 6), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.view(inp, -1)


@pytest.mark.view
def test_view_rejects_non_tensor_self():
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.view([[0.0] * 6] * 4, (4, 6))


@pytest.mark.view
def test_view_rejects_non_tensor_size():
    inp = tu.make_input(torch.float32, (4, 6), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.view(inp, "3x8")


@pytest.mark.view
@pytest.mark.parametrize(
    "storage_shape,layout,expand_shape,target", _UNVIEWABLE_SIZE_ROWS
)
def test_view_size_rejects_unviewable_layout(
    storage_shape, layout, expand_shape, target
):
    inp = _apply_layout(
        tu.make_input(torch.float32, storage_shape, ["-1", "1"]), layout, expand_shape
    )

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.view(inp, target)


@pytest.mark.view
@pytest.mark.parametrize("storage_shape,layout,expand_shape,dst", _DTYPE_INVALID_ROWS)
def test_view_dtype_rejects_invalid(storage_shape, layout, expand_shape, dst):
    inp = _apply_layout(
        tu.make_input(torch.float32, storage_shape, ["-1", "1"]), layout, expand_shape
    )

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.view(inp, dst)


@pytest.mark.view
@pytest.mark.parametrize("dtype", _COMPLEX_VIEW_DTYPES)
def test_view_dtype_rejects_conjugate_input(dtype):
    # Converting to a different dtype is the case the lazy conjugate bit blocks:
    # the trailing size is divisible, so only the flag can raise here.
    inp = tu.make_input(dtype, (2, 3), ["-1", "1"]).conj()

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.view(inp, torch.uint8)


@pytest.mark.view
def test_view_dtype_rejects_negative_bit_input():
    # Same rule for the lazy negative bit: torch._neg_view keeps the storage
    # untouched and only flips the flag, which the dtype conversion rejects.
    inp = torch._neg_view(tu.make_input(torch.float32, (2, 3), ["-1", "1"]))

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.view(inp, torch.int8)
