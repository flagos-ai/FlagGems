# Copyright 2026, The FlagOS Contributors.
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

# aten::swapaxes_(Tensor(a!) self, int axis0, int axis1) -> Tensor(a!)
# In-place view: two axes exchange strides and ``self`` is returned; no element
# moves and no storage is allocated. Checked contract: identity, shape/stride/
# offset, lazy conj/neg flags, storage retention, write-through and the shared
# value comparison. The operator is unary, so no broadcast dimension applies.


def _axis_pairs(shape):
    # Axis pairs in range for ``shape``. A 0-D or 1-D tensor only accepts
    # (0, 0): any other pair raises IndexError natively.
    rank = len(shape)
    if rank < 2:
        return ((0, 0),)
    pairs = {(0, rank - 1), (rank - 2, rank - 1)}
    if rank >= 3:
        pairs.add((0, 1))
    if rank >= 4:
        pairs.add((1, rank - 2))
    return tuple(sorted(pairs))


def _primary_pair(shape):
    return (0, 0) if len(shape) < 2 else (0, len(shape) - 1)


_BASE_DTYPES = list(tu.REQUIRED_DTYPES)
_BASE_DTYPES = [dtype for dtype in _BASE_DTYPES if _DTYPE_FLAGS.get(dtype, True)]
_EXTRA_DTYPES = [torch.bool, torch.complex64]
if utils.fp64_is_supported:
    _EXTRA_DTYPES.append(torch.float64)
_DTYPES = _BASE_DTYPES + _EXTRA_DTYPES
_DTYPES = [dtype for dtype in _DTYPES if _DTYPE_FLAGS.get(dtype, True)]

# One row per (spec shape, in-range axis pair, dtype): every rank covered by the
# spec shape set is exercised with each axis pair it accepts, so the stride
# permutation itself is checked, not only an axis-pair-free no-op.
_CASES = [
    (tuple(shape), axis0, axis1, dtype)
    for shape in tu.selected_shapes()
    for axis0, axis1 in _axis_pairs(shape)
    for dtype in _BASE_DTYPES
] + [
    (tuple(shape), _primary_pair(shape)[0], _primary_pair(shape)[1], dtype)
    for shape in tu.selected_shapes()
    for dtype in _EXTRA_DTYPES
]
# Empty tensor: the permuted stride metadata still has to be reported.
_CASES.append(((0, 3, 4), 0, 2, torch.float32))

_GRID = tu.selected_cases(
    _CASES,
    quick=[((2, 19, 7), 0, 2, dtype) for dtype in _DTYPES],
)


@pytest.mark.swapaxes_
@pytest.mark.parametrize("shape,axis0,axis1,dtype", _GRID)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_swapaxes_(shape, axis0, axis1, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    storage = inp.untyped_storage().data_ptr()
    nbytes = inp.untyped_storage().nbytes()
    base = inp._base

    ref_version, version = ref_inp._version, inp._version
    ref_out = torch.ops.aten.swapaxes_(ref_inp, axis0, axis1)
    res_out = flag_gems.swapaxes_(inp, axis0, axis1)
    assert inp._version - version == ref_inp._version - ref_version

    # In place: the candidate returns its own input and keeps the original
    # allocation (same data pointer, byte count and base tensor).
    assert res_out is inp
    assert inp.untyped_storage().data_ptr() == storage
    assert inp.untyped_storage().nbytes() == nbytes
    assert inp._base is base
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.is_neg() == ref_out.is_neg()
    tu.assert_result_equal(res_out, ref_out)
    # The input itself carries the swapped metadata, not a copy of it.
    assert inp.shape == ref_inp.shape
    assert inp.stride() == ref_inp.stride()
    assert inp.storage_offset() == ref_inp.storage_offset()
    tu.assert_result_equal(inp, ref_inp)


def _layout_input(layout, dtype):
    # Non-default layouts whose stored values survive the metadata swap.
    if layout == "slice":  # non-zero offset and non-contiguous strides
        return tu.make_input(dtype, (6, 8), ["-1", "1"])[1:4, ::2]
    if layout == "transposed":  # already-swapped strides
        return tu.make_input(dtype, (3, 5), ["-1", "1"]).t()
    if layout == "expanded":  # zero stride
        return tu.make_input(dtype, (1, 6), ["-1", "1"]).expand(4, 6)
    if layout == "conj":  # lazy conjugate bit
        return tu.make_input(dtype, (3, 4), ["-1", "1"]).conj()
    if layout == "neg":  # lazy negation bit
        return torch._neg_view(tu.make_input(dtype, (3, 4), ["-1", "1"]))
    raise AssertionError(f"unknown layout {layout!r}")


# All six rows are tiny, so the whole group runs in both the quick and the
# default suite: dropping the lazy-bit and offset layouts from quick would hide
# view-composition semantics from the smoke run.
_LAYOUTS = [
    ("slice", torch.float32),
    ("slice", torch.int8),
    ("transposed", torch.float16),
    ("expanded", torch.float32),
    ("conj", torch.complex64),
    ("neg", torch.float32),
]


@pytest.mark.swapaxes_
@pytest.mark.parametrize("layout,dtype", _LAYOUTS)
def test_swapaxes__layout(layout, dtype):
    inp = _layout_input(layout, dtype)
    ref_inp = tu.to_reference(inp)
    storage = inp.untyped_storage().data_ptr()
    offset = inp.storage_offset()

    ref_out = torch.ops.aten.swapaxes_(ref_inp, 0, 1)
    res_out = flag_gems.swapaxes_(inp, 0, 1)

    assert res_out is inp
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert inp.untyped_storage().data_ptr() == storage
    assert res_out.storage_offset() == offset
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.is_neg() == ref_out.is_neg()
    tu.assert_result_equal(res_out, ref_out)


_MUTATION = tu.selected_cases(
    [((2, 3, 4), 0, 2), ((20, 320, 15), 0, 1)],
    quick=[((2, 19, 7), 0, 2)],
)


@pytest.mark.swapaxes_
@pytest.mark.parametrize("shape,axis0,axis1", _MUTATION)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _BASE_DTYPES)
def test_swapaxes__write_through(shape, axis0, axis1, value_range, dtype):
    # The returned view aliases the original storage, so writes through it are
    # visible on the input tensor.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    storage = inp.untyped_storage().data_ptr()

    ref_version, version = ref_inp._version, inp._version
    ref_out = torch.ops.aten.swapaxes_(ref_inp, axis0, axis1)
    res_out = flag_gems.swapaxes_(inp, axis0, axis1)
    assert inp._version - version == ref_inp._version - ref_version

    tu.assert_result_equal(res_out, ref_out)

    # The swap permutes the shape, so the patch follows the swapped result.
    patch = tu.make_input(dtype, tuple(res_out.shape), ["0", "1"])
    res_out.copy_(patch)
    ref_out.copy_(tu.to_reference(patch))

    assert inp.untyped_storage().data_ptr() == storage
    assert inp.shape == ref_inp.shape
    assert inp.stride() == ref_inp.stride()
    tu.assert_result_equal(inp, ref_inp)


_ROUNDTRIP = tu.selected_cases(
    [((2, 3, 4), 0, 2), ((3, 5), 0, 1), ((20, 320, 15), 1, 2)],
    quick=[((2, 19, 7), 0, 2)],
)
_ROUNDTRIP_DTYPES = [torch.float32, torch.int8, torch.bool]


@pytest.mark.swapaxes_
@pytest.mark.parametrize("shape,axis0,axis1", _ROUNDTRIP)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _ROUNDTRIP_DTYPES)
def test_swapaxes__roundtrip(shape, axis0, axis1, value_range, dtype):
    # Swapping the same pair twice restores the original layout and values.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    original_shape = inp.shape
    original_stride = inp.stride()
    original_offset = inp.storage_offset()

    ref_mid = torch.ops.aten.swapaxes_(ref_inp, axis0, axis1)
    res_mid = flag_gems.swapaxes_(inp, axis0, axis1)
    ref_out = torch.ops.aten.swapaxes_(ref_mid, axis1, axis0)
    res_out = flag_gems.swapaxes_(res_mid, axis1, axis0)

    assert res_mid is inp
    assert res_out is inp
    assert res_out.shape == ref_out.shape
    assert res_out.shape == original_shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.stride() == original_stride
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.storage_offset() == original_offset
    tu.assert_result_equal(res_out, ref_out)


# Axis sweep: positive, negative, equal-axis ``i == i`` no-ops and the scalar /
# empty rank boundaries. The quick subset keeps one row per semantic form
# instead of collapsing the group to a single positive pair.
_AXIS_ROWS = tu.selected_cases(
    [
        ((16, 128, 64, 60), 1, 2),
        ((16, 128, 64, 60), 3, 0),
        ((16, 128, 64, 60), -1, 0),
        ((16, 128, 64, 60), -4, -2),
        ((16, 128, 64, 60), 0, 0),
        ((16, 128, 64, 60), 1, 1),
        ((16, 128, 64, 60), 0, 2),
        ((2, 19, 7), 0, 2),
        ((2, 19, 7), -1, 0),
        ((2, 19, 7), 1, 1),
        ((), 0, 0),
        ((4,), 0, 0),
        ((0, 3, 4), 0, 2),
    ],
    quick=[
        ((2, 19, 7), 0, 2),
        ((2, 19, 7), -1, 0),
        ((2, 19, 7), 1, 1),
        ((), 0, 0),
        ((0, 3, 4), 0, 2),
    ],
)
_AXIS_DTYPES = [torch.float32, torch.int32]


@pytest.mark.swapaxes_
@pytest.mark.parametrize("shape,axis0,axis1", _AXIS_ROWS)
@pytest.mark.parametrize("dtype", _AXIS_DTYPES)
def test_swapaxes__axis_values(shape, axis0, axis1, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_version, version = ref_inp._version, inp._version
    ref_out = torch.ops.aten.swapaxes_(ref_inp, axis0, axis1)
    res_out = flag_gems.swapaxes_(inp, axis0, axis1)
    assert inp._version - version == ref_inp._version - ref_version

    assert res_out is inp
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    tu.assert_result_equal(res_out, ref_out)


_SPECIAL_DTYPES = [dtype for dtype in _DTYPES if dtype.is_floating_point]
_SPECIAL = tu.selected_cases(
    [
        (dtype, scenario, shape)
        for dtype, scenario in tu.special_value_cases(_SPECIAL_DTYPES)
        for shape in ((5, 1), (1, 1, 5))
    ],
    quick=[],
)


@pytest.mark.swapaxes_
@pytest.mark.parametrize("dtype,scenario,shape", _SPECIAL)
def test_swapaxes__special_values(dtype, scenario, shape):
    inp = tu.make_special_input(dtype, scenario).reshape(shape)
    ref_inp = tu.to_reference(inp)
    axis1 = len(shape) - 1

    ref_out = torch.ops.aten.swapaxes_(ref_inp, 0, axis1)
    res_out = flag_gems.swapaxes_(inp, 0, axis1)

    assert res_out is inp
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


_BACKWARD = tu.selected_cases(
    [((2, 3, 4), 0, 2), ((20, 320, 15), 1, 2)],
    quick=[],
)
_BACKWARD_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
if utils.fp64_is_supported:
    _BACKWARD_DTYPES.append(torch.float64)
_BACKWARD_DTYPES = [
    dtype for dtype in _BACKWARD_DTYPES if _DTYPE_FLAGS.get(dtype, True)
]


@pytest.mark.swapaxes_
@pytest.mark.parametrize("shape,axis0,axis1", _BACKWARD)
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
def test_swapaxes__backward(shape, axis0, axis1, dtype):
    # torch rejects an in-place metadata change on a leaf that requires grad, so
    # the gradient reaches the differentiable leaf through a non-leaf that the
    # candidate and the reference each transpose in place.
    leaf = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_leaf = tu.to_reference(leaf)

    ref_inp = ref_leaf * 1.0
    inp = leaf * 1.0
    ref_version, version = ref_inp._version, inp._version
    ref_out = torch.ops.aten.swapaxes_(ref_inp, axis0, axis1)
    res_out = flag_gems.swapaxes_(inp, axis0, axis1)
    assert inp._version - version == ref_inp._version - ref_version

    assert res_out is inp
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()

    tu.assert_result_equal(res_out, ref_out)

    # The upstream gradient follows the swapped output shape, so the loss is an
    # elementwise product and the leaf gradient is that gradient permuted back.
    upstream = tu.make_input(dtype, tuple(res_out.shape), ["0", "1"])
    ref_upstream = tu.to_reference(upstream)
    ref_grad = torch.autograd.grad((ref_out * ref_upstream).sum(), ref_leaf)[0]
    res_grad = torch.autograd.grad((res_out * upstream).sum(), leaf)[0]

    assert res_grad.shape == ref_grad.shape
    tu.assert_result_equal(res_grad, ref_grad)


# Out-of-range axes raise IndexError natively before any metadata change; every
# negative row runs in both quick and default mode.
_NEGATIVE_AXES = [
    ((2, 3, 4), 3, 0),
    ((2, 3, 4), 0, -4),
    ((4,), 0, 1),
    ((), 0, 1),
]


@pytest.mark.swapaxes_
@pytest.mark.parametrize("shape,axis0,axis1", _NEGATIVE_AXES)
def test_swapaxes__invalid_axis(shape, axis0, axis1):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    with pytest.raises(IndexError):
        flag_gems.swapaxes_(inp, axis0, axis1)
    assert inp.shape == torch.Size(shape)


_NEGATIVE_AXIS_TYPES = [
    ((2, 3, 4), 1, 1.5),
    ((2, 3, 4), 1, None),
]


@pytest.mark.swapaxes_
@pytest.mark.parametrize("shape,axis0,axis1", _NEGATIVE_AXIS_TYPES)
def test_swapaxes__invalid_axis_type(shape, axis0, axis1):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.swapaxes_(inp, axis0, axis1)
    assert inp.shape == torch.Size(shape)


@pytest.mark.swapaxes_
def test_swapaxes__rejects_grad_leaf():
    inp = tu.make_input(torch.float32, (2, 3), ["-1", "1"]).requires_grad_(True)
    with pytest.raises(RuntimeError):
        flag_gems.swapaxes_(inp, 0, 1)
