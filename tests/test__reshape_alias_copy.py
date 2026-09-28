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

# ``_reshape_alias_copy`` is the materialising sibling of ``_reshape_alias``: it
# reads the requested ``size`` / ``stride`` view and returns a fresh dense tensor
# holding those values, so the result neither aliases the input storage nor keeps
# the requested strides or storage offset.
#
# Exempted dimensions, with the mechanism:
# * broadcast: the schema is (Tensor self, SymInt[] size, SymInt[] stride). The
#   only tensor operand is ``self``, so there is no second tensor to broadcast
#   against.
# * tensor vs scalar: the schema has no tensor vs scalar overload; it is one
#   tensor plus two integer metadata lists. The size/stride integer metadata are
#   covered by the layout cases below (identity, rank change, empty, zero-stride,
#   overlapping and offset reads) instead of by a scalar operand.
# * negative-stride and out-of-storage requests: the operator does no bounds
#   checking, so such a request reads undefined memory and returns different
#   values on every run, leaving no deterministic oracle to assert against. That
#   evidence is recorded in the local protocol-gap notes and those requests stay
#   unresolved rather than being asserted here. Every request below stays inside
#   the input storage, which is why each explicit size/stride row carries its
#   reach as a comment.
# * float8_e4m3fn has no infinity: the frozen ``tu.special_value_cases`` emits
#   nan only for it and keeps the inf/mixed cases for float8_e5m2.
#
# ``ref_inp`` is the independent oracle built by ``tu.to_reference`` before the
# candidate runs, so comparing the input against it afterwards doubles as the
# input-immutability check and no second storage clone is needed.

_GRID_DTYPES = [torch.float32, torch.float16, torch.int8, torch.uint8, torch.int32]
if utils.bf16_is_supported:
    _GRID_DTYPES.append(torch.bfloat16)
if utils.int64_is_supported:
    _GRID_DTYPES.append(torch.int64)
if utils.fp8_is_supported:
    _GRID_DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]
if utils.fp64_is_supported:
    _GRID_DTYPES.append(torch.float64)

_FLOAT_DTYPES = [torch.float32, torch.float16]
if utils.bf16_is_supported:
    _FLOAT_DTYPES.append(torch.bfloat16)
if utils.fp64_is_supported:
    _FLOAT_DTYPES.append(torch.float64)

_LAYOUT_DTYPES = [torch.float32, torch.int32, torch.uint8]

_OUT_DTYPES = [torch.float32]
if utils.bf16_is_supported:
    _OUT_DTYPES.append(torch.bfloat16)
_OUT_DTYPES.append(torch.int32)


def _contiguous_strides(shape):
    strides = []
    acc = 1
    for dim in reversed(shape):
        strides.append(acc)
        acc *= dim
    return list(reversed(strides))


def _permuted_spec(shape):
    # Rotating the dimensions keeps the request inside the input storage while
    # making the requested strides differ from the contiguous layout of the copy,
    # so a candidate that returns the input unchanged or ignores the strides
    # cannot match the reference.
    shape = tuple(shape)
    if not shape:
        return [1], [1]
    strides = _contiguous_strides(shape)
    if len(shape) < 2:
        return list(shape), strides
    order = list(range(1, len(shape))) + [0]
    return [shape[dim] for dim in order], [strides[dim] for dim in order]


def _assert_materialized_copy(res_out, ref_out, inp):
    # Operator-specific checks the shared value assertions do not cover: the
    # ``_copy`` variant materialises the request, so the result is a fresh dense
    # tensor that drops the requested strides and offset and must not alias the
    # input storage.
    assert res_out.device == inp.device
    assert res_out is not inp
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == 0
    if inp.numel() and res_out.numel():
        # Storage-base comparison: a result sharing the input storage at a
        # different offset still shares the same base pointer.
        assert res_out.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()


_GRID_ROWS = tu.selected_cases(
    [(shape, dtype) for dtype in _GRID_DTYPES for shape in tu.selected_shapes()],
    quick=[(tu.QUICK_SHAPES[0], dtype) for dtype in _GRID_DTYPES],
)


@pytest.mark.reshape_alias_copy
@pytest.mark.parametrize("shape,dtype", _GRID_ROWS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__reshape_alias_copy_value_range(shape, value_range, dtype):
    size, stride = _permuted_spec(shape)
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._reshape_alias_copy(ref_inp, size, stride)
    res_out = flag_gems._reshape_alias_copy(inp, size, stride)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    _assert_materialized_copy(res_out, ref_out, inp)


# Layout rows exercise the size/stride metadata the operator actually takes: the
# identity request on every spec shape, rank changes, empty inputs, zero-stride
# and overlapping reads, a rank-0 result, and a read that ends on the last
# element of the storage. Every request stays in bounds, so the reference is
# deterministic.
_LAYOUT_ROWS = tu.selected_cases(
    [
        # (input_shape, size, stride); the trailing number is the highest
        # element offset the request reaches, always below the input numel.
        ((1024, 1024), [1024, 1024], [1024, 1]),  # 1048575
        ((), [1], [1]),  # 0-dim input holding a single element
        ((1,), [1], [1]),
        ((1, 1), [1], [1]),
        ((2, 19, 7), [2, 19, 7], [133, 7, 1]),  # 265
        ((20, 320, 15), [20, 320, 15], [4800, 15, 1]),  # 95999
        ((16, 128, 64, 60), [16, 128, 64, 60], [491520, 3840, 60, 1]),  # 7864319
        ((16, 7, 57, 32, 29), [16, 7, 57, 32, 29], [370272, 52896, 928, 29, 1]),
        ((256,), [16, 16], [16, 1]),  # 255
        ((256,), [4, 8, 8], [64, 8, 1]),  # 255
        ((0,), [0], [1]),  # empty input reads nothing
        ((0, 4), [0, 4], [4, 1]),
        ((4, 0, 3), [4, 0, 3], [0, 3, 1]),
        ((6, 6), [3, 3], [7, 1]),  # strided read with holes, 16
        ((2, 2, 4), [2, 2, 4], [4, 8, 1]),  # 15
        ((2, 3, 5), [3, 5, 2], [5, 1, 15]),  # 29
        ((8, 8), [2, 4], [4, 1]),  # 7
        ((256,), [4, 4], [0, 1]),  # zero stride, one element repeated, 3
        ((8,), [4, 2], [1, 1]),  # overlapping read, 4
        ((256,), [], []),  # rank-0 result
        ((256,), [1], [255]),  # extent-1 read at offset 0, stride unused
        ((1024, 1024), [2, 1], [1048575, 0]),  # last element, zero stride
    ],
    quick=[],
)


@pytest.mark.reshape_alias_copy
@pytest.mark.parametrize("shape,size,stride", _LAYOUT_ROWS)
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test__reshape_alias_copy_layout(shape, size, stride, dtype):
    inp = tu.make_input(dtype, shape, ("-1", "1"))
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._reshape_alias_copy(ref_inp, size, stride)
    res_out = flag_gems._reshape_alias_copy(inp, size, stride)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    _assert_materialized_copy(res_out, ref_out, inp)


# Sliced inputs carry a storage offset and/or non-unit strides; 'dense' asks for
# that very view, 'permuted' asks for a rotated dense read of the same storage.
_STRIDED_INPUT_ROWS = tu.selected_cases(
    [
        # (base_shape, slicer, mode)
        ((16, 32), (slice(3, 11), slice(None)), "dense"),
        ((1, 64), (slice(None), slice(7, 43)), "dense"),
        ((8, 16), (slice(None), slice(None, None, 2)), "dense"),
        ((2, 19, 7), (slice(None), slice(1, 3), slice(None)), "dense"),
        ((16, 32), (slice(2, 10), slice(0, 32, 2)), "permuted"),
        ((4, 32, 8), (slice(None, None, 2), slice(None), slice(2, 8, 3)), "permuted"),
    ],
    quick=[],
)


def _view_spec(inp, mode):
    shape = list(inp.shape)
    strides = list(inp.stride())
    if mode == "dense" or len(shape) < 2:
        return shape, strides
    order = list(range(1, len(shape))) + [0]
    return [shape[dim] for dim in order], [strides[dim] for dim in order]


@pytest.mark.reshape_alias_copy
@pytest.mark.parametrize("base_shape,slicer,mode", _STRIDED_INPUT_ROWS)
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test__reshape_alias_copy_strided_input(base_shape, slicer, mode, dtype):
    base = tu.make_input(dtype, base_shape, ("-1", "1"))
    ref_base = tu.to_reference(base)
    inp = base[slicer]
    ref_inp = ref_base[slicer]
    size, stride = _view_spec(inp, mode)

    ref_out = torch.ops.aten._reshape_alias_copy(ref_inp, size, stride)
    res_out = flag_gems._reshape_alias_copy(inp, size, stride)

    tu.assert_result_equal(res_out, ref_out)
    # Whole-parent comparison: a write into the sliced-away padding, not just
    # into the view itself, makes the two parents differ.
    tu.assert_result_equal(base, ref_base)
    _assert_materialized_copy(res_out, ref_out, inp)


# bool and complex64 are outside the required nine-dtype list but natively valid.
# ``tu.make_input`` already supports both (randint for bool, and make_tensor
# fills BOTH the real and the imaginary part for complex), so the shared
# value-range generator is used here as well; the complex rows therefore carry
# varying nonzero imaginary values, and a candidate that drops the imaginary
# part or ignores the lazy conjugate flag cannot pass the exact comparison.
_EXTRA_ROWS = tu.selected_cases(
    [
        # (dtype, shape, size, stride, conj)
        (torch.bool, (16,), [16], [1], False),
        (torch.bool, (4, 8), [8, 4], [1, 8], False),
        (torch.bool, (2, 19, 7), [7, 19, 2], [1, 7, 133], False),
        (torch.complex64, (4, 8), [8, 4], [1, 8], False),
        (torch.complex64, (20, 320), [320, 20], [1, 20], True),
    ],
    quick=[],
)


@pytest.mark.reshape_alias_copy
@pytest.mark.parametrize("dtype,shape,size,stride,conj", _EXTRA_ROWS)
def test__reshape_alias_copy_extra_dtypes(dtype, shape, size, stride, conj):
    base = tu.make_input(dtype, shape, ("-1", "1"))
    ref_base = tu.to_reference(base)
    if conj:
        # The lazy conjugate bit is data to copy, not a property of the result.
        # Building the reference view from the transferred base keeps the same
        # lazy geometry under --refcpu.
        inp = base.conj()
        ref_inp = ref_base.conj()
    else:
        inp = base
        ref_inp = ref_base

    ref_out = torch.ops.aten._reshape_alias_copy(ref_inp, size, stride)
    res_out = flag_gems._reshape_alias_copy(inp, size, stride)

    if conj:
        assert inp.is_conj()
        assert not res_out.is_conj()
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(base, ref_base)
    _assert_materialized_copy(res_out, ref_out, inp)


_SENTINEL = -12345


def _numel(size):
    total = 1
    for dim in size:
        total *= dim
    return total


def _buffer_geometry(buf):
    return (buf.data_ptr(), tuple(buf.shape), tuple(buf.stride()), buf.storage_offset())


def _make_out(size, dtype, device, layout):
    # Sentinel-filled buffers: a silently reallocated, partially written or
    # wrong-offset result then shows up as a deterministic value mismatch
    # instead of undefined memory. The parent storage is returned so the caller
    # can compare the elements outside the view it handed to the operator.
    if layout == "contiguous":
        return torch.full(size, _SENTINEL, dtype=dtype, device=device), None
    if layout == "transposed":
        # Rank agnostic: allocate with the dimensions reversed and permute them
        # back, which gives the requested shape with non-contiguous strides for
        # any rank (.t() only accepts 2-D).
        order = tuple(range(len(size) - 1, -1, -1))
        base = torch.full(
            tuple(size[dim] for dim in order), _SENTINEL, dtype=dtype, device=device
        )
        return base.permute(order), base
    if layout == "offset":
        base = torch.full((_numel(size) + 8,), _SENTINEL, dtype=dtype, device=device)
        return base[4 : 4 + _numel(size)].view(size), base
    raise ValueError(f"unknown out layout {layout!r}")


# The out overload fills the caller buffer, keeps its geometry and returns that
# same buffer. The four original rows are preserved; the added rows cover a
# permuted buffer for another rank and nonzero-storage-offset buffers. Every
# requested reach stays inside the input storage (annotated per row).
_OUT_ROWS = tu.selected_cases(
    [
        # (input_shape, size, stride, out_layout)
        ((1024, 1024), [1024, 1024], [1024, 1], "contiguous"),
        ((20, 320, 15), [15, 320, 20], [1, 15, 4800], "contiguous"),
        (
            (16, 7, 57, 32, 29),
            [29, 32, 57, 7, 16],
            [1, 29, 928, 52896, 370272],
            "contiguous",
        ),
        ((256,), [16, 16], [1, 16], "transposed"),
        ((1024, 1024), [1024, 1024], [1, 1024], "transposed"),  # 1048575
        ((20, 320, 15), [15, 320, 20], [1, 15, 4800], "offset"),  # 95999
        ((256,), [16, 16], [1, 16], "offset"),  # 255
    ],
    quick=[((2, 19, 7), [7, 19, 2], [1, 7, 133], "contiguous")],
)


@pytest.mark.reshape_alias_copy
@pytest.mark.parametrize("shape,size,stride,out_layout", _OUT_ROWS)
@pytest.mark.parametrize("dtype", _OUT_DTYPES)
def test__reshape_alias_copy_out(shape, size, stride, out_layout, dtype):
    inp = tu.make_input(dtype, shape, ("-1", "1"))
    ref_inp = tu.to_reference(inp)

    # Build the reference buffer with the same requested geometry on the
    # reference device before the native out call, so the whole-parent
    # comparison below also covers the sentinel padding outside the view.
    ref_out, ref_parent = _make_out(size, ref_inp.dtype, ref_inp.device, out_layout)
    torch.ops.aten._reshape_alias_copy.out(ref_inp, size, stride, out=ref_out)

    out, parent = _make_out(size, dtype, flag_gems.device, out_layout)
    geometry = _buffer_geometry(out)
    res = flag_gems._reshape_alias_copy(inp, size, stride, out=out)

    assert res is out
    assert _buffer_geometry(out) == geometry
    tu.assert_result_equal(out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    if parent is not None:
        assert out.untyped_storage().data_ptr() == parent.untyped_storage().data_ptr()
        tu.assert_result_equal(parent, ref_parent)


# Pure permutation / copy requests: the gradient is the upstream gradient read
# with the same layout, so no arithmetic accumulation happens and the exact
# shared assertion applies. The upstream is created on the candidate device with
# the requested OUTPUT size, and the reference side gets its own device copy.
_BACKWARD_ROWS = tu.selected_cases(
    [
        ((1024, 1024), [1024, 1024], [1, 1024]),
        ((20, 320, 15), [15, 320, 20], [1, 15, 4800]),
        ((16, 7, 57, 32, 29), [29, 32, 57, 7, 16], [1, 29, 928, 52896, 370272]),
    ],
    quick=[],
)


@pytest.mark.reshape_alias_copy
@pytest.mark.parametrize("shape,size,stride", _BACKWARD_ROWS)
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test__reshape_alias_copy_backward(shape, size, stride, dtype):
    inp = tu.make_input(dtype, shape, ("-1", "1")).requires_grad_(True)
    ref_inp = tu.to_reference(inp.detach())
    ref_inp.requires_grad_(True)
    upstream = tu.make_input(dtype, size, ("-1", "1"))
    ref_upstream = tu.to_reference(upstream)

    ref_out = torch.ops.aten._reshape_alias_copy(ref_inp, size, stride)
    res_out = flag_gems._reshape_alias_copy(inp, size, stride)
    tu.assert_result_equal(res_out, ref_out)

    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)[0]
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]

    assert res_in_grad.device == inp.device
    tu.assert_result_equal(res_in_grad, ref_in_grad)
    tu.assert_result_equal(inp, ref_inp)


# The special-value matrix is derived from the dtypes the operator accepts, not
# from the ALL_FLOAT_DTYPES convenience list; the shared generator already omits
# the inf/mixed scenarios float8_e4m3fn cannot represent and keeps them for
# float8_e5m2.
_SPECIAL_DTYPES = [torch.float32, torch.float16]
if utils.bf16_is_supported:
    _SPECIAL_DTYPES.append(torch.bfloat16)
if utils.fp64_is_supported:
    _SPECIAL_DTYPES.append(torch.float64)
if utils.fp8_is_supported:
    _SPECIAL_DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]
_SPECIAL_ROWS = tu.selected_cases(tu.special_value_cases(_SPECIAL_DTYPES), quick=[])


@pytest.mark.reshape_alias_copy
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_ROWS)
def test__reshape_alias_copy_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario).reshape(1, 5)
    ref_inp = tu.to_reference(inp)
    # Requested reach is 4 < 5, so the NaN/Inf positions are read directly.
    size, stride = [5, 1], [1, 5]

    ref_out = torch.ops.aten._reshape_alias_copy(ref_inp, size, stride)
    res_out = flag_gems._reshape_alias_copy(inp, size, stride)

    # A copy is exact: NaN slots must match by position and must not be
    # replaced by finite values.
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    _assert_materialized_copy(res_out, ref_out, inp)


_COMPLEX_SCENARIOS = ("nan", "inf", "mixed")
_COMPLEX_SPECIAL_ROWS = tu.selected_cases(
    [(scenario, conj) for scenario in _COMPLEX_SCENARIOS for conj in (False, True)],
    quick=[],
)


def _complex_special_input(scenario):
    # Pair each scenario with a different one on the imaginary part, so the
    # imaginary values stay nonzero and varying instead of an all-zero part.
    index = _COMPLEX_SCENARIOS.index(scenario)
    imag = _COMPLEX_SCENARIOS[(index + 1) % len(_COMPLEX_SCENARIOS)]
    return torch.complex(
        tu.make_special_input(torch.float32, scenario),
        tu.make_special_input(torch.float32, imag),
    )


@pytest.mark.reshape_alias_copy
@pytest.mark.parametrize("scenario,conj", _COMPLEX_SPECIAL_ROWS)
def test__reshape_alias_copy_complex_special_values(scenario, conj):
    base = _complex_special_input(scenario)
    ref_base = tu.to_reference(base)
    if conj:
        inp = base.conj()
        ref_inp = ref_base.conj()
    else:
        inp = base
        ref_inp = ref_base
    size, stride = _permuted_spec(base.shape)

    ref_out = torch.ops.aten._reshape_alias_copy(ref_inp, size, stride)
    res_out = flag_gems._reshape_alias_copy(inp, size, stride)

    if conj:
        assert inp.is_conj()
        assert not res_out.is_conj()
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(base, ref_base)
    _assert_materialized_copy(res_out, ref_out, inp)


@pytest.mark.reshape_alias_copy
def test__reshape_alias_copy_rejects_mismatched_rank():
    inp = tu.make_input(torch.float32, (4, 6), ("-1", "1"))
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems._reshape_alias_copy(inp, [6, 4], [1])


@pytest.mark.reshape_alias_copy
def test__reshape_alias_copy_rejects_non_tensor_input():
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems._reshape_alias_copy([1.0, 2.0, 3.0], [3], [1])


@pytest.mark.reshape_alias_copy
def test__reshape_alias_copy_rejects_non_integer_stride():
    inp = tu.make_input(torch.float32, (4, 6), ("-1", "1"))
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems._reshape_alias_copy(inp, [6, 4], [1.5, 6])


@pytest.mark.reshape_alias_copy
def test__reshape_alias_copy_out_rejects_dtype_mismatch():
    inp = tu.make_input(torch.float32, (4, 6), ("-1", "1"))
    out = torch.empty(6, 4, dtype=torch.int32, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._reshape_alias_copy(inp, [6, 4], [4, 1], out=out)
