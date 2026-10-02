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

# aten::view_as(self, other) is the view-only sibling of reshape_as: the result
# takes its shape from 'other' and its element count from 'self', shares the
# storage of 'self' at the offset of 'self', and raises instead of materializing
# a copy when that target is not stride-expressible. No value is rounded, so
# values are compared exactly while the strides, the storage offset, the
# storage-sharing relation and the lazy conjugate / negative bits are checked
# separately.
#
# 'other' contributes its shape only: its dtype, its layout, its device and its
# values never reach the result, so those axes vary in OTHER_CASES.

# Capability flags, read while this module is imported: no tensor is allocated
# and no operator is called at collection time. A dtype outside the map is a
# baseline type the backend always handles.
_DTYPE_CAPABILITY = {
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name))


# complex64 rides the same 32-bit float path as float32, so it needs no
# capability flag of its own.
_VIEW_AS_DTYPES = [
    dtype
    for dtype in tu.REQUIRED_DTYPES
    + [torch.float64, torch.bool, torch.complex64, torch.complex128]
    if _dtype_supported(dtype)
]
_BACKWARD_DTYPES = [
    dtype for dtype in _VIEW_AS_DTYPES if dtype.is_floating_point or dtype.is_complex
]
# Expanded-base gradients reduce over repeated storage positions. Keep their
# existing real-dtype exact-sum fixtures separate from pure view gradients.
_ACCUMULATING_DTYPES = [
    dtype
    for dtype in (torch.float16, torch.float32, torch.bfloat16, torch.float64)
    if _dtype_supported(dtype)
]
_LAZY_DTYPES = [
    dtype for dtype in (torch.complex64, torch.complex128) if _dtype_supported(dtype)
]
_SCALAR_DTYPES = [
    dtype
    for dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int64,
    )
    if _dtype_supported(dtype)
]

# Numel-preserving target for each spec shape. Only the shape of 'other' is read,
# and every target below is a merge or a split of 'self' (never a permutation),
# which is what keeps the result a view of a dense input.
_TARGET_SHAPES = {
    (): (),
    (1,): (1, 1),
    (256,): (16, 16),
    (1024, 1024): (1048576,),
    (20, 320, 15): (20, 4800),
    (16, 128, 64, 60): (16, 128, 3840),
    (16, 7, 57, 32, 29): (16, 7, 57, 928),
    (2, 19, 7): (2, 133),
}


def _apply_layout(base, layout, expand_shape=None):
    if layout == "asis":
        return base
    if layout == "transposed":
        return base.transpose(-1, -2)
    if layout == "column_step":
        return base[..., ::2]
    if layout == "offset_window":
        return base[2:6, 1:5]
    if layout == "expanded":
        return base.expand(tuple(expand_shape))
    raise ValueError("unsupported layout " + repr(layout))


# Candidate operand, independently cloned reference operand, backing base and
# the backing base of the reference.
def _layout_pair(storage_shape, layout, dtype, expand_shape=None):
    base = tu.make_input(dtype, storage_shape, ["-1", "1"])
    ref_base = tu.to_reference(base.detach())
    inp = _apply_layout(base, layout, expand_shape)
    ref_inp = _apply_layout(ref_base, layout, expand_shape)
    return inp, ref_inp, base, ref_base


def _assert_view(res_out, ref_out, inp):
    assert res_out is not inp
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.is_neg() == ref_out.is_neg()


def _assert_operands_unchanged(
    inp, inp_before, other, other_before, base=None, base_before=None
):
    # Building a view never writes to its operands. Snapshots come from
    # tu.to_reference so they sit on the configured reference device, the
    # placement the shared accuracy helpers expect for the reference operand.
    tu.assert_result_equal(inp, inp_before)
    tu.assert_result_equal(other, other_before)
    for operand, before in ((inp, inp_before), (other, other_before)):
        assert operand.shape == before.shape
        assert operand.stride() == before.stride()
        assert operand.storage_offset() == before.storage_offset()
    if base is not None and base is not inp:
        # Sliced and expanded operands share storage with a full backing tensor,
        # which has to stay intact as well.
        tu.assert_result_equal(base, base_before)


def _exact_upstream(shape, dtype):
    # 1..numel and their partial sums are exactly representable in every tested
    # gradient dtype, so an accumulated gradient can be compared exactly.
    steps = torch.arange(1, math.prod(shape) + 1, dtype=dtype, device=flag_gems.device)
    return steps.reshape(shape)


# Rank and emptiness coverage: a 0-dim operand, unflatten, re-batch, flatten and
# zero-sized operands whose 0 elements still satisfy the numel contract.
_SIZE_ROWS = [
    ((1,), (1, 1)),
    ((24,), (2, 12)),
    ((24,), (2, 3, 4)),
    ((24,), (24,)),
    ((2, 3, 4), (24,)),
    ((0, 3), (0,)),
    ((0, 3), (3, 0)),
    ((5, 0, 7), (0,)),
]
# All small rank and emptiness boundaries run in both modes.
SIZE_CASES = _SIZE_ROWS

# 'other' is a tensor, never a Python number, so its scalar-shaped form is a
# 0-dim tensor. A 0-dim 'self' is the other role of the same signature.
_SCALAR_ROWS = [((), ()), ((), (1,)), ((1,), ())]
SCALAR_CASES = _SCALAR_ROWS

# Non-contiguous inputs. A step on the trailing extent keeps a stride description
# every target below can still express, and a window keeps a non-zero storage
# offset that the result has to keep too; a transposed operand is viewable only
# into shapes that keep its axis order.
_STRIDED_ROWS = [
    ((4, 12), "column_step", (4, 6)),
    ((4, 12), "column_step", (24,)),
    ((4, 12), "column_step", (8, 3)),
    ((2, 19, 14), "column_step", (266,)),
    ((4, 6), "transposed", (6, 4)),
    ((4, 6), "transposed", (6, 2, 2)),
    ((10, 8), "offset_window", (4, 4)),
    ((10, 8), "offset_window", (4, 2, 2)),
]
STRIDED_CASES = _STRIDED_ROWS

# Broadcast-shaped operands. Only a fully expanded operand (both extents greater
# than 1) has stride 0 everywhere, which is the layout a flat view can still
# express; a partially expanded operand is rejected and covered by the negative
# rows below. Expanding a (1, 1) base allocates nothing, so the spec-scale rows
# cost no storage for the operand itself.
_BROADCAST_ROWS = [
    ((1, 1), (3, 5), (15,)),
    ((1, 1), (3, 5), (1, 15)),
    ((1, 1), (3, 5), (5, 3)),
    ((1, 1), (1024, 1024), (1048576,)),
    ((1, 1), (16, 128, 64, 60), (16, 128, 3840)),
]
BROADCAST_CASES = tu.selected_cases(_BROADCAST_ROWS, quick=_BROADCAST_ROWS[:3])

# 'other' is shape-only: its dtype, its storage layout, its device and its values
# must not change the result, so a NaN-sentinel operand is exposed by the exact
# comparison below if a candidate reads it as values. Both operands need their
# declared storage, so both dtypes are gated on the static capability flags.
_OTHER_ROWS = [
    row
    for row in (
        ((4, 6), torch.float32, (24,), torch.bool, "contiguous"),
        ((4, 6), torch.float32, (6, 4), torch.float64, "nan"),
        ((2, 3, 4), torch.int32, (12, 2), torch.int64, "transposed"),
        ((4, 6), torch.float16, (12, 2), torch.float16, "transposed"),
        ((4, 6), torch.float32, (6, 4), torch.float32, "cpu"),
    )
    if _dtype_supported(row[1]) and _dtype_supported(row[3])
]
# Finite shape-only boundaries run in quick; the NaN sentinel is default-only.
OTHER_CASES = tu.selected_cases(
    _OTHER_ROWS, quick=[row for row in _OTHER_ROWS if row[-1] != "nan"]
)

# view_as always returns a view, so a write through the result has to reach the
# input and, for a sliced input, the values the slice does not cover have to stay
# untouched.
_MUTATION_ROWS = [
    ((24,), "asis", (2, 12)),
    ((3, 5, 2), "asis", (15, 2)),
    ((4, 12), "column_step", (12, 2)),
]
MUTATION_CASES = _MUTATION_ROWS

# A lazy conjugate / negative bit is metadata: view_as reuses the storage of
# 'self' instead of materializing it, so the bit survives on both a dense and a
# transposed operand.
_LAZY_ROWS = [
    ("asis", (24,), "conj"),
    ("transposed", (6, 4), "conj"),
    ("asis", (24,), "neg"),
    ("transposed", (6, 4), "neg"),
]
LAZY_CASES = [
    (layout, target, bit, dtype)
    for layout, target, bit in _LAZY_ROWS
    for dtype in _LAZY_DTYPES
]

# Backward. Differentiating with respect to the operand handed to the operator
# makes every gradient a pure relayout of a non-uniform upstream gradient, which
# is compared exactly rather than within a tolerance. Autograd is default-only.
_BACKWARD_ROWS = [
    ((4, 6), (24,)),
    ((7, 13, 29), (7, 377)),
    ((256,), (16, 16)),
]
BACKWARD_CASES = tu.selected_cases(
    [
        (shape, target, dtype)
        for shape, target in _BACKWARD_ROWS
        for dtype in _BACKWARD_DTYPES
    ],
    quick=[],
)
EXPANDED_BACKWARD_CASES = tu.selected_cases(_ACCUMULATING_DTYPES, quick=[])
NO_GRAD_CASES = tu.selected_cases(_BACKWARD_DTYPES, quick=[])

# Positive special values are default-only as well.
SPECIAL_VALUE_CASES = tu.selected_cases(
    tu.special_value_cases(_VIEW_AS_DTYPES), quick=[]
)

# An 'other' whose numel differs from 'self' is rejected natively with
# RuntimeError, and neither operand may be a non-tensor. These negatives stay in
# both modes.
_NUMEL_MISMATCH_ROWS = [
    ((4, 6), (5,)),
    ((3,), (2, 2)),
    ((24,), (4, 7)),
    ((0, 3), (4,)),
]

# Targets that share the element count but are not stride-expressible: the native
# operator raises for these instead of copying, and so must the candidate.
_NON_VIEWABLE_ROWS = [
    ((4, 6), "transposed", None, (24,)),
    ((10, 8), "offset_window", None, (16,)),
    ((1, 4096), "expanded", (2, 4096), (8192,)),
]


@pytest.mark.view_as
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VIEW_AS_DTYPES)
def test_view_as(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    other = torch.zeros(_TARGET_SHAPES[shape], dtype=dtype, device=flag_gems.device)
    inp_before = tu.to_reference(inp.detach())
    other_before = tu.to_reference(other.detach())

    ref_out = torch.ops.aten.view_as(inp_before, other_before)
    res_out = flag_gems.view_as(inp, other)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view(res_out, ref_out, inp)
    _assert_operands_unchanged(inp, inp_before, other, other_before)


@pytest.mark.view_as
@pytest.mark.parametrize("shape,target", SIZE_CASES)
def test_view_as_with_size(shape, target):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    other = torch.zeros(target, dtype=torch.float32, device=flag_gems.device)
    inp_before = tu.to_reference(inp.detach())
    other_before = tu.to_reference(other.detach())

    ref_out = torch.ops.aten.view_as(inp_before, other_before)
    res_out = flag_gems.view_as(inp, other)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view(res_out, ref_out, inp)
    _assert_operands_unchanged(inp, inp_before, other, other_before)


@pytest.mark.view_as
@pytest.mark.parametrize("self_shape,other_shape", SCALAR_CASES)
@pytest.mark.parametrize("dtype", _SCALAR_DTYPES)
def test_view_as_scalar_operand(self_shape, other_shape, dtype):
    # Both operands stay tensors: the scalar form of this signature is a 0-dim
    # 'other', and the complementary 0-dim 'self' supplies the element count.
    inp = tu.make_input(dtype, self_shape, ["-1", "1"])
    other = torch.zeros(other_shape, dtype=dtype, device=flag_gems.device)
    inp_before = tu.to_reference(inp.detach())
    other_before = tu.to_reference(other.detach())

    ref_out = torch.ops.aten.view_as(inp_before, other_before)
    res_out = flag_gems.view_as(inp, other)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view(res_out, ref_out, inp)
    _assert_operands_unchanged(inp, inp_before, other, other_before)


@pytest.mark.view_as
@pytest.mark.parametrize("storage,layout,target", STRIDED_CASES)
def test_view_as_strided_input(storage, layout, target):
    inp, ref_inp, base, ref_base = _layout_pair(storage, layout, torch.float32)
    other = torch.zeros(target, dtype=torch.float32, device=flag_gems.device)
    inp_before = tu.to_reference(inp.detach())
    base_before = tu.to_reference(base.detach())
    other_before = tu.to_reference(other.detach())

    ref_out = torch.ops.aten.view_as(ref_inp, other_before)
    res_out = flag_gems.view_as(inp, other)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view(res_out, ref_out, inp)
    _assert_operands_unchanged(inp, inp_before, other, other_before, base, base_before)


@pytest.mark.view_as
@pytest.mark.parametrize("storage,expand,target", BROADCAST_CASES)
def test_view_as_broadcast_input(storage, expand, target):
    inp, ref_inp, base, ref_base = _layout_pair(
        storage, "expanded", torch.float32, expand
    )
    other = torch.zeros(target, dtype=torch.float32, device=flag_gems.device)
    inp_before = tu.to_reference(inp.detach())
    base_before = tu.to_reference(base.detach())
    other_before = tu.to_reference(other.detach())

    ref_out = torch.ops.aten.view_as(ref_inp, other_before)
    res_out = flag_gems.view_as(inp, other)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view(res_out, ref_out, inp)
    _assert_operands_unchanged(inp, inp_before, other, other_before, base, base_before)


@pytest.mark.view_as
@pytest.mark.parametrize("shape,dtype,other_shape,other_dtype,other_kind", OTHER_CASES)
def test_view_as_other_operand(shape, dtype, other_shape, other_dtype, other_kind):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    other = torch.ones(other_shape, dtype=other_dtype, device=flag_gems.device)
    if other_kind == "nan":
        other.fill_(float("nan"))
    elif other_kind == "transposed":
        other = other.t()
    elif other_kind == "cpu":
        # Only the shape of 'other' is read, so its device must not matter
        # either.
        other = other.cpu()
    inp_before = tu.to_reference(inp.detach())
    other_before = tu.to_reference(other.detach())

    ref_out = torch.ops.aten.view_as(inp_before, other_before)
    res_out = flag_gems.view_as(inp, other)

    # An exact comparison against a native result built from a sentinel operand
    # detects any candidate that reads 'other' as values or as its own dtype
    # instead of taking its shape.
    tu.assert_result_equal(res_out, ref_out)
    _assert_view(res_out, ref_out, inp)
    _assert_operands_unchanged(inp, inp_before, other, other_before)


@pytest.mark.view_as
@pytest.mark.parametrize("storage,layout,target", MUTATION_CASES)
def test_view_as_view_writes_through(storage, layout, target):
    inp, ref_inp, base, ref_base = _layout_pair(storage, layout, torch.float32)
    other = torch.zeros(target, dtype=torch.float32, device=flag_gems.device)
    inp_before = tu.to_reference(inp.detach())
    base_before = tu.to_reference(base.detach())
    other_before = tu.to_reference(other.detach())

    ref_out = torch.ops.aten.view_as(ref_inp, other_before)
    res_out = flag_gems.view_as(inp, other)

    # Compare the untouched result first, then the operand contract.
    tu.assert_result_equal(res_out, ref_out)
    _assert_view(res_out, ref_out, inp)
    _assert_operands_unchanged(inp, inp_before, other, other_before, base, base_before)

    # The result is a view, so a write through it has to reach the candidate
    # input and its backing storage. The same write is applied to the reference
    # operands and both sides are compared afterwards.
    res_out.fill_(3.0)
    ref_out.fill_(3.0)
    tu.assert_result_equal(inp, ref_inp)
    if base is not inp:
        tu.assert_result_equal(base, ref_base)


@pytest.mark.view_as
@pytest.mark.parametrize("layout,target,bit,dtype", LAZY_CASES)
def test_view_as_lazy_bits(layout, target, bit, dtype):
    base = tu.make_input(dtype, (4, 6), ["-1", "1"])
    ref_base = tu.to_reference(base.detach())
    inp = _apply_layout(base, layout)
    ref_inp = _apply_layout(ref_base, layout)
    if bit == "conj":
        inp = inp.conj()
        ref_inp = ref_inp.conj()
    else:
        inp = torch._neg_view(inp)
        ref_inp = torch._neg_view(ref_inp)
    other = torch.zeros(target, dtype=dtype, device=flag_gems.device)
    inp_before = tu.to_reference(inp.detach())
    base_before = tu.to_reference(base.detach())
    other_before = tu.to_reference(other.detach())

    ref_out = torch.ops.aten.view_as(ref_inp, other_before)
    res_out = flag_gems.view_as(inp, other)

    # These bits are metadata of 'self': a candidate that materialized a copy
    # would drop them and be caught here.
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.is_neg() == ref_out.is_neg()
    tu.assert_result_equal(res_out, ref_out)
    _assert_view(res_out, ref_out, inp)
    _assert_operands_unchanged(inp, inp_before, other, other_before, base, base_before)


@pytest.mark.view_as
@pytest.mark.parametrize("shape,target,dtype", BACKWARD_CASES)
def test_view_as_backward(shape, target, dtype):
    base = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_base = tu.to_reference(base.detach()).requires_grad_(True)
    other = torch.zeros(target, dtype=dtype, device=flag_gems.device)
    upstream = tu.make_input(dtype, target, ["-1", "1"])
    ref_upstream = tu.to_reference(upstream.detach())
    base_before = tu.to_reference(base.detach())
    other_before = tu.to_reference(other.detach())

    ref_out = torch.ops.aten.view_as(ref_base, other_before)
    res_out = flag_gems.view_as(base, other)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view(res_out, ref_out, base)
    _assert_operands_unchanged(base, base_before, other, other_before)

    res_grad = torch.autograd.grad(res_out, base, grad_outputs=upstream)[0]
    ref_grad = torch.autograd.grad(ref_out, ref_base, grad_outputs=ref_upstream)[0]

    tu.assert_result_equal(res_grad, ref_grad)


@pytest.mark.view_as
@pytest.mark.parametrize("dtype", EXPANDED_BACKWARD_CASES)
def test_view_as_expanded_base_backward(dtype):
    # Differentiating with respect to the 1x1 base rather than the expanded
    # operand makes autograd accumulate the upstream over the expanded axes. The
    # upstream is 1..15, so every partial sum stays exactly representable and the
    # native accumulation is matched exactly instead of within a tolerance.
    base = tu.make_input(dtype, (1, 1), ["-1", "1"]).requires_grad_(True)
    ref_base = tu.to_reference(base.detach()).requires_grad_(True)
    inp = base.expand(3, 5)
    ref_inp = ref_base.expand(3, 5)

    other = torch.zeros((15,), dtype=dtype, device=flag_gems.device)
    upstream = _exact_upstream((15,), dtype)
    ref_upstream = tu.to_reference(upstream.detach())
    base_before = tu.to_reference(base.detach())
    inp_before = tu.to_reference(inp.detach())
    other_before = tu.to_reference(other.detach())

    ref_out = torch.ops.aten.view_as(ref_inp, other_before)
    res_out = flag_gems.view_as(inp, other)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view(res_out, ref_out, inp)
    _assert_operands_unchanged(inp, inp_before, other, other_before, base, base_before)

    res_grad = torch.autograd.grad(res_out, base, grad_outputs=upstream)[0]
    ref_grad = torch.autograd.grad(ref_out, ref_base, grad_outputs=ref_upstream)[0]

    tu.assert_result_equal(res_grad, ref_grad)


@pytest.mark.view_as
@pytest.mark.parametrize("dtype", NO_GRAD_CASES)
def test_view_as_other_has_no_gradient(dtype):
    inp = tu.make_input(dtype, (4, 6), ["-1", "1"])
    other = tu.make_input(dtype, (24,), ["-1", "1"])

    ref_out = torch.ops.aten.view_as(
        tu.to_reference(inp.detach()), tu.to_reference(other.detach())
    )
    inp.requires_grad_(True)
    other.requires_grad_(True)
    inp_before = tu.to_reference(inp.detach())
    other_before = tu.to_reference(other.detach())

    res_out = flag_gems.view_as(inp, other)

    # The candidate still has to return the right values and to leave both
    # operands untouched, before the gradient contract is checked.
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, inp_before)
    tu.assert_result_equal(other, other_before)

    res_other_grad = torch.autograd.grad(
        res_out, other, grad_outputs=torch.ones_like(res_out), allow_unused=True
    )[0]

    # 'other' supplies shape only, so it stays out of the autograd graph.
    assert res_other_grad is None


@pytest.mark.view_as
@pytest.mark.parametrize("dtype,scenario", SPECIAL_VALUE_CASES)
def test_view_as_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    other = torch.zeros((1, inp.numel()), dtype=dtype, device=flag_gems.device)
    inp_before = tu.to_reference(inp.detach())
    other_before = tu.to_reference(other.detach())

    ref_out = torch.ops.aten.view_as(inp_before, other_before)
    res_out = flag_gems.view_as(inp, other)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view(res_out, ref_out, inp)
    _assert_operands_unchanged(inp, inp_before, other, other_before)


@pytest.mark.view_as
@pytest.mark.parametrize("shape,target", _NUMEL_MISMATCH_ROWS)
def test_view_as_numel_mismatch(shape, target):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    other = torch.zeros(target, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.view_as(inp, other)


@pytest.mark.view_as
@pytest.mark.parametrize("storage,layout,expand,target", _NON_VIEWABLE_ROWS)
def test_view_as_non_viewable_target(storage, layout, expand, target):
    # These targets share the element count of the operand but are not
    # stride-expressible, so a view cannot exist and the operator has to raise
    # instead of silently materializing a copy.
    inp = _apply_layout(
        tu.make_input(torch.float32, storage, ["-1", "1"]), layout, expand
    )
    other = torch.zeros(target, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.view_as(inp, other)


@pytest.mark.view_as
@pytest.mark.parametrize("bad_operand", ["self", "other"])
def test_view_as_non_tensor_operand(bad_operand):
    inp = tu.make_input(torch.float32, (4, 6), ["-1", "1"])
    other = torch.zeros((2, 12), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError)):
        if bad_operand == "self":
            flag_gems.view_as([[0.0] * 6] * 4, other)
        else:
            flag_gems.view_as(inp, [2, 12])
