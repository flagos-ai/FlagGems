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

# aten::reshape_as(self, other) takes its result shape from 'other' and its
# element count from 'self'. When that shape is stride-expressible the result is
# a view sharing storage with 'self'; otherwise the operator materializes a copy.
# Neither path rounds, so values are compared exactly while the strides, the
# storage offset and the storage-sharing relation are checked separately.
#
# 'other' contributes its shape only: its dtype, its storage layout and its
# values never reach the result, so those three axes vary in OTHER_CASES.
# Cross-device operands are not exercised here.
#
# Every native reference is evaluated on independent operands and BEFORE the
# candidate runs, so neither the layout facts nor the values used for comparison
# can be contaminated by a candidate that wrote to its input.

# Static capability flags, read while this module is imported: no tensor is
# allocated and no operator is called at collection time, and no test probes
# support or skips at run time. A dtype outside the map is a baseline type the
# backend always handles; a mapped dtype stays in the grid only when its flag is
# enabled.
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
    return bool(getattr(flag_gems.runtime.device, flag_name, False))


# complex64 rides the same 32-bit float path as float32, so it needs no
# capability flag of its own.
_RESHAPE_AS_DTYPES = [
    dtype
    for dtype in tu.REQUIRED_DTYPES
    + [torch.float64, torch.bool, torch.complex64, torch.complex128]
    if _dtype_supported(dtype)
]
_BACKWARD_DTYPES = [
    dtype
    for dtype in (torch.float16, torch.float32, torch.bfloat16, torch.float64)
    if _dtype_supported(dtype)
]
_CONJ_DTYPES = [
    dtype for dtype in (torch.complex64, torch.complex128) if _dtype_supported(dtype)
]

# Numel-preserving target for each spec shape; only the shape of 'other' is read.
_TARGET_SHAPES = {
    (): (),
    (1,): (1,),
    (256,): (16, 16),
    (1024, 1024): (1048576,),
    (20, 320, 15): (15, 320, 20),
    (16, 128, 64, 60): (60, 64, 128, 16),
    (16, 7, 57, 32, 29): (29, 32, 57, 7, 16),
    (2, 19, 7): (7, 38),
}


def _apply_layout(base, transform, expand_shape=None):
    if transform == "asis":
        return base
    if transform == "transposed":
        return base.transpose(-1, -2)
    if transform == "column_step":
        return base[:, ::2]
    if transform == "both_steps":
        return base[::3, ::4]
    if transform == "offset_window":
        return base[2:6, 1:5]
    if transform == "expanded":
        return base.expand(tuple(expand_shape))
    raise ValueError("unsupported layout " + repr(transform))


# Candidate operand, independently cloned reference operand and backing base.
def _layout_pair(storage_shape, transform, dtype, expand_shape=None):
    base = tu.make_input(dtype, storage_shape, ["-1", "1"])
    ref_base = tu.to_reference(base.detach())
    inp = _apply_layout(base, transform, expand_shape)
    ref_inp = _apply_layout(ref_base, transform, expand_shape)
    return inp, ref_inp, base


# Native result on independent operands plus the layout facts to match.
def _native_facts(ref_inp, ref_other):
    ref_out = torch.ops.aten.reshape_as(ref_inp, ref_other)
    aliases = None
    if ref_inp.numel():
        aliases = ref_out.data_ptr() == ref_inp.data_ptr()
    return ref_out, (ref_out.stride(), ref_out.storage_offset(), aliases)


def _assert_layout_facts(res_out, inp, facts):
    # Value comparison cannot show whether a stride-expressible reshape still
    # shares storage with 'self', so match the candidate against facts the native
    # reference produced before the candidate ran. An empty operand owns no
    # element to alias, so its data pointer carries no information.
    stride, offset, aliases = facts
    assert res_out.stride() == stride
    assert res_out.storage_offset() == offset
    if aliases is not None:
        assert (res_out.data_ptr() == inp.data_ptr()) == aliases


def _assert_operands_unchanged(
    inp, inp_before, other, other_before, base=None, base_before=None
):
    # Reshaping never writes to its operands.
    tu.assert_result_equal(inp, inp_before)
    tu.assert_result_equal(other, other_before)
    if base is not None and base is not inp:
        # Sliced and expanded operands share storage with a full backing tensor,
        # which has to stay intact as well.
        tu.assert_result_equal(base, base_before)


def _exact_upstream(shape, dtype):
    # 1..numel and their partial sums are exactly representable in every tested
    # gradient dtype, so an accumulated gradient can be compared exactly. The
    # steps are built straight in the requested dtype on the configured device,
    # so no int64 or float64 intermediate is allocated.
    steps = torch.arange(1, math.prod(shape) + 1, dtype=dtype, device=flag_gems.device)
    return steps.reshape(shape)


# View-shape coverage: 0-dim, unflatten, re-batch, flatten, rank change and
# zero-sized inputs whose 0 elements still satisfy the numel contract.
_SIZE_ROWS = [
    ((1,), ()),
    ((), (1,)),
    ((24,), (2, 12)),
    ((24,), (2, 3, 4)),
    ((24,), (24,)),
    ((2, 3, 4), (24,)),
    ((0, 3), (0,)),
    ((0, 3), (3, 0)),
    ((5, 0, 7), (0,)),
]
SIZE_CASES = tu.selected_cases(_SIZE_ROWS, quick=[])

# Non-contiguous inputs. 'column_step' keeps a stride description the reshape can
# still express, so the result stays a view; 'transposed', 'offset_window' and
# 'both_steps' are not stride-expressible for their target and must materialize a
# copy on both sides.
_STRIDED_ROWS = [
    ((4, 6), "transposed", (24,)),
    ((4, 12), "column_step", (2, 12)),
    ((10, 8), "offset_window", (2, 8)),
    ((12, 24), "both_steps", (2, 12)),
]
STRIDED_CASES = tu.selected_cases(_STRIDED_ROWS, quick=[])

# A stride-0 expanded 'self' is a valid input. Rows 1 and 4 reshape into a target
# that is still stride-expressible (view); rows 2 and 3 need a materializing copy,
# so both branches of the operator are exercised.
_BROADCAST_ROWS = [
    ((1, 1), (3, 5), (15,)),
    ((1, 128), (2, 128), (16, 16)),
    ((128, 1), (128, 2), (16, 16)),
    ((1, 1, 1), (4, 5, 6), (10, 12)),
]
BROADCAST_CASES = tu.selected_cases(_BROADCAST_ROWS, quick=[])

# 'other' is shape-only: its dtype, its storage layout and its values must not
# change the result, so a NaN-sentinel operand would be exposed by the exact
# comparison below if a candidate read it as values. Both operands need their
# declared storage, so both dtypes are gated on the static capability flags.
_OTHER_ROWS = [
    row
    for row in (
        ((4, 6), torch.float32, (24,), torch.bool, "contiguous"),
        ((4, 6), torch.float32, (6, 4), torch.float64, "nan"),
        ((2, 3, 4), torch.int32, (12, 2), torch.int64, "transposed"),
        ((4, 6), torch.float16, (12, 2), torch.float16, "transposed"),
    )
    if _dtype_supported(row[1]) and _dtype_supported(row[3])
]
OTHER_CASES = tu.selected_cases(_OTHER_ROWS, quick=[])

# Contiguous layouts reshape into a view, so a write through the result must be
# observable in the input.
_MUTATION_ROWS = [
    ((24,), (2, 12)),
    ((3, 5, 2), (15, 2)),
    ((4, 6), (2, 12)),
]
MUTATION_CASES = tu.selected_cases(_MUTATION_ROWS, quick=[])

# Materializing layouts: the result owns fresh storage, which metadata equality
# alone cannot demonstrate.
_COPY_ROWS = [
    ((4, 6), "transposed", (24,)),
    ((10, 8), "offset_window", (2, 8)),
    ((12, 24), "both_steps", (2, 12)),
]
COPY_CASES = tu.selected_cases(_COPY_ROWS, quick=[])

# Conjugate layouts: a contiguous conjugate view is still stride-expressible, so
# the lazy bit survives the reshape; transposing first forces a materializing copy
# that resolves it.
_CONJ_ROWS = [("asis", (24,)), ("transposed", (24,))]
CONJ_CASES = tu.selected_cases(
    [
        (layout, target, dtype)
        for layout, target in _CONJ_ROWS
        for dtype in _CONJ_DTYPES
    ],
    quick=[],
)

# Backward. Each row differentiates with respect to the operand handed to the
# operator, so its gradient is a pure relayout of a non-uniform, independently
# placed upstream gradient and is compared exactly. Accumulation across expanded
# axes is covered separately by test_reshape_as_expanded_base_backward.
_BACKWARD_ROWS = [
    ((4, 6), "asis", None, (24,)),
    ((7, 13, 29), "asis", None, (7, 377)),
    ((4, 6), "transposed", None, (24,)),
    ((4, 12), "column_step", None, (2, 12)),
    ((1, 1), "expanded", (3, 5), (15,)),
]
BACKWARD_CASES = tu.selected_cases(
    [
        (shape, layout, expand, target, dtype)
        for shape, layout, expand, target in _BACKWARD_ROWS
        for dtype in _BACKWARD_DTYPES
    ],
    quick=[],
)
BASE_GRAD_CASES = tu.selected_cases(_BACKWARD_DTYPES, quick=[])
NO_GRAD_CASES = tu.selected_cases(_BACKWARD_DTYPES, quick=[])

# Positive special values are default-only as well.
SPECIAL_VALUE_CASES = tu.selected_cases(
    tu.special_value_cases(_RESHAPE_AS_DTYPES), quick=[]
)

# An 'other' whose numel differs from 'self' is rejected natively with
# RuntimeError, and neither operand may be a non-tensor.
_NUMEL_MISMATCH_ROWS = [
    ((4, 6), (5,)),
    ((3,), (2, 2)),
    ((24,), (4, 7)),
    ((0, 3), (4,)),
]


@pytest.mark.reshape_as
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _RESHAPE_AS_DTYPES)
def test_reshape_as(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    other = torch.zeros(_TARGET_SHAPES[shape], dtype=dtype, device=flag_gems.device)
    inp_before = inp.detach().clone()
    other_before = other.detach().clone()

    ref_out, facts = _native_facts(
        tu.to_reference(inp.detach()), tu.to_reference(other.detach())
    )
    res_out = flag_gems.reshape_as(inp, other)

    tu.assert_result_equal(res_out, ref_out)
    _assert_layout_facts(res_out, inp, facts)
    _assert_operands_unchanged(inp, inp_before, other, other_before)


@pytest.mark.reshape_as
@pytest.mark.parametrize("shape,target", SIZE_CASES)
def test_reshape_as_with_size(shape, target):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    other = torch.zeros(target, dtype=torch.float32, device=flag_gems.device)
    inp_before = inp.detach().clone()
    other_before = other.detach().clone()

    ref_out, facts = _native_facts(
        tu.to_reference(inp.detach()), tu.to_reference(other.detach())
    )
    res_out = flag_gems.reshape_as(inp, other)

    tu.assert_result_equal(res_out, ref_out)
    _assert_layout_facts(res_out, inp, facts)
    _assert_operands_unchanged(inp, inp_before, other, other_before)


@pytest.mark.reshape_as
@pytest.mark.parametrize("shape,layout,target", STRIDED_CASES)
def test_reshape_as_strided_input(shape, layout, target):
    inp, ref_inp, base = _layout_pair(shape, layout, torch.float32)
    other = torch.zeros(target, dtype=torch.float32, device=flag_gems.device)
    inp_before = inp.detach().clone()
    base_before = base.detach().clone()
    other_before = other.detach().clone()

    ref_out, facts = _native_facts(ref_inp, tu.to_reference(other.detach()))
    res_out = flag_gems.reshape_as(inp, other)

    tu.assert_result_equal(res_out, ref_out)
    _assert_layout_facts(res_out, inp, facts)
    _assert_operands_unchanged(inp, inp_before, other, other_before, base, base_before)


@pytest.mark.reshape_as
@pytest.mark.parametrize("storage,expand,target", BROADCAST_CASES)
def test_reshape_as_broadcast_input(storage, expand, target):
    inp, ref_inp, base = _layout_pair(storage, "expanded", torch.float32, expand)
    other = torch.zeros(target, dtype=torch.float32, device=flag_gems.device)
    inp_before = inp.detach().clone()
    base_before = base.detach().clone()
    other_before = other.detach().clone()

    ref_out, facts = _native_facts(ref_inp, tu.to_reference(other.detach()))
    res_out = flag_gems.reshape_as(inp, other)

    tu.assert_result_equal(res_out, ref_out)
    _assert_layout_facts(res_out, inp, facts)
    _assert_operands_unchanged(inp, inp_before, other, other_before, base, base_before)


@pytest.mark.reshape_as
@pytest.mark.parametrize("shape,dtype,other_shape,other_dtype,other_kind", OTHER_CASES)
def test_reshape_as_other_operand(shape, dtype, other_shape, other_dtype, other_kind):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    other = torch.ones(other_shape, dtype=other_dtype, device=flag_gems.device)
    if other_kind == "nan":
        other.fill_(float("nan"))
    elif other_kind == "transposed":
        other = other.t()
    inp_before = inp.detach().clone()
    other_before = other.detach().clone()

    ref_out, facts = _native_facts(
        tu.to_reference(inp.detach()), tu.to_reference(other.detach())
    )
    res_out = flag_gems.reshape_as(inp, other)

    # An exact comparison against a native result built from a sentinel operand
    # detects any candidate that reads 'other' as values instead of its shape.
    tu.assert_result_equal(res_out, ref_out)
    _assert_layout_facts(res_out, inp, facts)
    _assert_operands_unchanged(inp, inp_before, other, other_before)


@pytest.mark.reshape_as
@pytest.mark.parametrize("shape,target", MUTATION_CASES)
def test_reshape_as_view_writes_through(shape, target):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    other = torch.zeros(target, dtype=torch.float32, device=flag_gems.device)
    inp_before = inp.detach().clone()
    other_before = other.detach().clone()

    ref_out, facts = _native_facts(
        tu.to_reference(inp.detach()), tu.to_reference(other.detach())
    )
    res_out = flag_gems.reshape_as(inp, other)

    # Compare the untouched result first, then the operand contract.
    tu.assert_result_equal(res_out, ref_out)
    _assert_layout_facts(res_out, inp, facts)
    _assert_operands_unchanged(inp, inp_before, other, other_before)

    # These layouts are stride-expressible, so a write through the candidate
    # result has to reach the candidate input: that is what makes it a view.
    res_out.fill_(3.0)
    assert bool((inp == 3.0).all())


@pytest.mark.reshape_as
@pytest.mark.parametrize("shape,layout,target", COPY_CASES)
def test_reshape_as_materialized_copy_isolated(shape, layout, target):
    inp, ref_inp, base = _layout_pair(shape, layout, torch.float32)
    other = torch.zeros(target, dtype=torch.float32, device=flag_gems.device)
    inp_before = inp.detach().clone()
    base_before = base.detach().clone()
    other_before = other.detach().clone()

    ref_out, facts = _native_facts(ref_inp, tu.to_reference(other.detach()))
    res_out = flag_gems.reshape_as(inp, other)

    tu.assert_result_equal(res_out, ref_out)
    _assert_layout_facts(res_out, inp, facts)
    # Prove the candidate itself left its operands intact ...
    _assert_operands_unchanged(inp, inp_before, other, other_before, base, base_before)

    res_before = res_out.detach().clone()
    inp.fill_(7.0)
    base_filled = base.detach().clone()

    # ... then that a materializing result owns its storage: mutating the input
    # afterwards cannot change the already recorded result ...
    tu.assert_result_equal(res_out, res_before)

    res_out.fill_(-1.0)
    # ... and writing the result cannot reach the input or its backing storage.
    assert bool((inp == 7.0).all())
    tu.assert_result_equal(base, base_filled)


@pytest.mark.reshape_as
@pytest.mark.parametrize("layout,target,dtype", CONJ_CASES)
def test_reshape_as_conjugate_input(layout, target, dtype):
    base = tu.make_input(dtype, (4, 6), ["-1", "1"])
    ref_base = tu.to_reference(base.detach())
    if layout == "asis":
        # A contiguous conjugate view is still stride-expressible, so the lazy
        # conjugate bit survives the reshape.
        inp = base.conj()
        ref_inp = ref_base.conj()
        expect_conj = True
    else:
        # Transposing first forces a materializing copy, which resolves the bit.
        inp = base.t().conj()
        ref_inp = ref_base.t().conj()
        expect_conj = False
    other = torch.zeros(target, dtype=dtype, device=flag_gems.device)
    inp_before = inp.detach().clone()
    base_before = base.detach().clone()
    other_before = other.detach().clone()

    ref_out, facts = _native_facts(ref_inp, tu.to_reference(other.detach()))
    res_out = flag_gems.reshape_as(inp, other)

    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.is_conj() is expect_conj
    tu.assert_result_equal(res_out, ref_out)
    _assert_layout_facts(res_out, inp, facts)
    _assert_operands_unchanged(inp, inp_before, other, other_before, base, base_before)


@pytest.mark.reshape_as
@pytest.mark.parametrize("shape,layout,expand,target,dtype", BACKWARD_CASES)
def test_reshape_as_backward(shape, layout, expand, target, dtype):
    base = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_base = tu.to_reference(base.detach()).requires_grad_(True)
    inp = _apply_layout(base, layout, expand)
    ref_inp = _apply_layout(ref_base, layout, expand)

    other = torch.zeros(target, dtype=dtype, device=flag_gems.device)
    upstream = tu.make_input(dtype, target, ["-1", "1"])
    ref_upstream = tu.to_reference(upstream.detach())
    inp_before = inp.detach().clone()
    base_before = base.detach().clone()
    other_before = other.detach().clone()

    ref_out, facts = _native_facts(ref_inp, tu.to_reference(other.detach()))
    res_out = flag_gems.reshape_as(inp, other)

    # Check the forward result first. Every gradient below is a pure relayout of
    # the non-uniform upstream gradient, so it is compared exactly.
    tu.assert_result_equal(res_out, ref_out)
    _assert_layout_facts(res_out, inp, facts)
    _assert_operands_unchanged(inp, inp_before, other, other_before, base, base_before)

    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]
    ref_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)[0]

    tu.assert_result_equal(res_grad, ref_grad)


@pytest.mark.reshape_as
@pytest.mark.parametrize("dtype", BASE_GRAD_CASES)
def test_reshape_as_expanded_base_backward(dtype):
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
    inp_before = inp.detach().clone()
    base_before = base.detach().clone()
    other_before = other.detach().clone()

    ref_out, facts = _native_facts(ref_inp, tu.to_reference(other.detach()))
    res_out = flag_gems.reshape_as(inp, other)

    tu.assert_result_equal(res_out, ref_out)
    _assert_layout_facts(res_out, inp, facts)
    _assert_operands_unchanged(inp, inp_before, other, other_before, base, base_before)

    res_grad = torch.autograd.grad(res_out, base, grad_outputs=upstream)[0]
    ref_grad = torch.autograd.grad(ref_out, ref_base, grad_outputs=ref_upstream)[0]

    tu.assert_result_equal(res_grad, ref_grad)


@pytest.mark.reshape_as
@pytest.mark.parametrize("dtype", NO_GRAD_CASES)
def test_reshape_as_other_has_no_gradient(dtype):
    inp = tu.make_input(dtype, (4, 6), ["-1", "1"])
    other = tu.make_input(dtype, (24,), ["-1", "1"])

    ref_out = torch.ops.aten.reshape_as(
        tu.to_reference(inp.detach()), tu.to_reference(other.detach())
    )
    inp.requires_grad_(True)
    other.requires_grad_(True)
    inp_before = inp.detach().clone()
    other_before = other.detach().clone()

    res_out = flag_gems.reshape_as(inp, other)

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


@pytest.mark.reshape_as
@pytest.mark.parametrize("dtype,scenario", SPECIAL_VALUE_CASES)
def test_reshape_as_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    other = torch.zeros((1, inp.numel()), dtype=dtype, device=flag_gems.device)
    inp_before = inp.detach().clone()
    other_before = other.detach().clone()

    ref_out, facts = _native_facts(
        tu.to_reference(inp.detach()), tu.to_reference(other.detach())
    )
    res_out = flag_gems.reshape_as(inp, other)

    tu.assert_result_equal(res_out, ref_out)
    _assert_layout_facts(res_out, inp, facts)
    _assert_operands_unchanged(inp, inp_before, other, other_before)


@pytest.mark.reshape_as
@pytest.mark.parametrize("shape,target", _NUMEL_MISMATCH_ROWS)
def test_reshape_as_numel_mismatch(shape, target):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    other = torch.zeros(target, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.reshape_as(inp, other)


@pytest.mark.reshape_as
@pytest.mark.parametrize("bad_operand", ["self", "other"])
def test_reshape_as_non_tensor_operand(bad_operand):
    inp = tu.make_input(torch.float32, (4, 6), ["-1", "1"])
    other = torch.zeros((2, 12), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError)):
        if bad_operand == "self":
            flag_gems.reshape_as([[0.0] * 6] * 4, other)
        else:
            flag_gems.reshape_as(inp, [2, 12])
