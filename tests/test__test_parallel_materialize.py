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

"""Correctness tests for aten::_test_parallel_materialize.

The operator materializes a lazily conj/neg viewed operand: an already
materialized operand is returned as the very same object, while a lazy one comes
back as a fresh contiguous tensor on the operand device with the lazy bit
cleared. num_parallel and skip_first only describe how the work is segmented and
never change the result.
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu

_DEVICE = flag_gems.runtime.device

# Static capability flags: a False flag drops the dtype from the collected grid.
# The four names below are real runtime attributes on every backend.
_DTYPE_FLAGS = {
    torch.bfloat16: _DEVICE.support_bf16,
    torch.float64: _DEVICE.support_fp64,
    torch.complex128: _DEVICE.support_fp64,
    torch.int64: _DEVICE.support_int64,
    torch.float8_e4m3fn: _DEVICE.support_fp8,
    torch.float8_e5m2: _DEVICE.support_fp8,
}
_SPEC_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
)
# Additional native-supported families, filtered by their static capabilities.
_EXTRA_DTYPES = (
    torch.float64,
    torch.int16,
    torch.complex64,
    torch.complex128,
    torch.bool,
)
SUPPORTED_DTYPES = tuple(
    dtype for dtype in _SPEC_DTYPES + _EXTRA_DTYPES if _DTYPE_FLAGS.get(dtype, True)
)
FLOAT_DTYPES = tuple(dtype for dtype in SUPPORTED_DTYPES if dtype.is_floating_point)
# Parameter workloads use the dtype list of the supplement spec.
PARAM_DTYPES = tuple(
    dtype
    for dtype in (
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.int32,
        torch.int64,
        torch.int8,
        torch.uint8,
        torch.float8_e4m3fn,
    )
    if _DTYPE_FLAGS.get(dtype, True)
)

_NUM_PARALLEL = 2
# num_parallel is a segmentation hint: 0, negative and very large values all
# materialize the same result natively, so the whole sweep stays in the grid.
_NUM_PARALLEL_VALUES = [0, 1, -1, 3, 10**6]
_SKIP_FIRST_VALUES = [False, True]
_PARAM_SHAPE = (1024, 1024)
_PARAM_RANGE = ["-1", "1"]
_LAZY_SHAPE = (2, 19, 7)
_ALIAS_SHAPE = (256,)
_BACKWARD_SHAPE = (4, 5)

# Identity rows: the spec shape x range grid plus layout variants that share the
# same call form (zero elements, a stride-0 expansion, a permutation, a nonzero
# storage offset, channels-last). Layout rows pin [-1, 1] because they cover
# layout, not values; the scalar shape () stays in the grid with all five ranges.
_LAYOUT_ROWS = (
    ("empty", (0, 7)),
    ("expanded", (7,)),
    ("transposed", (5, 7, 3)),
    ("offset", (3, 4, 5)),
    ("channels_last", (2, 4, 6, 6)),
)

IDENTITY_ROWS = tu.selected_cases(
    [
        (None, shape, value_range)
        for shape in tu.REQUIRED_SHAPES
        for value_range in tu.REQUIRED_RANGES
    ]
    + [(kind, shape, _PARAM_RANGE) for kind, shape in _LAYOUT_ROWS],
    quick=[
        (None, shape, value_range)
        for shape in tu.QUICK_SHAPES
        for value_range in tu.QUICK_RANGES
    ]
    + [(kind, shape, _PARAM_RANGE) for kind, shape in _LAYOUT_ROWS],
)


def _layout_operand(kind, dtype, shape, value_range):
    """Dense operand that carries a non-default layout into the identity call."""
    base = tu.make_input(dtype, shape, value_range)
    if kind == "expanded":
        return base.expand(4, shape[0])
    if kind == "transposed":
        return base.permute(2, 0, 1)
    if kind == "offset":
        return base[1:3]
    if kind == "channels_last":
        return base.to(memory_format=torch.channels_last)
    return base  # "empty": a zero-element operand


@pytest.mark.test_parallel_materialize
@pytest.mark.parametrize("kind,shape,value_range", IDENTITY_ROWS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__test_parallel_materialize_identity(kind, shape, value_range, dtype):
    if kind is None:
        inp = tu.make_input(dtype, shape, value_range)
    else:
        inp = _layout_operand(kind, dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_parallel_materialize(ref_inp, _NUM_PARALLEL)
    res_out = flag_gems._test_parallel_materialize(inp, _NUM_PARALLEL)

    # A materialized operand comes back as the same object, so storage, layout
    # and autograd state are all preserved by the call.
    assert res_out is inp
    tu.assert_result_equal(res_out, ref_out)


# torch._neg_view needs a native negation kernel, which CUDA has none for on FP8
# ("neg_cuda" not implemented for 'Float8_e4m3fn' / 'Float8_e5m2') or bool, and a
# lazy conj bit only exists for complex, so those dtypes keep their dense
# coverage above and are not used to build lazy operands.
LAZY_ROWS = (
    ("neg", torch.float32),
    ("neg", torch.int32),
    ("neg", torch.complex64),
    ("conj", torch.complex64),
)
_LAZY_PARAMS = [
    (num_parallel, skip)
    for num_parallel in _NUM_PARALLEL_VALUES + [_NUM_PARALLEL]
    for skip in _SKIP_FIRST_VALUES
]


def _lazy_operand(tensor, kind):
    """Attach the lazy bit the way the native materialization paths see it."""
    return tensor.conj() if kind == "conj" else torch._neg_view(tensor)


def _is_lazy(tensor, kind):
    return tensor.is_conj() if kind == "conj" else tensor.is_neg()


@pytest.mark.test_parallel_materialize
@pytest.mark.parametrize("kind,dtype", LAZY_ROWS)
@pytest.mark.parametrize("offset", [False, True])
@pytest.mark.parametrize("num_parallel,skip_first", _LAZY_PARAMS)
def test__test_parallel_materialize_lazy(kind, dtype, offset, num_parallel, skip_first):
    base = tu.make_input(dtype, _LAZY_SHAPE, _PARAM_RANGE)
    source = base[1:] if offset else base
    ref_source = tu.to_reference(source)
    inp = _lazy_operand(source, kind)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_parallel_materialize(
        ref_inp, num_parallel, skip_first
    )
    res_out = flag_gems._test_parallel_materialize(inp, num_parallel, skip_first)

    # The lazy bit is cleared on a fresh contiguous tensor on the same device ...
    assert res_out is not inp
    assert not _is_lazy(res_out, kind)
    assert res_out.device == inp.device
    assert res_out.is_contiguous()
    tu.assert_result_equal(res_out, ref_out)
    # ... while the operand and the tensor it views stay untouched.
    assert _is_lazy(inp, kind)
    tu.assert_result_equal(source, ref_source)


@pytest.mark.test_parallel_materialize
@pytest.mark.parametrize("num_parallel", _NUM_PARALLEL_VALUES)
@pytest.mark.parametrize("dtype", PARAM_DTYPES)
def test__test_parallel_materialize_num_parallel(num_parallel, dtype):
    inp = tu.make_input(dtype, _PARAM_SHAPE, _PARAM_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_parallel_materialize(ref_inp, num_parallel)
    res_out = flag_gems._test_parallel_materialize(inp, num_parallel)

    assert res_out is inp
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.test_parallel_materialize
@pytest.mark.parametrize("skip_first", _SKIP_FIRST_VALUES)
@pytest.mark.parametrize("dtype", PARAM_DTYPES)
def test__test_parallel_materialize_skip_first(skip_first, dtype):
    inp = tu.make_input(dtype, _PARAM_SHAPE, _PARAM_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_parallel_materialize(
        ref_inp, _NUM_PARALLEL, skip_first
    )
    res_out = flag_gems._test_parallel_materialize(inp, _NUM_PARALLEL, skip_first)

    assert res_out is inp
    tu.assert_result_equal(res_out, ref_out)


_CALL_FORMS = (
    ((_NUM_PARALLEL,), {}),
    ((_NUM_PARALLEL, True), {}),
    ((_NUM_PARALLEL,), {"skip_first": True}),
    ((), {"num_parallel": _NUM_PARALLEL, "skip_first": False}),
)


@pytest.mark.test_parallel_materialize
@pytest.mark.parametrize(
    "args,kwargs",
    _CALL_FORMS,
    ids=[
        "skip_first_default",
        "positional",
        "skip_first_keyword",
        "num_parallel_keyword",
    ],
)
def test__test_parallel_materialize_call_forms(args, kwargs):
    inp = tu.make_input(torch.float32, _PARAM_SHAPE, _PARAM_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_parallel_materialize(ref_inp, *args, **kwargs)
    res_out = flag_gems._test_parallel_materialize(inp, *args, **kwargs)

    assert res_out is inp
    tu.assert_result_equal(res_out, ref_out)


# fill_ works for every dtype an identity result can carry, so the mutation is
# observable for floats, ints, bool and complex alike.
_ALIAS_DTYPES = (torch.float32, torch.float16, torch.int32, torch.bool, torch.complex64)


@pytest.mark.test_parallel_materialize
@pytest.mark.parametrize("dtype", _ALIAS_DTYPES)
def test__test_parallel_materialize_alias_storage(dtype):
    inp = tu.make_input(dtype, _ALIAS_SHAPE, _PARAM_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_parallel_materialize(ref_inp, _NUM_PARALLEL)
    res_out = flag_gems._test_parallel_materialize(inp, _NUM_PARALLEL)

    # Native returns the operand itself, so a write through the result shows up
    # in the operand: mutate the independent native and candidate results and
    # compare the candidate operand against the independent native operand.
    res_out.fill_(1)
    ref_out.fill_(1)
    tu.assert_result_equal(inp, ref_inp)


# The dense identity path never inspects the payload, so the special-value matrix
# is also run through the materializing (lazy) path for the float dtypes that can
# carry a lazy bit, where the copy has to preserve NaN/Inf exactly.
_SPECIAL_LAZY_DTYPES = [
    dtype
    for dtype in (torch.float16, torch.float32, torch.bfloat16, torch.float64)
    if dtype in SUPPORTED_DTYPES
]
SPECIAL_CASES = tu.selected_cases(
    [
        (None, dtype, scenario)
        for dtype, scenario in tu.special_value_cases(FLOAT_DTYPES)
    ]
    + [
        ("neg", dtype, scenario)
        for dtype, scenario in tu.special_value_cases(_SPECIAL_LAZY_DTYPES)
    ],
    quick=[],
)


@pytest.mark.test_parallel_materialize
@pytest.mark.parametrize("kind,dtype,scenario", SPECIAL_CASES)
def test__test_parallel_materialize_special_values(kind, dtype, scenario):
    source = tu.make_special_input(dtype, scenario)
    inp = source if kind is None else _lazy_operand(source, kind)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_parallel_materialize(ref_inp, _NUM_PARALLEL)
    res_out = flag_gems._test_parallel_materialize(inp, _NUM_PARALLEL)

    if kind is None:
        assert res_out is inp
    else:
        assert res_out is not inp
        assert not _is_lazy(res_out, kind)
    tu.assert_result_equal(res_out, ref_out)


# Native registers no derivative for this operator: the operator call itself
# succeeds on a lazy operand, but differentiating the materialized result with
# respect to the original leaf raises. That negative contract stays in quick.
_BACKWARD_ROWS = (("neg", torch.float32), ("conj", torch.complex64))


@pytest.mark.test_parallel_materialize
@pytest.mark.parametrize("kind,dtype", _BACKWARD_ROWS)
def test__test_parallel_materialize_backward_not_implemented(kind, dtype):
    inp = tu.make_input(dtype, _BACKWARD_SHAPE, _PARAM_RANGE).requires_grad_(True)
    ref_inp = tu.to_reference(inp).detach().clone().requires_grad_(True)
    operand = _lazy_operand(inp, kind)
    ref_operand = _lazy_operand(ref_inp, kind)

    ref_out = torch.ops.aten._test_parallel_materialize(ref_operand, _NUM_PARALLEL)
    res_out = flag_gems._test_parallel_materialize(operand, _NUM_PARALLEL)
    tu.assert_result_equal(res_out, ref_out)

    upstream = tu.make_input(dtype, _BACKWARD_SHAPE, _PARAM_RANGE)

    with pytest.raises(
        RuntimeError,
        match="derivative for aten::_test_parallel_materialize is not implemented",
    ):
        torch.autograd.grad(ref_out, ref_inp, grad_outputs=tu.to_reference(upstream))
    with pytest.raises(
        RuntimeError,
        match="derivative for aten::_test_parallel_materialize is not implemented",
    ):
        torch.autograd.grad(res_out, inp, grad_outputs=upstream)


# No dtype is rejected natively (all covered dtypes work) and the rank is free,
# so the negative rows cover the operand kind and the parameter types.
_NON_TENSOR_OPERANDS = (1, None, [1.0, 2.0])


@pytest.mark.test_parallel_materialize
@pytest.mark.parametrize("operand", _NON_TENSOR_OPERANDS, ids=["int", "none", "list"])
def test__test_parallel_materialize_rejects_non_tensor(operand):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_parallel_materialize(operand, _NUM_PARALLEL)


@pytest.mark.test_parallel_materialize
@pytest.mark.parametrize("num_parallel", [1.5, "3", None])
def test__test_parallel_materialize_rejects_invalid_num_parallel(num_parallel):
    inp = tu.make_input(torch.float32, _ALIAS_SHAPE, _PARAM_RANGE)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_parallel_materialize(inp, num_parallel)


@pytest.mark.test_parallel_materialize
def test__test_parallel_materialize_requires_num_parallel():
    inp = tu.make_input(torch.float32, _ALIAS_SHAPE, _PARAM_RANGE)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_parallel_materialize(inp)


@pytest.mark.test_parallel_materialize
def test__test_parallel_materialize_rejects_invalid_skip_first():
    inp = tu.make_input(torch.float32, _ALIAS_SHAPE, _PARAM_RANGE)

    # Natively skip_first accepts 1.5 / None (no bool conversion failure), so
    # only a value such as "yes" is a rejected parameter value.
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_parallel_materialize(inp, _NUM_PARALLEL, "yes")


@pytest.mark.test_parallel_materialize
def test__test_parallel_materialize_rejects_sparse_operand():
    indices = torch.tensor([[0, 1], [1, 2]], device=flag_gems.device)
    values = torch.tensor([1.0, 2.0], device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, (4, 5), device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_parallel_materialize(inp, _NUM_PARALLEL)
