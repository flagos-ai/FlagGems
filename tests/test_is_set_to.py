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

# aten::is_set_to(Tensor self, Tensor tensor) -> bool reports whether both
# operands describe the same storage view: identical untyped storage, storage
# offset, sizes and strides, with no lazy conjugate/negative bit. Only allocation
# metadata is inspected, so the elementwise value-range x shape x dtype grid is
# replaced by (aliasing relation, shape, dtype) workloads checked against the
# native operator, plus explicit value-invariance and NaN/Inf payload tests.

_VALUE_RANGE = ["-1", "1"]

# (aliasing relation, expected result). Each relation is rebuilt from the
# candidate operand and, separately, from the reference operand; transferring the
# two halves of a pair independently would destroy the aliasing they carry.
_KIND_ROWS = [
    ("same_object", True),
    ("detached", True),
    ("same_geometry_view", True),
    ("storage_set_twin", True),
    ("clone", False),
    ("narrowed", False),
    ("strided_slice", False),
    ("transposed", False),
    ("expanded", False),
    # reshape(-1) preserves the geometry only of a 1-D input (0-D gains a
    # dimension, rank >= 2 changes sizes or strides).
    ("flattened_same_geometry", True),
    ("flattened_rank_change", False),
]

_DTYPES = list(tu.REQUIRED_DTYPES) + [torch.bool]
if utils.fp64_is_supported:
    _DTYPES.append(torch.float64)
_DTYPES = [dtype for dtype in _DTYPES if _DTYPE_FLAGS.get(dtype, True)]


def _build_pair(kind, base):
    """Return the two operands describing one aliasing relation for base."""
    if kind == "same_object":
        return base, base
    if kind == "detached":
        return base, base.detach()
    if kind == "same_geometry_view":
        return base, base.as_strided(base.shape, base.stride(), base.storage_offset())
    if kind == "storage_set_twin":
        # A second TensorImpl over the same storage: same view, new object.
        twin = torch.empty(0, dtype=base.dtype, device=base.device)
        twin.set_(
            base.untyped_storage(), base.storage_offset(), base.size(), base.stride()
        )
        return base, twin
    if kind == "clone":
        return base, base.clone()
    if kind == "narrowed":
        return base, base[1:]
    if kind == "strided_slice":
        return base, base[:, ::2]
    if kind == "transposed":
        return base, base.transpose(0, -1)
    if kind == "expanded":
        return base, base.unsqueeze(0).expand((2,) + tuple(base.shape))
    if kind in ("flattened_same_geometry", "flattened_rank_change"):
        return base, base.reshape(-1)
    raise ValueError(f"unknown aliasing relation {kind!r}")


def _kind_applies(kind, shape):
    """Rank/extent restrictions of the relation built by _build_pair."""
    if kind == "narrowed":
        return len(shape) >= 1 and shape[0] >= 2
    if kind == "strided_slice":
        return len(shape) >= 2 and shape[1] >= 2
    if kind == "transposed":
        return len(shape) >= 2
    if kind == "flattened_same_geometry":
        return len(shape) == 1
    if kind == "flattened_rank_change":
        return len(shape) != 1
    return True


def _shape_tag(shape):
    return "scalar" if len(shape) == 0 else "x".join(str(dim) for dim in shape)


def _dtype_tag(dtype):
    return str(dtype).split(".")[-1]


def _layout_facts(tensor):
    """Allocation metadata of one operand: identity and geometry, not contents."""
    return (
        tensor.data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.size()),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.layout,
        tensor.is_conj(),
        tensor.is_neg(),
    )


def _assert_operands_intact(op_a, op_b, facts):
    # A metadata query must not modify the views it inspects.
    assert (_layout_facts(op_a), _layout_facts(op_b)) == facts


_CASES = [
    pytest.param(
        kind,
        expected,
        shape,
        dtype,
        id=f"{kind}-{_shape_tag(shape)}-{_dtype_tag(dtype)}",
    )
    for shape in list(tu.selected_shapes()) + [(0,), (0, 3)]
    for kind, expected in _KIND_ROWS
    if _kind_applies(kind, shape)
    for dtype in _DTYPES
]


@pytest.mark.is_set_to
@pytest.mark.parametrize("kind,expected,shape,dtype", _CASES)
def test_is_set_to_alias_pairs(kind, expected, shape, dtype):
    inp = tu.make_input(dtype, shape, _VALUE_RANGE)
    ref_inp = tu.to_reference(inp)
    cand_self, cand_other = _build_pair(kind, inp)
    ref_self, ref_other = _build_pair(kind, ref_inp)
    facts = (_layout_facts(cand_self), _layout_facts(cand_other))

    ref_out = torch.ops.aten.is_set_to(ref_self, ref_other)
    res_out = flag_gems.is_set_to(cand_self, cand_other)

    # A Python bool, not a tensor: type and value are compared directly.
    assert type(res_out) is bool
    assert res_out is expected
    assert res_out == ref_out

    _assert_operands_intact(cand_self, cand_other, facts)


_RANGE_CASES = [
    pytest.param(kind, value_range, id=f"{kind}-{value_range[0]}_{value_range[1]}")
    for kind in ("same_object", "clone")
    for value_range in tu.selected_ranges()
]


@pytest.mark.is_set_to
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("kind,value_range", _RANGE_CASES)
def test_is_set_to_ignores_element_values(kind, value_range, dtype):
    # Only allocation metadata is compared, so the answer must not depend on the
    # spec's five value ranges; the other relations run once in the pair grid.
    inp = tu.make_input(dtype, (20, 320, 15), value_range)
    ref_inp = tu.to_reference(inp)
    cand_self, cand_other = _build_pair(kind, inp)
    ref_self, ref_other = _build_pair(kind, ref_inp)
    facts = (_layout_facts(cand_self), _layout_facts(cand_other))

    ref_out = torch.ops.aten.is_set_to(ref_self, ref_other)
    res_out = flag_gems.is_set_to(cand_self, cand_other)

    assert type(res_out) is bool
    assert res_out is (kind == "same_object")
    assert res_out == ref_out

    _assert_operands_intact(cand_self, cand_other, facts)


# Positive NaN/Inf payloads are default-only: a NaN element must not turn an
# identical view pair into 'not set to'.
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_DTYPES), quick=[])


@pytest.mark.is_set_to
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_is_set_to_ignores_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario).reshape(1, -1)
    ref_inp = tu.to_reference(inp)
    cand_self, cand_other = _build_pair("same_object", inp)
    ref_self, ref_other = _build_pair("same_object", ref_inp)
    facts = (_layout_facts(cand_self), _layout_facts(cand_other))

    ref_out = torch.ops.aten.is_set_to(ref_self, ref_other)
    res_out = flag_gems.is_set_to(cand_self, cand_other)

    assert type(res_out) is bool
    assert res_out is True
    assert res_out == ref_out

    _assert_operands_intact(cand_self, cand_other, facts)


_VIEW_DTYPE_PAIRS = [
    pytest.param(torch.int8, torch.uint8, id="int8-uint8"),
    pytest.param(torch.int32, torch.float32, id="int32-float32"),
    pytest.param(torch.float8_e4m3fn, torch.int8, id="fp8e4m3fn-int8"),
    *(
        [pytest.param(torch.int64, torch.float64, id="int64-float64")]
        if utils.fp64_is_supported
        else []
    ),
]


@pytest.mark.is_set_to
@pytest.mark.parametrize("src_dtype,view_dtype", _VIEW_DTYPE_PAIRS)
def test_is_set_to_dtype_reinterpretation_view(src_dtype, view_dtype):
    # view(dtype) keeps the storage, offset and byte strides, so the two operands
    # still describe one storage view although their dtypes differ.
    inp = tu.make_input(src_dtype, (4, 8), _VALUE_RANGE)
    ref_inp = tu.to_reference(inp)
    cand_view = inp.view(view_dtype)
    ref_view = ref_inp.view(view_dtype)

    ref_out = torch.ops.aten.is_set_to(ref_inp, ref_view)
    res_out = flag_gems.is_set_to(inp, cand_view)

    assert type(res_out) is bool
    assert res_out is True
    assert res_out == ref_out


@pytest.mark.is_set_to
def test_is_set_to_lazy_conjugate_bit():
    # An unresolved conjugate bit is part of the compared metadata, so a lazy
    # conj view is not set to anything, not even to itself; materializing it
    # (resolve_conj) yields a plain tensor that is set to itself.
    inp = tu.make_input(torch.complex64, (4, 8), _VALUE_RANGE)
    ref_inp = tu.to_reference(inp)
    lazy = torch.conj(inp)
    ref_lazy = torch.conj(ref_inp)

    ref_out = torch.ops.aten.is_set_to(ref_lazy, ref_lazy)
    res_out = flag_gems.is_set_to(lazy, lazy)

    assert type(res_out) is bool
    assert res_out is False
    assert res_out == ref_out

    resolved = lazy.resolve_conj()
    ref_resolved = ref_lazy.resolve_conj()
    ref_out = torch.ops.aten.is_set_to(ref_resolved, ref_resolved)
    res_out = flag_gems.is_set_to(resolved, resolved)

    assert type(res_out) is bool
    assert res_out is True
    assert res_out == ref_out


@pytest.mark.is_set_to
def test_is_set_to_lazy_negative_bit():
    # Same lazy-bit rule as for the conjugate flag, for the negative bit that
    # neg views carry on real dtypes.
    inp = tu.make_input(torch.float32, (4, 8), _VALUE_RANGE)
    ref_inp = tu.to_reference(inp)
    lazy = torch._neg_view(inp)
    ref_lazy = torch._neg_view(ref_inp)

    ref_out = torch.ops.aten.is_set_to(ref_lazy, ref_lazy)
    res_out = flag_gems.is_set_to(lazy, lazy)

    assert type(res_out) is bool
    assert res_out is False
    assert res_out == ref_out

    resolved = lazy.resolve_neg()
    ref_resolved = ref_lazy.resolve_neg()
    ref_out = torch.ops.aten.is_set_to(ref_resolved, ref_resolved)
    res_out = flag_gems.is_set_to(resolved, resolved)

    assert type(res_out) is bool
    assert res_out is True
    assert res_out == ref_out


# Invalid operand types, kept in both the default and the --quick run.
_NEGATIVE_ROWS = [
    pytest.param("self", 3, id="self-int"),
    pytest.param("self", 2.5, id="self-float"),
    pytest.param("self", "x", id="self-str"),
    pytest.param("self", None, id="self-none"),
    pytest.param("other", 3, id="other-int"),
    pytest.param("other", 2.5, id="other-float"),
    pytest.param("other", "x", id="other-str"),
    pytest.param("other", None, id="other-none"),
]


@pytest.mark.is_set_to
@pytest.mark.parametrize("position,bad_value", _NEGATIVE_ROWS)
def test_is_set_to_rejects_non_tensor_operands(position, bad_value):
    inp = tu.make_input(torch.float32, (4,), _VALUE_RANGE)
    args = [inp, inp]
    args[0 if position == "self" else 1] = bad_value

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_set_to(*args)


@pytest.mark.is_set_to
@pytest.mark.parametrize("offset", [0, 1])
def test_is_set_to_empty_views_with_offset(offset):
    base = tu.make_input(torch.float32, (8,), _VALUE_RANGE)
    ref_base = tu.to_reference(base)
    inp, other = base[:0], base[offset:offset]
    ref_inp, ref_other = ref_base[:0], ref_base[offset:offset]

    ref_out = torch.ops.aten.is_set_to(ref_inp, ref_other)
    res_out = flag_gems.is_set_to(inp, other)

    assert type(res_out) is bool
    assert res_out == ref_out
    assert res_out is (offset == 0)
