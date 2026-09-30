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

# aten::is_conj(Tensor self) -> bool only reads the tensor's lazy conjugate bit,
# so coverage varies that state (and the dtype, because only complex tensors can
# carry the bit) instead of comparing computed values. Broadcast and backward do
# not apply: there is no second operand and no differentiable output.

_COMPLEX_DTYPES = [torch.complex32, torch.complex64] + (
    [torch.complex128] if utils.fp64_is_supported else []
)

_DTYPE_FLAGS = {
    torch.bfloat16: "bf16_is_supported",
    torch.float8_e4m3fn: "fp8_is_supported",
    torch.float8_e5m2: "fp8_is_supported",
    torch.float64: "fp64_is_supported",
}


def _dtype_supported(dtype):
    flag = _DTYPE_FLAGS.get(dtype)
    return getattr(utils, flag) if flag else True


# Probed on the active backend: every REQUIRED_DTYPES entry, float64, int16 and
# bool are accepted at all five spec ranges.
_DTYPES = (
    [d for d in tu.REQUIRED_DTYPES if _dtype_supported(d)]
    + ([torch.float64] if utils.fp64_is_supported else [])
    + [torch.int16, torch.bool]
    + _COMPLEX_DTYPES
)


def _conj_state(base, state):
    """Return a tensor in the requested lazy-flag state, preserving layout."""
    if state == "plain":
        return base
    if state == "conj":
        return base.conj()
    if state == "conj_transposed":
        return base.conj().transpose(0, 1)
    if state == "transposed_conj":
        return base.transpose(0, 1).conj()
    if state == "conj_sliced":
        return base.conj()[1:-1]
    if state == "conj_strided":
        return base.conj().transpose(1, 2)
    if state == "conj_narrowed":
        return base.conj().narrow(0, 1, base.shape[0] - 2)
    if state == "conj_expanded":
        # expand only broadcasts size-1 dimensions, so insert one first: the
        # leading stride becomes 0 while the conjugate bit and inner strides stay.
        return base.conj().unsqueeze(0).expand(2, *base.shape)
    if state == "conj_axis_swapped":
        return base.conj().transpose(0, -1)
    if state == "conj_resolved":
        return base.conj().resolve_conj()
    if state == "conj_cloned":
        return base.conj().clone()
    if state == "conj_contiguous":
        return base.conj().contiguous()
    if state == "conj_detached":
        return base.conj().detach()
    if state == "neg":
        return torch._neg_view(base)
    if state == "neg_of_conj":
        return torch._neg_view(base.conj())
    if state == "conj_of_neg":
        return torch._neg_view(base).conj()
    if state == "neg_of_conj_transposed":
        return torch._neg_view(base.conj()).transpose(0, 1)
    raise AssertionError(f"unknown state {state}")


@pytest.mark.is_conj
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_is_conj(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.is_conj(ref_inp)
    res_out = flag_gems.is_conj(inp)

    assert type(res_out) is bool
    assert res_out == ref_out
    # Querying must not set the flag it reads.
    assert inp.is_conj() == ref_inp.is_conj()


# State rows: each row keeps a distinct lazy-flag composition or view layout.
_STATE_ROWS = [
    ("plain", (1024, 1024)),
    ("conj", (1024, 1024)),
    ("conj_transposed", (20, 320, 15)),
    ("transposed_conj", (20, 320, 15)),
    ("conj_sliced", (20, 320, 15)),
    ("conj_strided", (16, 128, 64, 60)),
    ("conj_narrowed", (1024, 1024)),
    ("conj_expanded", (16, 128, 64, 60)),
    ("conj_axis_swapped", (16, 7, 57, 32, 29)),
    ("conj_resolved", (1024, 1024)),
    ("conj_cloned", (1024, 1024)),
    ("conj_contiguous", (16, 128, 64, 60)),
    ("conj_detached", (1024, 1024)),
    ("neg", (1024, 1024)),
    ("neg_of_conj", (20, 320, 15)),
    ("conj_of_neg", (20, 320, 15)),
    ("neg_of_conj_transposed", (16, 128, 64, 60)),
]

# Quick keeps every lazy-flag state on the smallest shapes the compositions
# accept, so no state is dropped: slice and narrow need dim0 >= 3 to stay
# non-empty, everything else fits the quick shape.
_QUICK_STATE_ROWS = [
    (state, (4, 19, 7) if state in ("conj_sliced", "conj_narrowed") else (2, 19, 7))
    for state, _ in _STATE_ROWS
]

_STATE_CASES = tu.selected_cases(_STATE_ROWS, quick=_QUICK_STATE_ROWS)


@pytest.mark.is_conj
@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES + [torch.float32])
@pytest.mark.parametrize("state,shape", _STATE_CASES)
def test_is_conj_state(state, shape, dtype):
    base = tu.make_input(dtype, shape, tu.selected_ranges()[0])
    ref_base = tu.to_reference(base)

    inp = _conj_state(base, state)
    ref_inp = _conj_state(ref_base, state)

    ref_out = torch.ops.aten.is_conj(ref_inp)
    res_out = flag_gems.is_conj(inp)

    assert type(res_out) is bool
    assert res_out == ref_out
    assert inp.is_conj() == ref_inp.is_conj()
    assert inp.is_neg() == ref_inp.is_neg()


_BOUNDARY_DTYPES = _COMPLEX_DTYPES + [torch.float32, torch.int64, torch.bool]


@pytest.mark.is_conj
@pytest.mark.parametrize("dtype", _BOUNDARY_DTYPES)
@pytest.mark.parametrize("state", ["plain", "conj"])
@pytest.mark.parametrize("shape", [(), (1,), (0,), (3, 0)])
def test_is_conj_boundary_shapes(shape, state, dtype):
    base = tu.make_input(dtype, shape, tu.selected_ranges()[0])
    ref_base = tu.to_reference(base)

    inp = _conj_state(base, state)
    ref_inp = _conj_state(ref_base, state)

    ref_out = torch.ops.aten.is_conj(ref_inp)
    res_out = flag_gems.is_conj(inp)

    assert type(res_out) is bool
    assert res_out == ref_out


# Complex special values are added explicitly; the shared generator already
# exempts e4m3fn inf/mixed because fp8 cannot represent inf.
_SPECIAL_DTYPES = [d for d in _DTYPES if d.is_floating_point]
_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases(_SPECIAL_DTYPES)
    + [(d, s) for d in _COMPLEX_DTYPES for s in ("nan", "inf", "mixed")],
    quick=[],
)


@pytest.mark.is_conj
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
@pytest.mark.parametrize("state", ["plain", "conj"])
def test_is_conj_special_values(state, dtype, scenario):
    base = tu.make_special_input(dtype, scenario).to(flag_gems.device)
    ref_base = tu.to_reference(base)

    inp = _conj_state(base, state)
    ref_inp = _conj_state(ref_base, state)

    ref_out = torch.ops.aten.is_conj(ref_inp)
    res_out = flag_gems.is_conj(inp)

    assert type(res_out) is bool
    assert res_out == ref_out


@pytest.mark.is_conj
@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES + [torch.float32])
@pytest.mark.parametrize("shape", [(256,), (20, 320, 15)])
def test_is_conj_query_leaves_input_untouched(shape, dtype):
    # Querying a lazy-conj view must neither resolve the flag nor touch the
    # storage it aliases.
    inp = _conj_state(tu.make_input(dtype, shape, tu.selected_ranges()[0]), "conj")
    # to_reference already re-applies the lazy conjugate bit, so a second .conj()
    # would toggle it back off and compare against a plain tensor.
    ref_inp = tu.to_reference(inp)

    before_ptr = inp.data_ptr()
    before_offset = inp.storage_offset()
    before_stride = inp.stride()
    before_storage = (
        inp.untyped_storage().data_ptr(),
        inp.untyped_storage().nbytes(),
    )
    before_state = (inp.is_conj(), inp.is_neg(), inp.shape)
    before_payload = tu.to_reference(inp)

    ref_out = torch.ops.aten.is_conj(ref_inp)
    res_out = flag_gems.is_conj(inp)

    assert type(res_out) is bool
    assert res_out == ref_out
    assert inp.data_ptr() == before_ptr
    assert inp.storage_offset() == before_offset
    assert inp.stride() == before_stride
    assert (inp.untyped_storage().data_ptr(), inp.untyped_storage().nbytes()) == (
        before_storage
    )
    assert (inp.is_conj(), inp.is_neg(), inp.shape) == before_state
    tu.assert_result_equal(tu.to_reference(inp), before_payload)


@pytest.mark.is_conj
@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_is_conj_flag_is_per_tensor_not_per_storage(shape, dtype):
    base = tu.make_input(dtype, shape, tu.selected_ranges()[0])
    ref_base = tu.to_reference(base)

    # A lazy-conj alias and a plain alias of the same storage.
    inp_view = base.conj()
    ref_view = ref_base.conj()
    inp_alias = base.view(shape)
    ref_alias = ref_base.view(shape)

    assert inp_view.data_ptr() == base.data_ptr()
    assert ref_view.data_ptr() == ref_base.data_ptr()

    view_ref = torch.ops.aten.is_conj(ref_view)
    view_res = flag_gems.is_conj(inp_view)
    alias_ref = torch.ops.aten.is_conj(ref_alias)
    alias_res = flag_gems.is_conj(inp_alias)

    assert type(view_res) is bool
    assert type(alias_res) is bool
    assert view_res == view_ref
    assert alias_res == alias_ref
    assert view_res != alias_res


# Negative cases stay in every mode. None is not a row: native
# aten::is_conj(None) binds the optional Tensor argument and returns False, so
# it is a valid call rather than invalid input.
_NON_TENSOR_ARGS = [3.14, 7, "abc", [1.0, 2.0], (1, 2)]


@pytest.mark.is_conj
@pytest.mark.parametrize("bad_arg", _NON_TENSOR_ARGS)
def test_is_conj_rejects_non_tensor_args(bad_arg):
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.is_conj(bad_arg)


@pytest.mark.is_conj
def test_is_conj_requires_an_argument():
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.is_conj()
