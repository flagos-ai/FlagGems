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


"""Correctness tests for ``aten::_foobar``."""

import pytest
import torch

import flag_gems

from . import test_utils as tu

# Only a CPU kernel is registered for this placeholder operator
# (torch._C._dispatch_dump lists CPU alone; CUDA dispatch raises
# NotImplementedError), so the native reference and the injected candidate both
# receive CPU tensors. Inputs are built on the framework device and moved to CPU.
_CPU = torch.device("cpu")

# The nine required spec dtypes all work on the CPU kernel, plus float64, bool and
# the complex types tu can build. Nothing is gated on GPU capability flags here:
# the registered kernel is CPU-only.
DTYPES = tu.REQUIRED_DTYPES + [
    torch.float64,
    torch.bool,
    torch.complex64,
    torch.complex128,
]

_PARAM_SHAPE = (20, 320, 15)
_PARAM_DTYPES = [torch.bfloat16, torch.int32, torch.uint8]

# Every bool parameter is exercised with the schema default and with an explicit
# True/False; the empty dict is the omit-the-argument call that checks the public
# signature defaults. These are cheap call forms, so they run in both modes.
_PARAM_CASES = [
    {},
    {"arg1": True},
    {"arg1": False},
    {"arg2": True},
    {"arg2": False},
    {"arg3": True},
    {"arg3": False},
    {"arg1": False, "arg2": False, "arg3": False},
    {"arg1": True, "arg2": True, "arg3": True},
]

_OUT_SHAPE = (20, 320, 15)

# One contiguous buffer per supported dtype plus two strided (permuted) buffers
# that check the in-place write respects the caller's layout.
_OUT_CASES = [(dtype, False) for dtype in DTYPES] + [
    (torch.float32, True),
    (torch.int64, True),
]

# Strided, expanded, empty and lazy-bit inputs; all cheap, so all stay in quick.
_LAYOUT_CASES = [
    pytest.param(
        lambda: torch.arange(24.0, device=_CPU).reshape(2, 3, 4).transpose(0, 2),
        id="transposed",
    ),
    pytest.param(
        lambda: torch.arange(30.0, device=_CPU).reshape(5, 6)[1:4, ::2],
        id="offset-noncontiguous",
    ),
    pytest.param(
        lambda: torch.arange(3.0, device=_CPU).expand(4, 3), id="expanded-zero-stride"
    ),
    pytest.param(lambda: torch.tensor(2.5, device=_CPU).expand(2), id="expanded-0dim"),
    pytest.param(
        lambda: torch.randn(2, 3, dtype=torch.complex64, device=_CPU).conj(),
        id="lazy-conj",
    ),
    pytest.param(
        lambda: torch._neg_view(torch.randn(2, 3, device=_CPU)), id="lazy-neg"
    ),
    pytest.param(lambda: torch.zeros(3, 0, 2, device=_CPU), id="empty"),
]

_SPECIAL_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]
# Positive special-value workloads are default-only.
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_SPECIAL_DTYPES), quick=[])

# Native accepts graph-recording inputs but raises for backward through a non-leaf.
_GRAPH_DTYPES = [torch.float32, torch.float64, torch.bfloat16]

# Negatives run in both modes.
_NON_TENSOR_ARGUMENTS = [None, 3.14, [1.0, 2.0]]

_OUT_DTYPE_MISMATCH = [
    (torch.float32, torch.float64),
    (torch.float32, torch.int64),
    (torch.int64, torch.int8),
    (torch.int64, torch.bool),
]


@pytest.mark.foobar
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", DTYPES)
def test__foobar(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range).to(_CPU)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._foobar(ref_inp)
    res_out = flag_gems._foobar(inp)

    tu.assert_result_equal(res_out, ref_out)
    # The CPU kernel returns the strided input object itself, so the candidate has
    # to preserve that identity; because res_out *is* the input, the exact value
    # comparison above also covers the no-mutation contract.
    assert res_out is inp


@pytest.mark.foobar
@pytest.mark.parametrize("kwargs", _PARAM_CASES)
@pytest.mark.parametrize("dtype", _PARAM_DTYPES)
def test__foobar_bool_params(dtype, kwargs):
    inp = tu.make_input(dtype, _PARAM_SHAPE, ["-1", "1"]).to(_CPU)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._foobar(ref_inp, **kwargs)
    res_out = flag_gems._foobar(inp, **kwargs)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.foobar
@pytest.mark.parametrize("dtype,non_contiguous", _OUT_CASES)
def test__foobar_out(dtype, non_contiguous):
    inp = tu.make_input(dtype, _OUT_SHAPE, ["-1", "1"]).to(_CPU)
    ref_inp = tu.to_reference(inp)

    if non_contiguous:
        act_out = torch.full(_OUT_SHAPE[::-1], -1, dtype=dtype, device=_CPU).permute(
            2, 1, 0
        )
        ref_buf = torch.full(_OUT_SHAPE[::-1], -1, dtype=dtype, device=_CPU).permute(
            2, 1, 0
        )
    else:
        act_out = torch.full(_OUT_SHAPE, -1, dtype=dtype, device=_CPU)
        ref_buf = torch.full(_OUT_SHAPE, -1, dtype=dtype, device=_CPU)
    buffer_stride = ref_buf.stride()
    buffer_offset = ref_buf.storage_offset()

    ref_out = torch.ops.aten._foobar.out(ref_inp, out=ref_buf)
    res_out = flag_gems._foobar(inp, out=act_out)

    # ``Tensor(a!) out``: the candidate writes the caller's buffer in place and
    # returns that same object with its layout intact, rather than a re-strided or
    # reallocated result. Compare right after the call, before anything else
    # touches either buffer.
    assert res_out is act_out
    assert res_out.stride() == buffer_stride
    assert res_out.storage_offset() == buffer_offset
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.foobar
@pytest.mark.parametrize("dtype", DTYPES)
def test__foobar_out_guard_storage(dtype):
    inp = tu.make_input(dtype, (3, 4), ["-1", "1"]).cpu()
    ref_inp = tu.to_reference(inp)
    backing = torch.full((5, 10), 7, dtype=dtype, device=_CPU)
    ref_backing = tu.to_reference(backing)
    out = backing[1:4, 1:9:2]
    ref_out = ref_backing[1:4, 1:9:2]
    torch.ops.aten._foobar.out(ref_inp, out=ref_out)
    res = flag_gems._foobar(inp, out=out)
    assert res is out
    assert res.stride() == ref_out.stride()
    assert res.storage_offset() == ref_out.storage_offset()
    tu.assert_result_equal(res, ref_out)
    tu.assert_result_equal(backing, ref_backing)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.foobar
@pytest.mark.parametrize("builder", _LAYOUT_CASES)
def test__foobar_layouts(builder):
    inp = builder()
    ref_inp = tu.to_reference(inp)
    # Classify from the pre-call state so an incorrect mutation cannot change
    # which result contract this test checks.
    flags_before = (inp.is_conj(), inp.is_neg())
    lazy_bits = any(flags_before)

    ref_out = torch.ops.aten._foobar(ref_inp)
    res_out = flag_gems._foobar(inp)

    if lazy_bits:
        # A lazy conj/neg bit is not stored data: the kernel resolves it and
        # returns a fresh tensor with both bits cleared.
        assert not res_out.is_conj()
        assert not res_out.is_neg()
    else:
        assert res_out is inp

    tu.assert_result_equal(res_out, ref_out)
    assert (inp.is_conj(), inp.is_neg()) == flags_before
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.foobar
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test__foobar_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario).to(_CPU)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._foobar(ref_inp)
    res_out = flag_gems._foobar(inp)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out is inp


@pytest.mark.foobar
@pytest.mark.parametrize("dtype", _GRAPH_DTYPES)
def test__foobar_rejects_backward(dtype):
    # A non-leaf input ensures backward reaches the missing native derivative.
    base = tu.make_input(dtype, _PARAM_SHAPE, ["-1", "1"]).to(_CPU).requires_grad_(True)
    ref_base = tu.to_reference(base)

    inp = base * 2.0
    ref_inp = ref_base * 2.0

    ref_out = torch.ops.aten._foobar(ref_inp)
    res_out = flag_gems._foobar(inp)

    tu.assert_result_equal(res_out, ref_out)
    # The native op keeps the result connected to the graph instead of dropping it
    # with a detach, so a candidate that returns a detached copy deviates here.
    assert res_out.requires_grad == ref_out.requires_grad
    upstream = torch.ones_like(res_out)
    with pytest.raises(
        RuntimeError, match="derivative for aten::_foobar is not implemented"
    ):
        torch.autograd.grad(ref_out, ref_base, grad_outputs=upstream)
    with pytest.raises(
        RuntimeError, match="derivative for aten::_foobar is not implemented"
    ):
        torch.autograd.grad(res_out, base, grad_outputs=upstream)


@pytest.mark.foobar
@pytest.mark.parametrize("bad_self", _NON_TENSOR_ARGUMENTS)
def test__foobar_rejects_non_tensor_self(bad_self):
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._foobar(bad_self)


@pytest.mark.foobar
@pytest.mark.parametrize("inp_dtype,out_dtype", _OUT_DTYPE_MISMATCH)
def test__foobar_rejects_mismatched_out_dtype(inp_dtype, out_dtype):
    inp = tu.make_input(inp_dtype, (5,), ["-1", "1"]).to(_CPU)
    out = torch.zeros(5, dtype=out_dtype, device=_CPU)
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._foobar(inp, out=out)
