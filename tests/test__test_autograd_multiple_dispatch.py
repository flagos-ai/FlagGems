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

# aten::_test_autograd_multiple_dispatch is a pure copy op with three call
# forms: fullcoverage(self), ntonly(self, bool b) and the keyword-only
# fullcoverage_out(self, *, out).
_FLOAT_DTYPES = (
    [torch.float16, torch.float32]
    + ([torch.bfloat16] if utils.bf16_is_supported else [])
    + ([torch.float64] if utils.fp64_is_supported else [])
)
_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2] if utils.fp8_is_supported else []
_INT_DTYPES = [torch.int8, torch.uint8, torch.int32] + (
    [torch.int64] if utils.int64_is_supported else []
)
_DTYPES = _FLOAT_DTYPES + _FP8_DTYPES + [torch.complex64] + _INT_DTYPES + [torch.bool]

# fullcoverage's native backward adds a constant to the incoming gradient, which
# needs an add kernel that fp8 does not provide; ntonly's backward is a pure copy
# and needs none, so fp8 stays eligible there. Fullcoverage rejects complex
# autograd outputs; ntonly supports them.
_FULLCOVERAGE_GRAD_DTYPES = _FLOAT_DTYPES
_NTONLY_GRAD_DTYPES = _FLOAT_DTYPES + _FP8_DTYPES + [torch.complex64]

_SCALAR_SHAPES = [(), (1,)]

# Non-contiguous inputs: a strided slice, a transposed view and a stride-0
# expanded view of a size-1 axis.
_LAYOUT_ROWS = [
    ((8, 16, 32), "sliced"),
    ((8, 16, 32), "transposed"),
    ((1, 16), "expanded"),
    ((4, 8, 16, 32), "sliced"),
]
_NONCONTIG_ROWS = _LAYOUT_ROWS + [((2, 19, 7), "sliced")]
_OUT_NONCONTIG_SHAPES = tu.selected_cases(
    [(8, 16, 32), (4, 8, 16, 32), (2, 19, 7)],
    quick=[(8, 16, 32), (4, 8, 16, 32), (2, 19, 7)],
)
_EMPTY_SHAPES = [(0,), (2, 0, 3)]


def _layout_view(tensor, kind):
    if kind == "sliced":
        return tensor[..., ::2]
    if kind == "transposed":
        return tensor.transpose(0, 1)
    if kind == "expanded":
        return tensor.expand(8, 16)
    raise ValueError(f"Unknown layout kind: {kind}")


def _assert_copy_result(res_out, ref_out, inp):
    tu.assert_result_equal(res_out, ref_out)
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out._is_view() == ref_out._is_view()
    if inp.numel():
        assert res_out.data_ptr() != inp.data_ptr()


@pytest.mark.test_autograd_multiple_dispatch
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_test_autograd_multiple_dispatch(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_autograd_multiple_dispatch.fullcoverage(ref_inp)
    res_out = flag_gems._test_autograd_multiple_dispatch(inp)

    _assert_copy_result(res_out, ref_out, inp)


@pytest.mark.test_autograd_multiple_dispatch
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_test_autograd_multiple_dispatch_ntonly(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_autograd_multiple_dispatch.ntonly(ref_inp, True)
    res_out = flag_gems._test_autograd_multiple_dispatch(inp, True)

    _assert_copy_result(res_out, ref_out, inp)


@pytest.mark.test_autograd_multiple_dispatch
@pytest.mark.parametrize(
    "shape",
    tu.selected_cases([(256,), (7, 13, 29), (2, 19, 7)], quick=[(256,), (2, 19, 7)]),
)
@pytest.mark.parametrize("b", [True, False])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_test_autograd_multiple_dispatch_ntonly_flag(shape, b, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_autograd_multiple_dispatch.ntonly(ref_inp, b)
    res_out = flag_gems._test_autograd_multiple_dispatch(inp, b)

    _assert_copy_result(res_out, ref_out, inp)


@pytest.mark.test_autograd_multiple_dispatch
@pytest.mark.parametrize("shape", _SCALAR_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_test_autograd_multiple_dispatch_scalar(shape, dtype):
    # 0-dim and single-element tensors are the cheapest rank boundary and stay in
    # the quick smoke subset as well.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_autograd_multiple_dispatch.fullcoverage(ref_inp)
    res_out = flag_gems._test_autograd_multiple_dispatch(inp)

    _assert_copy_result(res_out, ref_out, inp)


@pytest.mark.test_autograd_multiple_dispatch
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_test_autograd_multiple_dispatch_out(shape, dtype):
    # fullcoverage_out writes into the caller's buffer and returns that same
    # object; the sentinel fill proves the buffer was actually written.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    inp_before = tu.to_reference(inp.detach())
    sentinel = True if dtype == torch.bool else 7
    out = torch.full(shape, sentinel, dtype=dtype, device=flag_gems.device)
    ref_out = tu.to_reference(out)

    torch.ops.aten._test_autograd_multiple_dispatch.fullcoverage_out(
        ref_inp, out=ref_out
    )
    res_ret = flag_gems._test_autograd_multiple_dispatch(inp, out=out)

    assert res_ret is out
    tu.assert_result_equal(out, ref_out)
    assert out.stride() == ref_out.stride()
    assert out.storage_offset() == ref_out.storage_offset()
    tu.assert_result_equal(inp, inp_before)


@pytest.mark.test_autograd_multiple_dispatch
@pytest.mark.parametrize("shape", _OUT_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_test_autograd_multiple_dispatch_out_non_contiguous(shape, dtype):
    # A strided out buffer must keep its own layout instead of being replaced by a
    # fresh contiguous allocation, and the input must stay untouched.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    inp_before = tu.to_reference(inp.detach())
    sentinel = True if dtype == torch.bool else 7
    wide_shape = list(shape)
    wide_shape[-1] *= 2
    wide = torch.full(wide_shape, sentinel, dtype=dtype, device=flag_gems.device)
    out = wide[..., ::2]
    ref_wide = tu.to_reference(wide)
    ref_out = ref_wide[..., ::2]

    torch.ops.aten._test_autograd_multiple_dispatch.fullcoverage_out(
        ref_inp, out=ref_out
    )
    res_ret = flag_gems._test_autograd_multiple_dispatch(inp, out=out)

    assert res_ret is out
    tu.assert_result_equal(out, ref_out)
    assert out.stride() == ref_out.stride()
    tu.assert_result_equal(wide, ref_wide)
    tu.assert_result_equal(inp, inp_before)


@pytest.mark.test_autograd_multiple_dispatch
@pytest.mark.parametrize("shape, kind", _NONCONTIG_ROWS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_test_autograd_multiple_dispatch_non_contiguous_input(shape, kind, dtype):
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = _layout_view(base, kind)
    ref_inp = _layout_view(ref_base, kind)

    ref_out = torch.ops.aten._test_autograd_multiple_dispatch.fullcoverage(ref_inp)
    res_out = flag_gems._test_autograd_multiple_dispatch(inp)

    _assert_copy_result(res_out, ref_out, inp)


@pytest.mark.test_autograd_multiple_dispatch
@pytest.mark.parametrize("shape", _EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_test_autograd_multiple_dispatch_empty(shape, dtype):
    # No element values exist to compare; the copy still has to reproduce dtype,
    # shape and strides.
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_autograd_multiple_dispatch.fullcoverage(ref_inp)
    res_out = flag_gems._test_autograd_multiple_dispatch(inp)

    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()


@pytest.mark.test_autograd_multiple_dispatch
@pytest.mark.parametrize("shape", tu.selected_cases([(256,), (2, 19, 7)], quick=[]))
@pytest.mark.parametrize(
    "dtype", tu.selected_cases(_FULLCOVERAGE_GRAD_DTYPES, quick=[])
)
def test_test_autograd_multiple_dispatch_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    upstream = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_autograd_multiple_dispatch.fullcoverage(ref_inp)
    res_out = flag_gems._test_autograd_multiple_dispatch(inp)
    tu.assert_result_equal(res_out, ref_out)

    # fullcoverage's backward formula is dispatched per backend, so the native
    # oracle runs on the candidate's own device instead of through to_reference.
    native_inp = inp.detach().clone().requires_grad_(True)
    native_out = torch.ops.aten._test_autograd_multiple_dispatch.fullcoverage(
        native_inp
    )
    native_grad = torch.autograd.grad(native_out, native_inp, grad_outputs=upstream)[0]

    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]
    tu.assert_result_equal(res_grad, tu.to_reference(native_grad))


@pytest.mark.test_autograd_multiple_dispatch
@pytest.mark.parametrize("shape", tu.selected_cases([(256,), (2, 19, 7)], quick=[]))
@pytest.mark.parametrize("dtype", tu.selected_cases(_NTONLY_GRAD_DTYPES, quick=[]))
def test_test_autograd_multiple_dispatch_ntonly_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    upstream = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_autograd_multiple_dispatch.ntonly(ref_inp, True)
    res_out = flag_gems._test_autograd_multiple_dispatch(inp, True)
    tu.assert_result_equal(res_out, ref_out)

    # ntonly's backward is a pure copy of the incoming gradient.
    native_inp = inp.detach().clone().requires_grad_(True)
    native_out = torch.ops.aten._test_autograd_multiple_dispatch.ntonly(
        native_inp, True
    )
    native_grad = torch.autograd.grad(native_out, native_inp, grad_outputs=upstream)[0]

    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]
    tu.assert_result_equal(res_grad, tu.to_reference(native_grad))


@pytest.mark.test_autograd_multiple_dispatch
@pytest.mark.parametrize(
    "dtype, scenario",
    tu.selected_cases(tu.special_value_cases(_DTYPES), quick=[]),
)
def test_test_autograd_multiple_dispatch_special_scenarios(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_autograd_multiple_dispatch.fullcoverage(ref_inp)
    res_out = flag_gems._test_autograd_multiple_dispatch(inp)

    _assert_copy_result(res_out, ref_out, inp)


@pytest.mark.test_autograd_multiple_dispatch
@pytest.mark.parametrize("bad_self", [3.14, [1.0, 2.0]])
def test_test_autograd_multiple_dispatch_rejects_non_tensor(bad_self):
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._test_autograd_multiple_dispatch(bad_self)


@pytest.mark.test_autograd_multiple_dispatch
def test_test_autograd_multiple_dispatch_rejects_non_bool_flag():
    inp = tu.make_input(torch.float32, (8,), ["-1", "1"])
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._test_autograd_multiple_dispatch(inp, "not-a-bool")


@pytest.mark.test_autograd_multiple_dispatch
def test_test_autograd_multiple_dispatch_rejects_out_dtype_mismatch():
    inp = tu.make_input(torch.float32, (8,), ["-1", "1"])
    out = torch.empty(8, dtype=torch.float64, device=flag_gems.device)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._test_autograd_multiple_dispatch(inp, out=out)
