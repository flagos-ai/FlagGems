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

from . import test_utils as tu

# aten::_test_functorch_fallback(self, other) returns a fresh copy of 'self' and
# never reads 'other', so other's shape/dtype cannot reach the result (there is no
# broadcast dimension) and both parameters are Tensors (no scalar-operand form).
# Only CPU and Meta kernels exist -- CUDA operands raise NotImplementedError -- so
# the reference and the injected candidate both receive CPU tensors. There is no
# autograd formula, so backward is covered as a negative case.

_TFF_DTYPES = tu.REQUIRED_DTYPES + [
    torch.float64,
    torch.bool,
    torch.complex64,
    torch.complex128,
]

_TFF_OUT_SHAPES = tu.selected_cases(
    [(256,), (20, 320, 15), (2, 19, 7)], quick=[(256,), (2, 19, 7)]
)
_TFF_OUT_DTYPES = [torch.float32, torch.float16, torch.int32]

# Caller buffers for the out overload: the native kernel returns the caller's
# buffer by identity, writes it through its own strides and storage offset, and
# resizes one that is too small for the input.
_TFF_OUT_BUFFERS = [
    "contiguous",
    "non_contiguous",
    "offset",
    "resize_empty",
    "resize_mismatch",
]
_TFF_OUT_BUFFER_SHAPE = (4, 8)

# A dense source keeps its strides in the result, while stepped/offset/expanded
# sources are materialized and the lazy conjugate bit is dropped; the expected
# layout is read from the native reference instead of recomputed here.
_TFF_LAYOUTS = ["transposed", "stepped", "offset", "expanded", "conj"]
_TFF_LAYOUT_BASE_SHAPE = (8, 4)

# 'other' variants; none of them may influence the result.
_TFF_OTHER_CASES = [(torch.float32, ()), (torch.int64, (5,)), (torch.float32, (2, 3))]

_TFF_INVALID_CALLS = ["missing_other", "non_tensor_self", "non_tensor_other"]

_TFF_BACKWARD_ROWS = [
    (dtype, form)
    for dtype in _TFF_DTYPES
    if dtype.is_floating_point or dtype.is_complex
    for form in (
        ("leaf", "clone")
        if dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        else ("leaf", "nonleaf")
    )
]

# Positive NaN/Inf cases belong to the default suite only.
_TFF_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases([dtype for dtype in _TFF_DTYPES if dtype.is_floating_point]),
    quick=[],
)


@pytest.mark.test_functorch_fallback
@pytest.mark.parametrize("shape", tu.selected_shapes() + [(0,), (2, 0, 3)])
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _TFF_DTYPES)
def test_test_functorch_fallback(shape, value_range, dtype):
    # CPU-only operator: both operands use the CPU tensor type the native kernel
    # and the injected candidate accept.
    inp = tu.make_input(dtype, shape, value_range).cpu()
    other = tu.make_input(dtype, shape, value_range).cpu()
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten._test_functorch_fallback(ref_inp, ref_other)
    res_out = flag_gems._test_functorch_fallback(inp, other)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(other, ref_other)
    assert res_out.stride() == ref_out.stride()
    assert res_out.device == inp.device
    if inp.numel():
        assert res_out.data_ptr() != inp.data_ptr()


@pytest.mark.test_functorch_fallback
@pytest.mark.parametrize("shape", _TFF_OUT_SHAPES)
@pytest.mark.parametrize("dtype", _TFF_OUT_DTYPES)
def test_test_functorch_fallback_out(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).cpu()
    other = tu.make_input(dtype, shape, ["-1", "1"]).cpu()
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)
    ref_buf = torch.zeros(shape, dtype=dtype)
    res_buf = torch.zeros(shape, dtype=dtype)

    torch.ops.aten._test_functorch_fallback.out(ref_inp, ref_other, out=ref_buf)
    res_ret = flag_gems._test_functorch_fallback(inp, other, out=res_buf)

    # The out overload writes into and returns the caller's buffer itself.
    assert res_ret is res_buf
    tu.assert_result_equal(res_ret, ref_buf)
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(other, ref_other)


def _out_buffer(kind, shape, dtype):
    if kind == "contiguous":
        return torch.empty(shape, dtype=dtype), None
    if kind == "non_contiguous":
        return torch.empty(shape[1], shape[0], dtype=dtype).t(), None
    if kind == "offset":
        base = torch.full((shape[0] * shape[1] + 8,), 7, dtype=dtype)
        return base[8:].view(shape), base
    if kind == "resize_empty":
        return torch.empty(0, dtype=dtype), None
    # A non-empty buffer that is too small is resized by the native kernel.
    return torch.empty(shape[0] // 2, dtype=dtype), None


@pytest.mark.test_functorch_fallback
@pytest.mark.parametrize("buffer_kind", _TFF_OUT_BUFFERS)
def test_test_functorch_fallback_out_buffer(buffer_kind):
    shape = _TFF_OUT_BUFFER_SHAPE
    dtype = torch.float32
    inp = tu.make_input(dtype, shape, ["-1", "1"]).cpu()
    other = tu.make_input(dtype, shape, ["-1", "1"]).cpu()
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)
    ref_buf, ref_base = _out_buffer(buffer_kind, shape, dtype)
    res_buf, res_base = _out_buffer(buffer_kind, shape, dtype)

    torch.ops.aten._test_functorch_fallback.out(ref_inp, ref_other, out=ref_buf)
    res_ret = flag_gems._test_functorch_fallback(inp, other, out=res_buf)

    assert res_ret is res_buf
    # Values must land in the caller's layout (a candidate assuming a contiguous
    # buffer would scatter them), and a short buffer is resized like the native
    # kernel does.
    assert res_buf.shape == ref_buf.shape
    assert res_buf.stride() == ref_buf.stride()
    assert res_buf.storage_offset() == ref_buf.storage_offset()
    tu.assert_result_equal(res_ret, ref_buf)
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(other, ref_other)
    if res_base is not None:
        tu.assert_result_equal(res_base, ref_base)


def _strided_view(base, layout):
    if layout == "transposed":
        return base.t()
    if layout == "stepped":
        return base[::2]
    if layout == "offset":
        return base[2:]
    if layout == "expanded":
        return base[:1].expand(3, base.shape[1])
    return base.conj()  # lazy conjugate bit on a complex input


@pytest.mark.test_functorch_fallback
@pytest.mark.parametrize("layout", _TFF_LAYOUTS)
def test_test_functorch_fallback_layout(layout):
    dtype = torch.complex64 if layout == "conj" else torch.float32
    source = tu.make_input(dtype, _TFF_LAYOUT_BASE_SHAPE, ["-1", "1"]).cpu()
    inp = _strided_view(source, layout)
    # A single-element 'other' of an unrelated shape must not reach the result.
    other = tu.make_input(dtype, (1,), ["-1", "1"]).cpu()
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten._test_functorch_fallback(ref_inp, ref_other)
    res_out = flag_gems._test_functorch_fallback(inp, other)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(other, ref_other)
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.data_ptr() != source.data_ptr()


@pytest.mark.test_functorch_fallback
@pytest.mark.parametrize("other_dtype,other_shape", _TFF_OTHER_CASES)
def test_test_functorch_fallback_ignores_other(other_dtype, other_shape):
    inp = tu.make_input(torch.float32, (4, 8), ["-1", "1"]).cpu()
    # Sentinel values: a candidate that read 'other' would return them.
    fill = float("nan") if other_dtype.is_floating_point else 2**20
    other = torch.full(other_shape, fill, dtype=other_dtype)
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten._test_functorch_fallback(ref_inp, ref_other)
    res_out = flag_gems._test_functorch_fallback(inp, other)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(other, ref_other)


@pytest.mark.test_functorch_fallback
def test_test_functorch_fallback_returns_fresh_copy():
    inp = tu.make_input(torch.float32, (8, 4), ["-1", "1"]).cpu()
    other = tu.make_input(torch.float32, (8, 4), ["-1", "1"]).cpu()
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten._test_functorch_fallback(ref_inp, ref_other)
    res_out = flag_gems._test_functorch_fallback(inp, other)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(other, ref_other)
    assert not res_out._is_view()
    if inp.numel():
        assert res_out.data_ptr() != inp.data_ptr()
    # Writing the result must not reach the operand storage.
    res_out.zero_()
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.test_functorch_fallback
@pytest.mark.parametrize("dtype,form", _TFF_BACKWARD_ROWS)
def test_test_functorch_fallback_backward_unsupported(form, dtype):
    leaf = tu.make_input(dtype, (4, 8), ["-1", "1"]).cpu().requires_grad_(True)
    ref_leaf = tu.to_reference(leaf)
    other = tu.make_input(dtype, (4, 8), ["-1", "1"]).cpu()
    # Clone creates a non-leaf without requiring numeric FP8 multiplication.
    if form == "leaf":
        inp, ref_inp = leaf, ref_leaf
    elif form == "clone":
        inp, ref_inp = leaf.clone(), ref_leaf.clone()
    else:
        inp, ref_inp = leaf * 2.0, ref_leaf * 2.0
    upstream = tu.make_input(dtype, (4, 8), ["-1", "1"]).cpu()

    ref_out = torch.ops.aten._test_functorch_fallback(ref_inp, other)
    res_out = flag_gems._test_functorch_fallback(inp, other)
    tu.assert_result_equal(res_out, ref_out)
    with pytest.raises(
        RuntimeError,
        match="derivative for aten::_test_functorch_fallback is not implemented",
    ):
        torch.autograd.grad(res_out, leaf, grad_outputs=upstream)


@pytest.mark.test_functorch_fallback
def test_test_functorch_fallback_undefined_other():
    inp = tu.make_input(torch.float32, (4, 8), ["-1", "1"]).cpu()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_functorch_fallback(ref_inp, None)
    res_out = flag_gems._test_functorch_fallback(inp, None)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    assert not torch._C._is_alias_of(res_out, inp)


@pytest.mark.test_functorch_fallback
@pytest.mark.parametrize("dtype,scenario", _TFF_SPECIAL_CASES)
def test_test_functorch_fallback_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario).cpu()
    other = tu.make_special_input(dtype, scenario).cpu()
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten._test_functorch_fallback(ref_inp, ref_other)
    res_out = flag_gems._test_functorch_fallback(inp, other)

    # Exact copy semantics: NaN payloads must match, so NaN compares equal.
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(other, ref_other)


def _invalid_call_args(case, inp, other):
    if case == "missing_other":
        return (inp,)
    if case == "non_tensor_self":
        return (3.5, other)
    return (inp, 3.5)


@pytest.mark.test_functorch_fallback
@pytest.mark.parametrize("case", _TFF_INVALID_CALLS)
def test_test_functorch_fallback_invalid_input(case):
    inp = tu.make_input(torch.float32, (4, 8), ["-1", "1"]).cpu()
    other = tu.make_input(torch.float32, (4, 8), ["-1", "1"]).cpu()

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_functorch_fallback(*_invalid_call_args(case, inp, other))


@pytest.mark.test_functorch_fallback
def test_test_functorch_fallback_out_dtype_mismatch():
    inp = tu.make_input(torch.float32, (4, 8), ["-1", "1"]).cpu()
    other = tu.make_input(torch.float32, (4, 8), ["-1", "1"]).cpu()
    bad_buf = torch.zeros(4, 8, dtype=torch.float64)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_functorch_fallback(inp, other, out=bad_buf)
