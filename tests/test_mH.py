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

# aten::mH returns a zero-copy conjugate-transpose view: it swaps the last two
# dimensions and toggles the lazy conjugate bit (never materialized; a no-op for
# real dtypes). There is no kernel arithmetic to compare, so values are compared
# exactly and the view contract (shared storage, strides, storage offset,
# conjugate bit, write-through) is asserted separately. Only rank >= 2 is a
# transpose: rank 1 is rejected by aten and 0-D is the deprecated conj() path,
# so the value grid keeps the ranks the operator accepts.
_MH_DTYPES = (
    tu.REQUIRED_DTYPES
    + [torch.bool, torch.complex64]
    + ([torch.float64, torch.complex128] if utils.fp64_is_supported else [])
)

_MH_COMPLEX_DTYPES = [
    dtype for dtype in (torch.complex64, torch.complex128) if dtype in _MH_DTYPES
]

_MH_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) >= 2]

# Inputs that are already transposed, strided or offset windows stay views, so
# the result has to re-derive strides and keep the shared storage offset. Every
# row is a small tensor, so all of them stay in quick mode.
_MH_LAYOUT_ROWS = [
    ((6, 8), "transposed"),
    ((6, 8), "column_step"),
    ((6, 8), "offset_window"),
    ((4, 6, 10), "offset_window"),
]

_MH_WRITE_ROWS = [
    ((4, 6), "asis"),
    ((6, 8), "transposed"),
    ((6, 8), "offset_window"),
]

# "input already conjugated or not" x "input already transposed or not": mH
# toggles the conjugate bit instead of materializing the conjugation.
_MH_CONJ_ROWS = [
    (layout, dtype)
    for layout in ("asis", "transposed", "conj", "transposed_conj")
    for dtype in _MH_COMPLEX_DTYPES
]

_MH_BACKWARD_ROWS = tu.selected_cases(
    [
        (shape, dtype)
        for shape in ((4, 6), (7, 13, 29), (2, 3, 4))
        for dtype in (
            torch.float16,
            torch.float32,
            torch.bfloat16,
            torch.float64,
            torch.complex64,
        )
        if dtype in _MH_DTYPES
    ],
    quick=[],
)

_MH_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_MH_DTYPES), quick=[])

# Rank-1 forms (including the empty (0,) tensor) raise natively.
_MH_REJECT_SHAPES = [(5,), (1,), (0,)]


def _apply_layout(base, layout):
    if layout == "asis":
        return base
    if layout == "transposed":
        return base.transpose(-1, -2)
    if layout == "column_step":
        return base[..., ::2]
    if layout == "offset_window":
        return base[..., 1:4, 1:4]
    if layout == "conj":
        return base.conj()
    if layout == "transposed_conj":
        return base.transpose(-1, -2).conj()
    raise ValueError("unsupported layout " + repr(layout))


def _layout_pair(shape, layout, dtype):
    # Candidate operand, independently stored reference operand, and both
    # backing tensors so a write through either view can be traced back.
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base.detach())
    return _apply_layout(base, layout), _apply_layout(ref_base, layout), base, ref_base


def _assert_view_semantics(res_out, ref_out, inp, ref_inp):
    assert (res_out is inp) == (ref_out is ref_inp)
    assert inp.shape == ref_inp.shape
    assert inp.stride() == ref_inp.stride()
    assert inp.storage_offset() == ref_inp.storage_offset()
    # mH aliases its input instead of copying, so shared storage, strides,
    # storage offset and the lazy conjugate bit must match aten exactly.
    assert res_out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
    assert res_out._is_view() == ref_out._is_view()
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.is_conj() == ref_out.is_conj()


@pytest.mark.mH
@pytest.mark.parametrize("shape", _MH_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _MH_DTYPES)
def test_mH(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.mH(ref_inp)
    res_out = flag_gems.mH(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.mH
@pytest.mark.parametrize("shape,layout", _MH_LAYOUT_ROWS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_mH_strided_input(shape, layout, dtype):
    inp, ref_inp, _, _ = _layout_pair(shape, layout, dtype)

    ref_out = torch.ops.aten.mH(ref_inp)
    res_out = flag_gems.mH(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.mH
@pytest.mark.parametrize("layout,dtype", _MH_CONJ_ROWS)
def test_mH_conjugate_state(layout, dtype):
    inp, ref_inp, _, _ = _layout_pair((6, 8), layout, dtype)

    ref_out = torch.ops.aten.mH(ref_inp)
    res_out = flag_gems.mH(inp)

    assert res_out.is_conj() is ref_out.is_conj()
    tu.assert_result_equal(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.mH
@pytest.mark.parametrize("shape,layout", _MH_WRITE_ROWS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_mH_write_through(shape, layout, dtype):
    inp, ref_inp, base, ref_base = _layout_pair(shape, layout, dtype)

    res_out = flag_gems.mH(inp)
    ref_out = torch.ops.aten.mH(ref_inp)

    # mH shares storage, so a write through the returned view must reach the
    # input and its backing tensor exactly like the native view does.
    res_out.fill_(2.5)
    ref_out.fill_(2.5)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(base, ref_base)


@pytest.mark.mH
@pytest.mark.parametrize("dtype", _MH_DTYPES)
def test_mH_0d_scalar(dtype):
    # aten::mH warns and degrades to conj() on 0-D input: a real tensor comes
    # back as itself, a complex one as a lazy conjugate view on the same storage.
    inp = tu.make_input(dtype, (), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.mH(ref_inp)
    res_out = flag_gems.mH(inp)

    tu.assert_result_equal(res_out, ref_out)
    assert (res_out is inp) == (ref_out is ref_inp)
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.data_ptr() == inp.data_ptr()


@pytest.mark.mH
@pytest.mark.parametrize("shape,dtype", _MH_BACKWARD_ROWS)
def test_mH_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_inp = tu.to_reference(inp.detach()).requires_grad_(True)
    upstream = tu.make_input(dtype, shape[:-2] + (shape[-1], shape[-2]), ["-1", "1"])
    ref_upstream = tu.to_reference(upstream)

    ref_out = torch.ops.aten.mH(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)[0]

    res_out = flag_gems.mH(inp)
    tu.assert_result_equal(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)

    # The adjoint of a transpose/conjugate view is the same relayout applied to
    # the upstream gradient, so it is compared exactly.
    assert res_out.requires_grad
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]
    tu.assert_result_equal(res_in_grad, ref_in_grad)


@pytest.mark.mH
@pytest.mark.parametrize("dtype,scenario", _MH_SPECIAL_CASES)
def test_mH_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario).reshape(1, -1)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.mH(ref_inp)
    res_out = flag_gems.mH(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.mH
@pytest.mark.parametrize("shape", _MH_REJECT_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_mH_rejects_1d(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems.mH(inp)
