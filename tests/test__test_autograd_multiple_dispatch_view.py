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

# aten::_test_autograd_multiple_dispatch_view(Tensor(a) self) -> Tensor(a) is a
# test-only CompositeExplicitAutograd op whose body is self.view(-1): the result
# aliases the input storage, keeps the input's lazy conjugate/negative bits, and
# the call raises when the layout admits no flat view. It takes one tensor and
# no parameter, so the scalar, broadcast and parameter dimensions do not apply
# here; the layout, alias and mutation checks below replace them.

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


# Static capability gate for dtypes the active backend may not provide.
def _supported(dtype):
    if dtype in _FP8_DTYPES:
        return utils.fp8_is_supported
    if dtype == torch.float64:
        return utils.fp64_is_supported
    if dtype == torch.bfloat16:
        return utils.bf16_is_supported
    if dtype == torch.int64:
        return utils.int64_is_supported
    return True


def _supported_rows(rows):
    return [row for row in rows if _supported(row[0])]


# One dtype set serves both modes; --quick only shrinks shapes and ranges.
_VIEW_DTYPES = [
    dtype
    for dtype in tu.REQUIRED_DTYPES + [torch.bool, torch.complex64, torch.float64]
    if _supported(dtype)
]

# The forward view accepts complex64, but native autograd rejects complex
# outputs. Its device-dispatched backward adds one on CUDA, where native FP8
# addition is unavailable; gradients therefore use the supported real dtypes.
_BACKWARD_DTYPES = [
    dtype
    for dtype in (torch.float16, torch.float32, torch.bfloat16, torch.float64)
    if _supported(dtype)
]

_SPECIAL_DTYPES = [
    dtype
    for dtype in (
        torch.float16,
        torch.float32,
        torch.bfloat16,
        torch.float64,
        *_FP8_DTYPES,
    )
    if _supported(dtype)
]

# One row per accepted input layout: a strided slice, a slice starting at a
# nonzero storage offset, a stride-0 expand, empty inputs and the two lazy view
# bits the operator propagates. A column-stepped slice has no flat view when the
# remaining rows are not back-to-back, so the stepped dimension must stay even
# (an odd one raises the native RuntimeError).
_LAYOUT_ROWS = _supported_rows(
    [
        (torch.float32, (4, 6), "column_step"),
        (torch.int8, (4, 6), "column_step"),
        (torch.float16, (8,), "column_step"),
        (torch.bfloat16, (3, 5, 4), "column_step"),
        (torch.uint8, (1024, 1024), "column_step"),
        (torch.float32, (20, 320, 16), "column_step"),
        (torch.complex64, (20, 320, 15), "window"),
        (torch.float8_e4m3fn, (6, 5), "window"),
        (torch.int32, (6, 5), "window"),
        (torch.float64, (16, 128, 64, 60), "window"),
        (torch.float32, (1, 1), "expanded"),
        (torch.float16, (1, 1), "expanded"),
        (torch.int64, (1, 1), "expanded"),
        (torch.complex64, (4, 6), "conj"),
        (torch.complex64, (3, 5, 7), "conj"),
        (torch.float32, (4, 6), "neg"),
        (torch.float16, (2, 3), "neg"),
        (torch.complex64, (2, 3), "neg"),
        (torch.float32, (0,), "asis"),
        (torch.int32, (0, 3), "asis"),
        (torch.bfloat16, (3, 0), "asis"),
    ]
)

# --quick keeps every cheap layout row and drops only the two large-shape ones.
_LARGE_LAYOUT_SHAPES = ((1024, 1024), (16, 128, 64, 60))
_LAYOUT_QUICK = [row for row in _LAYOUT_ROWS if row[1] not in _LARGE_LAYOUT_SHAPES]

# The write must land in the input storage: the whole base tensor is compared
# afterwards, so the elements the view does not cover have to stay untouched.
_MUTATION_ROWS = _supported_rows(
    [
        (torch.float32, (3, 4), "asis"),
        (torch.float16, (4, 6), "column_step"),
        (torch.bfloat16, (6, 5), "window"),
        (torch.float8_e4m3fn, (3, 4), "asis"),
        (torch.float8_e5m2, (3, 4), "asis"),
        (torch.int8, (2, 3), "asis"),
        (torch.uint8, (2, 3), "asis"),
        (torch.int32, (2, 3), "asis"),
        (torch.int64, (2, 3), "asis"),
        (torch.bool, (2, 3), "asis"),
        (torch.float64, (2, 3), "asis"),
        (torch.complex64, (2, 3), "asis"),
    ]
)

# Every mutation row is small, so --quick keeps them all.
_MUTATION_QUICK = _MUTATION_ROWS

_BACKWARD_ROWS = [
    (dtype, shape, layout)
    for dtype in _BACKWARD_DTYPES
    for shape, layout in (
        ((4, 6), "asis"),
        ((3, 5, 7), "asis"),
        ((4, 6), "column_step"),
        ((6, 5), "window"),
        ((1, 1), "expanded"),
    )
]

# Layouts that admit no flat view; the native op raises RuntimeError for each.
# Kept in both modes.
_NON_VIEW_KINDS = ("t", "transpose", "permute", "odd_step")


def _apply_layout(base, layout):
    if layout == "column_step":
        return base[..., ::2]
    if layout == "window":
        return base[1:4]
    if layout == "expanded":
        return base.expand(3, 5)
    if layout == "conj":
        return base.conj()
    if layout == "neg":
        return torch._neg_view(base)
    return base


def _assert_view_contract(res_out, ref_out, inp):
    # Tensor(a) -> Tensor(a): a new view object over the input storage that
    # keeps the input's lazy conjugate and negative bits.
    assert res_out.dtype == inp.dtype
    assert res_out.device == inp.device
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.is_conj() == inp.is_conj()
    assert res_out.is_neg() == inp.is_neg()
    assert res_out._is_view()
    assert res_out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
    assert res_out.data_ptr() == inp.data_ptr()


def _non_view_input(kind):
    if kind == "t":
        return torch.zeros(2, 3, device=flag_gems.device).t()
    if kind == "transpose":
        return torch.zeros(2, 3, 4, device=flag_gems.device).transpose(0, 1)
    if kind == "permute":
        return torch.zeros(2, 3, 4, device=flag_gems.device).permute(2, 0, 1)
    return torch.zeros(4, 5, device=flag_gems.device)[..., ::2]


@pytest.mark.test_autograd_multiple_dispatch_view
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VIEW_DTYPES)
def test_autograd_multiple_dispatch_view(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_autograd_multiple_dispatch_view(ref_inp)
    res_out = flag_gems._test_autograd_multiple_dispatch_view(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_contract(res_out, ref_out, inp)


@pytest.mark.test_autograd_multiple_dispatch_view
@pytest.mark.parametrize(
    "dtype,shape,layout", tu.selected_cases(_LAYOUT_ROWS, quick=_LAYOUT_QUICK)
)
def test_autograd_multiple_dispatch_view_strided_views(dtype, shape, layout):
    base_inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base_inp = tu.to_reference(base_inp)
    view = _apply_layout(base_inp, layout)
    ref_view = _apply_layout(ref_base_inp, layout)

    ref_out = torch.ops.aten._test_autograd_multiple_dispatch_view(ref_view)
    res_out = flag_gems._test_autograd_multiple_dispatch_view(view)

    tu.assert_result_equal(res_out, ref_out)
    # The expected strides/offset come from the native view of the same layout,
    # so a materialized contiguous copy is rejected here.
    _assert_view_contract(res_out, ref_out, view)


@pytest.mark.test_autograd_multiple_dispatch_view
@pytest.mark.parametrize(
    "dtype,shape,layout", tu.selected_cases(_MUTATION_ROWS, quick=_MUTATION_QUICK)
)
def test_autograd_multiple_dispatch_view_mutation(dtype, shape, layout):
    base_inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base_inp = tu.to_reference(base_inp)
    view = _apply_layout(base_inp, layout)
    ref_view = _apply_layout(ref_base_inp, layout)

    ref_out = torch.ops.aten._test_autograd_multiple_dispatch_view(ref_view)
    res_out = flag_gems._test_autograd_multiple_dispatch_view(view)
    tu.assert_result_equal(res_out, ref_out)

    res_out.copy_(torch.zeros_like(res_out))
    ref_out.copy_(torch.zeros_like(ref_out))

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(base_inp, ref_base_inp)


@pytest.mark.test_autograd_multiple_dispatch_view
@pytest.mark.parametrize(
    "dtype,shape,layout", tu.selected_cases(_BACKWARD_ROWS, quick=[])
)
def test_autograd_multiple_dispatch_view_backward(dtype, shape, layout):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    upstream = tu.make_input(dtype, (_apply_layout(inp, layout).numel(),), ["-1", "1"])

    res_out = flag_gems._test_autograd_multiple_dispatch_view(
        _apply_layout(inp, layout)
    )
    assert res_out.requires_grad

    # The forward view is value-preserving, so its oracle follows the configured
    # reference device.
    ref_out = torch.ops.aten._test_autograd_multiple_dispatch_view(
        _apply_layout(tu.to_reference(inp.detach()), layout)
    )
    tu.assert_result_equal(res_out, ref_out)

    # The native autograd entry is device-dispatched (its CUDA registration adds
    # 1 to the incoming gradient), so the gradient oracle runs the same op on
    # the candidate's device; only the comparison follows the reference device.
    grad_inp = inp.detach().clone().requires_grad_()
    grad_out = torch.ops.aten._test_autograd_multiple_dispatch_view(
        _apply_layout(grad_inp, layout)
    )
    (ref_grad,) = torch.autograd.grad(grad_out, grad_inp, grad_outputs=upstream)
    (res_grad,) = torch.autograd.grad(res_out, inp, grad_outputs=upstream)

    tu.assert_result_close(res_grad, tu.to_reference(ref_grad))


@pytest.mark.test_autograd_multiple_dispatch_view
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_SPECIAL_DTYPES), quick=[]),
)
def test_autograd_multiple_dispatch_view_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_autograd_multiple_dispatch_view(ref_inp)
    res_out = flag_gems._test_autograd_multiple_dispatch_view(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_contract(res_out, ref_out, inp)


@pytest.mark.test_autograd_multiple_dispatch_view
@pytest.mark.parametrize("kind", _NON_VIEW_KINDS)
def test_autograd_multiple_dispatch_view_invalid_layout(kind):
    with pytest.raises(RuntimeError):
        flag_gems._test_autograd_multiple_dispatch_view(_non_view_input(kind))
