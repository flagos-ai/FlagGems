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

# aten::matrix_H is the (conjugate) transpose view (Tensor(a) -> Tensor(a)): it
# swaps the last two dimensions and, for complex input, toggles the lazy
# conjugate bit instead of materialising a conjugate copy, so the tested
# contract is layout plus aliasing. Rank 2 is the supported class; rank 0 is a
# deprecated scalar path and rank 1/3+ raise, which drives the negative cases.
# No element is read, so every dtype the operator accepts is covered here.


def _supported_dtypes():
    dtypes = [
        dtype
        for dtype in tu.REQUIRED_DTYPES
        if utils.fp8_is_supported
        or dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
    ]
    if utils.fp64_is_supported:
        dtypes += [torch.float64, torch.complex128]
    return dtypes + [torch.bool, torch.complex64]


_MATRIX_H_DTYPES = _supported_dtypes()
_COMPLEX_DTYPES = [torch.complex64] + (
    [torch.complex128] if utils.fp64_is_supported else []
)

# Only matrices and the deprecated scalar form are valid positive inputs.
_MATRIX_H_SHAPES = tu.selected_cases(
    [(), (1, 1), (16, 16), (1024, 1024), (20, 320), (256, 8192), (8192, 32)],
    quick=[(), (1, 1), (2, 19)],
)

_LAYOUT_CASES = [
    # kind, shape, dtype: strides, offset and expansion are layout, not values.
    ("transposed", (4, 6), torch.float32),
    ("transposed", (7, 3), torch.bfloat16),
    ("column_slice", (4, 6), torch.float32),
    ("column_slice", (8, 5), torch.float16),
    ("offset_window", (6, 8), torch.float32),
    ("offset_window", (5, 7), torch.complex64),
    ("expanded", (4, 6), torch.float32),
    ("expanded", (3, 5), torch.int64),
    ("empty", (0, 0), torch.float32),
    ("empty", (3, 0), torch.int32),
    ("empty", (0, 1), torch.complex64),
]

_CONJUGATE_CASES = [
    ("plain", (4, 6)),
    ("column_slice", (4, 8)),
]

_WRITE_CASES = [
    ((4, 6), torch.float32),
    ((8, 3), torch.float16),
    ((5, 7), torch.complex64),
]

_BACKWARD_CASES = [
    ((4, 6), torch.float32),
    ((7, 13), torch.float16),
    ((16, 32), torch.bfloat16),
]
if utils.fp64_is_supported:
    _BACKWARD_CASES.append(((8, 8), torch.float64))

_SPECIAL_VALUE_CASES = tu.special_value_cases(_MATRIX_H_DTYPES)

# tu.special_value_cases covers floating dtypes only (is_floating_point is
# False for complex), so the complex payloads are spelled out here.
_COMPLEX_SPECIAL_PAYLOADS = {
    "nan": [complex(float("nan"), 0.0), complex(0.0, float("nan")), 1.0 - 1.0j],
    "inf": [complex(float("inf"), float("-inf")), 1.0j, -1.0 + 0.0j],
    "mixed": [complex(float("nan"), float("inf")), complex(float("-inf"), 0.0), 0j],
}

_INVALID_RANK_SHAPES = [(1,), (2, 3, 5), (2, 3, 4, 6), (16, 7, 57, 32, 29)]

_OVERLOAD_SHAPES = tu.selected_cases(
    [(4, 6), (64, 64), (1, 4096), (256, 8192)],
    quick=[(4, 6)],
)


def _assert_view_semantics(res_out, ref_out, inp, ref_inp):
    assert (res_out is inp) == (ref_out is ref_inp)
    assert inp.shape == ref_inp.shape
    assert inp.stride() == ref_inp.stride()
    assert inp.storage_offset() == ref_inp.storage_offset()
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out._is_view() == ref_out._is_view()
    # A view keeps addressing the very storage of its input.
    assert res_out.data_ptr() == inp.data_ptr()


def _layout_input(kind, shape, dtype):
    """Build an input with the requested layout; tu.to_reference keeps strides,
    offset and size, so the candidate and the reference see the same layout."""
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    if kind == "transposed":
        inp = inp.t()
    elif kind == "column_slice":
        inp = inp[:, ::2]
    elif kind == "offset_window":
        base = tu.make_input(dtype, (2 * shape[0], 2 * shape[1]), ["-1", "1"])
        inp = base[2 : 2 + shape[0], 1 : 1 + shape[1]]
    elif kind == "expanded":
        inp = tu.make_input(dtype, (1, shape[1]), ["-1", "1"]).expand(shape)
    elif kind != "empty":
        raise AssertionError(f"unknown layout {kind!r}")
    return inp


@pytest.mark.matrix_H
@pytest.mark.parametrize("shape", _MATRIX_H_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _MATRIX_H_DTYPES)
def test_matrix_H(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.matrix_H(ref_inp)
    res_out = flag_gems.matrix_H(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.matrix_H
@pytest.mark.parametrize(
    "kind,shape,dtype",
    _LAYOUT_CASES,
)
def test_matrix_H_layout(kind, shape, dtype):
    inp = _layout_input(kind, shape, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.matrix_H(ref_inp)
    res_out = flag_gems.matrix_H(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.matrix_H
@pytest.mark.parametrize("form,shape", _CONJUGATE_CASES)
@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
def test_matrix_H_conjugate_state(form, shape, dtype):
    # The lazy conjugate bit is this operator's primary semantic, so both the
    # complex dtypes and the conjugate rows are collected in every mode.
    # An already-set lazy conjugate bit must be toggled, not materialised.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    if form == "column_slice":
        inp = inp[:, ::2]
    inp = inp.conj()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.matrix_H(ref_inp)
    res_out = flag_gems.matrix_H(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.matrix_H
@pytest.mark.parametrize(
    "shape,dtype",
    _WRITE_CASES,
)
def test_matrix_H_writes_through(shape, dtype):
    # The result addresses the input storage, so a write through it has to be
    # visible in the input (a materialised copy would silently drop the write).
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.matrix_H(ref_inp)
    res_out = flag_gems.matrix_H(inp)
    res_out.fill_(2.5)
    ref_out.fill_(2.5)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.matrix_H
@pytest.mark.parametrize("shape,dtype", tu.selected_cases(_BACKWARD_CASES, quick=[]))
def test_matrix_H_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, (shape[1], shape[0]), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_grad = tu.to_reference(grad)

    ref_out = torch.ops.aten.matrix_H(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    res_out = flag_gems.matrix_H(inp)
    assert res_out.requires_grad
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]

    tu.assert_result_equal(res_in_grad, ref_in_grad)


@pytest.mark.matrix_H
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(_SPECIAL_VALUE_CASES, quick=[])
)
def test_matrix_H_special_values(dtype, scenario):
    # make_special_input emits a 1-D payload; matrix_H requires rank 2, so the
    # payload is reshaped without touching its values.
    inp = tu.make_special_input(dtype, scenario).reshape(5, 1)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.matrix_H(ref_inp)
    res_out = flag_gems.matrix_H(inp)

    # A view passes NaN/Inf through unchanged, so an exact comparison with
    # equal_nan matching is the right oracle.
    tu.assert_result_equal(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.matrix_H
@pytest.mark.parametrize(
    "scenario", tu.selected_cases(list(_COMPLEX_SPECIAL_PAYLOADS), quick=[])
)
@pytest.mark.parametrize("dtype", tu.selected_cases(_COMPLEX_DTYPES, quick=[]))
def test_matrix_H_special_values_complex(scenario, dtype):
    inp = torch.tensor(
        _COMPLEX_SPECIAL_PAYLOADS[scenario], dtype=dtype, device=flag_gems.device
    ).reshape(3, 1)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.matrix_H(ref_inp)
    res_out = flag_gems.matrix_H(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.matrix_H
@pytest.mark.parametrize("shape", _OVERLOAD_SHAPES)
@pytest.mark.parametrize("dtype", _MATRIX_H_DTYPES)
def test_matrix_H_overload_a(shape, dtype):
    # aten::matrix_H.a has the same single-tensor signature as .default, so the
    # candidate is reached through the one public name for both overloads.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.matrix_H.a(ref_inp)
    res_out = flag_gems.matrix_H(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.matrix_H
@pytest.mark.parametrize("shape", _INVALID_RANK_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_matrix_H_rejects_invalid_rank(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.matrix_H(inp)


@pytest.mark.matrix_H
@pytest.mark.parametrize("value", [3.14, 7, [1.0, 2.0], "matrix"])
def test_matrix_H_rejects_non_tensor(value):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.matrix_H(value)
