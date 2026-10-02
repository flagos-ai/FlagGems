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

# aten::is_floating_point(Tensor self) -> bool reads self.dtype only, so the
# answer must not depend on values, shape, strides, storage offset or lazy
# conj/neg state. Broadcast and backward do not apply (one operand, Python bool
# result) and the operator has no .out schema (probed absent natively).
_IS_FLOATING_POINT_DTYPES = (
    list(tu.REQUIRED_DTYPES)
    + ([torch.float64] if utils.fp64_is_supported else [])
    + [torch.bool, torch.complex64, torch.complex32]
)


def _layout_view(base, layout):
    """Build the operand of one probed layout; it keeps sharing base storage."""
    if layout == "transposed":
        return base.transpose(0, 1)
    if layout == "column_slice":
        return base[:, ::2]
    if layout == "window":
        return base[2:6, 1:5]
    if layout == "expanded":
        return base[:1].expand(3, base.shape[1])
    if layout == "neg_view":
        return torch._neg_view(base)
    if layout == "conj":
        return base.conj()
    raise AssertionError(f"unknown layout {layout}")


# Small stride, offset and lazy-flag cases stay in quick mode.
_LAYOUT_CASES = [
    ((4, 6), "transposed", torch.float32),
    ((8, 12), "column_slice", torch.float16),
    ((10, 8), "window", torch.int8),
    ((1, 8), "expanded", torch.bfloat16),
    ((4, 6), "neg_view", torch.float32),
    ((4, 6), "conj", torch.complex64),
]

_EMPTY_CASES = [
    (shape, dtype)
    for shape in [(0,), (0, 4), (2, 0, 3)]
    for dtype in (torch.float32, torch.int8)
]

# One row per representable nan/inf scenario of each floating dtype: e4m3fn has
# no inf encoding and therefore contributes only its nan row.
_SPECIAL_VALUE_CASES = tu.selected_cases(
    tu.special_value_cases(
        [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]
        + ([torch.float64] if utils.fp64_is_supported else [])
    ),
    quick=[],
)

_NON_TENSOR_CASES = [
    pytest.param(3.14, id="python_float"),
    pytest.param([1.0, 2.0], id="list"),
    pytest.param("float32", id="string"),
]


@pytest.mark.is_floating_point
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _IS_FLOATING_POINT_DTYPES)
def test_is_floating_point(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.is_floating_point(ref_inp)
    res_out = flag_gems.is_floating_point(inp)

    assert isinstance(res_out, bool)
    assert res_out == ref_out


@pytest.mark.is_floating_point
@pytest.mark.parametrize("shape,layout,dtype", _LAYOUT_CASES)
def test_is_floating_point_layout(shape, layout, dtype):
    base = tu.make_input(dtype, shape, ["-1", "1"])
    inp = _layout_view(base, layout)
    # The snapshots keep strides, offset and the lazy conj/neg bits, so they catch
    # a candidate that rewrote or materialised the operand it was handed.
    base_snapshot = tu.to_reference(base)
    view_snapshot = tu.to_reference(inp.detach())
    ref_inp = _layout_view(base_snapshot, layout)

    ref_out = torch.ops.aten.is_floating_point(ref_inp)
    res_out = flag_gems.is_floating_point(inp)

    assert isinstance(res_out, bool)
    assert res_out == ref_out
    assert inp.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    assert (inp.stride(), inp.storage_offset()) == (
        view_snapshot.stride(),
        view_snapshot.storage_offset(),
    )
    tu.assert_result_equal(inp, view_snapshot)
    tu.assert_result_equal(base, base_snapshot)


@pytest.mark.is_floating_point
@pytest.mark.parametrize("shape,dtype", _EMPTY_CASES)
def test_is_floating_point_empty_operand(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])

    ref_out = torch.ops.aten.is_floating_point(tu.to_reference(inp))
    res_out = flag_gems.is_floating_point(inp)

    assert isinstance(res_out, bool)
    assert res_out == ref_out


@pytest.mark.is_floating_point
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_VALUE_CASES)
def test_is_floating_point_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)

    ref_out = torch.ops.aten.is_floating_point(tu.to_reference(inp))
    res_out = flag_gems.is_floating_point(inp)

    assert isinstance(res_out, bool)
    assert res_out == ref_out


@pytest.mark.is_floating_point
def test_is_floating_point_optional_none():
    # The schema's optional-Tensor form accepts None and reports it as not
    # floating, so this is a valid workload, not an invalid input.
    ref_out = torch.ops.aten.is_floating_point(None)
    res_out = flag_gems.is_floating_point(None)

    assert isinstance(res_out, bool)
    assert res_out == ref_out


@pytest.mark.is_floating_point
@pytest.mark.parametrize("bad", _NON_TENSOR_CASES)
def test_is_floating_point_rejects_non_tensor(bad):
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.is_floating_point(bad)


@pytest.mark.is_floating_point
def test_is_floating_point_requires_argument():
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.is_floating_point()
