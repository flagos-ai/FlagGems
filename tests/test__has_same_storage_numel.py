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

import math

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Register underscore-prefixed pytest markers explicitly.
setattr(
    pytest.mark,
    "_has_same_storage_numel",
    MarkDecorator(
        Mark("_has_same_storage_numel", (), {}, _ispytest=True), _ispytest=True
    ),
)

# Compare storage element counts, which may differ from logical tensor sizes.
_HAS_SAME_STORAGE_NUMEL_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool]
)

# (kind, shape) pairs separate logical sizes from backing storage sizes.
_HAS_SAME_STORAGE_NUMEL_CASES = [
    pytest.param(("plain", (4, 4)), ("plain", (4, 4)), id="same_shape_true"),
    pytest.param(("plain", (4, 4)), ("plain", (16,)), id="reshaped_same_storage_true"),
    pytest.param(("plain", (4, 4)), ("plain", (8,)), id="different_numel_false"),
    pytest.param(
        ("plain", (4, 4)), ("transposed", (4, 4)), id="transposed_same_storage_true"
    ),
    pytest.param(
        ("plain", (4, 4)), ("row_view", (4, 4)), id="row_slice_same_storage_true"
    ),
    pytest.param(
        ("plain", (4, 4)), ("narrowed", (4, 4)), id="narrowed_same_storage_true"
    ),
    pytest.param(
        ("plain", (4, 4)), ("expanded", (4, 4)), id="plain_larger_storage_false"
    ),
    pytest.param(
        ("expanded", (4, 4)), ("plain", (4,)), id="expanded_base_matches_false"
    ),
    pytest.param(
        ("row_view", (4, 4)), ("plain", (4,)), id="row_slice_larger_storage_false"
    ),
    pytest.param(
        ("narrowed", (16,)), ("plain", (4,)), id="narrowed_larger_storage_false"
    ),
    pytest.param(("plain", ()), ("plain", (1,)), id="scalar_vs_single_true"),
    pytest.param(("plain", (0,)), ("plain", (0, 5)), id="empty_same_storage_true"),
    pytest.param(("plain", (0,)), ("plain", (3,)), id="empty_vs_nonempty_false"),
]

_CROSS_DTYPE_CASES = [
    pytest.param(torch.float32, torch.int64, id="fp32_vs_int64"),
    pytest.param(torch.float16, torch.float32, id="fp16_vs_fp32"),
    pytest.param(torch.int8, torch.uint8, id="int8_vs_uint8"),
    pytest.param(torch.bfloat16, torch.float8_e4m3fn, id="bf16_vs_fp8"),
    pytest.param(torch.bool, torch.int32, id="bool_vs_int32"),
]

_INVALID_ARG_CASES = [
    pytest.param((1, 2), None, id="tuple_self"),
    pytest.param(1, None, id="int_self"),
    pytest.param(3.14, None, id="float_self"),
    pytest.param(None, 1, id="none_self"),
    pytest.param(None, None, id="none_both"),
    pytest.param("abc", "abc", id="str_both"),
]


def _make_tensor(spec, dtype, device):
    kind, shape = spec
    if kind == "plain":
        return torch.zeros(shape, dtype=dtype, device=device)
    if kind == "transposed":
        return torch.zeros((shape[1], shape[0]), dtype=dtype, device=device).t()
    if kind == "row_view":
        return torch.zeros(shape, dtype=dtype, device=device)[0]
    if kind == "expanded":
        base = torch.zeros((shape[0], 1), dtype=dtype, device=device)
        return base.expand(shape)
    if kind == "narrowed":
        base = torch.zeros(shape, dtype=dtype, device=device)
        return base.narrow(0, shape[0] // 4, max(shape[0] // 2, 1))
    raise ValueError(f"Unknown tensor spec kind: {kind!r}")


def _special_tensor(shape, dtype, scenario):
    numel = math.prod(shape)
    values = tu.make_special_input(dtype, scenario)
    repeats = (numel + values.numel() - 1) // values.numel()
    return values.repeat(repeats)[:numel].reshape(shape)


def _assert_result(res_out, ref_out):
    if isinstance(res_out, torch.Tensor):
        assert res_out.ndim == 0
        assert res_out.dtype == torch.bool
        res_out = res_out.item()
    assert type(res_out) is bool
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark._has_same_storage_numel
@pytest.mark.parametrize("self_spec,other_spec", _HAS_SAME_STORAGE_NUMEL_CASES)
@pytest.mark.parametrize("dtype", _HAS_SAME_STORAGE_NUMEL_DTYPES)
def test__has_same_storage_numel_layouts(self_spec, other_spec, dtype):
    self_t = _make_tensor(self_spec, dtype, flag_gems.device)
    other_t = _make_tensor(other_spec, dtype, flag_gems.device)

    # Build the reference from the same storage-layout spec on the reference
    # device: moving a view to CPU would compact its storage and change the
    # answer, so both sides must be constructed with identical layouts.
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device
    ref_self = _make_tensor(self_spec, dtype, ref_device)
    ref_other = _make_tensor(other_spec, dtype, ref_device)

    ref_out = torch.ops.aten._has_same_storage_numel(ref_self, ref_other)
    res_out = flag_gems._has_same_storage_numel(self_t, other_t)

    _assert_result(res_out, ref_out)


@pytest.mark._has_same_storage_numel
@pytest.mark.parametrize("self_dtype,other_dtype", _CROSS_DTYPE_CASES)
def test__has_same_storage_numel_cross_dtype(self_dtype, other_dtype):
    self_t = torch.zeros((4, 4), dtype=self_dtype, device=flag_gems.device)
    other_t = torch.zeros((16,), dtype=other_dtype, device=flag_gems.device)
    ref_self = self_t.to("cpu") if utils.TO_CPU else self_t
    ref_other = other_t.to("cpu") if utils.TO_CPU else other_t

    ref_out = torch.ops.aten._has_same_storage_numel(ref_self, ref_other)
    res_out = flag_gems._has_same_storage_numel(self_t, other_t)

    _assert_result(res_out, ref_out)


@pytest.mark._has_same_storage_numel
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _HAS_SAME_STORAGE_NUMEL_DTYPES)
def test__has_same_storage_numel_shapes(shape, dtype):
    self_t = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    other_t = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_self = tu.to_reference(self_t)
    ref_other = tu.to_reference(other_t)

    ref_out = torch.ops.aten._has_same_storage_numel(ref_self, ref_other)
    res_out = flag_gems._has_same_storage_numel(self_t, other_t)

    _assert_result(res_out, ref_out)


@pytest.mark._has_same_storage_numel
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _HAS_SAME_STORAGE_NUMEL_DTYPES)
def test__has_same_storage_numel_value_ranges(shape, value_range, dtype):
    self_t = tu.make_input(dtype, shape, value_range)
    other_t = tu.make_input(dtype, shape, value_range)
    ref_self = tu.to_reference(self_t)
    ref_other = tu.to_reference(other_t)

    ref_out = torch.ops.aten._has_same_storage_numel(ref_self, ref_other)
    res_out = flag_gems._has_same_storage_numel(self_t, other_t)

    _assert_result(res_out, ref_out)


@pytest.mark._has_same_storage_numel
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_HAS_SAME_STORAGE_NUMEL_DTYPES)),
)
def test__has_same_storage_numel_nan_inf(shape, dtype, scenario):
    self_t = _special_tensor(shape, dtype, scenario)
    other_t = _special_tensor(shape, dtype, scenario)
    ref_self = tu.to_reference(self_t)
    ref_other = tu.to_reference(other_t)

    ref_out = torch.ops.aten._has_same_storage_numel(ref_self, ref_other)
    res_out = flag_gems._has_same_storage_numel(self_t, other_t)

    _assert_result(res_out, ref_out)


@pytest.mark._has_same_storage_numel
def test__has_same_storage_numel_ignores_autograd():
    self_t = torch.zeros((4, 4), device=flag_gems.device).requires_grad_()
    other_t = torch.zeros((4, 4), device=flag_gems.device).requires_grad_()
    ref_self = self_t.detach()
    ref_other = other_t.detach()

    ref_out = torch.ops.aten._has_same_storage_numel(ref_self, ref_other)
    res_out = flag_gems._has_same_storage_numel(self_t, other_t)

    _assert_result(res_out, ref_out)
    assert not isinstance(res_out, torch.Tensor) or not res_out.requires_grad


@pytest.mark._has_same_storage_numel
@pytest.mark.parametrize("self_arg,other_arg", _INVALID_ARG_CASES)
def test__has_same_storage_numel_rejects_non_tensor(self_arg, other_arg):
    with pytest.raises(RuntimeError):
        torch.ops.aten._has_same_storage_numel(self_arg, other_arg)
    # The reference raises RuntimeError at the dispatcher level; a
    # plain-Python candidate naturally raises AttributeError (or a
    # TypeError/ValueError) for the same inputs, which is equally
    # acceptable.
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._has_same_storage_numel(self_arg, other_arg)


@pytest.mark._has_same_storage_numel
def test__has_same_storage_numel_rejects_missing_argument():
    inp = torch.zeros((4,), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten._has_same_storage_numel(inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._has_same_storage_numel(inp)
