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
try:
    pytest.mark._version
except AttributeError:
    setattr(
        pytest.mark,
        "_version",
        MarkDecorator(Mark("_version", (), {}, _ispytest=True), _ispytest=True),
    )

# Read the mutation counter shared by a tensor and its views.
_VERSION_SHAPES = tu.selected_cases(
    [(), (1,), (3, 4), (8, 16, 4), (2, 3, 4, 5), (4, 7, 5, 3, 2)], quick=[(2, 19, 7)]
)

_FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}

_VERSION_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool]
    + utils.COMPLEX_DTYPES
)

_MUTABLE_DTYPES = [
    dtype
    for dtype in _VERSION_DTYPES
    if dtype != torch.bool and dtype not in _FP8_DTYPES
]

# ATen accepts None and returns 0, so it is not an invalid-argument case.
_INVALID_ARG_CASES = [
    pytest.param(1, id="int"),
    pytest.param(3.14, id="float"),
    pytest.param("string", id="str"),
    pytest.param([1, 2], id="list"),
]


def _make_value_tensor(dtype, shape, value_range, device):
    # Build both sides the same way on their own devices; copying can reset the counter.
    low = tu.resolve_bound(value_range[0], dtype)
    high = tu.resolve_bound(value_range[1], dtype)

    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=device).bool()

    if not (dtype.is_floating_point or dtype.is_complex):
        low, high = int(low), int(high)
        dtype_min, _ = tu.dtype_bounds(dtype)
        if low < int(dtype_min):
            # Unsigned dtypes cannot represent the "-1" low symbol; snap it to
            # the representable minimum (the "min" symbol already resolves to
            # the dtype minimum through tu.resolve_bound).
            low = int(dtype_min)

    if low == high:
        return torch.full(shape, low, device=device, dtype=dtype)

    return torch.testing.make_tensor(
        shape, dtype=dtype, device=device, low=low, high=high
    )


def _special_tensor(shape, dtype, scenario, device):
    numel = math.prod(shape)
    values = tu.make_special_input(dtype, scenario).to(device)
    repeats = (numel + values.numel() - 1) // values.numel()
    return values.repeat(repeats)[:numel].reshape(shape)


def _as_int(value):
    # Normalize a Python scalar or a one-element tensor.
    if isinstance(value, torch.Tensor):
        assert value.numel() == 1, "candidate returned a non-scalar tensor"
        return value.item()
    return value


def _assert_result(res_out, ref_out):
    res_int = _as_int(res_out)
    ref_int = _as_int(ref_out)
    assert isinstance(res_int, int) and not isinstance(res_int, bool)
    utils.gems_assert_equal(res_int, ref_int)


@pytest.mark._version
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _VERSION_DTYPES)
def test__version_fresh(shape, dtype):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._version(ref_inp)
    res_out = flag_gems._version(inp)

    _assert_result(res_out, ref_out)


@pytest.mark._version
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", tu.REQUIRED_DTYPES)
def test__version_value_ranges(shape, value_range, dtype):
    inp = _make_value_tensor(dtype, shape, value_range, flag_gems.device)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device
    ref_inp = _make_value_tensor(dtype, shape, value_range, ref_device)

    ref_out = torch.ops.aten._version(ref_inp)
    res_out = flag_gems._version(inp)

    _assert_result(res_out, ref_out)


@pytest.mark._version
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_VERSION_DTYPES))
)
def test__version_nan_inf(shape, dtype, scenario):
    inp = _special_tensor(shape, dtype, scenario, flag_gems.device)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device
    ref_inp = _special_tensor(shape, dtype, scenario, ref_device)

    ref_out = torch.ops.aten._version(ref_inp)
    res_out = flag_gems._version(inp)

    _assert_result(res_out, ref_out)


@pytest.mark._version
@pytest.mark.parametrize("shape", _VERSION_SHAPES)
@pytest.mark.parametrize("bumps", [1, 2, 3, 5])
@pytest.mark.parametrize("dtype", _MUTABLE_DTYPES)
def test__version_after_inplace(shape, bumps, dtype):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    for _ in range(bumps):
        torch.ops.aten.add_.Tensor(inp, 1)
    if ref_inp is not inp:
        for _ in range(bumps):
            torch.ops.aten.add_.Tensor(ref_inp, 1)

    ref_out = torch.ops.aten._version(ref_inp)
    res_out = flag_gems._version(inp)

    _assert_result(res_out, ref_out)


@pytest.mark._version
@pytest.mark.parametrize("dtype", _VERSION_DTYPES)
def test__version_readonly(dtype):
    inp = torch.zeros((8, 16), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    data_before = inp.clone()
    version_before = torch.ops.aten._version(ref_inp)

    res_out = flag_gems._version(inp)

    _assert_result(res_out, version_before)
    assert torch.ops.aten._version(inp) == version_before
    assert torch.equal(inp, data_before)


@pytest.mark._version
@pytest.mark.parametrize("dtype", _VERSION_DTYPES)
def test__version_view(dtype):
    inp = torch.zeros((4, 6), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    view = inp.view(3, 8)
    ref_view = ref_inp.view(3, 8)

    ref_out = torch.ops.aten._version(ref_view)
    res_out = flag_gems._version(view)

    _assert_result(res_out, ref_out)


@pytest.mark._version
@pytest.mark.parametrize("dtype", _MUTABLE_DTYPES)
def test__version_view_inplace(dtype):
    inp = torch.zeros((4, 6), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    view = inp.view(3, 8)
    ref_view = ref_inp.view(3, 8)

    torch.ops.aten.add_.Tensor(view, 1)
    if ref_inp is not inp:
        torch.ops.aten.add_.Tensor(ref_view, 1)

    ref_out = torch.ops.aten._version(ref_view)
    res_out = flag_gems._version(inp)

    _assert_result(res_out, ref_out)


@pytest.mark._version
@pytest.mark.parametrize("dtype", _MUTABLE_DTYPES)
def test__version_detach_shares_counter(dtype):
    inp = torch.zeros((4,), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    detached = inp.detach()
    ref_detached = ref_inp.detach()

    torch.ops.aten.add_.Tensor(inp, 1)
    if ref_inp is not inp:
        torch.ops.aten.add_.Tensor(ref_inp, 1)

    ref_out = torch.ops.aten._version(ref_detached)
    res_out = flag_gems._version(detached)

    _assert_result(res_out, ref_out)


@pytest.mark._version
@pytest.mark.parametrize("dtype", _MUTABLE_DTYPES)
def test__version_independent_counters(dtype):
    first = torch.zeros((4,), dtype=dtype, device=flag_gems.device)
    second = torch.zeros((4,), dtype=dtype, device=flag_gems.device)
    ref_first = utils.to_reference(first)
    ref_second = utils.to_reference(second)

    torch.ops.aten.add_.Tensor(first, 1)
    torch.ops.aten.add_.Tensor(first, 1)
    torch.ops.aten.add_.Tensor(second, 1)
    if ref_first is not first:
        torch.ops.aten.add_.Tensor(ref_first, 1)
        torch.ops.aten.add_.Tensor(ref_first, 1)
        torch.ops.aten.add_.Tensor(ref_second, 1)

    _assert_result(flag_gems._version(first), torch.ops.aten._version(ref_first))
    _assert_result(flag_gems._version(second), torch.ops.aten._version(ref_second))


@pytest.mark._version
@pytest.mark.parametrize("bad_arg", _INVALID_ARG_CASES)
def test__version_rejects_non_tensor(bad_arg):
    with pytest.raises(RuntimeError):
        torch.ops.aten._version(bad_arg)

    # The reference raises RuntimeError at the dispatcher level; a plain
    # Python candidate naturally raises AttributeError / TypeError /
    # ValueError for the same inputs, which is equally acceptable.
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._version(bad_arg)


@pytest.mark._version
def test__version_rejects_wrong_arity():
    with pytest.raises((TypeError, RuntimeError)):
        torch.ops.aten._version()

    extra = torch.zeros(2, device=flag_gems.device)
    with pytest.raises((TypeError, RuntimeError)):
        torch.ops.aten._version(extra, 1)

    # The single Tensor argument may be passed by keyword.
    assert torch.ops.aten._version(self=extra) == 0

    # A candidate fails on a missing argument with whatever the runtime
    # raises for a wrong arity: the reference (packet / bound method) raises
    # RuntimeError, while a plain Python implementation raises TypeError.
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._version()
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._version(extra, 1)
