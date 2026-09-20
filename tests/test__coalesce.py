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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Register underscore-prefixed pytest markers explicitly.
setattr(
    pytest.mark,
    "_coalesce",
    MarkDecorator(Mark("_coalesce", (), {}, _ispytest=True), _ispytest=True),
)
setattr(
    pytest.mark,
    "_coalesce_out",
    MarkDecorator(Mark("_coalesce_out", (), {}, _ispytest=True), _ispytest=True),
)

# Merge duplicate COO coordinates and sum their stored values.
# FP8 inputs are exercised by the rejection test.
_COALESCE_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
]

# (shape, nnz); small coordinate spaces exercise duplicate reduction.
_COALESCE_CASES = tu.selected_cases(
    [
        ((4, 4), 20),
        ((5, 5), 30),
        ((8, 8), 80),
        ((16, 16), 300),
        ((64,), 200),
        ((2, 3, 4), 28),
        ((3, 5, 7), 120),
        ((4, 8, 16), 600),
        ((16, 7, 57), 2000),
        ((4, 4, 4, 4), 400),
    ],
    quick=[((4, 4), 20), ((3, 5, 7), 120)],
)

_VALUE_RANGE_CASES = tu.selected_cases(
    [((64,), 200), ((3, 5, 7), 120), ((4, 4, 4, 4), 400)], quick=[((3, 5, 7), 120)]
)

# Preserve the same-sign, non-extreme fp16/bf16 sweep: CPU/CUDA accumulation
# may differ on cancellation and overflow. Integer extremes remain covered.
_NARROW_FLOAT_DTYPES = (torch.float16, torch.bfloat16)
_EXTREME_RANGES = (("0", "max"), ("min", "0"))
_VALUE_CASES = [
    (value_range, dtype, case)
    for dtype in _COALESCE_DTYPES
    if dtype != torch.bool
    for value_range in tu.selected_ranges()
    if dtype not in _NARROW_FLOAT_DTYPES
    or (value_range != ["-1", "1"] and tuple(value_range) not in _EXTREME_RANGES)
    for case in _VALUE_RANGE_CASES
]


def _default_bounds(dtype):
    if dtype.is_floating_point or dtype == torch.bool:
        return 0.0, 1.0
    info = torch.iinfo(dtype)
    return (-5.0, 6.0) if info.min < 0 else (0.0, 6.0)


def _make_input(shape, nnz, dtype, value_range=None, low=None, high=None, seed=2026):
    # Seeded indices may repeat; default floats are non-negative to avoid cancellation.
    gen = torch.Generator("cpu").manual_seed(seed)
    if low is None or high is None:
        low, high = _default_bounds(dtype)
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in shape
        ]
    )
    if value_range is not None:
        values = tu.make_input(dtype, (nnz,), value_range).cpu()
    elif dtype == torch.bool:
        values = torch.randint(0, 2, (nnz,), dtype=torch.bool, generator=gen)
    elif dtype.is_floating_point:
        # Generate in float64 so that dtype-extreme bounds (e.g. finfo.min/max)
        # are representable while sampling; casting afterwards keeps extreme
        # float64 values from turning into inf/nan during the draw.
        values = (
            torch.rand(nnz, dtype=torch.float64, generator=gen)
            * (float(high) - float(low))
            + float(low)
        ).to(dtype)
    else:
        low_i, high_i = int(low), int(high)
        if high_i <= low_i:
            values = torch.full((nnz,), low_i, dtype=torch.int64, device="cpu").to(
                dtype
            )
        else:
            values = torch.randint(
                low_i, high_i, (nnz,), dtype=torch.int64, generator=gen
            ).to(dtype)
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


def _make_special_values(nnz, dtype, scenario):
    base = tu.make_special_input(dtype, scenario)
    return base.repeat((nnz + base.numel() - 1) // base.numel())[:nnz]


def _make_empty_out(shape, dtype, device):
    indices = torch.empty((len(shape), 0), dtype=torch.long, device=device)
    values = torch.empty((0,), dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def _assert_coalesced(res_out, ref_out, dtype, *, equal_nan=False):
    assert res_out.layout == torch.sparse_coo
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.is_coalesced()
    # Indices are int64 and must match exactly (unique, sorted coordinates).
    utils.gems_assert_equal(res_out.indices(), ref_out.indices())
    # Values are sums of duplicates: tolerance for float, exact for int/bool.
    if dtype.is_floating_point or dtype.is_complex:
        utils.gems_assert_close(
            res_out.values(), ref_out.values(), dtype, equal_nan=equal_nan
        )
    else:
        utils.gems_assert_equal(res_out.values(), ref_out.values())


@pytest.mark._coalesce
@pytest.mark.parametrize("case", _COALESCE_CASES)
@pytest.mark.parametrize("dtype", _COALESCE_DTYPES)
def test__coalesce(case, dtype):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype)
    assert not inp.is_coalesced()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._coalesce(ref_inp)
    res_out = flag_gems._coalesce(inp)

    _assert_coalesced(res_out, ref_out, dtype)
    # Coalescing returns a fresh tensor and must not mutate the input.
    assert res_out is not inp
    assert not inp.is_coalesced()


@pytest.mark._coalesce
@pytest.mark.parametrize("value_range,dtype,case", _VALUE_CASES)
def test__coalesce_value_ranges(value_range, dtype, case):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype, value_range=value_range)
    assert not inp.is_coalesced()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._coalesce(ref_inp)
    res_out = flag_gems._coalesce(inp)

    _assert_coalesced(res_out, ref_out, dtype)
    assert res_out is not inp
    assert not inp.is_coalesced()


@pytest.mark._coalesce
@pytest.mark.parametrize("case", _COALESCE_CASES)
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_COALESCE_DTYPES)),
)
def test__coalesce_nan_inf(case, dtype, scenario):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype)
    # Keep duplicate indices and vary the special-value scenario independently.
    inp = torch.sparse_coo_tensor(
        inp._indices(),
        _make_special_values(nnz, dtype, scenario),
        shape,
        device=flag_gems.device,
    )
    assert not inp.is_coalesced()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._coalesce(ref_inp)
    res_out = flag_gems._coalesce(inp)

    _assert_coalesced(res_out, ref_out, dtype, equal_nan=True)
    assert res_out is not inp
    assert not inp.is_coalesced()


@pytest.mark._coalesce_out
@pytest.mark.parametrize("case", _COALESCE_CASES)
@pytest.mark.parametrize("dtype", _COALESCE_DTYPES)
def test__coalesce_out(case, dtype):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype)
    assert not inp.is_coalesced()
    ref_inp = tu.to_reference(inp)
    out = _make_empty_out(shape, dtype, flag_gems.device)
    ref_out = _make_empty_out(shape, dtype, ref_inp.device)

    ref_ret = torch.ops.aten._coalesce.out(ref_inp, out=ref_out)
    res_ret = flag_gems._coalesce(inp, out=out)

    # The .out variant must write into and return the out tensor itself.
    assert res_ret is out
    _assert_coalesced(res_ret, ref_ret, dtype)
    assert not inp.is_coalesced()


@pytest.mark._coalesce
def test__coalesce_rejects_dense_input():
    inp = torch.randn(4, 4, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten._coalesce(inp)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        flag_gems._coalesce(inp)


@pytest.mark._coalesce
def test__coalesce_rejects_csr_input():
    inp = torch.randn(4, 4, dtype=torch.float32, device=flag_gems.device)
    inp = inp.to_sparse_csr()
    with pytest.raises(RuntimeError):
        torch.ops.aten._coalesce(inp)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        flag_gems._coalesce(inp)


@pytest.mark._coalesce
def test__coalesce_rejects_fp8_input():
    indices = torch.zeros((1, 3), dtype=torch.long, device=flag_gems.device)
    values = torch.zeros(3, dtype=torch.float8_e4m3fn, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, (4,), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten._coalesce(inp)
    with pytest.raises((RuntimeError, NotImplementedError)):
        flag_gems._coalesce(inp)
