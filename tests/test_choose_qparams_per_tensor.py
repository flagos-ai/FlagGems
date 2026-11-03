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

# ---------------------------------------------------------------------------
# Shared value-range / shape helpers (mirrors the operator-test spec
# conventions; QUICK_MODE selects the reduced sets via pytest --quick).
# ---------------------------------------------------------------------------
QUICK_MODE = False


def selected_cases(cases, *, quick=()):
    """Select parameter values; an empty quick subset contributes no cases."""
    return list(quick if QUICK_MODE else cases)


# Full set: ranks 0 through 5; the extents keep the whole-tensor min/max
# reduction bandwidth-relevant while staying allocation-friendly.
REQUIRED_SHAPES = [
    (),  # 0-dim scalar
    (1,),  # single-element 1-dim
    (256,),  # regular 1-dim
    (1024, 1024),  # regular 2-dim
    (20, 320, 15),  # regular 3-dim
    (16, 128, 64, 60),  # 4-dim
    (16, 7, 57, 32, 29),  # 5-dim
]


def selected_shapes():
    return [(2, 19, 7)] if QUICK_MODE else REQUIRED_SHAPES


REQUIRED_RANGES = [
    ["-1", "1"],
    ["0", "1"],
    ["-1", "0"],
    ["0", "max"],
    ["min", "0"],
]


def dtype_bounds(dtype):
    if dtype == torch.bool:
        return 0, 1
    if dtype.is_complex:
        real = {
            torch.complex32: torch.float16,
            torch.complex64: torch.float32,
            torch.complex128: torch.float64,
        }[dtype]
        finfo = torch.finfo(real)
        return finfo.min, finfo.max
    if dtype.is_floating_point:
        finfo = torch.finfo(dtype)
        return finfo.min, finfo.max
    iinfo = torch.iinfo(dtype)
    return iinfo.min, iinfo.max


def make_input(dtype, shape, value_range):
    """Build a tensor of ``shape`` / ``dtype`` with values in ``value_range``.

    Bounds are clamped to the dtype's representable range first, so a range a
    dtype cannot represent (e.g. uint8 and ``[-1, 0]``) degenerates to a
    constant fill instead of raising.
    """
    low, high = dtype_bounds(dtype)
    table = {
        "-1": -1.0,
        "0": 0.0,
        "1": 1.0,
        "max": high,
        "min": low,
        "max/2": high / 2,
        "min/2": low / 2,
    }
    low, high = table[value_range[0]], table[value_range[1]]

    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=flag_gems.device).bool()

    if not (dtype.is_floating_point or dtype.is_complex):
        low, high = int(low), int(high)
        info = torch.iinfo(dtype)
        low = max(low, int(info.min))
        high = min(high, int(info.max))
        low = min(low, high)

    if low == high:
        return torch.full(shape, low, device=flag_gems.device, dtype=dtype)

    return torch.testing.make_tensor(
        shape, dtype=dtype, device=flag_gems.device, low=low, high=high
    )


def make_special_input(dtype, scenario):
    payloads = {
        "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
        "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
        "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
    }
    return torch.tensor(
        payloads[scenario], device=flag_gems.device, dtype=torch.float32
    ).to(dtype)


def to_reference(inp, upcast=False):
    """Independent tensor oracle for these tests, retaining view metadata."""
    if inp is None:
        return None
    if inp.layout != torch.strided or inp.is_quantized or inp.is_nested:
        reference = inp.detach().clone()
    else:
        reference = torch.empty(0, dtype=inp.dtype, device=inp.device).set_(
            inp.untyped_storage().clone(),
            inp.storage_offset(),
            inp.size(),
            inp.stride(),
        )
        if inp.is_conj():
            reference = reference.conj()
    reference.requires_grad_(inp.requires_grad)
    return reference


def assert_result_close(result, reference, *, atol=1e-4):
    # NaN payloads feed the rejection tests and the comparison must ignore NaNs.
    utils.gems_assert_close(
        result, reference, dtype=result.dtype, atol=atol, equal_nan=True
    )


def special_value_cases(dtypes):
    """Representable special scenarios, kept separate in collected case IDs."""
    cases = []
    for dtype in dtypes:
        if not dtype.is_floating_point:
            continue
        cases.append((dtype, "nan"))
        if dtype not in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz,
        ):
            cases.extend((dtype, kind) for kind in ("inf", "mixed"))
    return cases


def resolve_bound(symbol, dtype):
    """Resolve a range-bound symbol to an actual value for ``dtype``."""
    low, high = dtype_bounds(dtype)
    table = {
        "-1": -1.0,
        "0": 0.0,
        "1": 1.0,
        "max": high,
        "min": low,
        "max/2": high / 2,
        "min/2": low / 2,
    }
    return table[symbol]


def selected_ranges():
    return [["-1", "1"]] if QUICK_MODE else REQUIRED_RANGES


# Reduce input extrema to Python (scale, zero_point) scalars.
_CQPT_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + [torch.int8, torch.uint8]
    + utils.ALL_INT_DTYPES
    + utils.BOOL_TYPES
)

_CQPT_REDUCE_RANGE = selected_cases([False, True], quick=[False])

# Clamp/rounding cases avoid exact half-integer zero points.
_CQPT_TINY_CASES = [
    [0.005],  # raw ~= 1.96e-5 -> clamped, zp = 0
    [-0.005],  # raw ~= 1.96e-5 -> clamped, zp = qmax
    [-0.0075, 0.0025, 0.0],  # raw ~= 3.92e-5 -> clamped, zp = 191 / 95
    [-0.002, 0.008, 0.0],  # raw ~= 3.92e-5 -> clamped, zp = 51 / 25
    [-0.008, 0.002, 0.0],  # raw ~= 3.92e-5 -> clamped, zp = 204 / 102
    [0.016],  # raw ~= 6.27e-5 -> just above the clamp, unclamped
]

_CQPT_CONSTANT_VALUES = [0.0, 5.0, -5.0, 1e-8, 1e-2]

_CQPT_INF_INPUT = [float("inf"), float("-inf"), 0.0]

# The reference reduces through fp32; keep fp64 extremes below its overflow bound.
_FP64_EXTREME_BOUND = 1e30


def _make_input(shape, dtype, value_range):
    # Keep uint8 bounds representable and limit fp64 extrema to the reference range.
    if dtype == torch.uint8:
        # Clamp negative symbols to 0 for unsigned dtypes.
        low = max(int(resolve_bound(value_range[0], dtype)), 0)
        high = max(int(resolve_bound(value_range[1], dtype)), 0)
        if low == high:
            return torch.full(shape, low, dtype=dtype, device=flag_gems.device)
        return torch.testing.make_tensor(
            shape, dtype=dtype, device=flag_gems.device, low=low, high=high
        )

    if dtype != torch.float64:
        return make_input(dtype, shape, value_range)

    # fp64: keep |value| <= 1e30 so the fp32-internal reference does not
    # overflow on the finfo-derived "min" / "max" symbols.
    table = {
        "-1": -1.0,
        "0": 0.0,
        "1": 1.0,
        "max": _FP64_EXTREME_BOUND,
        "min": -_FP64_EXTREME_BOUND,
        "max/2": _FP64_EXTREME_BOUND / 2,
        "min/2": -_FP64_EXTREME_BOUND / 2,
    }
    low = table[value_range[0]]
    high = table[value_range[1]]
    if low == high:
        return torch.full(shape, low, dtype=dtype, device=flag_gems.device)
    return torch.testing.make_tensor(
        shape, dtype=dtype, device=flag_gems.device, low=low, high=high
    )


def _assert_pair(res, ref):
    # Compare scale with float64 tolerance and zero_point exactly; preserve Python scalar types.
    res_scale, res_zp = res
    ref_scale, ref_zp = ref
    assert isinstance(res_scale, float), type(res_scale)
    assert isinstance(res_zp, int) and not isinstance(res_zp, bool), type(res_zp)
    assert_result_close(
        torch.tensor(res_scale, dtype=torch.float64),
        torch.tensor(ref_scale, dtype=torch.float64),
        atol=0,
    )
    assert res_zp == ref_zp, f"zero_point {res_zp} != {ref_zp}"


@pytest.mark.choose_qparams_per_tensor
@pytest.mark.parametrize("shape", selected_shapes())
@pytest.mark.parametrize("value_range", selected_ranges())
@pytest.mark.parametrize("dtype", _CQPT_DTYPES)
@pytest.mark.parametrize("reduce_range", _CQPT_REDUCE_RANGE)
def test_choose_qparams_per_tensor_value_ranges(
    shape, value_range, dtype, reduce_range
):
    utils.init_seed(0)
    inp = _make_input(shape, dtype, value_range)
    ref_inp = to_reference(inp)

    ref_pair = torch.ops.aten._choose_qparams_per_tensor(ref_inp, reduce_range)
    res_pair = flag_gems.choose_qparams_per_tensor(inp, reduce_range)

    _assert_pair(res_pair, ref_pair)


@pytest.mark.choose_qparams_per_tensor
@pytest.mark.parametrize("values", _CQPT_TINY_CASES)
@pytest.mark.parametrize("reduce_range", _CQPT_REDUCE_RANGE)
def test_choose_qparams_per_tensor_tiny_scale(values, reduce_range):
    # zero_point uses the raw scale even when the returned scale is clamped.
    inp = torch.tensor(values, dtype=torch.float32, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_pair = torch.ops.aten._choose_qparams_per_tensor(ref_inp, reduce_range)
    res_pair = flag_gems.choose_qparams_per_tensor(inp, reduce_range)

    _assert_pair(res_pair, ref_pair)


@pytest.mark.choose_qparams_per_tensor
@pytest.mark.parametrize("value", _CQPT_CONSTANT_VALUES)
# The constant and non-contiguous families pin the half/full/bfloat16 trio:
# one per precision class, so the clamp/rounding and strided paths are
# exercised on each without enlarging the grid.
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("reduce_range", _CQPT_REDUCE_RANGE)
def test_choose_qparams_per_tensor_constant(value, dtype, reduce_range):
    inp = torch.full((1024,), value, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_pair = torch.ops.aten._choose_qparams_per_tensor(ref_inp, reduce_range)
    res_pair = flag_gems.choose_qparams_per_tensor(inp, reduce_range)

    _assert_pair(res_pair, ref_pair)


@pytest.mark.choose_qparams_per_tensor
@pytest.mark.parametrize("reduce_range", _CQPT_REDUCE_RANGE)
def test_choose_qparams_per_tensor_inf(reduce_range):
    inp = torch.tensor(_CQPT_INF_INPUT, dtype=torch.float32, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_pair = torch.ops.aten._choose_qparams_per_tensor(ref_inp, reduce_range)
    res_pair = flag_gems.choose_qparams_per_tensor(inp, reduce_range)

    _assert_pair(res_pair, ref_pair)


@pytest.mark.choose_qparams_per_tensor
@pytest.mark.parametrize("layout", ["transpose", "slice"])
# The constant and non-contiguous families pin the half/full/bfloat16 trio:
# one per precision class, so the strided reduction path is exercised on
# every precision class without enlarging the grid.
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("reduce_range", _CQPT_REDUCE_RANGE)
def test_choose_qparams_per_tensor_non_contiguous(layout, dtype, reduce_range):
    utils.init_seed(0)
    base_inp = torch.randn((64, 32), dtype=dtype, device=flag_gems.device)
    inp = base_inp.t() if layout == "transpose" else base_inp[:, ::2]
    assert not inp.is_contiguous()

    ref_inp = to_reference(inp)
    ref_pair = torch.ops.aten._choose_qparams_per_tensor(ref_inp, reduce_range)
    res_pair = flag_gems.choose_qparams_per_tensor(inp, reduce_range)

    _assert_pair(res_pair, ref_pair)


@pytest.mark.choose_qparams_per_tensor
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_choose_qparams_per_tensor_default_reduce_range(dtype):
    utils.init_seed(0)
    inp = _make_input((20, 320, 15), dtype, ["-1", "1"])
    ref_inp = to_reference(inp)

    ref_pair = torch.ops.aten._choose_qparams_per_tensor(ref_inp)
    res_pair = flag_gems.choose_qparams_per_tensor(inp)

    _assert_pair(res_pair, ref_pair)


@pytest.mark.choose_qparams_per_tensor
def test_choose_qparams_per_tensor_rejects_nan():
    inp = torch.tensor(
        [float("nan"), 1.0, 2.0], dtype=torch.float32, device=flag_gems.device
    )
    ref_inp = to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten._choose_qparams_per_tensor(ref_inp, False)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.choose_qparams_per_tensor(inp, False)


@pytest.mark.choose_qparams_per_tensor
def test_choose_qparams_per_tensor_rejects_empty():
    inp = torch.empty(0, dtype=torch.float32, device=flag_gems.device)
    ref_inp = to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten._choose_qparams_per_tensor(ref_inp, False)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.choose_qparams_per_tensor(inp, False)


@pytest.mark.choose_qparams_per_tensor
def test_choose_qparams_per_tensor_rejects_complex():
    inp = torch.tensor([1.0 + 2.0j], dtype=torch.complex64, device=flag_gems.device)
    ref_inp = to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten._choose_qparams_per_tensor(ref_inp, False)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.choose_qparams_per_tensor(inp, False)


@pytest.mark.choose_qparams_per_tensor
def test_choose_qparams_per_tensor_rejects_fp8():
    inp = torch.zeros((4,), dtype=torch.float8_e4m3fn, device=flag_gems.device)
    ref_inp = to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten._choose_qparams_per_tensor(ref_inp, False)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.choose_qparams_per_tensor(inp, False)


@pytest.mark.choose_qparams_per_tensor
def test_choose_qparams_per_tensor_rejects_non_tensor():
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten._choose_qparams_per_tensor(3.14, False)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.choose_qparams_per_tensor(3.14, False)
