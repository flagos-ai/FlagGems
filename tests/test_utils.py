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


"""Shared test utilities for the regular-operator test spec.

Implements the value-range / shape-level / broadcast / backward conventions
from the "常规算子测试用例" spec (quick/default modes selected by the pytest
``--quick`` flag, matching FlagGems' own accuracy_utils convention). Tests
reference these helpers so the value-range and shape-selection logic lives in
one place instead of being copied into every ``tests/test_<op>.py`` file.

Reference example: the `add` sample (123.py) attached to the spec.
"""

import torch

import flag_gems

from .conftest import QUICK_MODE

# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------
# QUICK_MODE comes directly from pytest --quick; without it, use default cases.


def selected_cases(cases, *, quick=()):
    """Select parameter values; an empty quick subset contributes no cases."""
    return list(quick if QUICK_MODE else cases)


def selected_shapes():
    return QUICK_SHAPES if QUICK_MODE else REQUIRED_SHAPES


def selected_ranges():
    return QUICK_RANGES if QUICK_MODE else REQUIRED_RANGES


# ---------------------------------------------------------------------------
# dtype bounds and value-range resolution
# ---------------------------------------------------------------------------


def dtype_bounds(dtype):
    """Return the (min, max) value bounds of ``dtype``.

    - bool: fixed 0/1
    - complex: bounds of its real float dtype
    - floating / integer: finfo / iinfo min/max
    """
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


def resolve_bound(symbol, dtype):
    """Resolve a range-bound symbol (-1 / 0 / 1 / max / min / max/2 / min/2)
    to an actual value for ``dtype``."""
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


def make_input(dtype, shape, value_range):
    """Build a tensor of ``shape`` / ``dtype`` with values in ``value_range``.

    ``value_range`` is a [low_symbol, high_symbol] pair whose symbols resolve
    per-dtype (max/min are the dtype bounds). bool ignores the range; integer
    ranges are snapped to ints; a degenerate range (low == high) fills the
    constant; everything else uses torch.testing.make_tensor (complex fills
    both real and imaginary parts).

    Bounds are clamped to the dtype's representable range first, so the spec's
    five ranges work unchanged for dtypes that cannot represent a bound (e.g.
    uint8 cannot hold ``-1``, so ``[-1,0]`` becomes the degenerate ``[0,0]``
    → a constant zero fill). This keeps the caller from having to special-case
    unsigned dtypes.
    """
    low = resolve_bound(value_range[0], dtype)
    high = resolve_bound(value_range[1], dtype)

    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=flag_gems.device).bool()

    if not (dtype.is_floating_point or dtype.is_complex):
        low, high = int(low), int(high)
        dtype_min, dtype_max = dtype_bounds(dtype)
        # Clamp into the representable range (e.g. uint8 [-1,0] -> [0,0]).
        low = max(low, int(dtype_min))
        high = min(high, int(dtype_max))
        low = min(low, high)

    if low == high:
        return torch.full(shape, low, device=flag_gems.device, dtype=dtype)

    return torch.testing.make_tensor(
        shape, dtype=dtype, device=flag_gems.device, low=low, high=high
    )


_FP8_DTYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
}


def _comparison_values(result, reference):
    from . import accuracy_utils as utils

    assert result.dtype == reference.dtype
    result = utils.to_cpu(result, reference)
    # FP8 has no native comparison kernel. Conversion is lossless and happens
    # only after checking the original dtype; tolerance uses that dtype too.
    if result.dtype in _FP8_DTYPES:
        return result.float(), reference.float()
    return result, reference


def assert_result_close(result, reference, *, atol=1e-4):
    dtype = result.dtype
    result, reference = _comparison_values(result, reference)
    if dtype.is_floating_point or dtype.is_complex:
        torch.testing.assert_close(
            result,
            reference,
            rtol=flag_gems.testing.RESOLUTION[dtype],
            atol=atol,
            equal_nan=True,
        )
    else:
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def assert_result_equal(result, reference):
    result, reference = _comparison_values(result, reference)
    torch.testing.assert_close(result, reference, rtol=0, atol=0, equal_nan=True)


def to_reference(inp, upcast=False):
    """Independent tensor oracle for these tests, retaining view metadata.

    A plain clone can compact strides, reset offsets or materialize lazy flags.
    Copy dense storage to keep those tested properties. This does not reproduce
    inter-input aliases or autograd history; version-counter tests use their
    own reference setup.
    """
    from . import accuracy_utils as utils

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
        if inp.is_neg():
            reference = torch._neg_view(reference)
    reference.requires_grad_(inp.requires_grad)
    return utils.to_reference(reference, upcast)


# ---------------------------------------------------------------------------
# Shapes and value ranges by level
# ---------------------------------------------------------------------------

QUICK_SHAPES = [
    (2, 19, 7),
]

# Full set — the required 7 shapes from the team's operator-test spec
# (0~5 dims; a fixed-dim operator keeps only the ranks it accepts).
REQUIRED_SHAPES = [
    (),  # 0-dim scalar
    (1,),  # single-element 1-dim
    (256,),  # regular 1-dim
    (1024, 1024),  # regular 2-dim
    (20, 320, 15),  # regular 3-dim
    (16, 128, 64, 60),  # 4-dim
    (16, 7, 57, 32, 29),  # 5-dim
]

QUICK_RANGES = [
    ["-1", "1"],
]

# Full set — the required 5 value ranges from the spec
#   [-1,1], [0,1], [-1,0], [0,dtype_max], [dtype_min,0]
REQUIRED_RANGES = [
    ["-1", "1"],
    ["0", "1"],
    ["-1", "0"],
    ["0", "max"],
    ["min", "0"],
]

# Required dtype coverage (spec). int8/uint8/fp8 must be present for every
# operator whose CUDA kernel supports them; fp32/bf16/fp16/int32/int64 are
# added when the operator supports them.
REQUIRED_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
]

# Required minimum number of correctness cases per operator (spec: "at least
# 100"). 5 ranges x 7 shapes = 35 per dtype, so >=3 dtypes already clears it.
MIN_CASES = 100


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


def make_special_input(dtype, scenario):
    payloads = {
        "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
        "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
        "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
    }
    return torch.tensor(
        payloads[scenario], device=flag_gems.device, dtype=torch.float32
    ).to(dtype)


def is_extreme_range(value_range):
    return any(bound in {"min", "max", "min/2", "max/2"} for bound in value_range)
