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

"""Correctness tests for aten::_cufft_get_plan_cache_max_size."""

import pytest
import torch

import flag_gems

from . import test_utils as tu

_NATIVE_GET = torch.ops.aten._cufft_get_plan_cache_max_size
_NATIVE_SET = torch.ops.aten._cufft_set_plan_cache_max_size
_NATIVE_CLEAR = torch.ops.aten._cufft_clear_plan_cache

# Index 0 exists on every host that has a device; probing the native runtime
# during collection would run the operator before the candidate is available.
VALID_DEVICE_INDICES = [0]

# Capacities the native setter stores and reads back exactly.
_PLAN_CAPACITY_VALUES = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    15,
    16,
    17,
    31,
    32,
    33,
    63,
    64,
    65,
    100,
    127,
    128,
    129,
    255,
    256,
    257,
    511,
    512,
    513,
    1000,
    1023,
    1024,
    1025,
    2047,
    2048,
    2049,
    4095,
    4096,
    4097,
    8191,
    8192,
    8193,
    16383,
    16384,
    16385,
    32767,
    32768,
    32769,
    65535,
    65536,
    65537,
    131071,
    131072,
    131073,
    262143,
    262144,
    262145,
    524287,
    524288,
    524289,
    1048575,
    1048576,
    1048577,
    2097151,
    2097152,
    2097153,
    4194303,
    4194304,
    4194305,
    8388607,
    8388608,
    8388609,
    16777215,
    16777216,
    16777217,
    33554431,
    33554432,
    33554433,
    10**7,
    5 * 10**7,
    10**8,
    10**9,
    2**30,
    2**30 + 1,
    2**31 - 2,
    2**31 - 1,
]

# Quick keeps the disabling boundary, the smallest capacity and a cheap
# non-power value; every capacity is a host scalar, so no case is expensive.
PLAN_CACHE_CAPACITIES = tu.selected_cases(_PLAN_CAPACITY_VALUES, quick=[0, 1, 7])

# Clearing plans must not reset the capacity; checked in both modes.
POST_CLEAR_CAPACITIES = [0, 7]

# The driver truncates the index to its low byte before the range check, so
# these spellings address device 0 and the native result stays the oracle.
ALIAS_DEVICE_INDICES = [256, -256, 2**31, -(2**31)]

# Low bytes 231..255 name devices this host does not have.
OUT_OF_RANGE_DEVICE_INDICES = [-1, -2, -3, -999, 255, 1000, 2**31 - 1]

# Non-integer indices, including the nan/inf boundary of a float argument.
NON_INTEGER_INDICES = [
    1.5,
    float("nan"),
    float("inf"),
    float("-inf"),
    None,
    "0",
    [0],
    (0,),
    0j,
]

# The driver builds the out-of-range message from a non-UTF8 buffer, which
# surfaces as UnicodeDecodeError (a ValueError) for many indices and as
# RuntimeError for a few; both are the native rejection.
_OUT_OF_RANGE_ERRORS = (RuntimeError, ValueError)
_NON_INTEGER_ERRORS = (RuntimeError, TypeError, ValueError)
_ARITY_ERRORS = (TypeError, RuntimeError)


@pytest.fixture()
def plan_cache_capacity():
    """Configure the native capacity and always restore it, even on failure."""
    original = {index: _NATIVE_GET(index) for index in VALID_DEVICE_INDICES}
    try:
        yield _NATIVE_SET
    finally:
        for index, capacity in original.items():
            _NATIVE_SET(index, capacity)


def _assert_limit(result, reference):
    """The schema returns a Python int, so compare type and value directly."""
    assert type(result) is int
    assert result == reference


@pytest.mark.cufft_get_plan_cache_max_size
@pytest.mark.parametrize("device_index", VALID_DEVICE_INDICES)
def test__cufft_get_plan_cache_max_size(device_index):
    reference = _NATIVE_GET(device_index)
    result = flag_gems._cufft_get_plan_cache_max_size(device_index)
    _assert_limit(result, reference)


@pytest.mark.cufft_get_plan_cache_max_size
@pytest.mark.parametrize("device_index", VALID_DEVICE_INDICES)
@pytest.mark.parametrize("capacity", PLAN_CACHE_CAPACITIES)
def test__cufft_get_plan_cache_max_size_with_configured_capacity(
    device_index, capacity, plan_cache_capacity
):
    plan_cache_capacity(device_index, capacity)
    reference = _NATIVE_GET(device_index)
    result = flag_gems._cufft_get_plan_cache_max_size(device_index)
    _assert_limit(result, reference)


@pytest.mark.cufft_get_plan_cache_max_size
@pytest.mark.parametrize("device_index", VALID_DEVICE_INDICES)
@pytest.mark.parametrize("capacity", POST_CLEAR_CAPACITIES)
def test__cufft_get_plan_cache_max_size_after_clear(
    device_index, capacity, plan_cache_capacity
):
    plan_cache_capacity(device_index, capacity)
    _NATIVE_CLEAR(device_index)
    reference = _NATIVE_GET(device_index)
    result = flag_gems._cufft_get_plan_cache_max_size(device_index)
    _assert_limit(result, reference)


@pytest.mark.cufft_get_plan_cache_max_size
@pytest.mark.parametrize("device_index", VALID_DEVICE_INDICES)
def test__cufft_get_plan_cache_max_size_accepts_keyword_index(device_index):
    reference = _NATIVE_GET(device_index=device_index)
    result = flag_gems._cufft_get_plan_cache_max_size(device_index=device_index)
    _assert_limit(result, reference)


@pytest.mark.cufft_get_plan_cache_max_size
@pytest.mark.parametrize("device_index", VALID_DEVICE_INDICES)
def test__cufft_get_plan_cache_max_size_accepts_int_like_index(device_index):
    # A 0-dim integer tensor spells the same index argument, not an input tensor.
    index = torch.tensor(device_index, device=flag_gems.device)
    reference = _NATIVE_GET(index)
    result = flag_gems._cufft_get_plan_cache_max_size(index)
    _assert_limit(result, reference)


@pytest.mark.cufft_get_plan_cache_max_size
@pytest.mark.parametrize("device_index", ALIAS_DEVICE_INDICES)
def test__cufft_get_plan_cache_max_size_matches_native_index_aliasing(device_index):
    reference = _NATIVE_GET(device_index)
    result = flag_gems._cufft_get_plan_cache_max_size(device_index)
    _assert_limit(result, reference)


@pytest.mark.cufft_get_plan_cache_max_size
@pytest.mark.parametrize("device_index", OUT_OF_RANGE_DEVICE_INDICES)
def test__cufft_get_plan_cache_max_size_rejects_out_of_range_index(device_index):
    with pytest.raises(_OUT_OF_RANGE_ERRORS):
        flag_gems._cufft_get_plan_cache_max_size(device_index)


@pytest.mark.cufft_get_plan_cache_max_size
@pytest.mark.parametrize("device_index", NON_INTEGER_INDICES)
def test__cufft_get_plan_cache_max_size_rejects_non_integer_index(device_index):
    with pytest.raises(_NON_INTEGER_ERRORS):
        flag_gems._cufft_get_plan_cache_max_size(device_index)


@pytest.mark.cufft_get_plan_cache_max_size
@pytest.mark.parametrize("args", [(), (0, 0), (0, 0, 0)])
def test__cufft_get_plan_cache_max_size_rejects_wrong_arity(args):
    with pytest.raises(_ARITY_ERRORS):
        flag_gems._cufft_get_plan_cache_max_size(*args)
