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

"""Correctness tests for aten::_cufft_set_plan_cache_max_size.

Schema: ``_cufft_set_plan_cache_max_size(int device_index, int max_size) -> ()``.
The operator has no tensor operand and no return value; its contract is the
capacity it stores in the per-device cuFFT plan cache, observed through the
paired ``_cufft_get_plan_cache_max_size`` and ``_cufft_get_plan_cache_size``
entry points of that same cache.  The dtype, value-range, shape, broadcast,
backward and NaN/Inf grids therefore do not apply; the int64 capacity domain,
every native call form and the effect on a populated cache are covered instead.
"""

import contextlib

import pytest
import torch

import flag_gems

from . import test_utils as tu


def _plan_cache_index():
    """Ordinal of the device whose plan cache flag_gems.device refers to."""
    device = torch.device(flag_gems.device)
    return 0 if device.index is None else device.index


_PLAN_CACHE_INDEX = _plan_cache_index()

# Static vendor capability, read without touching a device or the driver during
# collection: this entry point drives the NVIDIA cuFFT runtime plan cache.
_PLAN_CACHE_AVAILABLE = (
    flag_gems.vendor_name == "nvidia"
    and hasattr(torch.ops.aten, "_cufft_set_plan_cache_max_size")
    and hasattr(torch.ops.aten, "_cufft_get_plan_cache_max_size")
)
pytestmark = pytest.mark.skipif(
    not _PLAN_CACHE_AVAILABLE,
    reason=(
        "the plan cache is an NVIDIA cuFFT runtime resource; backend "
        f"{flag_gems.vendor_name} has to expose its own runtime cache"
    ),
)


@contextlib.contextmanager
def _plan_cache_capacity():
    """Yield the cache index and put its ambient capacity back afterwards.

    The capacity is process-global and the user may have chosen it before the
    test, so the value read at entry is what is restored, in a ``finally`` block
    that also runs when a candidate call or an assertion raises.
    """
    index = _PLAN_CACHE_INDEX
    previous = torch.ops.aten._cufft_get_plan_cache_max_size(index)
    try:
        yield index
    finally:
        torch.ops.aten._cufft_set_plan_cache_max_size(index, previous)


def _distinct_capacity(capacity):
    """A valid capacity different from ``capacity``.

    Stored through the reference before the candidate runs, so a candidate that
    writes nothing cannot pass by leaving an earlier value in place.
    """
    return 1 if capacity != 1 else 2


# Boundary sweep of the int64 capacity domain, including the 4096 default.
_STORE_VALUES = [
    0,
    1,
    2,
    3,
    4,
    7,
    16,
    31,
    64,
    255,
    256,
    1024,
    4095,
    4096,
    4097,
    8192,
    65536,
    1 << 20,
    (1 << 31) - 1,
    1 << 31,
    (1 << 32) - 1,
    1 << 32,
    1 << 62,
    (1 << 63) - 1,
]
_QUICK_STORE_VALUES = _STORE_VALUES
_STORE_CASES = tu.selected_cases(_STORE_VALUES, quick=_QUICK_STORE_VALUES)

# Every Python-int call form the schema accepts for the single public entry
# point (both parameters are positional-or-keyword).
_CALL_FORMS = ("positional", "max_size_keyword", "both_keyword")


def _call_arguments(call_form, index, capacity):
    if call_form == "positional":
        return (index, capacity), {}
    if call_form == "max_size_keyword":
        return (index,), {"max_size": capacity}
    return (), {"device_index": index, "max_size": capacity}


@pytest.mark.cufft_set_plan_cache_max_size
@pytest.mark.parametrize("call_form", _CALL_FORMS)
@pytest.mark.parametrize("capacity", _STORE_CASES)
def test_set_plan_cache_max_size(capacity, call_form):
    args, kwargs = _call_arguments(call_form, _PLAN_CACHE_INDEX, capacity)
    with _plan_cache_capacity() as index:
        control = _distinct_capacity(capacity)
        torch.ops.aten._cufft_set_plan_cache_max_size(index, control)

        assert flag_gems._cufft_set_plan_cache_max_size(*args, **kwargs) is None
        assert torch.ops.aten._cufft_get_plan_cache_max_size(index) == capacity


# Overwriting an already stored capacity is the stateful core of this operator,
# including the shrink/grow pair, so the two cheap transitions stay in quick mode
# with small fixtures.
_TRANSITIONS = [(0, 4096), (4096, 0), ((1 << 63) - 1, 0), (0, (1 << 63) - 1)]
_TRANSITION_CASES = _TRANSITIONS


@pytest.mark.cufft_set_plan_cache_max_size
@pytest.mark.parametrize("first,second", _TRANSITION_CASES)
def test_repeated_set_overwrites_previous(first, second):
    with _plan_cache_capacity() as index:
        for capacity in (first, second):
            control = _distinct_capacity(capacity)
            torch.ops.aten._cufft_set_plan_cache_max_size(index, control)
            assert flag_gems._cufft_set_plan_cache_max_size(index, capacity) is None
            assert torch.ops.aten._cufft_get_plan_cache_max_size(index) == capacity


@pytest.mark.cufft_set_plan_cache_max_size
def test_bool_max_size_uses_int_subclass_semantics():
    # bool subclasses int, so the int parameter accepts True as capacity 1.
    with _plan_cache_capacity() as index:
        torch.ops.aten._cufft_set_plan_cache_max_size(index, _distinct_capacity(1))
        assert flag_gems._cufft_set_plan_cache_max_size(index, True) is None
        assert torch.ops.aten._cufft_get_plan_cache_max_size(index) == 1


# A 0-dim tensor is unpacked as an int by the native argument parser, while a
# multi-element or negative payload keeps its native error.  ``expected=None``
# marks the payloads the parser rejects.
_TENSOR_MAX_SIZE = 1024
_TENSOR_ARG_ROWS = [
    ("max_size", torch.int64, 1024, 1024),
    # A non-integral tensor is truncated towards zero.
    ("max_size", torch.float32, 1024.5, 1024),
    ("max_size", torch.bool, True, 1),
    ("device_index", torch.int64, 0, _TENSOR_MAX_SIZE),
    ("max_size", torch.int64, (1024, 1024), None),
    ("max_size", torch.int64, -1, None),
]


@pytest.mark.cufft_set_plan_cache_max_size
@pytest.mark.parametrize("argument,dtype,payload,expected", _TENSOR_ARG_ROWS)
def test_tensor_argument_follows_native_conversion(argument, dtype, payload, expected):
    # The tensor is built from parametrized metadata, so collection allocates
    # nothing and the bad argument is the one this row describes.
    value = torch.tensor(payload, dtype=dtype, device=flag_gems.device)
    kwargs = {"device_index": _PLAN_CACHE_INDEX, "max_size": _TENSOR_MAX_SIZE}
    kwargs[argument] = value

    if expected is None:
        with pytest.raises(RuntimeError):
            flag_gems._cufft_set_plan_cache_max_size(**kwargs)
        return

    with _plan_cache_capacity() as index:
        torch.ops.aten._cufft_set_plan_cache_max_size(
            index, _distinct_capacity(expected)
        )
        assert flag_gems._cufft_set_plan_cache_max_size(**kwargs) is None
        assert torch.ops.aten._cufft_get_plan_cache_max_size(index) == expected


_INVALID_SIZES = [
    -1,
    -2,
    -1024,
    -(1 << 31),
    -(1 << 63),
    1 << 63,
    1 << 64,
    1.5,
    -0.5,
    float("nan"),
    float("inf"),
    float("-inf"),
    "1024",
    None,
]


@pytest.mark.cufft_set_plan_cache_max_size
@pytest.mark.parametrize("max_size", _INVALID_SIZES)
def test_reject_invalid_max_size(max_size):
    with pytest.raises(RuntimeError):
        flag_gems._cufft_set_plan_cache_max_size(_PLAN_CACHE_INDEX, max_size)


# With one visible device the driver call itself fails and ATen reports
# RuntimeError.  Negative indices and >= 128 raise UnicodeDecodeError while
# decoding the driver's non-UTF-8 error byte, which is neither RuntimeError nor
# TypeError, so they are excluded from this negative set.
_OUT_OF_RANGE_DEVICE_INDICES = [1, 2, 3, 5, 100, 127]


@pytest.mark.cufft_set_plan_cache_max_size
@pytest.mark.parametrize("device_index", _OUT_OF_RANGE_DEVICE_INDICES)
def test_reject_unusable_device_index(device_index):
    with pytest.raises(RuntimeError):
        flag_gems._cufft_set_plan_cache_max_size(device_index, 4096)


_NON_INTEGER_DEVICE_INDICES = [1.5, "0", None]


@pytest.mark.cufft_set_plan_cache_max_size
@pytest.mark.parametrize("device_index", _NON_INTEGER_DEVICE_INDICES)
def test_reject_non_integer_device_index(device_index):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._cufft_set_plan_cache_max_size(device_index, 4096)


# Neither schema parameter has a default, so there is no native-valid call that
# omits one.  A wrong positional count is rejected by schema dispatch
# (RuntimeError), while missing or duplicated keyword arguments are rejected by
# Python argument binding (TypeError).
_ARITY_MISMATCH_CALLS = [
    ((), {}),
    ((_PLAN_CACHE_INDEX,), {}),
    ((_PLAN_CACHE_INDEX, 4096, 4096), {}),
    ((), {"max_size": 4096}),
    ((), {"device_index": _PLAN_CACHE_INDEX}),
    ((_PLAN_CACHE_INDEX,), {"device_index": _PLAN_CACHE_INDEX}),
]


@pytest.mark.cufft_set_plan_cache_max_size
@pytest.mark.parametrize("args,kwargs", _ARITY_MISMATCH_CALLS)
def test_reject_arity_mismatch(args, kwargs):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._cufft_set_plan_cache_max_size(*args, **kwargs)


# Real cuFFT plans populate the same cache through torch's FFT path.  Reference
# and candidate each start from an independently prepared cache, so a candidate
# that only records the number cannot reproduce the native occupancy that
# storing the capacity leaves behind.
_PREPARE_CAPACITY = 4096
_PLAN_LENGTHS = (8, 16, 32, 64)
_EVICTION_ROWS = [(2, 0), (2, 1), (3, 1), (3, 3), (4, 2)]
_EVICTION_CASES = _EVICTION_ROWS


def _run_capacity_scenario(index, plan_count, capacity, set_capacity):
    torch.ops.aten._cufft_clear_plan_cache(index)
    set_capacity(index, _PREPARE_CAPACITY)
    for length in _PLAN_LENGTHS[:plan_count]:
        torch.fft.fft(
            torch.zeros(length, dtype=torch.complex64, device=flag_gems.device)
        )
    occupied = torch.ops.aten._cufft_get_plan_cache_size(index)
    set_capacity(index, capacity)
    return (
        occupied,
        torch.ops.aten._cufft_get_plan_cache_size(index),
        torch.ops.aten._cufft_get_plan_cache_max_size(index),
    )


@pytest.mark.cufft_set_plan_cache_max_size
@pytest.mark.parametrize("plan_count,capacity", _EVICTION_CASES)
def test_set_plan_cache_max_size_affects_live_cache(plan_count, capacity):
    with _plan_cache_capacity() as index:
        reference = _run_capacity_scenario(
            index, plan_count, capacity, torch.ops.aten._cufft_set_plan_cache_max_size
        )
        candidate = _run_capacity_scenario(
            index, plan_count, capacity, flag_gems._cufft_set_plan_cache_max_size
        )

        assert candidate[0] == reference[0], "preparation did not reproduce the state"
        assert candidate[2] == capacity == reference[2]
        assert candidate[1] == reference[1]
