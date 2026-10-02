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
"""Correctness tests for ``aten::_cufft_clear_plan_cache``.

``aten::_cufft_clear_plan_cache(int device_index) -> ()`` drops every cuFFT plan cached
for one device and returns nothing. The operator has no tensor operand, so the spec's
value-range / shape / dtype grid has nothing to vary here: the device index and the state
of the vendor plan cache - read back with the native plan-cache query operators - are the
dimensions that change its behaviour.
"""

import contextlib

import pytest
import torch

import flag_gems

from . import test_utils as tu

_DEVICE_COUNT = flag_gems.runtime.device.device_count

# Static capability gate, evaluated at import time without calling an operator or
# allocating a tensor: the plan cache is a runtime resource of the CUDA API, so the native
# oracle and the plan-cache query operators exist exactly where the torch runtime is a
# CUDA-API runtime. Those vendors are taken from the runtime's vendor metadata rather than
# from a per-test device string, and the negative tests below are deliberately not gated,
# so a backend without the resource still has to reject invalid arguments.
_CUDA_API_VENDORS = frozenset({"nvidia", "amd", "hygon"})
_CUFFT_PLAN_CACHE_AVAILABLE = flag_gems.vendor_name in _CUDA_API_VENDORS
_SKIP_REASON = (
    "torch.ops.aten._cufft_clear_plan_cache and the plan-cache query operators are "
    "CUDA-API runtime resources and have no kernel on this vendor's backend, so there is "
    "no vendor plan cache to observe"
)

# The device index replaces the spec's shape level: it has to name a real device, and the
# last device is a distinct workload because the plan cache is per device.
_VALID_DEVICE_INDICES = tu.selected_cases(
    list(range(_DEVICE_COUNT)),
    quick=list(range(_DEVICE_COUNT)),
)

# The native oracle narrows an accepted argument to a signed byte before validating it, so
# these wide values alias device 0 instead of raising (measured: -(2**63), -(2**31), -256,
# 256). They are positive coverage: the aliased device's cache has to be cleared for real.
_ALIASED_DEVICE_INDICES = tu.selected_cases(
    [-(2**63), -(2**31), -256, 256],
    quick=[256],
)

# Invalid device indices: below the valid range (-(2**31) - 1 and -1), from the device
# count upwards, and beyond the schema's int64 conversion (2**63). The aliasing values
# above are excluded on purpose.
_OUT_OF_RANGE_INDICES = [
    -(2**31) - 1,
    -1,
    _DEVICE_COUNT,
    _DEVICE_COUNT + 5,
    2**31 - 1,
    2**63,
]

# Values that cannot name a device at all. ``bool`` is absent because True means 1 in the
# schema and is a valid index on a multi-device backend.
_NON_INT_INDICES = ["0", None, 1.5, [0], (0,)]

# Tensor arguments that hold no single device index; the tensors are built inside the test
# so that collecting cases allocates nothing. None of them depends on the device count.
_TENSOR_ARGUMENT_LABELS = ["int_vector", "float_vector", "bool_vector", "empty_int"]

# Signature violations: the argument has no default and the schema accepts exactly one
# positional argument, so the third row is an unknown keyword rather than a defaulted one.
_SIGNATURE_CASES = [
    ((), {}),
    ((0, 0), {}),
    ((), {"device_index": 0, "device": 0}),
]

# A rejected call must raise. The class is not pinned to a single one because a Python
# implementation reports an invalid value as TypeError/ValueError while ATen reports
# RuntimeError, and this build surfaces a rejected index as a malformed byte string (its
# UnicodeDecodeError is a ValueError subclass).
_REJECTIONS = (RuntimeError, TypeError, ValueError)

_PLAN_KINDS = ("c2c", "r2c", "c2r", "nd", "batched", "mixed")
_MIXED_KINDS = ("c2c", "r2c", "nd", "c2r", "batched")

# Cache states to clear: the transform kinds used to create plans, how many transforms were
# requested and how small the configured capacity is (None keeps the device default; 1 and
# 3 are capacities the larger fills cross, so the cache is full and then evicting). cuFFT
# plan creation is vendor-internal and one plan may serve two requests, so only emptiness
# before the call and the net effect after it are asserted instead of an exact plan count.
_FILL_LEVELS = (0, 1, 2, 3, 5)
_CAPACITIES = (None, 1, 3)
_CACHE_STATE_ROWS = tu.selected_cases(
    [
        (kind, fill, capacity)
        for kind in _PLAN_KINDS
        for fill in _FILL_LEVELS
        for capacity in _CAPACITIES
    ],
    quick=[
        # One plan of each kind, the empty cache and the smallest capacity (filled and
        # then evicting) are the cheap distinct branches of the observable effect.
        *[(kind, 1, None) for kind in _PLAN_KINDS],
        ("c2c", 0, None),
        ("mixed", 3, 1),
        ("mixed", 5, 3),
    ],
)


def _torch_device(device_index):
    return torch.device(flag_gems.device, device_index)


def _plan_cache_size(device_index):
    return torch.ops.aten._cufft_get_plan_cache_size(device_index)


def _create_plan(device_index, kind, length):
    """Create one cuFFT plan by running a real transform on ``device_index``."""
    device = _torch_device(device_index)
    if kind == "c2c":
        torch.fft.fft(torch.zeros(length, dtype=torch.complex64, device=device))
    elif kind == "r2c":
        torch.fft.rfft(torch.zeros(length, dtype=torch.float32, device=device))
    elif kind == "c2r":
        source = torch.zeros(length // 2 + 1, dtype=torch.complex64, device=device)
        torch.fft.irfft(source)
    elif kind == "nd":
        torch.fft.fft2(torch.zeros((length, 2), dtype=torch.complex64, device=device))
    elif kind == "batched":
        torch.fft.rfft(torch.zeros((2, length), dtype=torch.float32, device=device))
    else:
        raise ValueError(f"unknown plan kind {kind!r}")


def _populate_plan_cache(device_index, kind, count):
    for offset in range(count):
        variant = _MIXED_KINDS[offset % len(_MIXED_KINDS)] if kind == "mixed" else kind
        _create_plan(device_index, variant, 8 + offset)


@contextlib.contextmanager
def _plan_cache_capacity(device_index, capacity):
    """Temporarily set the plan cache capacity of one device and always restore it."""
    previous = torch.ops.aten._cufft_get_plan_cache_max_size(device_index)
    limit = max(previous, 1) if capacity is None else capacity
    torch.ops.aten._cufft_set_plan_cache_max_size(device_index, limit)
    try:
        yield
    finally:
        torch.ops.aten._cufft_set_plan_cache_max_size(device_index, previous)


def _observe_clear(clear_fn, device_index, kind, fill, capacity, argument=None):
    """Build a cache state on ``device_index``, run ``clear_fn`` and report the outcome.

    ``argument`` is what is passed to ``clear_fn`` - the device index itself unless a test
    exercises another accepted call form - while the cache is always read back on
    ``device_index``. Returns ``(result, size_after, capacity_after, capacity_before)``.
    """
    call_argument = device_index if argument is None else argument
    with _plan_cache_capacity(device_index, capacity):
        capacity_before = torch.ops.aten._cufft_get_plan_cache_max_size(device_index)
        # The native operator establishes the starting state, so a case never inherits a
        # plan cached by an earlier test.
        torch.ops.aten._cufft_clear_plan_cache(device_index)
        _populate_plan_cache(device_index, kind, fill)
        size_before = _plan_cache_size(device_index)
        if fill == 0:
            assert (
                size_before == 0
            ), f"expected an empty cache, got {size_before} plan(s)"
        else:
            assert size_before > 0, f"fill={fill} left no plan in the cache"
        result = clear_fn(call_argument)
        size_after = _plan_cache_size(device_index)
        capacity_after = torch.ops.aten._cufft_get_plan_cache_max_size(device_index)
    return result, size_after, capacity_after, capacity_before


def _assert_cleared(ref, res):
    """The candidate must return nothing, leave an empty cache and keep the capacity."""
    ref_result, ref_size, ref_capacity, capacity_before = ref
    res_result, res_size, res_capacity, _ = res
    assert res_result is ref_result is None, f"the operator returns {res_result!r}"
    assert res_size == ref_size == 0, f"{res_size} plan(s) left in the cache"
    assert (
        res_capacity == ref_capacity == capacity_before
    ), f"capacity is {res_capacity} after the call, expected {capacity_before}"


def _tensor_argument(label):
    """Build the invalid tensor argument named by ``label`` (at test time, so that
    collecting cases allocates nothing).
    """
    if label == "int_vector":
        return torch.tensor([0, 1], dtype=torch.int64, device=flag_gems.device)
    if label == "float_vector":
        return torch.tensor([0.0, 1.0], dtype=torch.float32, device=flag_gems.device)
    if label == "bool_vector":
        return torch.tensor([True, False], device=flag_gems.device)
    if label == "empty_int":
        return torch.tensor([], dtype=torch.int64, device=flag_gems.device)
    raise ValueError(f"unknown tensor argument label {label!r}")


@pytest.mark.cufft_clear_plan_cache
@pytest.mark.skipif(not _CUFFT_PLAN_CACHE_AVAILABLE, reason=_SKIP_REASON)
@pytest.mark.parametrize("kind,fill,capacity", _CACHE_STATE_ROWS)
@pytest.mark.parametrize("device_index", _VALID_DEVICE_INDICES)
def test__cufft_clear_plan_cache(kind, fill, capacity, device_index):
    """Clearing drops every plan cached for the device, keeps the configured capacity and
    returns nothing, for a cache built from each transform kind, fill level and capacity.
    """
    ref = _observe_clear(
        torch.ops.aten._cufft_clear_plan_cache, device_index, kind, fill, capacity
    )
    res = _observe_clear(
        flag_gems._cufft_clear_plan_cache, device_index, kind, fill, capacity
    )
    _assert_cleared(ref, res)


@pytest.mark.cufft_clear_plan_cache
@pytest.mark.skipif(not _CUFFT_PLAN_CACHE_AVAILABLE, reason=_SKIP_REASON)
@pytest.mark.parametrize("device_index", _VALID_DEVICE_INDICES)
def test__cufft_clear_plan_cache_keyword_device_index(device_index):
    """``device_index`` is the schema's parameter name, so the keyword form must clear the
    cache exactly like the positional form.
    """
    ref = _observe_clear(
        lambda index: torch.ops.aten._cufft_clear_plan_cache(device_index=index),
        device_index,
        "r2c",
        2,
        None,
    )
    res = _observe_clear(
        lambda index: flag_gems._cufft_clear_plan_cache(device_index=index),
        device_index,
        "r2c",
        2,
        None,
    )
    _assert_cleared(ref, res)


@pytest.mark.cufft_clear_plan_cache
@pytest.mark.skipif(not _CUFFT_PLAN_CACHE_AVAILABLE, reason=_SKIP_REASON)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("device_index", _VALID_DEVICE_INDICES)
def test__cufft_clear_plan_cache_index_tensor(dtype, device_index):
    """The schema converts a 0-D integer tensor to the ``int`` argument, so that call form
    is valid and has to clear the cache as well.
    """
    argument = torch.tensor(device_index, dtype=dtype, device=flag_gems.device)
    ref = _observe_clear(
        torch.ops.aten._cufft_clear_plan_cache,
        device_index,
        "r2c",
        2,
        None,
        argument=argument,
    )
    res = _observe_clear(
        flag_gems._cufft_clear_plan_cache,
        device_index,
        "r2c",
        2,
        None,
        argument=argument,
    )
    _assert_cleared(ref, res)


@pytest.mark.cufft_clear_plan_cache
@pytest.mark.skipif(not _CUFFT_PLAN_CACHE_AVAILABLE, reason=_SKIP_REASON)
@pytest.mark.parametrize("device_index", _ALIASED_DEVICE_INDICES)
def test__cufft_clear_plan_cache_aliased_device_index(device_index):
    """These wide ints are narrowed to a signed byte before validation, so they alias
    device 0 instead of raising and have to clear that device's cache for real.
    """
    ref = _observe_clear(
        torch.ops.aten._cufft_clear_plan_cache,
        0,
        "batched",
        2,
        None,
        argument=device_index,
    )
    res = _observe_clear(
        flag_gems._cufft_clear_plan_cache,
        0,
        "batched",
        2,
        None,
        argument=device_index,
    )
    _assert_cleared(ref, res)


@pytest.mark.cufft_clear_plan_cache
@pytest.mark.parametrize("args,kwargs", _SIGNATURE_CASES)
def test__cufft_clear_plan_cache_rejects_invalid_signature(args, kwargs):
    """Omitting the index, repeating it or spelling the keyword differently is not a valid
    call of the one-argument schema.
    """
    with pytest.raises(_REJECTIONS):
        flag_gems._cufft_clear_plan_cache(*args, **kwargs)


@pytest.mark.cufft_clear_plan_cache
@pytest.mark.parametrize("device_index", _OUT_OF_RANGE_INDICES)
def test__cufft_clear_plan_cache_rejects_out_of_range_device_index(device_index):
    """An index that names no device has to be rejected; a candidate that unconditionally
    reports success fails here.
    """
    with pytest.raises(_REJECTIONS):
        flag_gems._cufft_clear_plan_cache(device_index)


@pytest.mark.cufft_clear_plan_cache
@pytest.mark.parametrize("device_index", _NON_INT_INDICES)
def test__cufft_clear_plan_cache_rejects_non_int_device_index(device_index):
    """Only an int (or a 0-D integer tensor) names a device, so any other value is an
    invalid argument rather than a default.
    """
    with pytest.raises(_REJECTIONS):
        flag_gems._cufft_clear_plan_cache(device_index)


@pytest.mark.cufft_clear_plan_cache
@pytest.mark.parametrize("label", _TENSOR_ARGUMENT_LABELS)
def test__cufft_clear_plan_cache_rejects_tensor_device_index(label):
    """A tensor that does not hold exactly one integer cannot name a single device index
    and has to be rejected instead of being scanned for a first element.
    """
    argument = _tensor_argument(label)
    with pytest.raises(_REJECTIONS):
        flag_gems._cufft_clear_plan_cache(argument)
