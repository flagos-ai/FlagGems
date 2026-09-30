# Copyright 2026, The FlagOS Contributors.
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

import contextlib

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn

from . import test_utils as tu

# aten::_cufft_get_plan_cache_size(device_index) -> int reports how many plans the
# process-wide cuFFT plan cache holds for one device. It takes no tensor, so the
# value-range / shape / broadcast / backward grids do not apply; the dimensions
# exercised here are cache state, call form, index aliasing and the invalid-input
# contract. The cache is process-wide vendor state, so every case builds its own
# state and restores the saved capacity afterwards.
#
# Static gate: this is an NVIDIA/cuFFT resource. Other backends must report it
# unsupported or back it with their own runtime cache rather than simulate a
# vendor resource.
_NATIVE_AVAILABLE = flag_gems.vendor_name == "nvidia"

pytestmark = pytest.mark.skipif(
    not _NATIVE_AVAILABLE,
    reason=(
        "cuFFT plan cache is an NVIDIA resource; other backends must expose this "
        "operator unsupported or backed by their own runtime cache"
    ),
)

# The real device count, so a host without devices collects an empty
# parametrization instead of a fabricated index.
_DEVICE_INDICES = list(range(torch_device_fn.device_count()))

_LENGTH_STEP = 8
_MAX_WARM = 40
_SATURATION_CAPACITY = 8


def _warm_lengths(warm_count):
    # Distinct transform lengths, one cuFFT plan each, generated for the exact
    # requested count so a request is never silently capped.
    return [_LENGTH_STEP * (i + 1) for i in range(warm_count)]


# (warm plan count, cache capacity): the empty cache, a single cached plan,
# growing counts, and a cache smaller than the requested plan count, where
# further insertions evict and the reported count pins to the capacity.
_STATE_ROWS = (
    [(0, None), (1, None)]
    + [(count, None) for count in range(2, _MAX_WARM + 1)]
    + [(2 * _SATURATION_CAPACITY, _SATURATION_CAPACITY)]
)
_CACHE_STATE_ROWS = tu.selected_cases(
    _STATE_ROWS,
    quick=[
        (0, None),
        (1, None),
        (4, None),
        (2 * _SATURATION_CAPACITY, _SATURATION_CAPACITY),
    ],
)

_CALL_FORMS = ("positional", "keyword")

# The native bounds check reads only the low byte of device_index: probed on a
# one-device host, 1 reports byte 0x01 and -1 reports 0xff, while 256, 1024,
# INT_MIN and INT_MIN64 all report device 0's count. These rows require the
# candidate to follow that observed aliasing.
_ALIAS_INDICES = tu.selected_cases(
    [base + index for index in _DEVICE_INDICES for base in (256, 1024)]
    + ([-(2**31), -(2**63)] if _DEVICE_INDICES else []),
    quick=([256, 1024, -(2**31), -(2**63)] if _DEVICE_INDICES else []),
)

# Out-of-range indices are exactly the candidates whose low byte names no
# present device, which is what the native check rejects.
_INVALID_INDEX_CANDIDATES = [
    1,
    2,
    3,
    7,
    63,
    127,
    128,
    129,
    200,
    254,
    255,
    256,
    512,
    1024,
    257,
    513,
    1025,
    2**16 + 255,
    2**31 - 1,
    2**63 - 1,
    -1,
    -2,
    -3,
    -100,
    -255,
    -256,
    -(2**31),
    -(2**63),
    -(2**63) + 1,
]

_INVALID_INDICES = [
    index
    for index in _INVALID_INDEX_CANDIDATES
    if not 0 <= (index & 0xFF) < len(_DEVICE_INDICES)
]

# The schema parameter is int; these values have the wrong type for it.
_INVALID_ARGUMENTS = [0.0, 1.5, "0", "", None, [0], (0,), {0: 0}, complex(0, 0)]


@contextlib.contextmanager
def _isolated_plan_cache(device_index, warm_count, capacity=None):
    # Save the cache, build only the requested state, and restore the exact saved
    # capacity even when the case fails. When no capacity is requested the limit
    # is raised to hold the requested plans, because the ambient capacity can be
    # 0 (caching disabled), which would make a warm row depend on the environment.
    saved_capacity = torch.ops.aten._cufft_get_plan_cache_max_size(device_index)
    limit = capacity if capacity is not None else max(saved_capacity, warm_count)
    try:
        torch.ops.aten._cufft_clear_plan_cache(device_index)
        torch.ops.aten._cufft_set_plan_cache_max_size(device_index, limit)
        yield
    finally:
        torch.ops.aten._cufft_clear_plan_cache(device_index)
        torch.ops.aten._cufft_set_plan_cache_max_size(device_index, saved_capacity)


def _warm_plan_cache(warm_count, device_index):
    # Warm the exact device whose cache is queried, not a default device.
    device = torch.device(flag_gems.device, device_index)
    for length in _warm_lengths(warm_count):
        torch.ops.aten.fft_rfft(torch.ones(length, dtype=torch.float32, device=device))


@pytest.mark.cufft_get_plan_cache_size
@pytest.mark.parametrize("device_index", _DEVICE_INDICES)
@pytest.mark.parametrize("call_form", _CALL_FORMS)
@pytest.mark.parametrize("warm_count,capacity", _CACHE_STATE_ROWS)
def test_cufft_plan_cache_size_matches_native(
    warm_count, capacity, call_form, device_index
):
    with _isolated_plan_cache(device_index, warm_count, capacity):
        _warm_plan_cache(warm_count, device_index)

        ref = torch.ops.aten._cufft_get_plan_cache_size(device_index)
        if call_form == "keyword":
            res = flag_gems._cufft_get_plan_cache_size(device_index=device_index)
        else:
            res = flag_gems._cufft_get_plan_cache_size(device_index)

    # Pin the workload to the described state so a mismatched cache size cannot
    # make this pass, then compare the candidate against the native count.
    expected = warm_count if capacity is None else min(warm_count, capacity)
    assert ref == expected, f"native cache size {ref} != {expected}"
    assert type(res) is int
    assert res == ref


@pytest.mark.cufft_get_plan_cache_size
@pytest.mark.parametrize("device_index", _DEVICE_INDICES)
def test_cufft_plan_cache_size_query_leaves_the_cache_unchanged(device_index):
    with _isolated_plan_cache(device_index, 5):
        _warm_plan_cache(5, device_index)

        before = torch.ops.aten._cufft_get_plan_cache_size(device_index)
        res = flag_gems._cufft_get_plan_cache_size(device_index)
        after = torch.ops.aten._cufft_get_plan_cache_size(device_index)

    assert before == 5
    assert res == before
    # The candidate runs between the two native reads, so an unchanged count
    # shows the query neither added nor dropped plans.
    assert after == before


@pytest.mark.cufft_get_plan_cache_size
@pytest.mark.parametrize("alias_index", _ALIAS_INDICES)
def test_cufft_plan_cache_size_resolves_the_low_byte_alias(alias_index):
    device_index = alias_index & 0xFF
    with _isolated_plan_cache(device_index, 3):
        _warm_plan_cache(3, device_index)

        ref = torch.ops.aten._cufft_get_plan_cache_size(device_index)
        res_alias = flag_gems._cufft_get_plan_cache_size(alias_index)
        res = flag_gems._cufft_get_plan_cache_size(device_index)

    assert ref == 3
    assert res_alias == ref
    assert res == ref


@pytest.mark.cufft_get_plan_cache_size
@pytest.mark.parametrize("invalid_index", _INVALID_INDICES)
def test_cufft_plan_cache_size_rejects_out_of_range_index(invalid_index):
    # torch surfaces the malformed native range message as UnicodeDecodeError, a
    # ValueError, when the offending byte is >= 0x80.
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems._cufft_get_plan_cache_size(invalid_index)


@pytest.mark.cufft_get_plan_cache_size
@pytest.mark.parametrize("invalid_argument", _INVALID_ARGUMENTS)
def test_cufft_plan_cache_size_rejects_non_integer_argument(invalid_argument):
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems._cufft_get_plan_cache_size(invalid_argument)


@pytest.mark.cufft_get_plan_cache_size
def test_cufft_plan_cache_size_requires_the_device_index():
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems._cufft_get_plan_cache_size()


@pytest.mark.cufft_get_plan_cache_size
@pytest.mark.parametrize("device_index", _DEVICE_INDICES)
def test_cufft_plan_cache_size_rejects_extra_arguments(device_index):
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems._cufft_get_plan_cache_size(device_index, device_index)
