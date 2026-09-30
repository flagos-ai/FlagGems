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

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn

from . import base
from .generated_operator_utils import OperatorBenchmark

# aten::_cufft_get_plan_cache_size consumes a device index and reports process
# cache state, so the shape rows are integer cache-state descriptors rather than
# tensor shapes, and one dtype sentinel stands in for a dtype that never reaches
# the operator: no tensor shape cross-product and no per-dtype replicate of an
# identical workload.
_NATIVE_AVAILABLE = flag_gems.vendor_name == "nvidia"

_LENGTH_STEP = 8
_SATURATION_CAPACITY = 8
_STATE_DESCRIPTORS = [0, 1, 8, 32, (2 * _SATURATION_CAPACITY, _SATURATION_CAPACITY)]
_CALL_FORMS = ("positional", "keyword")

# Capacities changed while preparing cases are restored when the run ends. The
# dict stays empty during case listing, so listing performs no native call.
_SAVED_CAPACITIES = {}


def _warm_lengths(warm_count):
    return [_LENGTH_STEP * (i + 1) for i in range(warm_count)]


def _checked_descriptor(descriptor):
    # A bare int is a warm count under the ambient capacity; a pair also states an
    # explicit capacity.
    if isinstance(descriptor, int) and not isinstance(descriptor, bool):
        warm_count, capacity = descriptor, None
    else:
        try:
            warm_count, capacity = descriptor
        except (TypeError, ValueError):
            raise ValueError(
                "a cache-state descriptor must be an int or a (warm_count, capacity) pair"
            ) from None
    if (
        isinstance(warm_count, bool)
        or not isinstance(warm_count, int)
        or warm_count < 0
    ):
        raise ValueError("a cache-state descriptor needs a non-negative warm count")
    if capacity is not None and (
        isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0
    ):
        raise ValueError("a cache capacity must be a non-negative int")
    return warm_count, capacity


def _prepare_cache(device_index, warm_count, capacity):
    # State is built here, outside the timed region and on the exact device that
    # will be queried. The original capacity is remembered the first time a device
    # is touched so _restore_capacities can put it back.
    saved = torch.ops.aten._cufft_get_plan_cache_max_size(device_index)
    _SAVED_CAPACITIES.setdefault(device_index, saved)
    limit = capacity if capacity is not None else max(saved, warm_count)
    torch.ops.aten._cufft_clear_plan_cache(device_index)
    torch.ops.aten._cufft_set_plan_cache_max_size(device_index, limit)
    device = torch.device(flag_gems.device, device_index)
    for length in _warm_lengths(warm_count):
        torch.ops.aten.fft_rfft(torch.ones(length, dtype=torch.float32, device=device))


def _restore_capacities():
    # A no-op when nothing was prepared (case listing), so listing stays
    # tensor-free and free of native calls.
    for device_index, saved in _SAVED_CAPACITIES.items():
        torch.ops.aten._cufft_clear_plan_cache(device_index)
        torch.ops.aten._cufft_set_plan_cache_max_size(device_index, saved)
    _SAVED_CAPACITIES.clear()


class CufftPlanCacheSizeBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_STATE_DESCRIPTORS)

    def set_more_shapes(self):
        # The extra generic tensor shapes describe element counts this operator
        # never consumes, so none is contributed.
        return []


def _case_fn(descriptor, dtype):
    del dtype
    warm_count, capacity = _checked_descriptor(descriptor)
    for device_index in range(torch_device_fn.device_count()):
        for call_form in _CALL_FORMS:
            yield base.BenchmarkCasePlan(
                shape={"warm_plans": warm_count, "cache_capacity": capacity},
                params={"device_index": device_index, "call_form": call_form},
                builder_args=(device_index, call_form, warm_count, capacity),
            )


def _build_inputs_fn(plan, dtype, device):
    del dtype, device
    device_index, call_form, warm_count, capacity = plan.builder_args
    _prepare_cache(device_index, warm_count, capacity)
    if call_form == "keyword":
        # unpack_to_args_kwargs turns a dict element into call kwargs.
        return ({"device_index": device_index},)
    return (device_index, {})


@pytest.mark.cufft_get_plan_cache_size
@pytest.mark.skipif(
    not _NATIVE_AVAILABLE,
    reason=(
        "cuFFT plan cache is an NVIDIA resource; other backends must expose this "
        "operator unsupported or backed by their own runtime cache"
    ),
)
def test_cufft_get_plan_cache_size():
    bench = CufftPlanCacheSizeBenchmark(
        op_name="_cufft_get_plan_cache_size",
        torch_op=torch.ops.aten._cufft_get_plan_cache_size,
        gems_op=getattr(flag_gems, "_cufft_get_plan_cache_size", None),
        dtypes=[torch.float32],
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
    )
    try:
        bench.run()
    finally:
        _restore_capacities()
