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

"""Benchmark for aten::_cufft_set_plan_cache_max_size.

Schema: ``_cufft_set_plan_cache_max_size(int device_index, int max_size) -> ()``.
The operator stores a driver-level cuFFT plan-cache capacity and builds no
tensor, so the workload dimension is the cache state that capacity enforces:
each case is described by integer state descriptors, not by tensor shapes.
"""

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

_OP_NAME = "_cufft_set_plan_cache_max_size"

# The tensor-free operator still needs a dtype to drive the case list; it stays
# metadata only and no tensor of this dtype is ever built.
_BENCH_DTYPES = [torch.float32]

# Integer state descriptors handed to the shared shape machinery: one capacity
# per workload, so a user shape file can still select the capacities to time.
_STATE_DESCRIPTORS = [(0,), (1,), (4096,), (1 << 20,)]

# Each case starts from a populated cache, so storing the capacity has real
# contents to evict and the case metadata can state the occupancy left behind.
_CACHED_PLANS = 3
_PLAN_LENGTHS = (8, 16, 32)


def _plan_cache_index():
    """Ordinal of the device whose plan cache flag_gems.device refers to."""
    device = torch.device(flag_gems.device)
    return 0 if device.index is None else device.index


_DEVICE_INDEX = _plan_cache_index()


def _case_fn(descriptor, dtype):
    del dtype
    capacity = descriptor[0]
    yield base.BenchmarkCasePlan(
        shape={
            "plan_cache_capacity": [capacity],
            "cached_plans": _CACHED_PLANS,
            "retained_plans": min(capacity, _CACHED_PLANS),
        },
        params={"device_index": _DEVICE_INDEX, "max_size": capacity},
        builder_args=(capacity,),
    )


def _build_inputs_fn(plan, dtype, device):
    # The operator takes plain ints; filling the cache with real plans is input
    # preparation and stays outside the timed call.
    del dtype
    torch.ops.aten._cufft_clear_plan_cache(_DEVICE_INDEX)
    torch.ops.aten._cufft_set_plan_cache_max_size(_DEVICE_INDEX, _CACHED_PLANS)
    device = torch.device(flag_gems.device, _DEVICE_INDEX)
    for length in _PLAN_LENGTHS:
        torch.fft.fft(torch.zeros(length, dtype=torch.complex64, device=device))
    return int(plan.params["device_index"]), {"max_size": int(plan.params["max_size"])}


class PlanCacheMaxSizeBenchmark(OperatorBenchmark):
    """Tensor-free workload list described by integer cache-state descriptors."""

    DEFAULT_SHAPES = _STATE_DESCRIPTORS
    DEFAULT_SHAPE_DESC = "plan cache capacity"

    def set_shapes(self, shape_file_path=None, *, default_shapes=_STATE_DESCRIPTORS):
        # Shared shape-file handling: an entry for this operator still wins, the
        # integer descriptors are only the fallback.
        super().set_shapes(shape_file_path, default_shapes=default_shapes)

    def set_more_shapes(self):
        return []


@pytest.mark.cufft_set_plan_cache_max_size
def test_cufft_set_plan_cache_max_size():
    # A metadata-only listing or query must not touch the runtime, so the
    # ambient capacity is read and restored only for an actual run.
    running = not (
        getattr(base.Config, "list_cases", False)
        or getattr(base.Config, "query", False)
    )
    previous = (
        torch.ops.aten._cufft_get_plan_cache_max_size(_DEVICE_INDEX)
        if running
        else None
    )
    try:
        bench = PlanCacheMaxSizeBenchmark(
            op_name=_OP_NAME,
            case_fn=_case_fn,
            build_inputs_fn=_build_inputs_fn,
            torch_op=torch.ops.aten._cufft_set_plan_cache_max_size,
            gems_op=getattr(flag_gems, _OP_NAME, None),
            dtypes=_BENCH_DTYPES,
        )
        bench.run()
    finally:
        if previous is not None:
            torch.ops.aten._cufft_set_plan_cache_max_size(_DEVICE_INDEX, previous)
