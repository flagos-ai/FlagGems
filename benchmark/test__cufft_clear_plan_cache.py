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
"""Benchmark for ``aten::_cufft_clear_plan_cache``.

The operator takes no tensor operand: it clears the process-wide vendor plan cache of one
device and returns nothing, so a case measures that vendor call plus its dispatch. The
device index is the workload axis because the cache is per device, and the cache is filled
before the timed call by the input builder, which runs outside the timing loop.
"""

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

# The plan cache is per device, so every device index is its own workload. This is metadata
# only: nothing is allocated and no operator runs to build or list a case.
_DEVICE_COUNT = flag_gems.runtime.device.device_count
_DEFAULT_DEVICE_INDICES = list(range(_DEVICE_COUNT))
_SAVED_CAPACITIES = {}

# Static vendor gate read from the runtime's vendor metadata: the plan cache is a runtime
# resource of the CUDA API and the native operator has no kernel on another vendor's
# backend, so there is no reference latency to compare against there.
_CUDA_API_VENDORS = frozenset({"nvidia", "amd", "hygon"})
pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name not in _CUDA_API_VENDORS,
    reason=(
        "torch.ops.aten._cufft_clear_plan_cache clears the plan cache of the CUDA API "
        "and has no kernel elsewhere, so no reference latency can be measured"
    ),
)

# The plans created before a measured call, as (transform kind, length): the operator only
# does work against a populated cache. cuFFT plan creation is vendor-internal, so no plan
# count is asserted; the state only has to leave the cache non-empty.
_POPULATED_PLANS = (("r2c", 8), ("c2c", 8), ("batched", 9))


def _device_index(descriptor):
    """Read one case row as a device index.

    The operator has no tensor operand, so a shape-file entry names the device whose plan
    cache is measured rather than a tensor shape. A bare integer and a one-element list are
    accepted, and anything else is rejected instead of being reinterpreted.
    """
    extents = descriptor if isinstance(descriptor, (list, tuple)) else (descriptor,)
    if len(extents) != 1:
        raise ValueError(f"expected a single device index, got {descriptor!r}")
    (index,) = extents
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"invalid device index {index!r}")
    return index


def _case_fn(shape, dtype):
    # dtype is inert: the operator has no tensor operand, so it cannot change a case.
    del dtype
    index = _device_index(shape)
    yield base.BenchmarkCasePlan(
        shape={"device_index": index},
        params={"device_index": index, "plans": len(_POPULATED_PLANS)},
        builder_args=(index,),
    )


def _build_inputs_fn(plan, dtype, device):
    """Fill the device's plan cache and return the schema's single positional argument."""
    del dtype
    (index,) = plan.builder_args
    target = torch.device(device, index)
    if index not in _SAVED_CAPACITIES:
        _SAVED_CAPACITIES[index] = torch.ops.aten._cufft_get_plan_cache_max_size(index)
    torch.ops.aten._cufft_set_plan_cache_max_size(index, len(_POPULATED_PLANS))
    torch.ops.aten._cufft_clear_plan_cache(index)
    for kind, length in _POPULATED_PLANS:
        if kind == "c2c":
            torch.fft.fft(torch.zeros(length, dtype=torch.complex64, device=target))
        elif kind == "batched":
            torch.fft.rfft(torch.zeros((2, length), dtype=torch.float32, device=target))
        else:
            torch.fft.rfft(torch.zeros(length, dtype=torch.float32, device=target))
    # The state lives in the vendor runtime rather than in an argument, so it is rebuilt
    # here, once per case, before the timing loop; the operator then empties that cache.
    return (index,)


class CufftClearPlanCacheBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark for an operator without a tensor operand."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(
            shape_file_path, default_shapes=list(_DEFAULT_DEVICE_INDICES)
        )
        # The stored rows feed the inherited case iteration, so a requested shape-file
        # entry is normalized here as well.
        self.shapes = [_device_index(row) for row in self.shapes]

    def set_more_shapes(self):
        return []


@pytest.mark.cufft_clear_plan_cache
def test__cufft_clear_plan_cache():
    bench = CufftClearPlanCacheBenchmark(
        op_name="_cufft_clear_plan_cache",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._cufft_clear_plan_cache,
        gems_op=getattr(flag_gems, "_cufft_clear_plan_cache", None),
        # No tensor operand: one inert dtype avoids duplicate integer-only cases.
        dtypes=[torch.float32],
    )
    try:
        bench.run()
    finally:
        try:
            for index, capacity in _SAVED_CAPACITIES.items():
                torch.ops.aten._cufft_clear_plan_cache(index)
                torch.ops.aten._cufft_set_plan_cache_max_size(index, capacity)
        finally:
            _SAVED_CAPACITIES.clear()
