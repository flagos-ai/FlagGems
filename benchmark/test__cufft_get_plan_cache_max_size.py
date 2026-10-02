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

"""Benchmark for aten::_cufft_get_plan_cache_max_size."""

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

_OP_NAME = "_cufft_get_plan_cache_max_size"
_NATIVE_GET = torch.ops.aten._cufft_get_plan_cache_max_size
_NATIVE_SET = torch.ops.aten._cufft_set_plan_cache_max_size
_SAVED_CAPACITIES = {}

# Rows are integer state descriptors (device_index, plan-cache capacity, call
# form); the operator has no tensor operand, so no tensor shape is involved.
_STATE_ROWS = [
    (index, capacity, form)
    for index in range(flag_gems.runtime.device.device_count)
    for capacity, form in [
        (0, "positional"),
        (1, "keyword"),
        (7, "positional"),
        (4096, "keyword"),
    ]
]


# The harness dtype dimension is inert here; one int64 row keeps the standard
# collection and reporting path without an empty float cross product.
_BENCH_DTYPES = [torch.int64]


def _case_fn(state_row, dtype):
    """Describe one plan-cache state without touching native state."""
    del dtype
    device_index, capacity, call_form = state_row
    yield base.BenchmarkCasePlan(
        shape={},
        params={
            "device_index": device_index,
            "plan_cache_capacity": capacity,
            "call_form": call_form,
        },
        builder_args=(device_index, capacity, call_form),
    )


def _build_inputs_fn(plan, dtype, device):
    """Apply the case capacity and return flat positional arguments."""
    del dtype, device
    device_index, capacity, call_form = plan.builder_args
    if device_index not in _SAVED_CAPACITIES:
        _SAVED_CAPACITIES[device_index] = _NATIVE_GET(device_index)
    _NATIVE_SET(device_index, capacity)
    if call_form == "keyword":
        # A trailing dict is turned into keyword arguments by the harness.
        return ({"device_index": device_index},)
    return (device_index,)


def _restore_plan_cache_capacities():
    # Saved lazily by the builder, so listing touches no native state and custom
    # shape-file device indices are restored too.
    try:
        for index, capacity in _SAVED_CAPACITIES.items():
            _NATIVE_SET(index, capacity)
    finally:
        _SAVED_CAPACITIES.clear()


class CufftGetPlanCacheMaxSizeBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_STATE_ROWS)

    def set_more_shapes(self):
        return []


@pytest.mark.cufft_get_plan_cache_max_size
def test__cufft_get_plan_cache_max_size():
    bench = CufftGetPlanCacheMaxSizeBenchmark(
        op_name=_OP_NAME,
        torch_op=torch.ops.aten._cufft_get_plan_cache_max_size,
        dtypes=_BENCH_DTYPES,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        gems_op=getattr(flag_gems, _OP_NAME, None),
    )
    try:
        bench.run()
    finally:
        _restore_plan_cache_capacities()
