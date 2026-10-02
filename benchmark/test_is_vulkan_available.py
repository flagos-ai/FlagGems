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

"""Benchmark for ``aten::is_vulkan_available``.

The operator is nullary, so it has no tensor input, no shape and no dtype
dimension: the single placeholder dtype row exists only because the driver keys
its case list by dtype, and the tensor-free workload measures the host query
itself. ``torch_op`` is the ATen reference used as the perf baseline and
``gems_op`` is the injected FlagGems candidate; both are called as ``op()``.
"""

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark


def _case_fn(shape, dtype):
    del shape, dtype
    # A nullary operator has no shape-dependent work, so the reported case
    # carries no invented input description.
    yield base.BenchmarkCasePlan(shape={}, params={}, builder_args=())


def _build_inputs_fn(plan, dtype, device):
    del plan, dtype, device
    # The single empty mapping is the trailing kwargs dict of the flat input
    # list: it unpacks to zero positional arguments and zero keywords, so the
    # operator is invoked exactly as op().
    return ({},)


class IsVulkanAvailableBenchmark(OperatorBenchmark):
    """Nullary query: one tensor-free, shape-free workload.

    Shapes cannot parameterize this operator, so the driver default is a single
    empty shape while a caller-supplied --shape_file is still read by the base
    implementation. No additional COMPREHENSIVE shapes are contributed, so the
    core and comprehensive listings stay identical.
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=[()])

    def set_more_shapes(self):
        return []


@pytest.mark.is_vulkan_available
def test_is_vulkan_available():
    bench = IsVulkanAvailableBenchmark(
        op_name="is_vulkan_available",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.is_vulkan_available,
        gems_op=getattr(flag_gems, "is_vulkan_available", None),
        dtypes=[torch.float32],
    )
    bench.run()
