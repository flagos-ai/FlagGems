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

"""Benchmark for ``aten::_nnpack_available``.

The operator takes no operands (``() -> bool``), so a benchmark case carries no
tensor geometry and no dtype: the plan below describes an operand-free call and
keeps ``builder_args`` empty, which makes ``--list-cases`` tensor-free and lets
``--case-id`` replay the same call that normal execution builds.
"""

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

# No operands are involved, so no shape is measured. This single sentinel only
# drives the framework's shape loop; the emitted case metadata stays shape-free.
_NO_OPERAND_SHAPES = [()]

# A nullary query has no dtype dimension; one placeholder dtype keeps the
# framework's per-dtype case loop non-empty without pretending to sweep a type.
_PLACEHOLDER_DTYPE = [torch.float32]


def _case_fn(shape, dtype):
    del shape, dtype
    yield base.BenchmarkCasePlan(shape={}, params={}, builder_args=())


def _build_inputs_fn(plan, dtype, device):
    del plan, dtype, device
    # An empty argument list unpacks to op(): the operator takes no operands.
    return ()


class NnpackAvailableBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_NO_OPERAND_SHAPES)

    def set_more_shapes(self):
        # Without operands there is no additional geometry for comprehensive mode.
        return []


@pytest.mark.nnpack_available
def test__nnpack_available():
    bench = NnpackAvailableBenchmark(
        op_name="_nnpack_available",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._nnpack_available,
        gems_op=getattr(flag_gems, "_nnpack_available", None),
        dtypes=_PLACEHOLDER_DTYPE,
    )
    bench.run()
