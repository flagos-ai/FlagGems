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

import pytest
import torch

import flag_gems

from . import base, utils
from .generated_operator_utils import OperatorBenchmark

# A ger workload is a pair of 1-D operand lengths (self_len, vec2_len); the
# native operator rejects every non-1-D operand, so the generic multi-dim shape
# set does not apply and the length-1 entries keep the single-row and
# single-column edge cases in the timing set.
GER_SHAPES = [
    (1, 4096),
    (4096, 1),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (8192, 4096),
    (4096, 8192),
    (16384, 1024),
    (1024, 16384),
]

# One dtype list drives both case listing and execution.
BENCH_DTYPES = [torch.float16, torch.float32]
if flag_gems.runtime.device.support_bf16:
    BENCH_DTYPES.append(torch.bfloat16)


def _case_fn(shape, dtype):
    del dtype
    self_len, vec2_len = shape
    yield base.BenchmarkCasePlan(
        shape={"self": self_len, "vec2": vec2_len},
        params={},
        builder_args=(self_len, vec2_len),
    )


def _build_inputs_fn(plan, dtype, device):
    self_len, vec2_len = plan.builder_args
    self_vec = utils.generate_tensor_input((self_len,), dtype, device)
    vec2 = utils.generate_tensor_input((vec2_len,), dtype, device)
    return self_vec, vec2


class GerBenchmark(OperatorBenchmark):
    """Benchmark over rank-1 operand pairs, the only shapes ger accepts."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=GER_SHAPES)


@pytest.mark.ger
def test_ger():
    bench = GerBenchmark(
        op_name="ger",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.ger,
        gems_op=getattr(flag_gems, "ger", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
