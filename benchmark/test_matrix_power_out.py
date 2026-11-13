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

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

MATRIX_POWER_N = 5


class MatrixPowerBenchmark(base.BlasBenchmark, OperatorBenchmark):
    def supports_cases(self):
        return True

    def set_more_shapes(self):
        return [(1, m, m, m) for m in (256, 512, 1024, 2048)]

    get_input_iter = base.GenericBenchmark.get_input_iter
    get_case_iter = OperatorBenchmark.get_case_iter
    build_inputs = OperatorBenchmark.build_inputs


def _case_fn(shape, dtype):
    del dtype
    m = shape[-1]
    yield base.BenchmarkCasePlan(
        shape={"self": [m, m]},
        params={"n": MATRIX_POWER_N},
        builder_args=(m,),
    )


def _build_matrix(m, dtype, device):
    torch.manual_seed(0)
    return torch.randn((m, m), dtype=dtype, device=device)


def _build_inputs_fn(plan, dtype, device):
    (m,) = plan.builder_args
    matrix = _build_matrix(m, dtype, device)
    out = torch.empty_like(matrix)
    return matrix, MATRIX_POWER_N, {"out": out}


@pytest.mark.matrix_power_out
def test_matrix_power_out():
    MatrixPowerBenchmark(
        op_name="matrix_power_out",
        torch_op=torch.ops.aten.matrix_power.out,
        input_fn=None,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    ).run()
