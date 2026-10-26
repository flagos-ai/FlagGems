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

class BlasCaseBenchmark(base.BlasBenchmark, OperatorBenchmark):
    def supports_cases(self):
        return True

    get_case_iter = OperatorBenchmark.get_case_iter
    build_inputs = OperatorBenchmark.build_inputs


def _case_fn(shape, dtype):
    del dtype
    b, m, n, k = tuple(shape)
    for column_major in (False, True):
        yield base.BenchmarkCasePlan(
            shape={"self": [m, n], "mat1": [m, k], "mat2": [k, n]},
            params={"column_major": column_major},
            builder_args=(b, m, n, k, column_major),
        )


def _build_inputs_fn(plan, dtype, device):
    b, m, n, k, column_major = plan.builder_args
    torch.manual_seed(0)
    mat1 = torch.randn((m, k), dtype=dtype, device=device)
    if column_major:
        mat2 = torch.randn((n, k), dtype=dtype, device=device).t()
    else:
        mat2 = torch.randn((k, n), dtype=dtype, device=device)
    self = torch.randn((n,) if False else (m, n), dtype=dtype, device=device)
    return self, mat1, mat2


@pytest.mark.addmm_
def test_addmm_():
    BlasCaseBenchmark(
        op_name="addmm_",
        torch_op=lambda self, mat1, mat2: self.addmm_(mat1, mat2),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    ).run()
