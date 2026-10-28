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
from .test__addmm_activation import _B03H800ReferenceMixin


class BlasCaseBenchmark(_B03H800ReferenceMixin, base.BlasBenchmark, OperatorBenchmark):
    __init__ = OperatorBenchmark.__init__
    get_input_iter = OperatorBenchmark.get_input_iter

    def supports_cases(self):
        return True

    get_case_iter = OperatorBenchmark.get_case_iter
    build_inputs = OperatorBenchmark.build_inputs


def _case_fn(shape, dtype):
    del dtype
    b, m, n, k = tuple(shape)
    for use_gelu in (False, True):
        for column_major in (False, True):
            yield base.BenchmarkCasePlan(
                shape={"self": [n], "mat1": [m, k], "mat2": [k, n]},
                params={"column_major": column_major, "use_gelu": use_gelu},
                builder_args=(b, m, n, k, column_major, use_gelu),
            )


def _build_inputs_fn(plan, dtype, device):
    b, m, n, k, column_major, use_gelu = plan.builder_args
    torch.manual_seed(0)
    mat1 = torch.randn((m, k), dtype=dtype, device=device)
    if column_major:
        mat2 = torch.randn((n, k), dtype=dtype, device=device).t()
    else:
        mat2 = torch.randn((k, n), dtype=dtype, device=device)
    self = torch.randn((n,) if True else (m, n), dtype=dtype, device=device)
    return self, mat1, mat2, {"use_gelu": use_gelu}


@pytest.mark.addmm_activation
@pytest.mark.underscore_addmm_activation
def test_addmm_activation():
    BlasCaseBenchmark(
        op_name="_addmm_activation",
        torch_op=torch._addmm_activation,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
