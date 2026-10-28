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
    for column_major in (False, True):
        yield base.BenchmarkCasePlan(
            shape={"self": [m, n], "batch1": [b, m, k], "batch2": [b, k, n]},
            params={"column_major": column_major},
            builder_args=(b, m, n, k, column_major),
        )


def _build_inputs_fn(plan, dtype, device):
    b, m, n, k, column_major = plan.builder_args
    torch.manual_seed(0)
    batch1 = torch.randn((b, m, k), dtype=dtype, device=device)
    if column_major:
        batch2 = torch.randn((b, n, k), dtype=dtype, device=device).transpose(1, 2)
    else:
        batch2 = torch.randn((b, k, n), dtype=dtype, device=device)
    self = torch.randn((m, n), dtype=dtype, device=device)
    return self, batch1, batch2


@pytest.mark.addbmm
def test_addbmm():
    BlasCaseBenchmark(
        op_name="addbmm",
        torch_op=torch.addbmm,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
