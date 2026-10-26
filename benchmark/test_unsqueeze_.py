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

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark


def _case_fn(shape, dtype):
    del dtype
    shape = tuple(shape)
    dim = 0 if len(shape) <= 1 else 1
    result_shape = list(shape)
    result_shape.insert(dim, 1)
    yield base.BenchmarkCasePlan(
        shape={"self": list(shape), "result": result_shape},
        params={"dim": dim},
        builder_args=(shape, dim),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, dim = plan.builder_args
    return torch.empty(shape, dtype=dtype, device=device), dim


@pytest.mark.unsqueeze_
def test_unsqueeze_():
    OperatorBenchmark(
        op_name="unsqueeze_",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.unsqueeze_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    ).run()
