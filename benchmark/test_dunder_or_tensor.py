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


def _case_fn(shape, dtype):
    del dtype
    shape = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": list(shape), "other": list(shape)},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    high = 2 if dtype == torch.bool else 32767
    lhs = torch.randint(0, high, shape, dtype=dtype, device=device)
    rhs = torch.randint(0, high, shape, dtype=dtype, device=device)
    return lhs, rhs


@pytest.mark.dunder_or_tensor
@pytest.mark.bitwise_or_tensor
def test_dunder_or_tensor():
    OperatorBenchmark(
        op_name="bitwise_or_tensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.__or__.Tensor,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    ).run()
