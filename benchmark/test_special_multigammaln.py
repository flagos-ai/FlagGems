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

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

def _case_fn(shape, dtype):
    del dtype
    shape = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": list(shape)},
        params={"p": 5},
        builder_args=(shape, 5),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, p = plan.builder_args
    torch.manual_seed(0)
    inp = torch.rand(shape, dtype=dtype, device=device) + (p - 1) / 2 + 1.0
    return inp, p


@pytest.mark.special_multigammaln
def test_special_multigammaln():
    OperatorBenchmark(
        op_name="special_multigammaln",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.special_multigammaln,
        gems_op=getattr(flag_gems, "special_multigammaln", None),
        dtypes=consts.FLOAT_DTYPES,
    ).run()
