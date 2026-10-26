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

SHAPES = [(256, 256), (512, 512), (1024, 1024)]
LIST_LENGTH = 16
VALUE_KIND = lambda n: 0.5


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"tensors": [list(shape)] * LIST_LENGTH},
        params={"list_length": LIST_LENGTH},
        builder_args=(tuple(shape),),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    make = lambda: [
        torch.rand(shape, dtype=dtype, device=device) + 1.0
        for _ in range(LIST_LENGTH)
    ]
    self_tensors = make()
    tensor1 = make()
    tensor2 = make()
    value = VALUE_KIND(len(self_tensors))
    return self_tensors, tensor1, tensor2, value



@pytest.mark.foreach_addcdiv_scalar
def test_perf_foreach_addcdiv_scalar():
    bench = OperatorBenchmark(
        op_name="foreach_addcdiv_scalar",
        torch_op=torch.ops.aten._foreach_addcdiv.Scalar,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        fresh_inputs=False,
    )
    bench.run()
