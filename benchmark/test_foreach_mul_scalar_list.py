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


_ATEN_OP = torch.ops.aten._foreach_mul.ScalarList
_LIST_LEN = 4
_POSITIVE_OTHER = False


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": [list(dims)] * _LIST_LEN},
        params={"list_length": _LIST_LEN, "overload": "ScalarList"},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    tensors = [
        utils.generate_tensor_input(shape, dtype, device) for _ in range(_LIST_LEN)
    ]
    other = [1.5 + 0.25 * index for index in range(_LIST_LEN)]
    return tensors, other


@pytest.mark.foreach_mul_scalar_list
def test_foreach_mul_scalar_list_benchmark():
    bench = OperatorBenchmark(
        op_name="foreach_mul_scalar_list",
        torch_op=_ATEN_OP,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
