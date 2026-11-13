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


def case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape=shape,
        params={"dim": 0, "return_inverse": True, "return_counts": False},
        builder_args=(shape,),
    )


def build_inputs(plan, dtype, device):
    (shape,) = plan.builder_args
    return utils.generate_tensor_input(shape, dtype, device), dict(plan.params)


@pytest.mark.unique_dim_consecutive
def test_unique_dim_consecutive():
    bench = base.GenericBenchmark2DOnly(
        case_fn=case_fn,
        build_inputs_fn=build_inputs,
        op_name="unique_dim_consecutive",
        torch_op=torch.ops.aten.unique_dim_consecutive.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
