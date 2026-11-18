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


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)},
        params={"dim": [1], "keepdim": False},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    inp = utils.generate_tensor_input(plan.builder_args[0], dtype, device)
    return inp, [1], False


@pytest.mark.frobenius_norm
def test_frobenius_norm():
    bench = base.GenericBenchmarkExcluse1D(
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        op_name="frobenius_norm",
        torch_op=torch.ops.aten.frobenius_norm.dim,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
