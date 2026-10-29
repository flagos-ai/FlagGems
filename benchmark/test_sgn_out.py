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
from .generated_operator_utils import OperatorBenchmark as _BaseOperatorBenchmark
from .test_sgn import _SgnOutCurrentNativeReference


def _case_fn(shape, dtype):
    del dtype
    shape = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape), "out": list(shape)},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = utils.generate_tensor_input(shape, dtype, device)
    out = torch.empty_like(inp)
    return inp, {"out": out}


class OperatorBenchmark(_SgnOutCurrentNativeReference, _BaseOperatorBenchmark):
    _sgn_profile = "current-fresh"


@pytest.mark.sgn_out
@pytest.mark.skip_native(
    vendors=["enflame"],
    reason="Native SGNOut executes on CPU; use the exact H800 fresh-input profile",
)
def test_sgn_out():
    OperatorBenchmark(
        op_name="sgn_out",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.sgn.out,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    ).run()
