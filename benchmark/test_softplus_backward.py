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
from .test_softplus import _SoftplusStaticReference


class SoftplusBackwardBenchmark(_SoftplusStaticReference, OperatorBenchmark):
    _H800_PROFILE = "direct"

    def build_inputs(self, case):
        if flag_gems.vendor_name == "kunlunxin" and case.dtype == torch.bfloat16:
            base.Config.skipped_case_ids.add(case.case_id)
            pytest.skip("native BF16 softplus backward is not implemented on Kunlunxin")
        return super().build_inputs(case)


def _case_fn(shape, dtype):
    del dtype
    shape = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"grad_output": list(shape), "self": list(shape)},
        params={"beta": 1.5, "threshold": 20.0},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    inp = torch.randn(shape, dtype=dtype, device=device)
    return grad_output, inp, 1.5, 20.0


@pytest.mark.softplus_backward
@pytest.mark.skip_native(
    vendors=["enflame"],
    reason="native softplus backward is CPU fallback; require exact direct H800 profile",
)
def test_softplus_backward():
    SoftplusBackwardBenchmark(
        op_name="softplus_backward",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.softplus_backward,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
