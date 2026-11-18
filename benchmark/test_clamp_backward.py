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


def clamp_backward_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"min_value": -0.5, "max_value": 0.5},
        builder_args=(shape,),
    )


def clamp_backward_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    grad_output = utils.generate_tensor_input(shape, dtype, device)
    inp = utils.generate_tensor_input(shape, dtype, device).requires_grad_(True)
    # clamp_backward's ABI is (grad_output, self, min_value, max_value) with
    # scalar bounds only.
    return grad_output, inp, -0.5, 0.5


def clamp_backward_torch_op(grad_output, inp, min_value, max_value):
    out = torch.clamp(inp, min=min_value, max=max_value)
    (ref_in_grad,) = torch.autograd.grad(out, inp, grad_output)
    return ref_in_grad


@pytest.mark.clamp_backward
def test_clamp_backward():
    bench = base.GenericBenchmarkExcluse1D(
        case_fn=clamp_backward_case_fn,
        build_inputs_fn=clamp_backward_build_inputs_fn,
        op_name="clamp_backward",
        torch_op=clamp_backward_torch_op,
        gems_op=flag_gems.clamp_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
