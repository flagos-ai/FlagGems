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


def cosh_backward_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={},
        builder_args=(shape,),
    )


def cosh_backward_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    grad_output = utils.generate_tensor_input(shape, dtype, device)
    inp = utils.generate_tensor_input(shape, dtype, device).requires_grad_(True)
    return grad_output, inp


def cosh_backward_torch_op(grad_output, inp):
    (ref_in_grad,) = torch.autograd.grad(torch.cosh(inp), inp, grad_output)
    return ref_in_grad


@pytest.mark.cosh_backward
def test_cosh_backward():
    bench = base.GenericBenchmarkExcluse1D(
        case_fn=cosh_backward_case_fn,
        build_inputs_fn=cosh_backward_build_inputs_fn,
        op_name="cosh_backward",
        torch_op=cosh_backward_torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
