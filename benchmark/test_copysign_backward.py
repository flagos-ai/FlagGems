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


def _copysign_backward_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={},
        builder_args=(shape,),
    )


def _copysign_backward_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    # copysign's VJP w.r.t. ``other`` is identically zero, so only ``self`` is
    # differentiated; ``other`` is a plain sibling tensor.
    inp = utils.generate_tensor_input(shape, dtype, device)
    other = utils.generate_tensor_input(shape, dtype, device)
    grad_output = utils.generate_tensor_input(shape, dtype, device)
    return grad_output, inp, other


def _copysign_backward_torch_op(grad_output, inp, other):
    inp = inp.detach().requires_grad_(True)
    (ref_in_grad,) = torch.autograd.grad(torch.copysign(inp, other), inp, grad_output)
    return ref_in_grad


@pytest.mark.copysign_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_copysign_backward():
    bench = base.GenericBenchmarkExcluse1D(
        case_fn=_copysign_backward_case_fn,
        build_inputs_fn=_copysign_backward_build_inputs_fn,
        op_name="copysign_backward",
        torch_op=_copysign_backward_torch_op,
        gems_op=flag_gems.copysign_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
