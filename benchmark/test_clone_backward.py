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

# bfloat16 has no kernel on every backend; this is a static capability query, so
# case listing stays free of device work and tensor allocations.
_BENCH_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype is not torch.bfloat16 or flag_gems.runtime.device.support_bf16
]


def clone_backward_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={},
        builder_args=(shape,),
    )


def clone_backward_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    grad_output = utils.generate_tensor_input(shape, dtype, device)
    inp = utils.generate_tensor_input(shape, dtype, device).requires_grad_(True)
    # clone_backward's ABI is (grad_output, self).
    return grad_output, inp


def clone_backward_torch_op(grad_output, inp):
    out = torch.ops.aten.clone(inp)
    (ref_in_grad,) = torch.autograd.grad(out, inp, grad_output)
    return ref_in_grad


@pytest.mark.clone_backward
def test_clone_backward():
    bench = base.GenericBenchmarkExcluse1D(
        case_fn=clone_backward_case_fn,
        build_inputs_fn=clone_backward_build_inputs_fn,
        op_name="clone_backward",
        torch_op=clone_backward_torch_op,
        gems_op=flag_gems.clone_backward,
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
