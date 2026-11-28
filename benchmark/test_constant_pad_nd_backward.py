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

from . import base, consts

# The backward is the inverse crop of the forward: given grad_output shaped like
# the padded output, it returns the gradient at the original input's extent.
PAD_BACKWARD_CASES = [
    ((2, 3, 5, 7), [1, 2, 3, 1, 0, 2]),
    ((2, 3, 5, 7), [1, -2, 3, 0, -1, 2]),
    ((4, 6), [2, 1]),
    ((3, 4, 5), [0, 0, 2, 1]),
    ((2, 3, 4), [0, 0, 0, 0]),
    ((2, 3), [1, 2, 0, 0]),
    ((2, 3, 5), [0, 0, 2, 1]),
    ((7,), [2, 3]),
]

PAD_BACKWARD_VALUE = 1.5


def constant_pad_nd_backward_case_fn(shape, dtype):
    del dtype
    (case,) = (shape,)
    yield base.BenchmarkCasePlan(
        shape={"input": list(case[0]), "pad": list(case[1])},
        params={"value": PAD_BACKWARD_VALUE},
        builder_args=(case,),
    )


def constant_pad_nd_backward_build_inputs_fn(plan, dtype, device):
    (case,) = plan.builder_args
    shape, pad = case
    inp = torch.randn(shape, device=device, dtype=dtype)
    # grad_output has the padded output's shape; use the native forward to
    # derive it so negative pad entries stay consistent.
    out_shape = torch.constant_pad_nd(inp, pad, PAD_BACKWARD_VALUE).shape
    grad_output = torch.randn(out_shape, device=device, dtype=dtype)
    # ABI: (grad_output, self, pad, value).
    return grad_output, inp, list(pad), PAD_BACKWARD_VALUE


def constant_pad_nd_backward_torch_op(grad_output, inp, pad, value):
    ref_inp = inp.detach().requires_grad_(True)
    ref_out = torch.constant_pad_nd(ref_inp, pad, value)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, grad_output)
    return ref_in_grad


def constant_pad_nd_backward_gems_op(grad_output, inp, pad, value):
    # Mirror the reference: both sides build the forward graph and call
    # autograd.grad, so the comparison covers the same work.
    res_inp = inp.detach().requires_grad_(True)
    res_out = flag_gems.constant_pad_nd(res_inp, pad, value)
    (res_in_grad,) = torch.autograd.grad(res_out, res_inp, grad_output)
    return res_in_grad


class ConstantPadNdBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PAD_BACKWARD_CASES
        self.shape_desc = "(input shape, pad)"


@pytest.mark.constant_pad_nd_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_constant_pad_nd_backward():
    bench = ConstantPadNdBackwardBenchmark(
        case_fn=constant_pad_nd_backward_case_fn,
        build_inputs_fn=constant_pad_nd_backward_build_inputs_fn,
        op_name="constant_pad_nd_backward",
        torch_op=constant_pad_nd_backward_torch_op,
        gems_op=constant_pad_nd_backward_gems_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
