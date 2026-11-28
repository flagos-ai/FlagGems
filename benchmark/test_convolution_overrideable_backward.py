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

# Backward benchmark mirrors the accuracy cases at a small core-friendly size.
# Each entry is (input shape, weight shape, stride, padding, dilation,
# transposed, groups). Channel counts keep the im2col K >= 16.
CONV_BACKWARD_CASES = [
    ((2, 16, 10, 10), (16, 16, 3, 3), (1, 1), (1, 1), (1, 1), False, 1),
    ((2, 16, 12, 12), (16, 8, 3, 3), (2, 2), (2, 2), (2, 2), False, 2),
    ((2, 16, 11), (16, 16, 3), (1,), (1,), (1,), False, 1),
    ((2, 16, 6, 6, 6), (16, 16, 3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), False, 1),
    # Transposed forward: weight is (in_channels, out_channels/groups, *kernel).
    ((2, 16, 8, 8), (16, 16, 3, 3), (2, 1), (1, 1), (1, 1), True, 1),
    ((2, 16, 8), (16, 16, 3), (2,), (1,), (1,), True, 1),
]


def convolution_overrideable_backward_case_fn(shape, dtype):
    del dtype
    case = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"case": [list(x) if isinstance(x, tuple) else x for x in case]},
        params={
            "stride": list(case[2]),
            "padding": list(case[3]),
            "dilation": list(case[4]),
            "transposed": case[5],
            "groups": case[6],
        },
        builder_args=(case,),
    )


def convolution_overrideable_backward_build_inputs_fn(plan, dtype, device):
    (case,) = plan.builder_args
    in_shape, weight_shape, stride, padding, dilation, transposed, groups = case
    torch.manual_seed(0)
    torch.backends.cudnn.allow_tf32 = False
    inp = torch.randn(in_shape, dtype=dtype, device=device)
    weight = torch.randn(weight_shape, dtype=dtype, device=device)
    output_padding = [0] * (len(in_shape) - 2)
    # grad_output has the forward output's shape; derive it from the reference
    # convolution so the case parameters stay single-sourced.
    out_shape = torch.ops.aten.convolution(
        inp,
        weight,
        None,
        list(stride),
        list(padding),
        list(dilation),
        transposed,
        output_padding,
        groups,
    ).shape
    grad_output = torch.randn(out_shape, dtype=dtype, device=device)
    # ABI: (grad_output, input, weight, stride, padding, dilation, transposed,
    #       output_padding, groups).
    return (
        grad_output,
        inp,
        weight,
        list(stride),
        list(padding),
        list(dilation),
        transposed,
        output_padding,
        groups,
    )


def convolution_overrideable_backward_torch_op(
    grad_output,
    inp,
    weight,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
):
    # Only grad_input is compared; ATen's backward also returns weight/bias grads.
    inp = inp.detach().requires_grad_(True)
    out = torch.ops.aten.convolution(
        inp,
        weight,
        None,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )
    (grad_input,) = torch.autograd.grad(out, inp, grad_output)
    return grad_input


class ConvolutionOverrideableBackwardBenchmark(base.GenericBenchmarkExcluse1D):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CONV_BACKWARD_CASES
        self.shape_desc = (
            "(input shape, weight shape, stride, padding, dilation, transposed, groups)"
        )


@pytest.mark.convolution_overrideable_backward
def test_convolution_overrideable_backward():
    bench = ConvolutionOverrideableBackwardBenchmark(
        case_fn=convolution_overrideable_backward_case_fn,
        build_inputs_fn=convolution_overrideable_backward_build_inputs_fn,
        op_name="convolution_overrideable_backward",
        torch_op=convolution_overrideable_backward_torch_op,
        gems_op=flag_gems.convolution_overrideable_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
