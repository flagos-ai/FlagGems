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

from . import base
from .generated_operator_utils import OperatorBenchmark

SHAPES = [
    (32, 64, 128, 128, 32, 3, 3, 1, 1, 1),
    (32, 64, 64, 64, 16, 5, 5, 2, 1, 1),
    (16, 32, 24, 24, 24, 3, 3, 2, 1, 1),
]


class ConvBackwardBenchmark(OperatorBenchmark):
    DEFAULT_SHAPES = SHAPES

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=SHAPES)


def _case_fn(shape, dtype):
    del dtype
    batch, in_c, in_h, in_w, out_c, kh, kw, stride, padding, groups = shape
    input_shape = (batch, in_c, in_h, in_w)
    weight_shape = (out_c, in_c // groups, kh, kw)
    yield base.BenchmarkCasePlan(
        shape={"input": list(input_shape), "weight": list(weight_shape)},
        params={"stride": stride, "padding": padding, "groups": groups},
        builder_args=(input_shape, weight_shape, stride, padding, groups),
    )


def _build_inputs_fn(plan, dtype, device):
    input_shape, weight_shape, stride, padding, groups = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(input_shape, dtype=dtype, device=device)
    weight = torch.randn(weight_shape, dtype=dtype, device=device)
    forward = torch.nn.functional.conv2d(
        inp, weight, stride=stride, padding=padding, groups=groups
    )
    grad_output = torch.randn_like(forward)
    return (
        grad_output,
        inp,
        weight,
        [stride, stride],
        [padding, padding],
        [1, 1],
        False,
        [0, 0],
        groups,
        [True, True, True],
    )


def _native_backward(grad_output, inp, weight, stride, padding, dilation,
                     transposed, output_padding, groups, output_mask):
    # The overrideable extension hook has no CPU/CUDA native kernel. This is
    # the same ATen reference used by the operator's correctness suite.
    return torch.ops.aten.convolution_backward(
        grad_output, inp, weight, [grad_output.shape[1]], stride, padding,
        dilation, transposed, output_padding, groups, output_mask,
    )


@pytest.mark.convolution_backward_overrideable
def test_convolution_backward_overrideable():
    ConvBackwardBenchmark(
        op_name="convolution_backward_overrideable",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=_native_backward,
        dtypes=[torch.float16, torch.float32],
    ).run()
