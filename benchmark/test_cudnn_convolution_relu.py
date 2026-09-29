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

from . import base, utils
from .generated_operator_utils import OperatorBenchmark

# Case descriptors for the fused convolution + ReLU:
#   (input_shape, out_channels, kernel, stride, padding, dilation, groups, bias)
# The rows are realistic convolution workloads: 3x3 / 5x5 / 7x7 and 1x1 kernels,
# stride-2 downsampling, dilation 2 and grouped (including depthwise) forms, in
# both 2-D and 3-D. Every operand stays shape-consistent within its own row: the
# weight has shape (out_channels, in_channels // groups, ...), and groups divides
# both the input channels and the row's out_channels.
_BENCH_CASES = [
    ((16, 64, 128, 128), 64, 3, 1, 1, 1, 1, True),
    ((64, 64, 56, 56), 64, 3, 1, 1, 1, 1, True),
    ((128, 128, 28, 28), 256, 3, 1, 1, 1, 1, False),
    ((64, 256, 14, 14), 512, 3, 1, 1, 1, 1, True),
    ((16, 512, 7, 7), 512, 3, 1, 1, 1, 1, True),
    ((32, 64, 224, 224), 64, 7, 2, 3, 1, 1, True),
    ((16, 3, 224, 224), 64, 7, 2, 3, 1, 1, True),
    ((32, 128, 64, 64), 128, 1, 1, 0, 1, 1, True),
    ((32, 256, 32, 32), 256, 3, 2, 1, 1, 1, False),
    ((16, 256, 32, 32), 256, 3, 1, 1, 2, 2, True),
    ((4, 64, 32, 32, 32), 64, 3, 1, 1, 1, 1, True),
    ((2, 128, 16, 16, 16), 128, 3, 1, 1, 2, 1, True),
    ((4, 128, 8, 8, 8), 128, 1, 1, 0, 1, 1, False),
    ((4, 64, 16, 16, 16), 64, 3, 1, 1, 1, 2, True),
]

# cuDNN executes float16 / float32 / bfloat16 only; the bfloat16 entry follows
# the runtime capability attribute rather than a fixed dtype list. Reading the
# static flag is metadata only - no tensor is created and no dtype is probed
# here or anywhere else in this file.
_BENCH_DTYPES = [torch.float32, torch.float16]
if flag_gems.runtime.device.support_bf16:
    _BENCH_DTYPES.append(torch.bfloat16)


def _case_fn(case, dtype):
    del dtype
    shape, out_channels, kernel, stride, padding, dilation, groups, bias = case
    spatial = len(shape) - 2
    yield base.BenchmarkCasePlan(
        shape={
            "input": list(shape),
            "out_channels": out_channels,
            "kernel_size": kernel,
            "groups": groups,
        },
        params={
            "stride": [stride] * spatial,
            "padding": [padding] * spatial,
            "dilation": [dilation] * spatial,
            "groups": groups,
            "bias": bias,
        },
        builder_args=(
            shape,
            out_channels,
            kernel,
            stride,
            padding,
            dilation,
            groups,
            bias,
        ),
    )


def _build_inputs_fn(plan, dtype, device):
    (
        shape,
        out_channels,
        kernel,
        stride,
        padding,
        dilation,
        groups,
        bias,
    ) = plan.builder_args
    spatial = len(shape) - 2
    inp = utils.generate_tensor_input(shape, dtype, device)
    weight = utils.generate_tensor_input(
        (out_channels, shape[1] // groups) + (kernel,) * spatial, dtype, device
    )
    bias_t = (
        utils.generate_tensor_input((out_channels,), dtype, device) if bias else None
    )
    return (
        inp,
        weight,
        bias_t,
        [stride] * spatial,
        [padding] * spatial,
        [dilation] * spatial,
        groups,
        {},
    )


class CudnnConvolutionReluBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_BENCH_CASES)


@pytest.mark.cudnn_convolution_relu
def test_cudnn_convolution_relu():
    bench = CudnnConvolutionReluBenchmark(
        op_name="cudnn_convolution_relu",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.cudnn_convolution_relu,
        gems_op=getattr(flag_gems, "cudnn_convolution_relu", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
