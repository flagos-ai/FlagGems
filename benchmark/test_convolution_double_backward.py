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

from . import base
from .generated_operator_utils import OperatorBenchmark

# Shapes for the second-order (double) backward of a 2D convolution.
# Each tuple is
#   (N, Cin, H, W, Cout, kH, kW, stride, padding, dilation, groups, transposed, output_padding)
# and the weight layout follows the chosen ``transposed`` flag.
CONV_DOUBLE_BACKWARD_SHAPES = [
    # non-transposed convolutions
    (16, 32, 64, 64, 64, 3, 3, 1, 1, 1, 1, False, (0, 0)),
    (16, 32, 64, 64, 64, 3, 3, 2, 1, 1, 1, False, (0, 0)),
    (8, 64, 32, 32, 128, 3, 3, 2, 1, 1, 1, False, (0, 0)),
    (8, 32, 32, 32, 32, 3, 3, 1, 1, 1, 2, False, (0, 0)),
    # transposed convolutions (weight is (Cin, Cout//groups, kH, kW))
    (8, 64, 16, 16, 32, 3, 3, 2, 1, 1, 1, True, (0, 0)),
    (8, 64, 16, 16, 32, 3, 3, 2, 1, 1, 1, True, (1, 1)),
    (8, 32, 16, 16, 32, 3, 3, 2, 1, 1, 2, True, (0, 0)),
]

# Shapes for the second-order (double) backward of a 1D convolution. Each tuple
# is
#   (N, Cin, L, Cout, kL, stride, padding, dilation, groups, transposed, output_padding)
# with scalar spatial params; weight layout follows ``transposed``. 1D reuses
# the 2D path internally (unsqueeze the lone spatial axis), so this coverage
# mirrors ``CONV1D_CASES`` in the accuracy test at benchmark-appropriate sizes.
CONV_DOUBLE_BACKWARD_1D_SHAPES = [
    # non-transposed convolutions
    (16, 32, 1024, 64, 3, 1, 1, 1, 1, False, 0),
    (16, 32, 1024, 64, 3, 2, 1, 1, 1, False, 0),
    (8, 64, 512, 128, 3, 2, 1, 1, 1, False, 0),
    (8, 32, 512, 32, 3, 1, 1, 1, 2, False, 0),
    # transposed convolutions (weight is (Cin, Cout//groups, kL))
    (8, 64, 256, 32, 3, 2, 1, 1, 1, True, 0),
    (8, 64, 256, 32, 3, 2, 1, 1, 1, True, 1),
    (8, 32, 256, 32, 3, 2, 1, 1, 2, True, 0),
]


class ConvDoubleBackwardBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CONV_DOUBLE_BACKWARD_SHAPES + CONV_DOUBLE_BACKWARD_1D_SHAPES
        self.shape_desc = "batch, channels, spatial shape, convolution parameters"


def _case_fn(shape, dtype):
    case = tuple(shape)
    metadata = [*case[:-1], list(case[-1]) if len(case) == 13 else case[-1]]
    yield base.BenchmarkCasePlan(
        shape={"case": metadata}, params={}, builder_args=(case,)
    )


def _build_inputs_fn(plan, dtype, device):
    (case,) = plan.builder_args
    if len(case) == 13:
        n, cin, h, w, cout, kh, kw, *config = case
        input_spatial, kernel_spatial = (h, w), (kh, kw)
        conv, conv_transpose = (
            torch.nn.functional.conv2d,
            torch.nn.functional.conv_transpose2d,
        )
    else:
        n, cin, length, cout, kernel, *config = case
        input_spatial, kernel_spatial = (length,), (kernel,)
        conv, conv_transpose = (
            torch.nn.functional.conv1d,
            torch.nn.functional.conv_transpose1d,
        )
    stride, padding, dilation, groups, transposed, output_padding = config
    spatial_ndim = len(input_spatial)
    if spatial_ndim == 1:
        output_padding = (output_padding,)

    torch.manual_seed(0)
    x = torch.randn((n, cin, *input_spatial), dtype=dtype, device=device)
    weight_channels = (cin, cout // groups) if transposed else (cout, cin // groups)
    weight = torch.randn(
        (*weight_channels, *kernel_spatial), dtype=dtype, device=device
    )
    bias = torch.randn(cout, dtype=dtype, device=device)
    conv_kwargs = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)
    if transposed:
        y = conv_transpose(
            x, weight, bias, output_padding=output_padding, **conv_kwargs
        )
    else:
        y = conv(x, weight, bias, **conv_kwargs)
    gO = torch.randn_like(y)
    return (
        torch.randn_like(x),
        torch.randn_like(weight),
        torch.randn_like(bias),
        gO,
        weight,
        x,
        [stride] * spatial_ndim,
        [padding] * spatial_ndim,
        [dilation] * spatial_ndim,
        transposed,
        list(output_padding),
        groups,
        (True, True, True),
    )


@pytest.mark.convolution_double_backward
@pytest.mark.underscore_convolution_double_backward
def test_convolution_double_backward():
    bench = ConvDoubleBackwardBenchmark(
        op_name="_convolution_double_backward",
        torch_op=torch.ops.aten._convolution_double_backward.default,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32, torch.float16],
    )
    bench.set_gems(lambda *args: flag_gems._convolution_double_backward(*args))
    bench.run()
