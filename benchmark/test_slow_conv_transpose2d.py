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

from .generated_operator_utils import OperatorBenchmark
from . import base, consts, utils

# aten::slow_conv_transpose2d(self, weight, kernel_size, bias, stride, padding,
# output_padding, dilation) performs an im2col-based 2-D transposed convolution
# (groups=1) with output_padding/dilation support. The shared shape files have
# no transposed-conv input/weight pairs, so define local performance shapes whose
# output sizes stay in the tens-of-MB range. Each tuple is (inp_shape,
# weight_shape, kernel_size, stride, padding, output_padding, dilation); note the
# transposed weight layout (C_in, C_out, kH, kW). 1x1 (pure GEMM), 3x3/5x5
# (im2col-heavy), stride-2, output_padding and dilation-2 cases are represented,
# with C_in/C_out both kept >= 32 so the GEMM dimensions are substantial.
SLOW_CONV_TRANSPOSE2D_SHAPES = [
    ((32, 64, 128, 128), (64, 64, 3, 3), (3, 3), (1, 1), (0, 0), (0, 0), (1, 1)),
    ((32, 64, 56, 56), (64, 64, 3, 3), (3, 3), (2, 2), (1, 1), (1, 1), (1, 1)),
    ((16, 64, 56, 56), (64, 128, 3, 3), (3, 3), (1, 1), (1, 1), (0, 0), (2, 2)),
    ((8, 128, 32, 32), (128, 128, 3, 3), (3, 3), (1, 1), (1, 1), (0, 0), (1, 1)),
    ((8, 64, 32, 32), (64, 128, 5, 5), (5, 5), (2, 2), (2, 2), (1, 1), (1, 1)),
    ((16, 32, 64, 64), (32, 64, 3, 3), (3, 3), (2, 2), (1, 1), (1, 1), (2, 2)),
    ((16, 64, 56, 56), (64, 64, 1, 1), (1, 1), (1, 1), (0, 0), (0, 0), (1, 1)),
    ((8, 64, 64, 64), (64, 64, 3, 3), (3, 3), (2, 2), (2, 2), (1, 1), (1, 1)),
    ((16, 32, 112, 112), (32, 64, 3, 3), (3, 3), (2, 2), (1, 1), (0, 0), (1, 1)),
]


def _case_fn(shape, dtype):
    del dtype
    (
        inp_shape,
        weight_shape,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
    ) = shape
    yield base.BenchmarkCasePlan(
        shape={"input": inp_shape, "weight": weight_shape},
        params={
            "kernel_size": kernel_size,
            "bias": True,
            "stride": stride,
            "padding": padding,
            "output_padding": output_padding,
            "dilation": dilation,
        },
        builder_args=(
            inp_shape,
            weight_shape,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
        ),
    )


def _build_inputs_fn(plan, dtype, device):
    (
        inp_shape,
        weight_shape,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
    ) = plan.builder_args
    inp = utils.generate_tensor_input(inp_shape, dtype, device)
    weight = utils.generate_tensor_input(weight_shape, dtype, device)
    bias = utils.generate_tensor_input((weight_shape[1],), dtype, device)
    # Trailing dict is unpacked as call kwargs by Benchmark.unpack_to_args_kwargs.
    return (
        inp,
        weight,
        kernel_size,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        {},
    )


class SlowConvTranspose2dBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark over (input, weight, kernel, stride, padding, output_padding, dilation)."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=SLOW_CONV_TRANSPOSE2D_SHAPES)

    def set_more_shapes(self):
        # The generic 1D/2D/3D extra shapes cannot describe a conv workload;
        # this op only benchmarks its explicit (input, weight) shape pairs.
        return []


@pytest.mark.slow_conv_transpose2d
def test_slow_conv_transpose2d():
    bench = SlowConvTranspose2dBenchmark(
        op_name="slow_conv_transpose2d",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.slow_conv_transpose2d,
        gems_op=getattr(flag_gems, "slow_conv_transpose2d", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
