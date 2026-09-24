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

# aten::slow_conv_transpose3d(self, weight, kernel_size, bias, stride, padding,
# output_padding, dilation) performs an im2col-based transposed 3-D convolution
# (groups=1). ``self`` is (N, C_in, D, H, W) and ``weight`` is
# (C_in, C_out, kD, kH, kW); the output is (N, C_out, D_out, H_out, W_out) with
#   D_out = (D - 1)*sD - 2*pD + dil_d*(kD - 1) + out_pad_d + 1
# so transposed convolutions upsample. The default shape set has no transposed
# conv input/weight pairs, so define local performance shapes whose output sizes
# stay in the tens-of-MB range. Each tuple is (inp_shape, weight_shape,
# kernel_size, stride, padding, output_padding, dilation); the col2im cost grows
# with kernel volume, so 1x1x1 (pure GEMM), 3x3x3 (col2im-heavy), stride-2,
# output_padding, and dilation-2 cases are all represented.
SLOW_CONV_TRANSPOSE3D_SHAPES = [
    (
        (2, 16, 16, 16, 16),
        (16, 16, 1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (0, 0, 0),
        (0, 0, 0),
        (1, 1, 1),
    ),
    (
        (2, 32, 12, 12, 12),
        (32, 32, 3, 3, 3),
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (0, 0, 0),
        (1, 1, 1),
    ),
    (
        (1, 32, 8, 8, 8),
        (32, 32, 3, 3, 3),
        (3, 3, 3),
        (2, 2, 2),
        (1, 1, 1),
        (0, 0, 0),
        (1, 1, 1),
    ),
    (
        (1, 32, 8, 8, 8),
        (32, 32, 3, 3, 3),
        (3, 3, 3),
        (2, 2, 2),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    ),
    (
        (1, 64, 6, 6, 6),
        (64, 64, 3, 3, 3),
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (0, 0, 0),
        (2, 2, 2),
    ),
    (
        (1, 32, 10, 10, 10),
        (32, 64, 3, 3, 3),
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (0, 0, 0),
        (1, 1, 1),
    ),
    (
        (2, 16, 16, 16, 16),
        (16, 32, 3, 3, 3),
        (3, 3, 3),
        (2, 2, 2),
        (1, 1, 1),
        (1, 1, 1),
        (2, 2, 2),
    ),
    (
        (2, 8, 16, 16, 16),
        (8, 16, 3, 3, 3),
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (0, 0, 0),
        (1, 1, 1),
    ),
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
    # Transposed conv weight is (C_in, C_out, kD, kH, kW): bias has C_out
    # elements, i.e. the second weight dim.
    bias = utils.generate_tensor_input((weight_shape[1],), dtype, device)
    return (
        inp,
        weight,
        kernel_size,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
    )


class SlowConvTranspose3dBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark over the transposed conv3d parameter tuple."""

    DEFAULT_SHAPE_DESC = "input/weight shape"

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=SLOW_CONV_TRANSPOSE3D_SHAPES)


@pytest.mark.slow_conv_transpose3d
def test_slow_conv_transpose3d():
    bench = SlowConvTranspose3dBenchmark(
        op_name="slow_conv_transpose3d",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.slow_conv_transpose3d,
        # flag_gems.slow_conv_transpose3d may not exist until the candidate is
        # generated; Benchmark._candidate_call still finds the process-local KernelGen
        # override for this op_name.
        gems_op=getattr(flag_gems, "slow_conv_transpose3d", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
