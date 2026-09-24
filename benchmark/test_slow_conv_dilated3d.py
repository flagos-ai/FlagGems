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

# aten::slow_conv_dilated3d(self, weight, kernel_size, bias, stride, padding,
# dilation) performs an im2col-based 3-D convolution (groups=1) with dilation
# support. ``self`` is (N, C_in, D, H, W) and ``weight`` is
# (C_out, C_in, kD, kH, kW); the output is (N, C_out, D_out, H_out, W_out) with
#   D_out = (D + 2*pD - dil_d*(kD - 1) - 1) // sD + 1
# and likewise for H and W.
#
# Only the 5-D batched route is benchmarked: the native CUDA reference is
# non-deterministic for the 4-D unbatched route, so it is not a meaningful
# candidate target. The default shape set has no convolved input/weight pairs,
# so define local performance shapes whose output sizes stay in the
# tens-of-MB range. Each tuple is (inp_shape, weight_shape, kernel_size,
# stride, padding, dilation); the im2col cost grows with kernel volume, so
# 1x1x1 (pure GEMM), 3x3x3 (im2col-heavy), stride-2, dilation-2, channel
# expansion and asymmetric stride/padding cases are all represented.
SLOW_CONV_DILATED3D_SHAPES = [
    (
        (2, 16, 32, 32, 32),
        (16, 16, 1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (0, 0, 0),
        (1, 1, 1),
    ),
    (
        (2, 32, 16, 16, 16),
        (32, 32, 3, 3, 3),
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    ),
    (
        (1, 32, 16, 16, 16),
        (32, 32, 3, 3, 3),
        (3, 3, 3),
        (2, 2, 2),
        (1, 1, 1),
        (1, 1, 1),
    ),
    (
        (1, 64, 16, 16, 16),
        (64, 64, 3, 3, 3),
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (2, 2, 2),
    ),
    (
        (1, 32, 12, 12, 12),
        (64, 32, 3, 3, 3),
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    ),
    (
        (2, 16, 32, 32, 32),
        (32, 16, 3, 3, 3),
        (3, 3, 3),
        (2, 2, 2),
        (1, 1, 1),
        (2, 2, 2),
    ),
    (
        (2, 8, 32, 32, 32),
        (16, 8, 3, 3, 3),
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    ),
    (
        (2, 16, 24, 20, 28),
        (24, 16, 3, 3, 3),
        (3, 3, 3),
        (2, 1, 1),
        (1, 2, 0),
        (1, 1, 1),
    ),
]


def _case_fn(shape, dtype):
    del dtype
    inp_shape, weight_shape, kernel_size, stride, padding, dilation = shape
    yield base.BenchmarkCasePlan(
        shape={"input": inp_shape, "weight": weight_shape},
        params={
            "kernel_size": kernel_size,
            "bias": True,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
        },
        builder_args=(
            inp_shape,
            weight_shape,
            kernel_size,
            stride,
            padding,
            dilation,
        ),
    )


def _build_inputs_fn(plan, dtype, device):
    inp_shape, weight_shape, kernel_size, stride, padding, dilation = plan.builder_args
    inp = utils.generate_tensor_input(inp_shape, dtype, device)
    weight = utils.generate_tensor_input(weight_shape, dtype, device)
    # Conv weight is (C_out, C_in, kD, kH, kW): bias has C_out elements, i.e.
    # the first weight dim.
    bias = utils.generate_tensor_input((weight_shape[0],), dtype, device)
    return inp, weight, kernel_size, bias, stride, padding, dilation, {}


class SlowConvDilated3dBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark over (input, weight, kernel, stride, padding, dilation)."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=SLOW_CONV_DILATED3D_SHAPES)


@pytest.mark.slow_conv_dilated3d
def test_slow_conv_dilated3d():
    bench = SlowConvDilated3dBenchmark(
        op_name="slow_conv_dilated3d",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.slow_conv_dilated3d,
        gems_op=getattr(flag_gems, "slow_conv_dilated3d", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
