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

# aten::slow_conv_dilated2d(self, weight, kernel_size, bias, stride, padding,
# dilation) performs an im2col-based 2-D convolution (groups=1) with dilation
# support. The default shape set has no convolved input/weight pairs, so define
# local performance shapes whose output sizes stay in the tens-of-MB range.
# Each tuple is (inp_shape, weight_shape, kernel_size, stride, padding,
# dilation); the im2col cost grows with kernel area, so 1x1 (pure GEMM),
# 3x3/5x5 (im2col-heavy), stride-2, and dilation-2 cases are all represented.
SLOW_CONV_DILATED2D_SHAPES = [
    ((32, 64, 128, 128), (64, 64, 1, 1), (1, 1), (1, 1), (0, 0), (1, 1)),
    ((32, 64, 56, 56), (64, 64, 3, 3), (3, 3), (1, 1), (1, 1), (1, 1)),
    ((16, 64, 56, 56), (64, 64, 3, 3), (3, 3), (2, 2), (1, 1), (1, 1)),
    ((8, 128, 32, 32), (128, 128, 3, 3), (3, 3), (1, 1), (1, 1), (2, 2)),
    ((8, 64, 32, 32), (128, 64, 5, 5), (5, 5), (1, 1), (2, 2), (1, 1)),
    ((16, 32, 64, 64), (64, 32, 3, 3), (3, 3), (2, 2), (1, 1), (2, 2)),
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
    # weight is (C_out, C_in, kH, kW): bias has C_out elements.
    bias = utils.generate_tensor_input((weight_shape[0],), dtype, device)
    return inp, weight, kernel_size, bias, stride, padding, dilation, {}


class SlowConvDilated2dBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark over (input, weight, kernel, stride, padding, dilation)."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=SLOW_CONV_DILATED2D_SHAPES)


@pytest.mark.slow_conv_dilated2d
def test_slow_conv_dilated2d():
    bench = SlowConvDilated2dBenchmark(
        op_name="slow_conv_dilated2d",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.slow_conv_dilated2d,
        # Resolved from the flag_gems namespace when the candidate is registered;
        # the KernelGen override is picked up by Benchmark._candidate_call internally.
        gems_op=getattr(flag_gems, "slow_conv_dilated2d", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
