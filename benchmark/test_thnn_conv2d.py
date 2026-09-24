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

# aten::thnn_conv2d(self, weight, kernel_size, bias, stride, padding) performs
# an im2col-based 2-D convolution (groups=1, no dilation). No public benchmark
# family models a conv, so this uses the two-phase GenericBenchmark with both
# case_fn (one BenchmarkCasePlan per shape) and build_inputs_fn. ``torch_op`` is
# the perf reference and ``gems_op`` the candidate; both share the exact same
# call semantics (self, weight, kernel_size, bias, stride, padding).
#
# The default shape set has no convolved input/weight pairs, so the local
# performance shapes below are used. Each tuple is
# (inp_shape, weight_shape, kernel_size, stride, padding); the im2col cost grows
# with kernel area, so both 1x1 (pure GEMM) and 3x3/5x5 (im2col-heavy) kernels
# are represented, and output sizes stay in the tens-of-MB range.
THNN_CONV2D_SHAPES = [
    ((32, 64, 128, 128), (32, 64, 1, 1), (1, 1), (1, 1), (0, 0)),
    ((32, 64, 56, 56), (32, 64, 3, 3), (3, 3), (1, 1), (1, 1)),
    ((64, 32, 18, 18), (64, 32, 5, 5), (5, 5), (2, 2), (1, 1)),
    ((64, 32, 32, 32), (32, 32, 3, 3), (3, 3), (2, 2), (0, 0)),
    ((16, 128, 16, 16), (64, 128, 3, 3), (3, 3), (1, 1), (1, 1)),
    ((8, 256, 28, 28), (128, 256, 1, 1), (1, 1), (1, 1), (0, 0)),
]


def _case_fn(shape, dtype):
    del dtype
    inp_shape, weight_shape, kernel_size, stride, padding = shape
    yield base.BenchmarkCasePlan(
        shape={"input": inp_shape, "weight": weight_shape},
        params={
            "kernel_size": kernel_size,
            "bias": True,
            "stride": stride,
            "padding": padding,
        },
        builder_args=(inp_shape, weight_shape, kernel_size, stride, padding),
    )


def _build_inputs_fn(plan, dtype, device):
    inp_shape, weight_shape, kernel_size, stride, padding = plan.builder_args
    inp = utils.generate_tensor_input(inp_shape, dtype, device)
    weight = utils.generate_tensor_input(weight_shape, dtype, device)
    bias = utils.generate_tensor_input((weight_shape[0],), dtype, device)
    return inp, weight, kernel_size, bias, stride, padding, {}


class ThnnConv2dBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark over (input, weight, kernel, stride, padding)."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=THNN_CONV2D_SHAPES)


@pytest.mark.thnn_conv2d
def test_thnn_conv2d():
    bench = ThnnConv2dBenchmark(
        op_name="thnn_conv2d",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.thnn_conv2d,
        # The op has no registered flag_gems.thnn_conv2d entry point in this
        # checkout, so fall back to None: the candidate is supplied either by
        # KernelGen's --override thnn_conv2d:<file>:<function> or by the FlagGems
        # dispatcher fallback inside Benchmark._measure_input.
        gems_op=getattr(flag_gems, "thnn_conv2d", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.fixture(autouse=True)
def full_precision():
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
