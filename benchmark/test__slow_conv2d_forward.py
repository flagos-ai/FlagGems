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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from .generated_operator_utils import OperatorBenchmark
from . import base, consts, utils

# ``_slow_conv2d_forward`` starts with an underscore, and ``pytest.mark`` refuses
# to generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._slow_conv2d_forward`` and ``-m
# _slow_conv2d_forward`` both work.
setattr(
    pytest.mark,
    "_slow_conv2d_forward",
    MarkDecorator(
        Mark("_slow_conv2d_forward", (), {}, _ispytest=True),
        _ispytest=True,
    ),
)

# aten::_slow_conv2d_forward(self, weight, kernel_size, bias, stride, padding)
# performs an im2col-based 2-D convolution (groups=1, no dilation). The default
# shape set has no convolved input/weight pairs, so define local performance
# shapes whose output sizes stay in the tens-of-MB range. Each tuple is
# (inp_shape, weight_shape, kernel_size, stride, padding); the im2col cost grows
# with kernel area, so both 1x1 (pure GEMM) and 3x3/5x5 (im2col-heavy) kernels
# are represented.
SLOW_CONV2D_SHAPES = [
    ((32, 64, 128, 128), (32, 64, 1, 1), (1, 1), (1, 1), (0, 0)),
    ((32, 64, 56, 56), (32, 64, 3, 3), (3, 3), (1, 1), (1, 1)),
    ((64, 32, 18, 18), (64, 32, 5, 5), (5, 5), (2, 2), (1, 1)),
    ((64, 32, 32, 32), (32, 32, 3, 3), (3, 3), (2, 2), (0, 0)),
    ((16, 128, 16, 16), (64, 128, 3, 3), (3, 3), (1, 1), (1, 1)),
    ((16, 32, 24, 24), (24, 32, 3, 3), (3, 3), (1, 1), (1, 1)),
    ((8, 64, 64, 64), (64, 64, 3, 3), (3, 3), (2, 2), (1, 1)),
    ((4, 256, 32, 32), (128, 256, 1, 1), (1, 1), (1, 1), (0, 0)),
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


class SlowConv2dForwardBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark over (input, weight, kernel, stride, padding)."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=SLOW_CONV2D_SHAPES)


@pytest.mark._slow_conv2d_forward
def test__slow_conv2d_forward():
    # ``flag_gems._slow_conv2d_forward`` is only public once KernelGen registers
    # the candidate; before that the benchmark resolves the candidate through
    # the process-local override. Passing the attribute (when it exists) keeps
    # the same call semantics as the reference.
    bench = SlowConv2dForwardBenchmark(
        op_name="_slow_conv2d_forward",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._slow_conv2d_forward,
        gems_op=getattr(flag_gems, "_slow_conv2d_forward", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
