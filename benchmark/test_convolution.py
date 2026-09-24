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

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

CORE_SHAPES = [
    ((8, 32, 256), (32, 32, 3), (1,), (1,), (1,), 1, False, (0,)),
    ((8, 64, 128, 128), (32, 64, 3, 3), (1, 1), (1, 1), (1, 1), 1, False, (0, 0)),
    ((8, 32, 16, 16), (16, 32, 3, 3), (1, 1), (1, 1), (1, 1), 1, False, (0, 0)),
    (
        (8, 32, 16, 16, 16),
        (16, 32, 3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        1,
        False,
        (0, 0, 0),
    ),
]

EXTRA_SHAPES = [
    ((16, 32, 24, 24), (32, 32, 3, 3), (2, 2), (1, 1), (1, 1), 1, False, (0, 0)),
    ((16, 32, 24, 24), (32, 1, 3, 3), (1, 1), (1, 1), (1, 1), 32, False, (0, 0)),
    (
        (4, 8, 20, 320, 15),
        (16, 8, 3, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        1,
        False,
        (0, 0, 0),
    ),
    ((32, 64, 128, 128), (64, 32, 3, 3), (2, 2), (1, 1), (1, 1), 1, True, (1, 1)),
]


def _case_fn(shape, dtype):
    del dtype
    (
        input_shape,
        weight_shape,
        stride,
        padding,
        dilation,
        groups,
        transposed,
        output_padding,
    ) = shape
    yield base.BenchmarkCasePlan(
        shape={"input": input_shape, "weight": weight_shape},
        params={
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "transposed": transposed,
            "output_padding": output_padding,
            "bias": True,
        },
        builder_args=(
            input_shape,
            weight_shape,
            stride,
            padding,
            dilation,
            groups,
            transposed,
            output_padding,
        ),
    )


def _build_inputs_fn(plan, dtype, device):
    (
        input_shape,
        weight_shape,
        stride,
        padding,
        dilation,
        groups,
        transposed,
        output_padding,
    ) = plan.builder_args
    inp = utils.generate_tensor_input(input_shape, dtype, device)
    weight = utils.generate_tensor_input(weight_shape, dtype, device)
    bias = utils.generate_tensor_input(
        (weight_shape[1] * groups if transposed else weight_shape[0],), dtype, device
    )
    return (
        inp,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        {},
    )


class ConvolutionBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        shapes = list(CORE_SHAPES)
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            shapes.extend(EXTRA_SHAPES)
        super().set_shapes(shape_file_path, default_shapes=shapes)


@pytest.mark.convolution
def test_convolution():
    bench = ConvolutionBenchmark(
        op_name="convolution",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.convolution,
        gems_op=getattr(flag_gems, "convolution", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
