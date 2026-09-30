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

"""Benchmark for ``aten::mkldnn_linear_backward_weights``.

CPU-only operator (MkldnnCPU dispatch key): torch_op (the perf baseline) and
gems_op are timed through the same call
``op(grad_output, input, weight, bias_defined=...)`` on mkldnn-layout CPU
grad_output/input and a dense float32 CPU weight.
"""

import math

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

OUT_FEATURES = 8

# Native matrix shapes supplement the shared shape grid.
# (20, 320, 15) keeps the rank-3 reshape path in the suite.
LBW_SHAPES = [
    (1024, 1024),
    (4096, 4096),
    (8192, 1024),
    (20, 320, 15),
]


def _case_fn(shape, dtype):
    del dtype  # the weight stays float32 on both paths
    for bias_defined in (True, False):
        yield base.BenchmarkCasePlan(
            shape={
                "grad_output": (*shape[:-1], OUT_FEATURES),
                "input": shape,
                "weight": (OUT_FEATURES, shape[-1]),
            },
            params={"bias_defined": bias_defined},
            builder_args=(shape,),
        )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    del device  # oneDNN operands are CPU tensors.
    grad_output = torch.empty((*shape[:-1], OUT_FEATURES), dtype=dtype).to_mkldnn()
    inp = torch.empty(shape, dtype=dtype).to_mkldnn()
    weight = torch.empty((OUT_FEATURES, shape[-1]), dtype=torch.float32)
    # Flat positional args plus a trailing kwargs dict, matching
    # unpack_to_args_kwargs: op(grad_output, input, weight, bias_defined=...).
    return grad_output, inp, weight, {"bias_defined": plan.params["bias_defined"]}


class MkldnnLinearBackwardWeightsBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark with a CPU-sized native-legal shape set."""

    DEFAULT_SHAPE_DESC = "B, K"

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        shapes = []
        for shape in list(self.shapes) + LBW_SHAPES:
            shape = tuple(shape)
            if not shape:
                continue  # No rank-0 oneDNN operand exists.
            if len(shape) == 1:
                # The native backward needs rank >= 2. Preserve the shared
                # element count with a feature width dividing it, avoiding an
                # enormous one-row weight for large pointwise shape entries.
                features = math.gcd(shape[0], 1024)
                shape = (shape[0] // features, features)
            shapes.append(shape)
        self.shapes = list(dict.fromkeys(shapes))


@pytest.mark.mkldnn_linear_backward_weights
def test_mkldnn_linear_backward_weights():
    bench = MkldnnLinearBackwardWeightsBenchmark(
        op_name="mkldnn_linear_backward_weights",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.mkldnn_linear_backward_weights,
        gems_op=getattr(flag_gems, "mkldnn_linear_backward_weights", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
