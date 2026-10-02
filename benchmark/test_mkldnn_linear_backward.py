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

import math
from itertools import product

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# oneDNN linear backward with M = prod(shape[:-1]), K = shape[-1] and N = number
# of output features. The oneDNN operands are CPU-only, so the builders convert
# to the mkldnn layout; core_shapes.yaml has no entry for this operator, so these
# defaults (or a --shape-file override) are what gets benchmarked.
_MKLDNN_LINEAR_BACKWARD_SHAPES = [
    (4, 8),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (20, 320, 15),
    (16, 128, 64, 60),
]
_OUT_FEATURES = 8
_OUTPUT_MASK = [True, True, True]


def _case_fn(shape, dtype):
    del dtype
    masks = (
        list(product([False, True], repeat=3)) if shape == (4, 8) else [_OUTPUT_MASK]
    )
    for mask in masks:
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"N": _OUT_FEATURES, "output_mask": list(mask)},
            builder_args=(shape, _OUT_FEATURES),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, out_features = plan.builder_args
    rows = 1
    for dim in shape[:-1]:
        rows *= dim
    # self/grad_output are oneDNN tensors; the weight must be dense fp32.
    self_mkldnn = torch.empty(shape, dtype=dtype).to_mkldnn()
    grad_out_mkldnn = torch.empty((rows, out_features), dtype=dtype).to_mkldnn()
    weight = torch.empty((out_features, shape[-1]), dtype=torch.float32)
    return self_mkldnn, grad_out_mkldnn, weight, list(plan.params["output_mask"])


class MkldnnLinearBackwardBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark restricted to oneDNN-legal shapes."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        shapes = []
        for shape in list(self.shapes) + _MKLDNN_LINEAR_BACKWARD_SHAPES:
            shape = tuple(shape)
            if not shape:
                continue  # No rank-zero oneDNN operand exists.
            if len(shape) == 1:
                # Preserve generic element counts in native rank-two geometry.
                features = math.gcd(shape[0], 1024)
                shape = (shape[0] // features, features)
            shapes.append(shape)
        self.shapes = list(dict.fromkeys(shapes))


@pytest.mark.mkldnn_linear_backward
def test_mkldnn_linear_backward():
    bench = MkldnnLinearBackwardBenchmark(
        op_name="mkldnn_linear_backward",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.mkldnn_linear_backward,
        gems_op=getattr(flag_gems, "mkldnn_linear_backward", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
