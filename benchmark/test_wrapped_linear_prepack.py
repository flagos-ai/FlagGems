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

from . import base
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(4, 8), (64, 128), (256, 256), (512, 1024), (2048, 1024)]
_PACK_MAGIC = 260821.0
_PACK_VERSION = 1.0


def _torch_reference(weight, weight_scale, weight_zero_point, bias):
    quantized = torch.clamp(
        torch.round(weight / weight_scale) + weight_zero_point, -128, 127
    ).to(torch.int8)
    metadata = torch.stack(
        (
            weight_scale.to(torch.float32),
            weight_zero_point.to(torch.float32),
            weight_scale.new_tensor(_PACK_MAGIC),
            weight_scale.new_tensor(_PACK_VERSION),
        )
    )
    return torch.cat(
        (
            metadata.contiguous().view(torch.uint8),
            bias.to(torch.float32).contiguous().view(torch.uint8),
            quantized.contiguous().view(torch.uint8).flatten(),
        )
    )


class WrappedLinearPrepackBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "OUT_FEATURES, IN_FEATURES"


def _case_fn(shape, dtype):
    del dtype
    out_features, in_features = shape
    yield base.BenchmarkCasePlan(
        shape={"weight": [out_features, in_features], "bias": [out_features]},
        params={"scale": 0.03125, "zero_point": -3},
        builder_args=(out_features, in_features),
    )


def _build_inputs_fn(plan, dtype, device):
    del dtype
    out_features, in_features = plan.builder_args
    torch.manual_seed(0)
    return (
        torch.randn((out_features, in_features), dtype=torch.float32, device=device),
        torch.tensor(0.03125, dtype=torch.float32, device=device),
        torch.tensor(-3, dtype=torch.int64, device=device),
        torch.randn(out_features, dtype=torch.float32, device=device),
    )


@pytest.mark.wrapped_linear_prepack
@pytest.mark.underscore_wrapped_linear_prepack
def test_wrapped_linear_prepack_benchmark():
    WrappedLinearPrepackBenchmark(
        op_name="_wrapped_linear_prepack",
        torch_op=_torch_reference,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    ).run()
