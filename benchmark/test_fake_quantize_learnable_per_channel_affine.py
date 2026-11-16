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

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

PER_CHANNEL_SHAPES = [
    ((8,), 0),
    ((4, 8), 0),
    ((4, 8), 1),
    ((2, 19, 7), 1),
    ((16, 128, 64, 60), 1),
    ((16, 128, 64, 60), 3),
]
QUANT_RANGES = [(0, 255), (-128, 127)]
PER_CHANNEL_CASES = [
    (shape, axis, quant_min, quant_max)
    for shape, axis in PER_CHANNEL_SHAPES
    for quant_min, quant_max in QUANT_RANGES
]


def _case_fn(case, dtype):
    del dtype
    shape, axis, quant_min, quant_max = case
    yield base.BenchmarkCasePlan(
        shape={"self": list(shape), "scale": [shape[axis]], "zero_point": [shape[axis]]},
        params={"axis": axis, "quant_min": quant_min, "quant_max": quant_max, "grad_factor": 1.0},
        builder_args=(tuple(shape), axis, quant_min, quant_max),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, axis, quant_min, quant_max = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    channels = shape[axis]
    scale = torch.rand(channels, dtype=torch.float32, device=device) * 0.5 + 0.01
    zero_point = torch.randn(channels, dtype=torch.float32, device=device)
    return inp, scale, zero_point, axis, quant_min, quant_max, 1.0


class FakeQuantizeLearnablePerChannelAffineBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PER_CHANNEL_CASES


@pytest.mark.fake_quantize_learnable_per_channel_affine
def test_fake_quantize_learnable_per_channel_affine():
    bench = FakeQuantizeLearnablePerChannelAffineBenchmark(
        op_name="fake_quantize_learnable_per_channel_affine",
        torch_op=torch.ops.aten._fake_quantize_learnable_per_channel_affine,
        gems_op=flag_gems._fake_quantize_learnable_per_channel_affine,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
