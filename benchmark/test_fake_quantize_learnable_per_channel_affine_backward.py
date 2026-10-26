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

BENCH_CASES = [
    ((64, 1024), 0, -128, 127, 1.0),
    ((1024, 64), 1, -128, 127, 1.0),
    ((256, 512, 32), 1, -128, 127, 1.0),
    ((128, 256, 128), 2, -128, 127, 1.0),
    ((64, 1024), 0, 0, 255, 1.0),
    ((4096, 4096), 1, -128, 127, 1.0),
    ((16, 128, 64, 60), 2, -128, 127, 1.0),
    ((1024, 1024), 0, -7, 7, 1.0),
]


def _case_fn(case, dtype):
    del dtype
    shape, axis, quant_min, quant_max, grad_factor = case
    yield base.BenchmarkCasePlan(
        shape={"grad": list(shape), "self": list(shape), "scale": [shape[axis]], "zero_point": [shape[axis]]},
        params={"axis": axis, "quant_min": quant_min, "quant_max": quant_max, "grad_factor": grad_factor},
        builder_args=(tuple(shape), axis, quant_min, quant_max, grad_factor),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, axis, quant_min, quant_max, grad_factor = plan.builder_args
    torch.manual_seed(0)
    channels = shape[axis]
    scale = torch.rand(channels, dtype=dtype, device=device) * 0.4 + 0.05
    zero_point = torch.rand(channels, dtype=dtype, device=device) * (quant_max * 0.5)
    grad = torch.randn(shape, dtype=dtype, device=device)
    self_t = torch.randn(shape, dtype=dtype, device=device) * (float(quant_max) * 0.05)
    return grad, self_t, scale, zero_point, axis, quant_min, quant_max, grad_factor


class FakeQuantizeLearnablePerChannelAffineBackwardBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = BENCH_CASES

    def get_tflops(self, op, *args, **kwargs):
        del op, kwargs
        return args[0].numel()


@pytest.mark.fake_quantize_learnable_per_channel_affine_backward
@pytest.mark.underscore_fake_quantize_learnable_per_channel_affine_backward
def test_fake_quantize_learnable_per_channel_affine_backward():
    bench = FakeQuantizeLearnablePerChannelAffineBackwardBenchmark(
        op_name="_fake_quantize_learnable_per_channel_affine_backward",
        torch_op=torch.ops.aten._fake_quantize_learnable_per_channel_affine_backward,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    )
    bench.run()
