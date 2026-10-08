# Copyright 2026 FlagOS Contributors.
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

# The aten op `_fake_quantize_learnable_per_channel_affine_backward` is always
# computed in float32 (see its schema). We benchmark a representative set of
# per-channel shapes where the quantization axis carries a moderate number of
# channels, with the saturation factor tuned so that some channels saturate
# (exercising the saturation branch of the backward kernel).
ATEN_OP = torch.ops.aten._fake_quantize_learnable_per_channel_affine_backward

# (shape, axis, quant_min, quant_max, grad_factor)
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


class FakeQuantizeLearnablePerChannelAffineBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = BENCH_CASES

    def get_input_iter(self, cur_dtype):
        for shape, axis, quant_min, quant_max, grad_factor in self.shapes:
            C = shape[axis]
            scale = torch.rand(C, dtype=cur_dtype, device=self.device) * 0.4 + 0.05
            zero_point = torch.rand(C, dtype=cur_dtype, device=self.device) * (
                quant_max * 0.5
            )
            sat_factor = float(quant_max) * 0.05
            grad = torch.randn(shape, dtype=cur_dtype, device=self.device)
            self_t = (
                torch.randn(shape, dtype=cur_dtype, device=self.device) * sat_factor
            )
            yield grad, self_t, scale, zero_point, axis, quant_min, quant_max, grad_factor

    def get_tflops(self, op, *args, **kwargs):
        # `args` = (grad, self, scale, zero_point, axis, qmin, qmax, grad_factor)
        grad = args[0]
        numel = 1
        for s in grad.shape:
            numel *= s
        return torch.tensor(numel).item()


@pytest.mark.fake_quantize_learnable_per_channel_affine_backward
def test_fake_quantize_learnable_per_channel_affine_backward():
    bench = FakeQuantizeLearnablePerChannelAffineBackwardBenchmark(
        op_name="fake_quantize_learnable_per_channel_affine_backward",
        torch_op=ATEN_OP,
        # The aten op operates in float32 only.
        dtypes=[torch.float32],
    )
    bench.run()
