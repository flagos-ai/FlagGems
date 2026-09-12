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
from flag_gems.ops._wrapped_linear_prepack import unpack_linear_weight

from . import base


def torch_wrapped_quantized_linear_prepacked(
    input,
    input_scale,
    input_zero_point,
    packed_weight,
    output_scale,
    output_zero_point,
    out_channel,
):
    weight, metadata, bias = unpack_linear_weight(packed_weight, out_channel)
    quantized_input = torch.clamp(
        torch.round(input / input_scale) + input_zero_point, 0, 255
    )
    real_output = torch.matmul(
        quantized_input - input_zero_point,
        (weight.to(torch.float32) - metadata[1]).T,
    )
    real_output = real_output * input_scale * metadata[0] + bias
    quantized_output = torch.clamp(
        torch.round(real_output / output_scale) + output_zero_point, 0, 255
    )
    return (quantized_output - output_zero_point) * output_scale


class WrappedQuantizedLinearPrepackedBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (2, 4, 8),
            (16, 64, 128),
            (64, 256, 256),
            (128, 512, 1024),
            (512, 1024, 1024),
        ]

    def get_input_iter(self, dtype):
        for M, N, K in self.shapes:
            input = torch.randn((M, K), dtype=dtype, device=self.device)
            weight = torch.randn((N, K), dtype=dtype, device=self.device)
            bias = torch.randn((N,), dtype=dtype, device=self.device)
            input_scale = torch.tensor(0.05, dtype=dtype, device=self.device)
            input_zero_point = torch.tensor(127, device=self.device)
            weight_scale = torch.tensor(0.03125, dtype=dtype, device=self.device)
            weight_zero_point = torch.tensor(-3, device=self.device)
            output_scale = torch.tensor(0.08, dtype=dtype, device=self.device)
            output_zero_point = torch.tensor(121, device=self.device)
            packed_weight = flag_gems._wrapped_linear_prepack(
                weight, weight_scale, weight_zero_point, bias
            )
            yield (
                input,
                input_scale,
                input_zero_point,
                packed_weight,
                output_scale,
                output_zero_point,
                N,
            )


@pytest.mark.wrapped_quantized_linear_prepacked
def test_wrapped_quantized_linear_prepacked_benchmark():
    bench = WrappedQuantizedLinearPrepackedBenchmark(
        op_name="wrapped_quantized_linear_prepacked",
        torch_op=torch_wrapped_quantized_linear_prepacked,
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems._wrapped_quantized_linear_prepacked)
    bench.run()
