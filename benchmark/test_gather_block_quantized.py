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

# GatherBlockQuantized Benchmark
GATHER_BLOCK_QUANTIZED_SHAPES = [
    (1024,),
    (2048,),
    (4096,),
    (8192,),
    (16384,),
]


class GatherBlockQuantizedBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = GATHER_BLOCK_QUANTIZED_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            n_elements = shape[0]
            block_size = 128
            n_blocks = (n_elements + block_size - 1) // block_size

            # Create quantized data and scales
            quantized_data = torch.randint(
                -100, 100, shape, dtype=torch.int8, device=self.device
            )
            scales = (
                torch.rand(n_blocks, dtype=torch.float32, device=self.device) * 2 + 0.5
            )

            # Dequantize all values (no indices) using the composed aten baseline
            yield quantized_data, scales, None, block_size


def _aten_gather_block_quantized(quantized_data, scales, indices, block_size):
    """Reference composed from native aten ops (no native block-quantized
    gather exists in PyTorch): expand per-block scales to element level,
    then dequantize."""
    scale_expanded = scales.repeat_interleave(block_size)
    scale_expanded = scale_expanded[: quantized_data.numel()]
    return quantized_data.float() * scale_expanded


@pytest.mark.gather_block_quantized
def test_gather_block_quantized():
    bench = GatherBlockQuantizedBenchmark(
        op_name="gather_block_quantized",
        torch_op=_aten_gather_block_quantized,
        # gather_block_quantized consumes int8 data and float32 scales.
        dtypes=[torch.float32],
    )
    bench.run()
