# Copyright 2026, The FlagOS Contributors.
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

# Representative quantization shapes: 2D weight/activation tensors and a couple
# of higher-dim activations. The channel axis is the last one, so the number of
# scales/zero points grows with the innermost dimension.
QUANT_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
]


def _ref(inp, scales, zero_points, axis):
    return torch.ops.aten._make_per_channel_quantized_tensor(
        inp, scales, zero_points, axis
    )


def _int_input(shape, dtype, device):
    if dtype == torch.uint8:
        return torch.randint(0, 256, shape, dtype=dtype, device=device)
    if dtype == torch.int32:
        return torch.randint(-1000, 1000, shape, dtype=dtype, device=device)
    return torch.randint(-128, 128, shape, dtype=dtype, device=device)


def _pcq_input_fn(shape, dtype, device):
    """Yield (input, scales, zero_points, axis) for the benchmark harness."""
    axis = len(shape) - 1
    n = shape[axis]
    inp = _int_input(shape, dtype, device)
    scales = torch.rand(n, device=device) + 0.01
    zero_points = torch.randint(0, 10, (n,), device=device, dtype=torch.int64)
    yield inp, scales, zero_points, axis


class MakePerChannelQuantizedTensorBenchmark(base.GenericBenchmark):
    """Flat integer copy into per-channel quantized storage."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = QUANT_SHAPES

    def set_more_shapes(self):
        return []


@pytest.mark.make_per_channel_quantized_tensor
def test_make_per_channel_quantized_tensor():
    bench = MakePerChannelQuantizedTensorBenchmark(
        op_name="make_per_channel_quantized_tensor",
        torch_op=_ref,
        input_fn=_pcq_input_fn,
        dtypes=[torch.uint8, torch.int8, torch.int32],
    )
    bench.run()
