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

# quantize_per_channel is an element-wise quantization with per-channel scale
# and zero_point. The float input is always float32 (PyTorch rejects float16 /
# bfloat16). We benchmark across the quantized output dtypes instead.
QUANT_DTYPES = [torch.quint8, torch.qint8, torch.qint32]

# Representative quantization shapes: 2D weight/activation tensors and a couple
# of higher-dim activations. Kept small enough to avoid OOM while still
# exercising coalesced element-wise throughput.
QUANT_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
]


def _quantize_input_fn(shape, dtype, device):
    """Yield (input, scales, zero_points, axis, dtype) for torch.quantize_per_channel."""
    # `dtype` here is the *quantized* dtype from the benchmark parametrization.
    axis = len(shape) - 1
    n_channels = shape[axis]
    inp = torch.randn(shape, dtype=torch.float32, device=device) * 10.0
    scales = torch.rand(n_channels, device=device) * 0.5 + 0.01
    zero_points = torch.randint(0, 50, (n_channels,), device=device, dtype=torch.int32)
    yield inp, scales, zero_points, axis, dtype


class QuantizePerChannelBenchmark(base.GenericBenchmark):
    """Benchmark for quantize_per_channel using the quantized dtypes."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = QUANT_SHAPES

    def set_more_shapes(self):
        return []


@pytest.mark.quantize_per_channel
def test_quantize_per_channel():
    bench = QuantizePerChannelBenchmark(
        op_name="quantize_per_channel",
        torch_op=torch.quantize_per_channel,
        input_fn=_quantize_input_fn,
        dtypes=QUANT_DTYPES,
    )
    bench.run()


def _quantize_out_input_fn(shape, dtype, device):
    """Yield (input, scales, zero_points, axis, dtype, {'out': out})."""
    axis = len(shape) - 1
    n_channels = shape[axis]
    inp = torch.randn(shape, dtype=torch.float32, device=device) * 10.0
    scales = torch.rand(n_channels, device=device) * 0.5 + 0.01
    zero_points = torch.randint(0, 50, (n_channels,), device=device, dtype=torch.int32)
    # Pre-allocate the per-channel quantized ``out`` tensor via the public API
    # with matching scales/zero_points/axis; the FlagGems out kernel overwrites
    # its integer storage.
    out = torch.quantize_per_channel(
        torch.zeros(shape, dtype=torch.float32, device=device),
        scales.double(),
        zero_points.long(),
        axis,
        dtype,
    )
    yield inp, scales, zero_points, axis, dtype, {"out": out}


class QuantizePerChannelOutBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = QUANT_SHAPES

    def set_more_shapes(self):
        return []


@pytest.mark.quantize_per_channel_out
def test_quantize_per_channel_out():
    bench = QuantizePerChannelOutBenchmark(
        op_name="quantize_per_channel_out",
        torch_op=torch.ops.aten.quantize_per_channel.out,
        input_fn=_quantize_out_input_fn,
        dtypes=QUANT_DTYPES,
    )
    bench.run()
