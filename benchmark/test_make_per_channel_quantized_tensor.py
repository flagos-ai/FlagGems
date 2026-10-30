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

# _make_per_channel_quantized_tensor wraps a raw integer payload with
# per-channel qparams. The payload dtype decides the output quantized dtype,
# so the benchmark always runs the qint8 scheme (int8 payload) -- the only
# quantized dtype in this list (torch.qint8) mirrors the fixed int8 payload;
# no float dtype parametrization applies to this operator.
# Representative quantization shapes: 2D weight/activation tensors and a
# higher-dim activation, kept small enough to avoid OOM.
MPCQ_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
]


def _mpcq_input_fn(shape, dtype, device):
    """Yield (payload, scales, zero_points, axis) for the wrap benchmark."""
    axis = 0
    n_channels = shape[axis]
    # The payload is always int8 (qint8 scheme); INT_DTYPES would parametrize
    # over wider ints whose quantized schemes have no QuantizedCUDA kernel.
    payload = torch.randint(-100, 100, shape, device=device, dtype=torch.int8)
    scales = torch.rand(n_channels, device=device) * 0.5 + 0.01
    zero_points = torch.randint(-40, 40, (n_channels,), device=device, dtype=torch.int8)
    yield payload, scales, zero_points, axis


class MakePerChannelQuantizedTensorBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = MPCQ_SHAPES

    def set_more_shapes(self):
        return []


@pytest.mark.make_per_channel_quantized_tensor
def test_make_per_channel_quantized_tensor():
    bench = MakePerChannelQuantizedTensorBenchmark(
        op_name="make_per_channel_quantized_tensor",
        torch_op=torch.ops.aten._make_per_channel_quantized_tensor,
        input_fn=_mpcq_input_fn,
        dtypes=[torch.qint8],
    )
    bench.run()


def _mpcq_out_input_fn(shape, dtype, device):
    """Yield (payload, scales, zero_points, axis, {'out': out})."""
    axis = 0
    n_channels = shape[axis]
    # The payload is always int8 (qint8 scheme); INT_DTYPES would parametrize
    # over wider ints whose quantized schemes have no QuantizedCUDA kernel.
    payload = torch.randint(-100, 100, shape, device=device, dtype=torch.int8)
    scales = torch.rand(n_channels, device=device) * 0.5 + 0.01
    zero_points = torch.randint(-40, 40, (n_channels,), device=device, dtype=torch.int8)
    # Pre-allocate the per-channel quantized ``out`` through the public
    # quantize_per_channel factory on a zeros tensor with matching qparams;
    # the FlagGems out kernel overwrites its integer storage and adopts the
    # source quantizer (quantized copy_ semantics).
    out = torch.quantize_per_channel(
        torch.zeros(shape, dtype=torch.float32, device=device),
        scales.double(),
        zero_points.long(),
        axis,
        torch.qint8,
    )
    yield payload, scales, zero_points, axis, {"out": out}


class MakePerChannelQuantizedTensorOutBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = MPCQ_SHAPES

    def set_more_shapes(self):
        return []


@pytest.mark.make_per_channel_quantized_tensor_out
def test_make_per_channel_quantized_tensor_out():
    bench = MakePerChannelQuantizedTensorOutBenchmark(
        op_name="make_per_channel_quantized_tensor_out",
        torch_op=torch.ops.aten._make_per_channel_quantized_tensor.out,
        input_fn=_mpcq_out_input_fn,
        dtypes=[torch.qint8],
    )
    bench.run()
