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

from . import base, consts


class BenchmarkFakeQuantizePerChannelAffineCachemask(base.Benchmark):
    axis_configs = (0, 1)
    DEFAULT_SHAPES = [
        (4, 4),
        (64, 64),
        (128, 256),
        (512, 512),
        (1024, 1024),
        (2, 3, 128, 128),
        (8, 16, 64, 64),
    ]

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            for axis in self.axis_configs:
                input = torch.randn(shape, dtype=dtype, device=self.device)
                channels = shape[axis]
                scale = (
                    torch.rand(channels, dtype=torch.float32, device=self.device) + 0.1
                )
                zero_point = torch.zeros(
                    channels, dtype=torch.int32, device=self.device
                )
                yield input, scale, zero_point, axis, 0, 255


class BenchmarkFakeQuantizePerChannelAffineCachemaskOut(
    BenchmarkFakeQuantizePerChannelAffineCachemask
):
    def get_input_iter(self, dtype):
        for args in super().get_input_iter(dtype):
            input = args[0]
            out0 = torch.empty_like(input)
            out1 = torch.empty_like(input, dtype=torch.bool)
            yield *args, {"out0": out0, "out1": out1}


@pytest.mark.fake_quantize_per_channel_affine_cachemask
def test_fake_quantize_per_channel_affine_cachemask():

    bench = BenchmarkFakeQuantizePerChannelAffineCachemask(
        op_name="fake_quantize_per_channel_affine_cachemask",
        torch_op=torch.ops.aten.fake_quantize_per_channel_affine_cachemask,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.fake_quantize_per_channel_affine_cachemask_out
def test_fake_quantize_per_channel_affine_cachemask_out():
    bench = BenchmarkFakeQuantizePerChannelAffineCachemaskOut(
        op_name="fake_quantize_per_channel_affine_cachemask_out",
        torch_op=torch.ops.aten.fake_quantize_per_channel_affine_cachemask.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
