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


@pytest.mark.fake_quantize_per_channel_affine_cachemask
def test_fake_quantize_per_channel_affine_cachemask():
    class BenchmarkFakeQuantizePerChannelAffineCachemask(base.Benchmark):
        def set_more_shapes(self):
            self.shapes = [
                (4, 4),
                (64, 64),
                (128, 256),
                (512, 512),
                (1024, 1024),
                (2, 3, 128, 128),
                (8, 16, 64, 64),
            ]
            self.axis_configs = [0, 1]

        def get_input_iter(self, dtype):
            for shape in self.shapes:
                for axis in self.axis_configs:
                    input = torch.randn(shape, dtype=dtype, device=self.device)
                    channels = shape[axis]
                    scale = (
                        torch.rand(channels, dtype=torch.float32, device=self.device)
                        + 0.1
                    )
                    zero_point = torch.zeros(
                        channels, dtype=torch.int32, device=self.device
                    )
                    yield input, scale, zero_point, axis, 0, 255

    bench = BenchmarkFakeQuantizePerChannelAffineCachemask(
        op_name="fake_quantize_per_channel_affine_cachemask",
        torch_op=torch.ops.aten.fake_quantize_per_channel_affine_cachemask,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
