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


@pytest.mark.fake_quantize_per_channel_affine
def test_fake_quantize_per_channel_affine():
    class BenchmarkFakeQuantizePerChannelAffine(base.Benchmark):
        """
        Benchmark fake_quantize_per_channel_affine operator
        """

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

        def get_case_iter(self, dtype):
            ordinal = 0
            for shape in self.shapes:
                for axis in self.axis_configs:
                    if axis >= len(shape):
                        continue
                    yield self._case_from_plan(
                        dtype,
                        ordinal,
                        base.BenchmarkCasePlan(
                            shape={"input": list(shape)},
                            params={
                                "axis": axis,
                                "quant_min": 0,
                                "quant_max": 255,
                            },
                            builder_args=(shape, axis),
                        ),
                    )
                    ordinal += 1

        def build_inputs(self, case):
            plan = case.builder_args[0]
            shape, axis = plan.builder_args
            inp = torch.randn(shape, dtype=case.dtype, device=flag_gems.device)
            n_channels = shape[axis]
            scale = (
                torch.rand(n_channels, dtype=torch.float32, device=flag_gems.device) * 0.1 + 0.01
            )
            zero_point = torch.zeros(n_channels, dtype=torch.int32, device=flag_gems.device)
            quant_min = 0
            quant_max = 255
            return inp, scale, zero_point, axis, quant_min, quant_max

        def forward(self, inp, scale, zero_point, axis, quant_min, quant_max):
            return torch.fake_quantize_per_channel_affine(
                inp, scale, zero_point, axis, quant_min, quant_max
            )

    bench = BenchmarkFakeQuantizePerChannelAffine(
        op_name="fake_quantize_per_channel_affine",
        torch_op=torch.fake_quantize_per_channel_affine,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
