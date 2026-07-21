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

from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts


def adaptive_avg_pool2d_input_fn(shape, dtype, device):
    inp = base.generate_tensor_input(shape, dtype, device)
    # Common cases - output_size must be (H, W)
    yield inp, (7, 7)
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield inp, (1, 1)
        yield inp, (16, 16)
        yield inp, (14, 14)


class AdaptiveAvgPool2dBenchmark(base.GenericBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        shapes_4d = [
            (4, 3, 32, 32),  # Small input
            (8, 64, 56, 56),  # Medium input
            (16, 128, 28, 28),  # Medium input
            (1, 64, 224, 224),  # Typical image size
            (4, 128, 112, 112),  # Typical intermediate feature map
        ]

        for shape in shapes_4d:
            yield from self.input_fn(shape, cur_dtype, self.device)


@pytest.mark.adaptive_avg_pool2d
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_perf_adaptive_avg_pool2d():
    bench = AdaptiveAvgPool2dBenchmark(
        input_fn=adaptive_avg_pool2d_input_fn,
        op_name="adaptive_avg_pool2d",
        torch_op=torch.ops.aten._adaptive_avg_pool2d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
