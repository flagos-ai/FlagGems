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
from .conftest import Config

CORE_SHAPES = [
    (2, 2),
    (3, 2),
    (2, 3),
    (4, 4),
    (8, 4),
    (4, 8),
    (256, 2),
    (2, 256),
    (32, 4, 4),
]
COMPREHENSIVE_SHAPES = [
    (1024, 2),
    (2, 1024),
    (256, 4),
    (4, 256),
    (128, 4, 4),
]


class PinverseBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CORE_SHAPES)
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += self.set_more_shapes()

    def set_more_shapes(self):
        return list(COMPREHENSIVE_SHAPES)

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield (torch.randn(shape, dtype=cur_dtype, device=self.device),)


@pytest.mark.pinverse
def test_pinverse():
    bench = PinverseBenchmark(
        op_name="pinverse",
        torch_op=torch.pinverse,
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.pinverse)
    bench.run()
