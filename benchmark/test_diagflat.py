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

from typing import Generator

import pytest
import torch

from . import base, consts
from .conftest import Config

# ``diagflat`` allocates an (m, m) output for an n-element input, so cost grows
# as n ** 2. The sizes stay at or below 4096 to keep the largest output at
# 16 M elements. Both a flat and a multi-dimensional input are included because
# the op flattens first; the ranks only exercise the reshape.
CORE_CASES = [
    ((64,), 0),
    ((64,), 3),
    ((256,), 0),
    ((256,), 3),
    ((1024,), 0),
    ((1024,), 3),
    ((2048,), 0),
    ((2048,), 3),
    ((4096,), 0),
    ((4096,), 3),
    ((32, 32), 0),
    ((32, 32), 3),
    ((64, 64), 0),
    ((64, 64), 3),
    ((16, 16, 16), 0),
    ((16, 16, 16), 3),
]

# Negative offsets take the mirrored address path, and the 8192 entry pushes the
# output to 64 M elements, well past the fused-tile crossover.
COMPREHENSIVE_CASES = [
    ((64,), -3),
    ((1024,), -3),
    ((4096,), -3),
    ((8192,), 0),
    ((8192,), 3),
    ((128, 64), 0),
]


class DiagflatBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CORE_CASES)
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes.extend(COMPREHENSIVE_CASES)
        self.shape_desc = "input shape, diagonal offset"

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape, offset in self.shapes:
            inp = base.generate_tensor_input(shape, cur_dtype, self.device)
            yield inp, offset


@pytest.mark.diagflat
def test_diagflat_perf():
    bench = DiagflatBenchmark(
        op_name="diagflat",
        torch_op=torch.diagflat,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
