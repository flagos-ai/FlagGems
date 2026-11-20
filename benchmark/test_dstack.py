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

# Each entry is one tensor list. The 2-D entries promote to depth 1 per input,
# so they take the interleaving path; the 3-D entries have depth > 1 and take
# the block-copy path. The last entry has unequal depths, which exercises the
# running output offset.
CORE_CASES = [
    [(64, 64), (64, 64), (64, 64)],
    [(4096, 4096), (4096, 4096), (4096, 4096)],
    [(64, 512, 512), (64, 512, 512), (64, 512, 512)],
    [(64, 512, 512), (64, 512, 768), (64, 512, 256)],
]

# Adds a two-input list, a four-input list (the largest the interleaving path
# accepts), a 1-D list, and a rank-5 list where the inner extent is above 1.
COMPREHENSIVE_CASES = [
    [(1024, 1024), (1024, 1024)],
    [(512, 512), (512, 512), (512, 512), (512, 512)],
    [(1048576,), (1048576,)],
    [(16, 128, 64, 60), (16, 128, 64, 60)],
    [(13, 3, 64, 5, 2), (13, 3, 96, 5, 2), (13, 3, 32, 5, 2)],
]


class DstackBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Each shape entry is a whole tensor list rather than a single tensor
        # shape, so the cases are listed explicitly instead of being derived
        # from the shared shape file.
        self.shapes = [tuple(case) for case in CORE_CASES]
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes.extend(tuple(case) for case in COMPREHENSIVE_CASES)
        self.shape_desc = "shapes of the tensors being stacked"

    def get_input_iter(self, cur_dtype) -> Generator:
        for shapes in self.shapes:
            inputs = [
                base.generate_tensor_input(shape, cur_dtype, self.device)
                for shape in shapes
            ]
            yield (inputs,)


@pytest.mark.dstack
def test_dstack_perf():
    bench = DstackBenchmark(
        op_name="dstack",
        torch_op=torch.dstack,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
