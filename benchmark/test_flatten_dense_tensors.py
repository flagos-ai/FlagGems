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

# Each entry is one tensor list. The pairs and triples take the 2-D grid
# kernel; the five-input lists take the metadata fallback.
CORE_CASES = [
    [(4096, 4096), (4096, 4096)],
    [(1024, 1024), (1024, 1024), (1024, 1024)],
    [(4, 1024, 1024), (4, 1024, 1024)],
    [(1048576,), (1048576,), (1048576,), (1048576,), (1048576,)],
]

# Adds a four-input list (the largest the 2-D grid kernel accepts), unequal
# sizes, and a single-tensor case that takes the zero-copy view path.
COMPREHENSIVE_CASES = [
    [(512, 512), (512, 512), (512, 512), (512, 512)],
    [(8, 512, 512), (8, 512, 768), (8, 512, 256)],
    [(16, 128, 64, 60), (16, 128, 64, 60), (16, 128, 64, 60)],
    [(2048, 2048)],
]


class FlattenDenseTensorsBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Each shape entry is a whole tensor list rather than a single tensor
        # shape, so the cases are listed explicitly instead of being derived
        # from the shared shape file.
        self.shapes = [tuple(case) for case in CORE_CASES]
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes.extend(tuple(case) for case in COMPREHENSIVE_CASES)
        self.shape_desc = "shapes of the tensors being flattened"

    def get_input_iter(self, cur_dtype) -> Generator:
        for shapes in self.shapes:
            inputs = [
                base.generate_tensor_input(shape, cur_dtype, self.device)
                for shape in shapes
            ]
            yield (inputs,)


@pytest.mark.flatten_dense_tensors
def test_flatten_dense_tensors_perf():
    bench = FlattenDenseTensorsBenchmark(
        op_name="flatten_dense_tensors",
        torch_op=torch.ops.aten.flatten_dense_tensors.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
