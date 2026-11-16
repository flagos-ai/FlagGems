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

# Each entry is the pair of operand lengths (M, N); the output is M * N, so cost
# grows with the product rather than with either length. The pairs straddle the
# 1 << 22 element cutoff in _forward_config, so both tile geometries are
# measured, and the lopsided ones hold the product fixed while skewing the
# aspect ratio, which is what the program-order swizzle responds to.
CORE_CASES = [
    (1, 4096),
    (4096, 1),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (8192, 4096),
    (4096, 8192),
    (16384, 1024),
    (1024, 16384),
]

# Small shapes where the launch dominates, and a pair whose lengths divide
# neither tile size so every program takes the masked path.
COMPREHENSIVE_CASES = [
    (64, 64),
    (256, 256),
    (4095, 4097),
    (8192, 8192),
    (65536, 256),
]


class GerBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # A shape entry is a pair of 1-D operand lengths rather than one tensor
        # shape, so the cases are listed explicitly instead of being derived
        # from the shared shape file.
        self.shapes = list(CORE_CASES)
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes.extend(COMPREHENSIVE_CASES)
        self.shape_desc = "self length, vec2 length"

    def get_input_iter(self, cur_dtype) -> Generator:
        for m, n in self.shapes:
            inp = base.generate_tensor_input((m,), cur_dtype, self.device)
            vec2 = base.generate_tensor_input((n,), cur_dtype, self.device)
            yield inp, vec2


@pytest.mark.ger
def test_ger_perf():
    bench = GerBenchmark(
        op_name="ger",
        torch_op=torch.ger,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
