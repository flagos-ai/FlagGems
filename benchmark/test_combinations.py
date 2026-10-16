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

# (n, r) pairs: n is the input vector length, r the combination width. The
# default shape set cannot be reused, because the output has C(n, r) rows and
# the row count explodes with r -- a 1024-long vector at r=4 would already be
# 4.6e10 rows. Each entry below keeps the output under ~2e7 rows while still
# covering the r == 2 and r == 3 closed-form kernels and the generic
# binary-search kernel used for r >= 4.
CORE_CASES = [
    (4096, 2),
    (8192, 2),
    (512, 3),
    (1024, 3),
    (128, 4),
    (64, 5),
]

COMPREHENSIVE_CASES = [
    (16384, 2),
    (32768, 2),
    (2048, 3),
    (4096, 3),
    (256, 4),
    (96, 5),
]


class CombinationsBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CORE_CASES)
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes.extend(COMPREHENSIVE_CASES)
        self.shape_desc = "input vector length n, combination width r"

    def get_input_iter(self, cur_dtype) -> Generator:
        for n, r in self.shapes:
            inp = base.generate_tensor_input((n,), cur_dtype, self.device)
            yield inp, r


@pytest.mark.combinations
def test_combinations():
    bench = CombinationsBenchmark(
        op_name="combinations",
        torch_op=torch.combinations,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
