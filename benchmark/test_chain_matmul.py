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

import pytest
import torch

from . import base, consts
from .conftest import Config

# Chain dimensions (d0, d1, ..., dn): n matrices of shapes (d0, d1) ... and an
# (d0, dn) output. The default shape set cannot be reused because a chain is
# described by a dimension list rather than one tensor shape. The narrow inner
# dimensions are what make the optimal parenthesization non-trivial.
CORE_CHAINS = [
    (64, 256, 128),
    (256, 64, 512),
    (128, 512, 64, 256),
    (512, 64, 256, 128),
    (128, 256, 64, 512, 128),
    (256, 128, 512, 64, 256),
]

COMPREHENSIVE_CHAINS = [
    (1024, 64, 1024),
    (2048, 32, 2048, 32),
    (512, 2048, 64, 1024),
    (1024, 128, 2048, 64, 512),
    (256, 1024, 64, 2048, 128, 512),
    (512, 128, 1024, 64, 2048, 32, 512),
]


class ChainMatmulBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CORE_CHAINS)
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes.extend(COMPREHENSIVE_CHAINS)
        self.shape_desc = "matrix-chain dimensions (d0, d1, ..., dn)"

    def get_input_iter(self, cur_dtype):
        for dimensions in self.shapes:
            tensors = [
                torch.randn(
                    dimensions[index],
                    dimensions[index + 1],
                    dtype=cur_dtype,
                    device=self.device,
                )
                for index in range(len(dimensions) - 1)
            ]
            yield (tensors,)


@pytest.mark.chain_matmul
def test_chain_matmul():
    bench = ChainMatmulBenchmark(
        op_name="chain_matmul",
        torch_op=torch.ops.aten.chain_matmul.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
