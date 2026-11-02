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

"""Benchmark for ``aten::chain_matmul``.

Each shape is a chain of 2-D matrices; the measured operator is the whole
chain product, so the cost covers every intermediate matmul the optimal
parenthesization selects.
"""

import pytest
import torch

import flag_gems

from . import base, consts

CHAIN_SHAPES = [
    [(256, 256), (256, 256)],
    [(128, 256), (256, 128), (128, 64)],
    [(64, 128), (128, 192), (192, 96), (96, 64)],
    [(256, 128), (128, 256), (256, 64), (64, 128), (128, 128)],
]


class ChainMatmulBenchmark(base.Benchmark):
    """Benchmark for ``aten::chain_matmul`` over a matrix-chain grid."""

    DEFAULT_SHAPE_DESC = "matrix chain shapes"

    def set_shapes(self, shape_file_path=None):
        self.shapes = CHAIN_SHAPES

    def get_input_iter(self, cur_dtype):
        for shapes in self.shapes:
            yield (
                [
                    torch.randn(shape, dtype=cur_dtype, device=self.device)
                    for shape in shapes
                ],
            )


@pytest.mark.chain_matmul
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_chain_matmul(dtype):
    bench = ChainMatmulBenchmark(
        op_name="chain_matmul",
        torch_op=torch.ops.aten.chain_matmul,
        gems_op=flag_gems.chain_matmul,
        dtypes=[dtype],
    )
    bench.run()
