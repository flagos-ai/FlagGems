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

# Each entry lists the lengths of the 1-D input vectors, so the output has
# prod(entry) rows and len(entry) columns. The default shape set cannot be
# reused here: the cartesian product of three 1024-long vectors would already
# allocate 2**30 rows.
CARTESIAN_PROD_SHAPES = [
    (4096,),
    (1024, 1024),
    (64, 64, 64),
    (32, 256, 32),
    (16, 128, 64, 16),
    (4, 512, 8, 256),
]


class CartesianProdBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CARTESIAN_PROD_SHAPES
        self.shape_desc = "input vector lengths"

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            yield tuple(
                base.generate_tensor_input((size,), cur_dtype, self.device)
                for size in shape
            )


@pytest.mark.cartesian_prod
def test_cartesian_prod():
    bench = CartesianProdBenchmark(
        op_name="cartesian_prod",
        torch_op=torch.cartesian_prod,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
