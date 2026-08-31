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

from . import base, consts, utils

FILL_DIAGONAL_SHAPES = [
    ((4, 4), False),
    ((512, 512), False),
    ((4096, 4096), False),
    ((65536, 128), True),
    ((262144, 32), True),
    ((1048576, 8), True),
    ((64, 64, 64), False),
]


class FillDiagonalBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "input shape, wrap"

    def set_shapes(self, shape_file_path=None):
        self.shapes = FILL_DIAGONAL_SHAPES

    def get_input_iter(self, dtype) -> Generator:
        for shape, wrap in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            yield inp, 5.0, {"wrap": wrap}


@pytest.mark.fill_diagonal_
def test_fill_diagonal_():
    bench = FillDiagonalBenchmark(
        op_name="fill_diagonal_",
        torch_op=torch.Tensor.fill_diagonal_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
