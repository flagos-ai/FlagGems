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

from . import base, consts

# Cubic growth shapes for profiling diagonal copy bandwidth at varying sizes,
# plus small uniform shapes to cover edge-case performance
DIAGONAL_COPY_SHAPES = [
    (16, 32, 64),
    (32, 64, 128),
    (64, 128, 256),
    (128, 256, 512),
    (256, 512, 1024),
    (16, 16, 16),
    (32, 32, 32),
]


class DiagonalCopyBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = DIAGONAL_COPY_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield x, 0, 1, 2


@pytest.mark.diagonal_copy
def test_diagonal_copy():
    bench = DiagonalCopyBenchmark(
        op_name="diagonal_copy",
        torch_op=torch.diagonal_copy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
