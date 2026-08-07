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

from . import base, consts


class AsStridedScatterBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Dense, sparse-stride, and one-dimensional scatter layouts.
        self.shapes = [
            (128, 256, 256, 1, 0, 128 * 256),
            (64, 64, 130, 2, 5, 64 * 130 + 2),
            (1, 65536, 131072, 2, 1, 2 * 65536),
        ]
        self.shape_desc = "ROWS, COLS, ROW_STRIDE, COL_STRIDE, OFFSET, STORAGE_SIZE"

    def get_input_iter(self, cur_dtype) -> Generator:
        for rows, cols, row_stride, col_stride, offset, storage_size in self.shapes:
            inp = torch.randn(storage_size, dtype=cur_dtype, device=self.device)
            src = torch.randn((rows, cols), dtype=cur_dtype, device=self.device)
            yield inp, src, (rows, cols), (row_stride, col_stride), offset


@pytest.mark.as_strided_scatter
def test_as_strided_scatter():
    bench = AsStridedScatterBenchmark(
        op_name="as_strided_scatter",
        torch_op=torch.ops.aten.as_strided_scatter,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
