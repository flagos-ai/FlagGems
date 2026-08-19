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

from . import base


class CholeskySolveHelperBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Small-to-medium square systems exercise the sequential triangular solve.
        self.shapes = [(4, 1), (8, 2), (16, 4), (32, 4), (64, 8)]
        self.shape_desc = "N, NRHS"

    def get_input_iter(self, cur_dtype) -> Generator:
        for n, nrhs in self.shapes:
            matrix = torch.randn((n, n), dtype=cur_dtype, device=self.device)
            matrix = matrix @ matrix.mT + n * torch.eye(
                n, dtype=cur_dtype, device=self.device
            )
            factor = torch.linalg.cholesky(matrix)
            rhs = torch.randn((n, nrhs), dtype=cur_dtype, device=self.device)
            yield rhs, factor, False


@pytest.mark.cholesky_solve_helper
def test_cholesky_solve_helper():
    bench = CholeskySolveHelperBenchmark(
        op_name="cholesky_solve_helper",
        torch_op=torch.ops.aten._cholesky_solve_helper,
        # CUDA Cholesky solve does not support Half/BFloat16.
        dtypes=[torch.float32],
    )
    bench.run()
