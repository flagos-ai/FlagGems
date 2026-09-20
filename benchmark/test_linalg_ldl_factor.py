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

from . import base


# LDL Factorization benchmark
class LdlFactorBenchmark(base.Benchmark):
    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        # LDL factorization shapes (square matrices only)
        self.shapes = [
            (4, 4),
            (8, 8),
            (16, 16),
            (32, 32),
            (64, 64),
        ]

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            n = shape[0]
            # Create symmetric positive definite matrix
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            A = (
                A @ A.transpose(-2, -1)
                + torch.eye(n, dtype=cur_dtype, device=self.device) * n
            )
            yield (A,)

    def get_gems_input_iter(self, cur_dtype):
        return self.get_input_iter(cur_dtype)


@pytest.mark.linalg_ldl_factor
def test_linalg_ldl_factor():
    bench = LdlFactorBenchmark(
        op_name="linalg_ldl_factor",
        torch_op=torch.linalg.ldl_factor,
        # torch.linalg.ldl_factor on CUDA supports float32/float64 for this path.
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
