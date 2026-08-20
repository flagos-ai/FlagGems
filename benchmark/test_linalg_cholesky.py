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

# Cholesky decomposition benchmark shapes
# Square matrices from 2x2 to 256x256 covering small to medium-large use cases
CHOLESKY_SHAPES = [
    (2, 2),
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
]


class CholeskyBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CHOLESKY_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            n = shape[-1]
            # Create positive-definite matrix
            B = torch.randn(shape, dtype=cur_dtype, device=self.device)
            A = (
                B @ B.transpose(-2, -1)
                + torch.eye(n, dtype=cur_dtype, device=self.device) * 0.1
            )
            yield (A,)


@pytest.mark.linalg_cholesky
def test_linalg_cholesky():
    bench = CholeskyBenchmark(
        op_name="linalg_cholesky",
        torch_op=torch.ops.aten.linalg_cholesky,
        # Cholesky only supports float32/float64; fp16/bf16 not supported by PyTorch
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
