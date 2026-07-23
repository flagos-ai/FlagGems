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

# Benchmark shapes from the worktree PDIST_BACKWARD_SHAPES covering small to large matrices
PDIST_BACKWARD_SHAPES = [
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
]


class PdistBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PDIST_BACKWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            n, m = shape
            n_pairs = n * (n - 1) // 2
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            pdist_out = torch.pdist(x, p=2.0)
            grad = torch.ones(n_pairs, dtype=cur_dtype, device=self.device)
            yield grad, x, 2.0, pdist_out


@pytest.mark.pdist_backward
def test_pdist_backward():
    bench = PdistBackwardBenchmark(
        op_name="pdist_backward",
        torch_op=torch.ops.aten._pdist_backward,
        # pdist_backward limited to float32 for numerical stability
        dtypes=[torch.float32],
    )
    bench.run()
