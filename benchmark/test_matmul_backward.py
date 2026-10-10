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

import flag_gems

from . import base

# (M, K, N) shapes for matmul_backward benchmark: out = (M, K) @ (K, N)
MATMUL_BACKWARD_SHAPES = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]


def matmul_backward_baseline(grad, self, other, mask):
    """PyTorch baseline using component operations.

    matmul_backward computes gradients for: out = self @ other
    grad_self = grad @ other.T
    grad_other = self.T @ grad
    """
    grad_self = None
    grad_other = None

    if mask[0]:
        grad_self = torch.matmul(grad, other.transpose(-2, -1))

    if mask[1]:
        grad_other = torch.matmul(self.transpose(-2, -1), grad)

    return grad_self, grad_other


class MatmulBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = MATMUL_BACKWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for m, k, n in self.shapes:
            self_t = torch.randn(m, k, dtype=cur_dtype, device=self.device)
            other_t = torch.randn(k, n, dtype=cur_dtype, device=self.device)
            grad = torch.randn(m, n, dtype=cur_dtype, device=self.device)
            yield grad, self_t, other_t, [True, True]


@pytest.mark.matmul_backward
def test_matmul_backward():
    bench = MatmulBackwardBenchmark(
        op_name="matmul_backward",
        torch_op=matmul_backward_baseline,
        dtypes=[torch.float32, torch.float16],
        gems_op=flag_gems.matmul_backward,
    )
    bench.run()
