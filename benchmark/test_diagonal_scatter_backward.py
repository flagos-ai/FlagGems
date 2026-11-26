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

from . import base, consts

# Planes are square so ``grad_output`` stays (batch, N, N); the largest N here
# keeps a single case within ~2 GiB for fp32 while still reaching the copy
# bandwidth plateau on H20.
DIAGONAL_SCATTER_BACKWARD_SHAPES = [
    (8, 512, 512),
    (64, 512, 512),
    (1024, 256, 256),
    (16, 4096, 4096),
]


def diagonal_scatter_backward_input_fn(shape, dtype, device):
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    diag = torch.diagonal(grad_output, 0, -2, -1)
    src = torch.randn(diag.shape, dtype=dtype, device=device)
    yield grad_output, src, 0, -2, -1


def _diagonal_scatter_backward_ref(grad_output, src, offset, dim1, dim2):
    """Eager reference following ATen's ``DiagonalScatterBackward0`` formula."""
    zeros = torch.zeros_like(src)
    grad_input = torch.diagonal_scatter(grad_output, zeros, offset, dim1, dim2)
    grad_src = torch.diagonal(grad_output, offset, dim1, dim2).contiguous()
    return grad_input, grad_src


class DiagonalScatterBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = DIAGONAL_SCATTER_BACKWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield from diagonal_scatter_backward_input_fn(shape, cur_dtype, self.device)


@pytest.mark.diagonal_scatter_backward
def test_diagonal_scatter_backward():
    """Backward of ``diagonal_scatter``: both input and src gradients."""
    bench = DiagonalScatterBackwardBenchmark(
        op_name="diagonal_scatter_backward",
        torch_op=_diagonal_scatter_backward_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.diagonal_scatter_backward)
    bench.run()
