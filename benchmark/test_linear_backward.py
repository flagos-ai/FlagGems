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

# Shapes for linear_backward benchmark
LINEAR_BACKWARD_SHAPES = [
    (64, 512),
    (128, 1024),
    (256, 2048),
    (512, 4096),
]


class LinearBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LINEAR_BACKWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for batch, in_features in self.shapes:
            out_features = in_features * 2  # out_features = 2 * in_features
            input = torch.randn(batch, in_features, dtype=cur_dtype, device=self.device)
            weight = torch.randn(
                out_features, in_features, dtype=cur_dtype, device=self.device
            )
            grad_output = torch.randn(
                batch, out_features, dtype=cur_dtype, device=self.device
            )
            yield input, grad_output, weight, (True, True, True)


class _AtenLinearBackwardAdapter:
    """aten::linear_backward has no CUDA kernel in stock PyTorch (Meta-only
    registration); PyTorch's own linear autograd composes the backward from
    basic ops. Adapt the benchmark args to that native composition so the
    baseline measures real aten CUDA kernels."""

    def __call__(self, input, grad_output, weight, output_mask):
        grad_input = torch.mm(grad_output, weight) if output_mask[0] else None
        grad_weight = (
            torch.mm(grad_output.transpose(-2, -1), input) if output_mask[1] else None
        )
        grad_bias = grad_output.sum(dim=0) if output_mask[2] else None
        return grad_input, grad_weight, grad_bias


@pytest.mark.linear_backward
def test_linear_backward():
    bench = LinearBackwardBenchmark(
        op_name="linear_backward",
        torch_op=_AtenLinearBackwardAdapter(),
        # Keep the worktree benchmark dtype set; this backward benchmark uses only fp32/fp16 core cases.
        dtypes=[torch.float32, torch.float16],
    )
    bench.set_gems(flag_gems.linear_backward)
    bench.run()
