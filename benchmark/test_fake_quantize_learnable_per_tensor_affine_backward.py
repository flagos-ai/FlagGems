# Copyright 2026 FlagOS Contributors.
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


def _fake_quantize_learnable_per_tensor_affine_backward_input_fn(shape, dtype, device):
    # The ATen reference rejects float16 gradients, so only float32 is
    # benchmarked here (it is the dtype that actually flows through the
    # learnable fake-quantize autograd path in practice).
    self_t = torch.randn(shape, dtype=torch.float32, device=device)
    grad_t = torch.randn(shape, dtype=torch.float32, device=device)
    scale_t = torch.tensor([0.1], dtype=torch.float32, device=device)
    zero_point_t = torch.tensor([128.0], dtype=torch.float32, device=device)
    yield grad_t, self_t, scale_t, zero_point_t, 0, 255, 1.0


class _FakeQuantizeLearnablePerTensorAffineBackwardBenchmark(base.Benchmark):
    """Benchmark the backward op directly (no autograd graph needed)."""

    DEFAULT_METRICS = consts.DEFAULT_METRICS[:]

    def set_more_shapes(self):
        # Shape sweep that spans small to large 2D/3D tensors for benchmarking
        # the backward kernel across representative reduction sizes.
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        sp_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
        return special_shapes_2d + sp_shapes_3d

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield from (
                _fake_quantize_learnable_per_tensor_affine_backward_input_fn(
                    shape, cur_dtype, self.device
                )
            )

    def get_gbps(self, args, latency):
        # Read grad + self, write grad_self (scale/zero_point grads are negligible).
        grad, self_t = args[0], args[1]
        io = (
            grad.numel() * grad.element_size() * 2
            + self_t.numel() * self_t.element_size()
        )
        return io * 1e-9 / (latency * 1e-3)


@pytest.mark.fake_quantize_learnable_per_tensor_affine_backward
def test_fake_quantize_learnable_per_tensor_affine_backward():
    bench = _FakeQuantizeLearnablePerTensorAffineBackwardBenchmark(
        op_name="fake_quantize_learnable_per_tensor_affine_backward",
        torch_op=torch.ops.aten._fake_quantize_learnable_per_tensor_affine_backward,
        # The ATen reference rejects float16 gradients, so only float32 is
        # benchmarked here (it is the dtype that actually flows through the
        # learnable fake-quantize autograd path in practice).
        dtypes=[torch.float32],
    )
    bench.run()
