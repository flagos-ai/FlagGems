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


class EmbeddingSparseBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Mirror embedding_dense_backward shapes: (B, M, D, num_weights)
        # covering typical embedding-table gradient sizes on H20.
        self.shapes = [
            (32, 2048, 128, 8192),
            (16, 2048, 256, 16384),
            (8, 4096, 256, 32768),
        ]


def _input_fn(shape, dtype, device):
    B, M, D, num_weights = shape

    grad_output = torch.randn((B, M, D), device=device, dtype=dtype)
    indices = torch.randint(0, num_weights, (B, M), device=device, dtype=torch.long)

    def inject_padding_idx(cur_indices: torch.Tensor, padding_idx: int) -> torch.Tensor:
        if padding_idx < 0:
            return cur_indices
        mask = torch.rand((B, M), device=device) < 0.25
        return torch.where(mask, torch.full_like(cur_indices, padding_idx), cur_indices)

    # scale_grad_by_freq is always False: aten does not support it for sparse
    # gradients, so only padding_idx varies across cases.
    test_cases = [(-1, False), (0, False), (5, False)]
    for padding_idx, scale_grad_by_freq in test_cases:
        cur_indices = inject_padding_idx(indices, padding_idx)
        yield grad_output, cur_indices, num_weights, padding_idx, scale_grad_by_freq


@pytest.mark.skipif(
    (not torch.cuda.is_available()) or (flag_gems.device != "cuda"),
    reason="CUDA backend is not available for this benchmark.",
)
@pytest.mark.embedding_sparse_backward
def test_embedding_sparse_backward():
    bench = EmbeddingSparseBackwardBenchmark(
        input_fn=_input_fn,
        op_name="embedding_sparse_backward",
        torch_op=torch.ops.aten.embedding_sparse_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
