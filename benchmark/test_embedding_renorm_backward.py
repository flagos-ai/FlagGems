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


def _embedding_renorm_backward_ref(g, weight, indices, max_norm, norm_type):
    """Eager VJP baseline with the same signature as the Triton entry."""
    dtype = weight.dtype
    work = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    x = weight.to(work)
    g = g.to(work)
    ax = x.abs()
    n = ax.pow(norm_type).sum(dim=1).pow(1.0 / norm_type)
    rescaled = n > max_norm
    c = torch.where(rescaled, max_norm / (n + 1e-7), torch.ones_like(n))
    coef = torch.where(
        rescaled,
        max_norm / (n + 1e-7).pow(2) * n.pow(1.0 - norm_type),
        torch.zeros_like(n),
    )
    t = (g * ax.pow(norm_type - 1.0) * x.sign()).sum(dim=1)
    grad = (
        c[:, None] * g - coef[:, None] * t[:, None] * ax.pow(norm_type - 1.0) * x.sign()
    )
    grad = grad.to(dtype)
    out = g.to(dtype).clone()
    uniq = torch.unique(indices)
    out[uniq] = grad[uniq]
    return out


class EmbeddingRenormBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # (num_embeddings, embedding_dim) tables at nn.Embedding scale;
        # indices cover half of the rows (as in the embedding_renorm_
        # forward benchmark).
        self.shapes = [
            (1024, 128),
            (4096, 256),
            (16384, 256),
            (32768, 512),
            (50000, 256),
            (50000, 512),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype):
        for num_embeddings, embedding_dim in self.shapes:
            weight = (
                torch.randn(
                    num_embeddings, embedding_dim, dtype=cur_dtype, device=self.device
                )
                * 1.5
            )
            num_indices = num_embeddings // 2
            indices = torch.randint(
                0, num_embeddings, (num_indices,), dtype=torch.long, device=self.device
            )
            grad_output = torch.randn(
                num_embeddings, embedding_dim, dtype=cur_dtype, device=self.device
            )
            # max_norm=1.0, norm_type=2.0 matches nn.Embedding's defaults.
            yield grad_output, weight, indices, 1.0, 2.0


@pytest.mark.embedding_renorm_backward
def test_embedding_renorm_backward():
    bench = EmbeddingRenormBackwardBenchmark(
        op_name="embedding_renorm_backward",
        torch_op=_embedding_renorm_backward_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.embedding_renorm_backward)
    bench.run()
