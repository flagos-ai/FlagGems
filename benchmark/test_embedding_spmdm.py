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

"""Performance benchmark for :func:`flag_gems.embedding_spmdm`."""

import pytest
import torch
import torch.nn.functional as F

import flag_gems

from . import base


class EmbeddingSpmdmBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


def _torch_embedding_bag(table, indices, weights):
    """``F.embedding_bag`` reference matched to ``embedding_spmdm`` semantics."""
    # ``offsets=None`` together with a 2D ``indices`` matrix is the canonical
    # way to invoke ``F.embedding_bag`` for the fixed-length bag use case.
    return F.embedding_bag(
        indices,
        table,
        offsets=None,
        mode="sum",
        per_sample_weights=weights,
    )


def _input_fn(shape, dtype, device, weighted):
    num_embeddings, embedding_dim, batch_size, bag_size = shape
    table = torch.randn(num_embeddings, embedding_dim, dtype=dtype, device=device)
    indices = torch.randint(
        0,
        num_embeddings,
        (batch_size, bag_size),
        dtype=torch.int64,
        device=device,
    )
    weights = (
        torch.randn(batch_size, bag_size, dtype=dtype, device=device)
        if weighted
        else None
    )
    yield table, indices, weights


@pytest.mark.embedding_spmdm
@pytest.mark.parametrize("weighted", (False, True))
def test_embedding_spmdm_benchmark(weighted):
    op_name = "embedding_spmdm_weighted" if weighted else "embedding_spmdm"
    bench = EmbeddingSpmdmBenchmark(
        op_name=op_name,
        torch_op=_torch_embedding_bag,
        gems_op=flag_gems.embedding_spmdm,
        input_fn=lambda shape, dtype, device: _input_fn(shape, dtype, device, weighted),
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.run()
