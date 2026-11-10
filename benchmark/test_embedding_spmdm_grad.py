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

"""Performance benchmark for :func:`flag_gems.embedding_spmdm_grad`."""

import pytest
import torch

import flag_gems

from . import base


class EmbeddingSpmdmGradBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # See ``EmbeddingSpmdmBenchmark.set_more_shapes`` for rationale.
        return []


def _torch_backward(grad_output, table, indices, weights):
    """Compose the reference backward from PyTorch primitive ops.

    This mirrors the reference used in ``tests/test_embedding_spmdm_grad.py``
    so the benchmark measures the same numerical regime the accuracy tests
    check against.  Always returns ``(grad_table, grad_weights)`` to match
    the public signature of :func:`flag_gems.embedding_spmdm_grad`; the
    ``grad_weights`` slot is ``None`` when ``weights`` is ``None``.
    """
    num_embeddings, embedding_dim = table.shape
    batch_size, bag_size = indices.shape

    flat_indices = indices.reshape(-1)
    expanded_grad = (
        grad_output[:, None, :]
        .expand(batch_size, bag_size, embedding_dim)
        .reshape(-1, embedding_dim)
    )

    sample_grad = expanded_grad
    if weights is not None:
        sample_grad = sample_grad * weights.reshape(-1, 1)

    grad_table = torch.ops.aten.embedding_dense_backward(
        sample_grad,
        flat_indices,
        num_embeddings,
        -1,
        False,
    )
    if weights is None:
        return grad_table, None

    selected_table = torch.index_select(table, 0, flat_indices)
    grad_weights = (selected_table * expanded_grad).sum(dim=1).reshape_as(weights)
    return grad_table, grad_weights


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
    grad_output = torch.randn(batch_size, embedding_dim, dtype=dtype, device=device)
    yield grad_output, table, indices, weights


@pytest.mark.embedding_spmdm_grad
@pytest.mark.parametrize("weighted", (False, True))
def test_embedding_spmdm_grad_benchmark(weighted):
    op_name = "embedding_spmdm_grad_weighted" if weighted else "embedding_spmdm_grad"
    bench = EmbeddingSpmdmGradBenchmark(
        op_name=op_name,
        torch_op=_torch_backward,
        gems_op=flag_gems.embedding_spmdm_grad,
        input_fn=lambda shape, dtype, device: _input_fn(shape, dtype, device, weighted),
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.run()
