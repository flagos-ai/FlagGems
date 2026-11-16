# Copyright 2026, The FlagOS Contributors.
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

from . import accuracy_utils as utils
from . import conftest as cfg


@pytest.mark.embedding_sparse_backward
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.parametrize(
    "Batch, M, N, embeddingsize",
    [
        # (batch, seq_len, embed_dim, num_weights): small/medium/large tables.
        (2, 4, 8, 16),
        (4, 8, 32, 64),
        (1, 3, 64, 128),
    ],
)
@pytest.mark.parametrize("padding_idx", [-1, 0, 5])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("seed", [42])
def test_embedding_sparse_backward(
    Batch, M, N, embeddingsize, padding_idx, dtype, seed
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    grad_output = torch.randn((Batch, M, N), device=flag_gems.device, dtype=dtype)
    indices = torch.randint(
        0, embeddingsize, (Batch, M), device=flag_gems.device, dtype=torch.long
    )

    if padding_idx >= 0 and embeddingsize > 0:
        mask = torch.rand((Batch, M), device=flag_gems.device) < 0.25
        indices = torch.where(mask, torch.full_like(indices, padding_idx), indices)
    num_weights = embeddingsize
    # aten does not support scale_grad_by_freq for sparse gradients.
    scale_grad_by_freq = False
    ref_grad_output = utils.to_reference(grad_output)
    ref_indices = utils.to_reference(indices)
    ref_out = torch.ops.aten.embedding_sparse_backward(
        ref_grad_output,
        ref_indices,
        num_weights,
        padding_idx,
        scale_grad_by_freq,
    )

    res_out = flag_gems.embedding_sparse_backward(
        grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
    )

    assert res_out.is_sparse, "embedding_sparse_backward must return a sparse tensor."
    # COO output is uncoalesced with arbitrary ordering; compare dense forms.
    utils.gems_assert_close(res_out.to_dense(), ref_out.to_dense(), dtype)
