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

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

EMBEDDING_RENORM_SHAPES = (
    [(50, 64), (128, 256)]
    if QUICK_MODE
    else [(20, 32), (50, 64), (128, 256), (256, 512), (1024, 128)]
)
# ATen's CUDA embedding_renorm_ only agrees with the true p-norm for
# norm_type in {1, 2} (p=2 is nn.Embedding's default); other orders diverge
# from the CPU reference, so we validate the reliable, in-use cases.
EMBEDDING_RENORM_NORM_TYPE_LIST = [2.0] if QUICK_MODE else [1.0, 2.0]
EMBEDDING_RENORM_MAX_NORM_LIST = [1.0] if QUICK_MODE else [0.5, 1.0, 2.0]


@pytest.mark.embedding_renorm_
@pytest.mark.parametrize("shape", EMBEDDING_RENORM_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("norm_type", EMBEDDING_RENORM_NORM_TYPE_LIST)
@pytest.mark.parametrize("max_norm", EMBEDDING_RENORM_MAX_NORM_LIST)
def test_embedding_renorm_(shape, dtype, norm_type, max_norm):
    num_embeddings, embedding_dim = shape
    # Scale rows up so a good fraction exceed max_norm and get renormalized.
    inp = torch.randn(
        num_embeddings, embedding_dim, dtype=dtype, device=flag_gems.device
    )
    inp = inp * 1.5
    # Repeated indices exercise the dedup path; cover a subset of the rows.
    num_indices = max(1, num_embeddings // 2)
    indices = torch.randint(
        0, num_embeddings, (num_indices,), dtype=torch.long, device=flag_gems.device
    )

    ref_inp = utils.to_reference(inp)
    ref_indices = utils.to_reference(indices)

    ref_out = torch.ops.aten.embedding_renorm_(
        ref_inp, ref_indices, max_norm, norm_type
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.embedding_renorm_(inp, indices, max_norm, norm_type)

    # embedding_renorm_ is in-place: validate both the returned handle and the
    # mutated input tensor against the reference.
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(inp, ref_inp, dtype)
