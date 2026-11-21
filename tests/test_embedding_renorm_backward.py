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

EMBEDDING_RENORM_BACKWARD_SHAPES = (
    [(50, 64), (128, 256)]
    if QUICK_MODE
    else [(20, 32), (50, 64), (128, 256), (256, 512), (1024, 128)]
)
# ATen's CUDA embedding_renorm_ only agrees with the true p-norm for
# norm_type in {1, 2} (p=2 is nn.Embedding's default); other orders diverge
# from the CPU reference, so we validate the reliable, in-use cases.
EMBEDDING_RENORM_BACKWARD_NORM_TYPE_LIST = [2.0] if QUICK_MODE else [1.0, 2.0]
EMBEDDING_RENORM_BACKWARD_MAX_NORM_LIST = [1.0] if QUICK_MODE else [0.5, 1.0, 2.0]


def _renorm_vjp_reference(g, x, indices, max_norm, norm_type, eps=1e-7):
    """Eager VJP of the row renormalization (float64, against native autograd)."""
    x64 = utils.to_reference(x).to(torch.float64)
    g64 = utils.to_reference(g).to(torch.float64)
    ax = x64.abs()
    n = ax.pow(norm_type).sum(dim=1).pow(1.0 / norm_type)
    rescaled = n > max_norm
    c = torch.where(rescaled, max_norm / (n + eps), torch.ones_like(n))
    coef = torch.where(
        rescaled,
        max_norm / (n + eps).pow(2) * n.pow(1.0 - norm_type),
        torch.zeros_like(n),
    )
    t = (g64 * ax.pow(norm_type - 1.0) * x64.sign()).sum(dim=1)
    grad = (
        c[:, None] * g64
        - coef[:, None] * t[:, None] * ax.pow(norm_type - 1.0) * x64.sign()
    )

    out = g64.clone()
    uniq = torch.unique(indices)
    out[uniq] = grad[uniq]
    return out


@pytest.mark.embedding_renorm_backward
@pytest.mark.parametrize("shape", EMBEDDING_RENORM_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("norm_type", EMBEDDING_RENORM_BACKWARD_NORM_TYPE_LIST)
@pytest.mark.parametrize("max_norm", EMBEDDING_RENORM_BACKWARD_MAX_NORM_LIST)
def test_embedding_renorm_backward(shape, dtype, norm_type, max_norm):
    num_embeddings, embedding_dim = shape
    # Scale rows up so a good fraction exceed max_norm and take the
    # rescaled path; a mix of selected/renormed and untouched rows.
    weight = torch.randn(
        num_embeddings, embedding_dim, dtype=dtype, device=flag_gems.device
    )
    weight = weight * 1.5
    num_indices = max(1, num_embeddings // 2)
    indices = torch.randint(
        0, num_embeddings, (num_indices,), dtype=torch.long, device=flag_gems.device
    )
    grad_output = torch.randn(
        num_embeddings, embedding_dim, dtype=dtype, device=flag_gems.device
    )

    ref_grad_output = utils.to_reference(grad_output, True)
    ref_weight = utils.to_reference(weight, True)
    ref_indices = utils.to_reference(indices)

    res_grad = flag_gems.embedding_renorm_backward(
        grad_output, weight, indices, max_norm, norm_type
    )
    ref_grad = _renorm_vjp_reference(
        ref_grad_output, ref_weight, ref_indices, max_norm, norm_type
    )

    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.embedding_renorm_backward
def test_embedding_renorm_backward_use_gems_entry():
    """Dispatch path: the flag_gems entry point runs the Triton kernel; compare
    against the eager VJP computed from the same inputs (the op is
    non-differentiable in ATen, so there is no native autograd path to race
    against).
    """
    torch.manual_seed(0)
    V, D = 128, 256
    weight = torch.randn(V, D, dtype=torch.float32, device=flag_gems.device) * 1.5
    grad_output = torch.randn(V, D, dtype=torch.float32, device=flag_gems.device)
    indices = torch.randint(0, V, (V // 2,), device=flag_gems.device)

    res_grad = flag_gems.embedding_renorm_backward(
        grad_output, weight, indices, 1.0, 2.0
    )
    ref_grad_output = utils.to_reference(grad_output)
    ref_weight = utils.to_reference(weight)
    ref_indices = utils.to_reference(indices)
    ref_grad = _renorm_vjp_reference(ref_grad_output, ref_weight, ref_indices, 1.0, 2.0)
    utils.gems_assert_close(res_grad, ref_grad, torch.float32)
