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

DEVICE = flag_gems.device

# torch geqrf (cuSOLVER/LAPACK) only supports float32/float64 on CUDA.
_TEST_DTYPES = [torch.float32, torch.float64]

# Shapes covering both implementation paths: single-launch register kernel
# (small/batched) and blocked column-major panel path (large), tall/wide/
# square, plus degenerate 1-row/1-column and batched inputs.
GEQRF_SHAPES = [
    # degenerate
    (1, 1),
    (1, 8),
    (8, 1),
    # small path (single-program register kernel)
    (4, 4),
    (33, 33),
    (8, 3),
    (3, 8),
    # large path (blocked column-major panels)
    (256, 256),
    (300, 100),
    (100, 300),
    # batched, and batched with extra dims
    (4, 16, 16),
    (2, 3, 8, 8),
    (64, 8, 8),
    (2, 32, 8),
]


def _well_conditioned(shape, dtype, device):
    # A = randn + m * I (diagonal bump over the min(m, n) diagonal): well
    # conditioned for QR, mirrors the det/slogdet/solve_ex recipe.
    m, n = shape[-2], shape[-1]
    A = torch.randn(shape, dtype=dtype, device=device)
    return A + m * torch.eye(m, n, dtype=dtype, device=device)


def _reconstruction(a, tau, A):
    """A = Q @ R from the packed (a, tau) pair; works for tall, square and
    wide inputs.  Q = H_0 H_1 ... H_{k-1} (reflectors compose right to
    left), so apply them to the identity in reverse column order."""
    m, n = A.shape[-2], A.shape[-1]
    k = min(m, n)
    B = A.numel() // (m * n)
    A64 = A.detach().to("cpu").to(torch.float64).reshape(B, m, n)
    a64 = a.detach().to("cpu").to(torch.float64).reshape(B, m, n)
    tau64 = tau.detach().to("cpu").to(torch.float64).reshape(B, k)
    recon = 0.0
    for b in range(B):
        q = torch.eye(m, dtype=torch.float64)
        for j in range(k - 1, -1, -1):
            v = torch.zeros(m, dtype=torch.float64)
            v[j] = 1.0
            if j + 1 < m:
                v[j + 1 :] = a64[b, j + 1 :, j]
            tv = tau64[b, j]
            if tv != 0:
                q[:, j:] = q[:, j:] - torch.outer(v, tv * (v @ q[:, j:]))
        recon = max(recon, (q @ torch.triu(a64[b]) - A64[b]).abs().max().item())
    scale = A64.abs().max().item() + 1e-30
    return recon / scale


@pytest.mark.geqrf
@pytest.mark.parametrize("shape", GEQRF_SHAPES)
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
def test_geqrf(shape, dtype):
    A = _well_conditioned(shape, dtype, DEVICE)
    ref_A = utils.to_reference(A)

    ref_a, ref_tau = torch.ops.aten.geqrf(ref_A)
    res_a, res_tau = flag_gems.geqrf(A)

    assert res_a.shape == ref_a.shape
    assert res_tau.shape == ref_tau.shape

    # The packed reflectors and tau are implementation dependent (cuSOLVER vs
    # the Triton Householder chain produce different, equally valid
    # factorisations), so compare the deterministic reconstruction property
    # A = Q @ R instead of the factors themselves.
    max_rel = _reconstruction(res_a, res_tau, A)
    assert max_rel <= 2e-2, f"reconstruction max_rel={max_rel:.3e}"

    # R (= triu(a)) must match the reference R: for a full-rank well-
    # conditioned A the QR factor R is unique (positive/negative diagonal
    # convention may differ per column, so align signs first).
    ref_R = torch.triu(ref_a.to(torch.float64))
    res_R = torch.triu(utils.to_cpu(res_a, ref_a).to(torch.float64))
    k = min(shape[-2], shape[-1])
    sgn = torch.sign(res_R.diagonal(dim1=-2, dim2=-1))
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    ref_sgn = torch.sign(ref_R.diagonal(dim1=-2, dim2=-1))
    ref_sgn = torch.where(ref_sgn == 0, torch.ones_like(ref_sgn), ref_sgn)
    # Scale each column of R by its diagonal sign: for a full-rank A the QR
    # factor R is unique up to a per-column sign.  sgn is (..., k); the wide
    # case (n > k) leaves the sign-free tail columns (j >= k) untouched, so
    # compare only the leading k columns.
    kcols = min(res_R.shape[-1], k)
    res_R_aligned = res_R[..., :kcols] * sgn.unsqueeze(-2)
    ref_R_aligned = ref_R[..., :kcols] * ref_sgn.unsqueeze(-2)
    utils.gems_assert_close(res_R_aligned, ref_R_aligned, torch.float64, atol=1e-2)


@pytest.mark.geqrf
def test_geqrf_non_contiguous():
    # transposed (non-contiguous) input must still produce the QR of the
    # logical matrix
    base = _well_conditioned((16, 8), torch.float32, DEVICE)
    At = base.t().t().t()[:8]  # non-contiguous view with the same rows
    ref = torch.ops.aten.geqrf(At.contiguous())
    res_a, res_tau = flag_gems.geqrf(At)
    max_rel = _reconstruction(res_a, res_tau, At.contiguous())
    assert max_rel <= 2e-2, f"reconstruction max_rel={max_rel:.3e}"
    assert res_tau.shape == ref[1].shape


@pytest.mark.geqrf
def test_geqrf_return_structure():
    # multi-output contract: a tuple (a, tau) with the packed layout
    A = _well_conditioned((8, 8), torch.float32, DEVICE)
    out = flag_gems.geqrf(A)
    assert isinstance(out, tuple) and len(out) == 2
    a, tau = out
    assert a.shape == (8, 8) and tau.shape == (8,)
    # strict lower triangle of the first k columns holds the packed
    # reflectors with unit implicit diagonal -> a's diagonal holds R
    assert torch.all(torch.triu(a).diagonal()[1:] != 0)


@pytest.mark.geqrf
def test_geqrf_zero_and_tiny_sizes():
    # zero-size batch
    A = _well_conditioned((0, 4, 4), torch.float32, DEVICE)
    a, tau = flag_gems.geqrf(A)
    assert a.shape == (0, 4, 4) and tau.shape == (0, 4)
    # 1x1: no reflection, tau == 0
    A = torch.tensor([[3.0]], dtype=torch.float32, device=DEVICE)
    a, tau = flag_gems.geqrf(A)
    ref_a, ref_tau = torch.ops.aten.geqrf(utils.to_reference(A))
    assert tau.item() == ref_tau.item() == 0.0
    utils.gems_assert_close(a, ref_a, torch.float32)
