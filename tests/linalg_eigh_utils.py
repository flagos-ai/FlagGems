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

"""Shared helpers for the linalg_eigh / _linalg_eigh test files.

`torch.linalg.eigh` (the user-facing Python API) dispatches to
`aten::_linalg_eigh`, which is the operator registered to FlagGems. Both
`aten::_linalg_eigh` and `aten::linalg_eigh` are registered; both ultimately
run the on-device Triton paths:
  - n == 2 real                 -> closed-form `_eig_2x2_kernel`.
  - 3 <= n <= 64 real / 2n<=128 complex -> register-resident Jacobi.
  - n > 64 real / 2n > 128 complex      -> global-memory pair-wise Jacobi.
  - n < 2                       -> diagonal / identity on device.
Complex inputs use a real embedding (2n x 2n real symmetric) and recover
complex eigenpairs from it. Shapes below are split so each path is exercised.
"""

from contextlib import contextmanager

import torch

import flag_gems

from . import accuracy_utils as utils

# Path A: closed-form 2x2 kernel (real). fp16/bf16 are widened to fp32 on
# device, an enhancement over native torch.linalg.eigh (which raises
# NotImplementedError for Half/BFloat16).
EIG_2X2_SHAPES = [(2, 2)]
EIG_2X2_LOWDTYPE = [torch.float16, torch.bfloat16]

# Path B: register-resident Jacobi (n > 2, within the tile bound).
EIG_JACOBI_TILE_SHAPES = [
    (3, 3),
    (5, 5),
    (8, 8),
    (16, 16),
    (32, 32),
    (48, 48),
    (64, 64),
]

# Path B2: global-memory round-robin Jacobi (n > 64). Verified by
# reconstruction only; the trailing precision is worse than cuSOLVER, so
# element-wise eigenvalue comparison would not hold at these sizes.
EIG_JACOBI_GLOBAL_SHAPES = [
    (96, 96),
    (128, 128),
]

# Path C: n < 2 (0x0 / 1x1), diagonal/identity on device.
EIG_TRIVIAL_SHAPES = [(0, 0), (1, 1)]
EIG_TRIVIAL_COMPLEX_SHAPES = [(0, 0), (1, 1)]

# Complex inputs (any n): real embedding then the real Jacobi path.
# Up to n=32 (real embedding 2n<=64) hits the tile path; n>=48 (2n>=96)
# hits the global path.
EIG_COMPLEX_SHAPES = [(2, 2), (3, 3), (5, 5), (8, 8), (16, 16), (32, 32)]
EIG_COMPLEX_GLOBAL_SHAPES = [(48, 48), (64, 64)]

# Batched variants.
EIG_BATCH_2X2_SHAPES = [(2, 2, 2)]
EIG_BATCH_JACOBI_SHAPES = [(4, 3, 3), (1, 8, 8)]
EIG_BATCH_TRIVIAL_SHAPES = [(2, 0, 0), (3, 1, 1)]
EIG_BATCH_COMPLEX_SHAPES = [(2, 3, 3), (1, 4, 4)]

# UPLO is a core parameter of eigh; "L" is covered above, "U" separately.
EIG_UPLO_U_SHAPES = [(3, 3), (5, 5)]
EIG_UPLO_U_COMPLEX_SHAPES = [(3, 3), (5, 5)]

# Eigenvalue element-wise tolerance per path. The Jacobi path has slightly
# lower trailing precision than cuSOLVER, so the element-wise eigenvalue check
# uses a looser atol than the reconstruction check.
EIG_EVAL_ATOL = {torch.float32: 5e-4, torch.complex64: 5e-4}


@contextmanager
def gems_eigh_dispatch():
    """Temporarily dispatch the eigh aten ops to FlagGems for the block.

    Registers ``_linalg_eigh`` and ``linalg_eigh`` onto a private
    ``torch.library`` handle and tears the registration down on exit, so
    calls *outside* the block (the reference computation) keep dispatching to
    native aten while calls *inside* the block run on FlagGems.
    """
    lib = torch.library.Library("aten", "IMPL")
    flag_gems.only_enable(lib=lib, include=["_linalg_eigh", "linalg_eigh"])
    try:
        yield
    finally:
        if torch.__version__ >= "2.5":
            lib._destroy()
        del lib


@contextmanager
def ieee_float32_matmul():
    """Force IEEE float32 matmul (disable TF32) for the block, restore after.

    TF32 (10-bit mantissa) on Ampere+ GPUs introduces ~1e-3 errors in float32
    matmul, which corrupts the reconstruction verification of eigh
    (V @ diag(w) @ V.T). Disable it for the verification matmul only; the op
    under test is unaffected.
    """
    m = torch.backends.cuda.matmul
    use_new = hasattr(m, "fp32_precision")
    if use_new:
        old = m.fp32_precision
        m.fp32_precision = "ieee"
    else:
        old = m.allow_tf32
        m.allow_tf32 = False
    try:
        yield
    finally:
        if use_new:
            m.fp32_precision = old
        else:
            m.allow_tf32 = old


def make_symmetric_matrix(shape, dtype, device, symmetric_only=True):
    """Create a symmetric (or Hermitian) matrix for eigendecomposition.

    When ``symmetric_only`` is False the matrix is made asymmetric so that the
    UPLO selection (which triangle is used) is actually exercised.
    """
    A = torch.randn(shape, dtype=dtype, device=device)
    if A.is_complex():
        A = (A + A.mH) / 2
    else:
        A = (A + A.transpose(-2, -1)) / 2
    if not symmetric_only and A.shape[-1] >= 2:
        # Perturb the upper triangle only; UPLO="L" must ignore it, "U" must
        # use it. The reference is built with the matching UPLO below.
        A = A + torch.triu(0.1 * torch.randn_like(A), diagonal=1)
    return A


def assert_close(res, ref, dtype, atol=1e-4):
    """Wrapper around the accuracy utils that handles complex128.

    FlagGems' ``gems_assert_close`` tolerance table (``RESOLUTION``) has no
    ``complex128`` entry, so direct lookup raises ``KeyError``. Existing linalg
    tests (cholesky_solve, linalg_ldl_solve, linalg_cross) handle this by
    falling back to ``torch.testing.assert_close`` for complex128; mirror that
    pattern here.
    """
    if dtype == torch.complex128:
        res = utils.to_cpu(res, ref)
        torch.testing.assert_close(res, ref, atol=atol, rtol=1e-3)
    else:
        utils.gems_assert_close(res, ref, dtype, atol=atol)


def _assert_orthonormal(v, atol=1e-2):
    """Columns of v are eigenvectors: Vᴴ V ≈ I.

    Avoids comparing eigenvectors elementwise, since v and -v are both valid
    eigenvectors (sign ambiguity).
    """
    n = v.shape[-1]
    eye = torch.eye(n, dtype=v.dtype, device=v.device)
    v_h = v.mH if v.is_complex() else v.transpose(-2, -1)
    gram = v_h @ v
    expected = utils.to_reference(eye.expand_as(gram), False)
    assert_close(gram, expected, gram.dtype, atol=atol)


def symmetrise(inp, UPLO):
    """Mirror one triangle into the other, matching eigh's UPLO semantics.

    Used as the reconstruction target for asymmetric UPLO inputs: eigh returns
    eigenpairs of the symmetrised matrix, so the reconstruction must be checked
    against that matrix, not the raw asymmetric input.
    """
    idx = torch.arange(inp.shape[-1], device=inp.device)
    if UPLO == "U":
        tri_mask = idx[None, :] >= idx[:, None]
    else:
        tri_mask = idx[None, :] <= idx[:, None]
    tri = inp * tri_mask
    sym = tri + (tri.mH if inp.is_complex() else tri.transpose(-2, -1))
    sym = sym.clone()
    sym.diagonal(dim1=-2, dim2=-1).copy_(tri.diagonal(dim1=-2, dim2=-1))
    return sym


def check_eigh_decomposition(A, eigenvalues, eigenvectors, atol=1e-3):
    """Verify the eigendecomposition via the defining relation A = V diag(w) Vᴴ.

    For real inputs this is V diag(w) Vᵀ; for complex/Hermitian inputs it is
    V diag(w) Vᴴ (conjugate transpose). This is sign-ambiguous-free: any valid
    eigenbasis reconstructs A and is orthonormal, regardless of per-vector sign
    choices. Works for all Triton paths.
    """
    with ieee_float32_matmul():
        v_t = (
            eigenvectors.mH
            if eigenvectors.is_complex()
            else eigenvectors.transpose(-2, -1)
        )
        reconstructed = (
            eigenvectors @ torch.diag_embed(eigenvalues).to(eigenvectors.dtype) @ v_t
        )
        ref_A = utils.to_reference(A, False)
        assert_close(reconstructed, ref_A, reconstructed.dtype, atol=atol)
        _assert_orthonormal(eigenvectors)


def assert_ascending(eigenvalues, atol=1e-4):
    """Eigenvalues are returned in ascending order (torch.linalg.eigh contract).

    Ties are allowed; only a strictly descending adjacent pair is a failure.
    """
    w = utils.to_reference(eigenvalues, False)
    diffs = w[..., 1:] - w[..., :-1]
    torch.testing.assert_close(
        diffs,
        torch.clamp(diffs, min=-atol),
        atol=atol,
        rtol=0,
    )
