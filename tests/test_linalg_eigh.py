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

from contextlib import contextmanager

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@contextmanager
def _ieee_float32_matmul():
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


# linalg_eigh / _linalg_eigh registration.
#
# `torch.linalg.eigh` (the user-facing Python API) dispatches to
# `aten::_linalg_eigh`, which is the operator registered to FlagGems. Both
# `aten::_linalg_eigh` and `aten::linalg_eigh` are registered; both ultimately
# run the on-device Triton paths:
#   - n == 2 real                 -> closed-form `_eig_2x2_kernel`.
#   - 3 <= n <= 64 real / 2n<=128 complex -> register-resident Jacobi.
#   - n > 64 real / 2n > 128 complex      -> global-memory pair-wise Jacobi.
#   - n < 2                       -> diagonal / identity on device.
# Complex inputs use a real embedding (2n x 2n real symmetric) and recover
# complex eigenpairs from it. Shapes below are split so each path is exercised.

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


def _assert_close(res, ref, dtype, atol=1e-4):
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
    _assert_close(gram, expected, gram.dtype, atol=atol)


def _symmetrise(inp, UPLO):
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


def _check_eigh_decomposition(A, eigenvalues, eigenvectors, atol=1e-3):
    """Verify the eigendecomposition via the defining relation A = V diag(w) Vᴴ.

    For real inputs this is V diag(w) Vᵀ; for complex/Hermitian inputs it is
    V diag(w) Vᴴ (conjugate transpose). This is sign-ambiguous-free: any valid
    eigenbasis reconstructs A and is orthonormal, regardless of per-vector sign
    choices. Works for all Triton paths.
    """
    with _ieee_float32_matmul():
        v_t = (
            eigenvectors.mH
            if eigenvectors.is_complex()
            else eigenvectors.transpose(-2, -1)
        )
        reconstructed = (
            eigenvectors @ torch.diag_embed(eigenvalues).to(eigenvectors.dtype) @ v_t
        )
        ref_A = utils.to_reference(A, False)
        _assert_close(reconstructed, ref_A, reconstructed.dtype, atol=atol)
        _assert_orthonormal(eigenvectors)


def _assert_ascending(eigenvalues, atol=1e-4):
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


# ---------------------------------------------------------------------------
# User path: torch.linalg.eigh -> aten::_linalg_eigh (the registered op)
# ---------------------------------------------------------------------------


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_2X2_SHAPES,
    ids=[f"kernel_2x2-{s[0]}x{s[1]}" for s in EIG_2X2_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_2x2_kernel(shape, dtype):
    """n == 2 real: the closed-form `_eig_2x2_kernel`."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    _check_eigh_decomposition(inp, res_out[0], res_out[1])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_JACOBI_TILE_SHAPES,
    ids=[f"jacobi_n{s[0]}" for s in EIG_JACOBI_TILE_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_jacobi(shape, dtype):
    """n > 2 real on the user path: register-resident Jacobi."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    # Eigenvalues element-wise (Jacobi tolerance), plus reconstruction.
    utils.gems_assert_close(res_out[0], ref_out[0], dtype, atol=EIG_EVAL_ATOL[dtype])
    _check_eigh_decomposition(inp, res_out[0], res_out[1], atol=1e-2)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_COMPLEX_SHAPES,
    ids=[f"complex_n{s[0]}" for s in EIG_COMPLEX_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_linalg_eigh_complex(shape, dtype):
    """Complex inputs: real embedding then the real Jacobi path."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    # Eigenvalues of a Hermitian matrix are real.
    _assert_close(res_out[0], ref_out[0], res_out[0].dtype, atol=5e-4)
    _check_eigh_decomposition(inp, res_out[0], res_out[1], atol=1e-2)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_BATCH_2X2_SHAPES,
    ids=[f"kernel_2x2-batch{s[0]}" for s in EIG_BATCH_2X2_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_batch_2x2_kernel(shape, dtype):
    """Batched n == 2 real: each batch element hits the 2x2 kernel."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    _check_eigh_decomposition(inp, res_out[0], res_out[1])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_BATCH_JACOBI_SHAPES,
    ids=[f"jacobi-batch{s[-1]}" for s in EIG_BATCH_JACOBI_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_batch_jacobi(shape, dtype):
    """Batched n > 2 real: register-resident Jacobi per batch element."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    utils.gems_assert_close(res_out[0], ref_out[0], dtype, atol=EIG_EVAL_ATOL[dtype])
    _check_eigh_decomposition(inp, res_out[0], res_out[1], atol=1e-2)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_TRIVIAL_SHAPES,
    ids=[f"trivial_n{s[0]}" for s in EIG_TRIVIAL_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_trivial(shape, dtype):
    """n < 2 (0x0 / 1x1) real: diagonal as eigenvalues, identity as eigenvectors."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    _check_eigh_decomposition(inp, res_out[0], res_out[1])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_BATCH_TRIVIAL_SHAPES,
    ids=[f"trivial-batch_n{s[-1]}" for s in EIG_BATCH_TRIVIAL_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_batch_trivial(shape, dtype):
    """Batched n < 2 real on the user path: computed on device."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    _check_eigh_decomposition(inp, res_out[0], res_out[1])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_2X2_SHAPES,
    ids=[f"lowdtype_2x2-{s[0]}x{s[1]}" for s in EIG_2X2_SHAPES],
)
@pytest.mark.parametrize("dtype", EIG_2X2_LOWDTYPE)
def test_linalg_eigh_2x2_low_precision(shape, dtype):
    """n == 2 fp16/bf16: the 2x2 path widens to fp32 on device and casts back.
    cuSOLVER reference is unavailable for these dtypes, so validate via
    reconstruction only."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    _check_eigh_decomposition(inp, res_out[0], res_out[1], atol=1e-2)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_TRIVIAL_COMPLEX_SHAPES,
    ids=[f"trivial_complex_n{s[0]}" for s in EIG_TRIVIAL_COMPLEX_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_linalg_eigh_trivial_complex(shape, dtype):
    """Complex n < 2 (0x0 / 1x1): diagonal/identity on device."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    _assert_close(res_out[0], ref_out[0], res_out[0].dtype)
    if res_out[1].numel() > 0:
        _check_eigh_decomposition(inp, res_out[0], res_out[1], atol=1e-2)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_BATCH_COMPLEX_SHAPES,
    ids=[f"complex-batch{s[-1]}" for s in EIG_BATCH_COMPLEX_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_linalg_eigh_batch_complex(shape, dtype):
    """Batched complex n > 2: real embedding then the real Jacobi path."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    _assert_close(res_out[0], ref_out[0], res_out[0].dtype, atol=5e-4)
    _check_eigh_decomposition(inp, res_out[0], res_out[1], atol=1e-2)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_UPLO_U_SHAPES,
    ids=[f"uplo_u_n{s[0]}" for s in EIG_UPLO_U_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_uplo_upper(shape, dtype):
    """UPLO="U": the upper triangle is used and the lower ignored. Inputs are
    made asymmetric so the triangle selection is genuinely exercised."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device, symmetric_only=False)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp, UPLO="U")

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp, UPLO="U")

    utils.gems_assert_close(res_out[0], ref_out[0], dtype, atol=EIG_EVAL_ATOL[dtype])
    # Reconstruction against the symmetrised matrix (eigh uses one triangle).
    _check_eigh_decomposition(_symmetrise(inp, "U"), res_out[0], res_out[1], atol=1e-2)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_UPLO_U_COMPLEX_SHAPES,
    ids=[f"uplo_u_complex_n{s[0]}" for s in EIG_UPLO_U_COMPLEX_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_linalg_eigh_uplo_upper_complex(shape, dtype):
    """UPLO="U" on the complex path."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device, symmetric_only=False)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp, UPLO="U")

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp, UPLO="U")

    _assert_close(res_out[0], ref_out[0], res_out[0].dtype, atol=5e-4)
    _check_eigh_decomposition(_symmetrise(inp, "U"), res_out[0], res_out[1], atol=1e-2)


# ---------------------------------------------------------------------------
# Underlying entry: aten::_linalg_eigh (covers compute_v and the linalg_eigh
# public entry which mirrors the same paths).
# ---------------------------------------------------------------------------


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_2X2_SHAPES + EIG_JACOBI_TILE_SHAPES,
    ids=[f"ueigh-{s[0]}x{s[1]}" for s in EIG_2X2_SHAPES + EIG_JACOBI_TILE_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_underlying_linalg_eigh(shape, dtype):
    """Directly call aten::_linalg_eigh.default with compute_v=True.

    n == 2 hits the 2x2 kernel; n > 2 hits the Jacobi path.
    """
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_w, ref_v = torch.ops.aten._linalg_eigh.default(ref_inp, "L", True)

    with flag_gems.use_gems():
        res_w, res_v = torch.ops.aten._linalg_eigh.default(inp, "L", True)

    utils.gems_assert_close(res_w, ref_w, dtype, atol=EIG_EVAL_ATOL[dtype])
    _check_eigh_decomposition(inp, res_w, res_v, atol=1e-2)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_JACOBI_TILE_SHAPES,
    ids=[f"ueigh_no_v_n{s[0]}" for s in EIG_JACOBI_TILE_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_underlying_linalg_eigh_no_vectors(shape, dtype):
    """compute_v=False returns eigenvalues only (empty eigenvectors)."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_w, _ = torch.ops.aten._linalg_eigh.default(ref_inp, "L", False)

    with flag_gems.use_gems():
        res_w, res_v = torch.ops.aten._linalg_eigh.default(inp, "L", False)

    utils.gems_assert_close(res_w, ref_w, dtype, atol=EIG_EVAL_ATOL[dtype])
    # Eigenvectors tensor is empty when compute_v=False.
    assert res_v.numel() == 0


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_2X2_SHAPES,
    ids=[f"ueigh_no_v_2x2-{s[0]}x{s[1]}" for s in EIG_2X2_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_underlying_linalg_eigh_no_vectors_2x2(shape, dtype):
    """compute_v=False with n == 2: still returns eigenvalues only."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_w, _ = torch.ops.aten._linalg_eigh.default(ref_inp, "L", False)

    with flag_gems.use_gems():
        res_w, res_v = torch.ops.aten._linalg_eigh.default(inp, "L", False)

    utils.gems_assert_close(res_w, ref_w, dtype)
    assert res_v.numel() == 0


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_JACOBI_GLOBAL_SHAPES,
    ids=[f"ueigh_no_v_global_n{s[0]}" for s in EIG_JACOBI_GLOBAL_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_underlying_linalg_eigh_no_vectors_global(shape, dtype):
    """compute_v=False on the global-memory Jacobi path (n > 64).

    Eigenvalues only; eigenvector computation is skipped (no V sort), and
    the eigenvector tensor is empty.
    """
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_w, _ = torch.ops.aten._linalg_eigh.default(ref_inp, "L", False)

    with flag_gems.use_gems():
        res_w, res_v = torch.ops.aten._linalg_eigh.default(inp, "L", False)

    res_w_cpu = utils.to_cpu(res_w, ref_w)
    torch.testing.assert_close(
        res_w_cpu.sort().values,
        ref_w.sort().values,
        atol=2e-2,
        rtol=1e-3,
    )
    _assert_ascending(res_w)
    assert res_v.numel() == 0


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_COMPLEX_GLOBAL_SHAPES,
    ids=[f"ueigh_no_v_complex_global_n{s[0]}" for s in EIG_COMPLEX_GLOBAL_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.complex64])
def test_underlying_linalg_eigh_no_vectors_complex_global(shape, dtype):
    """compute_v=False on the complex global path (complex64, 2n > 128).

    The complex pick kernel is skipped; eigenvalues come from the real
    embedding's Jacobi path alone, and eigenvectors are empty.
    """
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_w, _ = torch.ops.aten._linalg_eigh.default(ref_inp, "L", False)

    with flag_gems.use_gems():
        res_w, res_v = torch.ops.aten._linalg_eigh.default(inp, "L", False)

    res_w_cpu = utils.to_cpu(res_w, ref_w)
    torch.testing.assert_close(
        res_w_cpu.sort().values,
        ref_w.sort().values,
        atol=2e-2,
        rtol=1e-3,
    )
    _assert_ascending(res_w)
    assert res_v.numel() == 0


# ---------------------------------------------------------------------------
# Global-memory round-robin path (n > 64 real, 2n > 128 complex): the large-n
# Jacobi kernels. Verified by reconstruction + ascending order only; the
# trailing precision is worse than cuSOLVER at these sizes.
# ---------------------------------------------------------------------------


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_JACOBI_GLOBAL_SHAPES,
    ids=[f"jacobi_global_n{s[0]}" for s in EIG_JACOBI_GLOBAL_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_jacobi_global(shape, dtype):
    """n > 64 real: the global-memory round-robin Jacobi path."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    _check_eigh_decomposition(inp, res_out[0], res_out[1], atol=1e-1)
    _assert_ascending(res_out[0])
    # Eigenvalue set agrees with the reference to the global-path tolerance.
    res_w = utils.to_cpu(res_out[0], ref_out[0])
    torch.testing.assert_close(
        res_w.sort().values,
        ref_out[0].sort().values,
        atol=2e-2,
        rtol=1e-3,
    )


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_COMPLEX_GLOBAL_SHAPES,
    ids=[f"complex_global_n{s[0]}" for s in EIG_COMPLEX_GLOBAL_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.complex64])
def test_linalg_eigh_complex_global(shape, dtype):
    """Complex n >= 48: real embedding (2n >= 96) hits the global Jacobi path."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    # Eigenvalues of a Hermitian matrix are real.
    _assert_close(res_out[0], ref_out[0], res_out[0].dtype, atol=2e-2)
    _check_eigh_decomposition(inp, res_out[0], res_out[1], atol=1e-1)
    _assert_ascending(res_out[0])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_linalg_eigh_ascending_order(dtype):
    """Eigenvalues are returned in ascending order for every path."""
    shapes = {
        torch.float32: [(2, 2), (8, 8), (96, 96)],
        torch.complex64: [(2, 2), (8, 8), (48, 48)],
    }[dtype]
    for shape in shapes:
        inp = make_symmetric_matrix(shape, dtype, flag_gems.device)
        with flag_gems.use_gems():
            res_out = torch.linalg.eigh(inp)
        _assert_ascending(res_out[0])


@pytest.mark.linalg_eigh
def test_linalg_eigh_nonsquare_raises():
    """A non-square input must raise ValueError on the Gems path."""
    A = torch.randn(3, 5, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(ValueError):
        with flag_gems.use_gems():
            torch.linalg.eigh(A)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_linalg_eigh_non_contiguous(dtype):
    """A non-contiguous (transposed-view) symmetric input is accepted."""
    n = 8
    A = make_symmetric_matrix((n, n), dtype, flag_gems.device)
    # A transposed view is non-contiguous but still symmetric (A == A.T/mH).
    view = A.transpose(-2, -1).contiguous().transpose(-2, -1)
    assert not view.is_contiguous()

    ref_inp = utils.to_reference(view)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(view)

    if dtype == torch.complex64:
        _assert_close(res_out[0], ref_out[0], res_out[0].dtype, atol=5e-4)
    else:
        utils.gems_assert_close(
            res_out[0], ref_out[0], dtype, atol=EIG_EVAL_ATOL[dtype]
        )
    _check_eigh_decomposition(view, res_out[0], res_out[1], atol=1e-2)
