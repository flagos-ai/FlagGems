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

"""Tests for the linalg_eigh operator family.

The user-facing ``torch.linalg.eigh`` (``aten::linalg_eigh``) and the
underlying ``aten::_linalg_eigh`` (distinguished by its ``compute_v``
argument) are both registered to FlagGems and share the same Triton
paths. The Gems results are obtained by calling ``flag_gems.linalg_eigh``
and ``flag_gems._linalg_eigh`` directly, while the reference is computed
with the native ``torch.linalg.eigh`` / ``torch.ops.aten._linalg_eigh``.
The ``test_underlying_*`` cases below exercise ``compute_v`` directly.
"""

import sys
from contextlib import contextmanager

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .accuracy_utils import _SKIP_FP64

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


# Aliases kept from the previous test__linalg_eigh.py / utils module split, so
# the test bodies below stay unchanged.
_assert_ascending = assert_ascending
_assert_close = assert_close
_check_eigh_decomposition = check_eigh_decomposition
_symmetrise = symmetrise


# ---------------------------------------------------------------------------
# User-facing path: torch.linalg.eigh (aten::linalg_eigh)
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

    res_out = flag_gems.linalg_eigh(inp)

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

    res_out = flag_gems.linalg_eigh(inp)

    # Eigenvalues element-wise (Jacobi tolerance), plus reconstruction.
    utils.gems_assert_close(res_out[0], ref_out[0], dtype, atol=EIG_EVAL_ATOL[dtype])
    _check_eigh_decomposition(inp, res_out[0], res_out[1])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_COMPLEX_SHAPES,
    ids=[f"complex_n{s[0]}" for s in EIG_COMPLEX_SHAPES],
)
@pytest.mark.parametrize(
    "dtype", [torch.complex64, pytest.param(torch.complex128, marks=_SKIP_FP64)]
)
def test_linalg_eigh_complex(shape, dtype):
    """Complex inputs: real embedding then the real Jacobi path."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    res_out = flag_gems.linalg_eigh(inp)

    # Eigenvalues of a Hermitian matrix are real.
    _assert_close(res_out[0], ref_out[0], res_out[0].dtype, atol=5e-4)
    _check_eigh_decomposition(inp, res_out[0], res_out[1])


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

    res_out = flag_gems.linalg_eigh(inp)

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

    res_out = flag_gems.linalg_eigh(inp)

    utils.gems_assert_close(res_out[0], ref_out[0], dtype, atol=EIG_EVAL_ATOL[dtype])
    _check_eigh_decomposition(inp, res_out[0], res_out[1])


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

    res_out = flag_gems.linalg_eigh(inp)

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

    res_out = flag_gems.linalg_eigh(inp)

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
    reconstruction only. The output-dtype quantization of w and V bounds the
    achievable reconstruction accuracy: fp16 stays within 1e-2, while bf16
    (eps ~ 7.8e-3, quantization step ~2e-2 at |w| ~ 4) needs 2e-2."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    res_out = flag_gems.linalg_eigh(inp)

    atol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    _check_eigh_decomposition(inp, res_out[0], res_out[1], atol=atol)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_TRIVIAL_COMPLEX_SHAPES,
    ids=[f"trivial_complex_n{s[0]}" for s in EIG_TRIVIAL_COMPLEX_SHAPES],
)
@pytest.mark.parametrize(
    "dtype", [torch.complex64, pytest.param(torch.complex128, marks=_SKIP_FP64)]
)
def test_linalg_eigh_trivial_complex(shape, dtype):
    """Complex n < 2 (0x0 / 1x1): diagonal/identity on device."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    res_out = flag_gems.linalg_eigh(inp)

    _assert_close(res_out[0], ref_out[0], res_out[0].dtype)
    if res_out[1].numel() > 0:
        _check_eigh_decomposition(inp, res_out[0], res_out[1])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_BATCH_COMPLEX_SHAPES,
    ids=[f"complex-batch{s[-1]}" for s in EIG_BATCH_COMPLEX_SHAPES],
)
@pytest.mark.parametrize(
    "dtype", [torch.complex64, pytest.param(torch.complex128, marks=_SKIP_FP64)]
)
def test_linalg_eigh_batch_complex(shape, dtype):
    """Batched complex n > 2: real embedding then the real Jacobi path."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    res_out = flag_gems.linalg_eigh(inp)

    _assert_close(res_out[0], ref_out[0], res_out[0].dtype, atol=5e-4)
    _check_eigh_decomposition(inp, res_out[0], res_out[1])


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

    res_out = flag_gems.linalg_eigh(inp, UPLO="U")

    utils.gems_assert_close(res_out[0], ref_out[0], dtype, atol=EIG_EVAL_ATOL[dtype])
    # Reconstruction against the symmetrised matrix (eigh uses one triangle).
    _check_eigh_decomposition(_symmetrise(inp, "U"), res_out[0], res_out[1])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_UPLO_U_COMPLEX_SHAPES,
    ids=[f"uplo_u_complex_n{s[0]}" for s in EIG_UPLO_U_COMPLEX_SHAPES],
)
@pytest.mark.parametrize(
    "dtype", [torch.complex64, pytest.param(torch.complex128, marks=_SKIP_FP64)]
)
def test_linalg_eigh_uplo_upper_complex(shape, dtype):
    """UPLO="U" on the complex path."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device, symmetric_only=False)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp, UPLO="U")

    res_out = flag_gems.linalg_eigh(inp, UPLO="U")

    _assert_close(res_out[0], ref_out[0], res_out[0].dtype, atol=5e-4)
    _check_eigh_decomposition(_symmetrise(inp, "U"), res_out[0], res_out[1])


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

    res_out = flag_gems.linalg_eigh(inp)

    _check_eigh_decomposition(inp, res_out[0], res_out[1], atol=1e-1)
    _assert_ascending(res_out[0])
    # Eigenvalue set agrees with the reference to the global-path tolerance.
    utils.gems_assert_close(
        res_out[0].sort().values, ref_out[0].sort().values, torch.float32, atol=2e-2
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

    res_out = flag_gems.linalg_eigh(inp)

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
        res_out = flag_gems.linalg_eigh(inp)
        _assert_ascending(res_out[0])


@pytest.mark.linalg_eigh
def test_linalg_eigh_nonsquare_raises():
    """A non-square input must raise ValueError on the Gems path."""
    A = torch.randn(3, 5, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(ValueError):
        flag_gems.linalg_eigh(A)


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

    res_out = flag_gems.linalg_eigh(view)

    if dtype == torch.complex64:
        _assert_close(res_out[0], ref_out[0], res_out[0].dtype, atol=5e-4)
    else:
        utils.gems_assert_close(
            res_out[0], ref_out[0], dtype, atol=EIG_EVAL_ATOL[dtype]
        )
    _check_eigh_decomposition(view, res_out[0], res_out[1])


# ---------------------------------------------------------------------------
# Underlying path: aten::_linalg_eigh.default (direct call, compute_v arg)
# ---------------------------------------------------------------------------


@pytest.mark.underscore_linalg_eigh
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

    res_w, res_v = flag_gems._linalg_eigh(inp, "L", True)

    utils.gems_assert_close(res_w, ref_w, dtype, atol=EIG_EVAL_ATOL[dtype])
    _check_eigh_decomposition(inp, res_w, res_v)


@pytest.mark.underscore_linalg_eigh
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

    res_w, res_v = flag_gems._linalg_eigh(inp, "L", False)

    utils.gems_assert_close(res_w, ref_w, dtype, atol=EIG_EVAL_ATOL[dtype])
    # Eigenvectors tensor is empty when compute_v=False.
    assert res_v.numel() == 0


@pytest.mark.underscore_linalg_eigh
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

    res_w, res_v = flag_gems._linalg_eigh(inp, "L", False)

    utils.gems_assert_close(res_w, ref_w, dtype)
    assert res_v.numel() == 0


@pytest.mark.underscore_linalg_eigh
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

    res_w, res_v = flag_gems._linalg_eigh(inp, "L", False)

    utils.gems_assert_close(
        res_w.sort().values, ref_w.sort().values, torch.float32, atol=2e-2
    )
    _assert_ascending(res_w)
    assert res_v.numel() == 0


@pytest.mark.underscore_linalg_eigh
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

    res_w, res_v = flag_gems._linalg_eigh(inp, "L", False)

    utils.gems_assert_close(
        res_w.sort().values, ref_w.sort().values, torch.float32, atol=2e-2
    )
    _assert_ascending(res_w)
    assert res_v.numel() == 0


# ---------------------------------------------------------------------------
# First-order autograd backward.
#
# ATen has no independent backward schema for eigh: the engine builds
# ``LinalgEighBackward0`` from ``derivatives.yaml`` and runs
# ``linalg_eig_backward(is_hermitian=true)``, whose closed form is
#
#   grad_A = V (diag(grad_w) + F * (V^H grad_V - grad_V^H V) / 2) V^H
#   F[i, j] = 1 / (w[j] - w[i])   (zero on the diagonal)
#
# Eigenvectors are only defined up to a per-column sign (a phase, when
# complex), so an upstream gradient ``grad_V`` is meaningful only for a loss
# that is invariant under that freedom.  Every comparison against native
# autograd below therefore uses such a loss -- the reconstruction loss
# ``||V diag(w) V^T - B||^2 + c.w`` or a pure eigenvalue loss -- so both sides
# may legitimately choose different eigenvector branches and still have the
# same mathematical gradient.
# ---------------------------------------------------------------------------


def _skip_if_cpu_reference(reason):
    """Skip an eigenvector-gradient test when the reference runs on the CPU.

    ``--ref cpu`` moves the reference eigen-decomposition to the host, which
    picks a different (equally valid) eigenvector branch than the device
    forward.  An eigenvector gradient is only meaningful for a loss invariant
    under that branch choice, and the branch-dependent end-to-end comparisons
    below cannot be rescued by a tolerance; they are covered on device by the
    default run.  The device-reference path is the authoritative one here.
    """
    if utils.TO_CPU:
        pytest.skip(reason)


@pytest.fixture(autouse=True)
def _ieee_matmul_for_eigh_backward(request):
    """Run backward tests with IEEE fp32 matmul.

    The reference gradients are matmul chains; TF32 (the default on Ampere+)
    truncates them to 10 mantissa bits and would mask real kernel error.  The
    device kernel already uses IEEE ``tl.dot``, so both sides must match.
    """
    if request.node.get_closest_marker("linalg_eigh_backward") is None:
        yield
        return
    with ieee_float32_matmul():
        yield


def eigh_backward_closed_form(grad_w, grad_v, w, v):
    """Evaluate the ATen closed form from explicit ``(w, v, grad_w, grad_v)``.

    Deliberately built from plain PyTorch matmuls so it shares no code with the
    Triton kernel.  The result is Hermitian for every input.
    """
    n = w.shape[-1]
    inner = v.transpose(-2, -1) @ grad_v
    ct = inner.mH if v.is_complex() else inner.transpose(-2, -1)
    inner = 0.5 * (inner - ct)
    gap = w[..., None, :] - w[..., None]
    eye = torch.eye(n, dtype=torch.bool, device=w.device)
    f = torch.where(eye, torch.zeros_like(gap), 1.0 / gap)
    scaled = f * inner + torch.diag_embed(grad_w.to(v.dtype))
    vt = v.mH if v.is_complex() else v.transpose(-2, -1)
    out = v @ scaled @ vt
    out_t = out.mH if v.is_complex() else out.transpose(-2, -1)
    return 0.5 * (out + out_t)


def reconstruction_loss(w, v, target, coef):
    """Sign/phase-invariant scalar loss in the eigenpairs of a symmetric matrix.

    ``target``/``coef`` are created by :func:`_loss_data` on ``v``'s device so
    the same loss works for both the device and the ``--ref cpu`` reference.
    """
    vt = v.mH if v.is_complex() else v.transpose(-2, -1)
    rebuilt = v @ torch.diag_embed(w.to(v.dtype)) @ vt
    return ((rebuilt - target) ** 2).sum() + (coef * w).sum()


def _loss_data(shape, dtype, device, seed):
    """Reconstruction target and eigenvalue coefficients on ``device``."""
    torch.manual_seed(seed)
    target = torch.randn(shape, dtype=dtype, device=device)
    target = (target + target.transpose(-2, -1)) / 2
    coef = torch.randn(shape[-1], dtype=dtype, device=device)
    return target, coef


def _make_eigh_input(shape, dtype, seed=0, asymmetric=False):
    torch.manual_seed(seed)
    a = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    a = (a + a.transpose(-2, -1)) / 2
    if asymmetric:
        a = a + torch.triu(0.1 * torch.randn_like(a), diagonal=1)
    return a


def _loss_grads(w, v, target, coef):
    """Upstream gradients of the invariant loss w.r.t. this implementation's w, v."""
    return torch.autograd.grad(reconstruction_loss(w, v, target, coef), (w, v))


# The backward reads (w, V) from the forward, so an end-to-end comparison
# against native autograd is bounded by the *forward's* eigenvector accuracy,
# not by the backward arithmetic.  An orthonormality residual eps in V perturbs
# the gap-scaled gradient by O(eps / min_gap), and that factor is large here:
# measured min gaps are ~1e-2 for a random 48x48 matrix.  The forward's own
# residual is ~4e-4 (fp32) and ~3e-5 (fp64) -- note the Jacobi forward runs in
# fp32 even for fp64 input -- so the end-to-end tolerance has to absorb
# eps/min_gap, and the fp64 forward is fp32-limited (eps ~ 3e-5, so with the
# smallest gaps this reaches ~ 1e-2).  Backward exactness is pinned separately
# and tightly by ``test_linalg_eigh_backward_matches_closed_form_exactly`` and
# by the eigenvalue-only test, both of which isolate the backward from this
# limit.
END_TO_END_RTOL = {torch.float32: 5e-3, torch.float64: 2e-2}


@pytest.mark.linalg_eigh_backward
@pytest.mark.parametrize(
    "shape", [(2, 2), (3, 3), (8, 8), (17, 17), (32, 32), (48, 48)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_eigh_backward_gradients(shape, dtype):
    """Invariant-loss gradients match native autograd in float32 and float64."""
    _skip_if_cpu_reference(
        "test_linalg_eigh_backward_gradients needs the device reference branch"
    )

    inp = _make_eigh_input(shape, dtype)
    target, coef = _loss_data(shape, dtype, inp.device, 1)

    ref_inp = utils.to_reference(inp).requires_grad_()
    inp = inp.clone().requires_grad_()
    ref_w, ref_v = torch.linalg.eigh(ref_inp)
    w, v = flag_gems.linalg_eigh(inp)

    # The Jacobi forward computes in fp32 registers even for fp64 input, so the
    # eigenvalue agreement is fp32-bounded; that is a property of the existing
    # forward, not of this backward and not something a gradient test can fix.
    utils.gems_assert_close(w, ref_w, dtype, atol=EIG_EVAL_ATOL[torch.float32])
    with ieee_float32_matmul():
        (ref_grad,) = torch.autograd.grad(
            reconstruction_loss(ref_w, ref_v, target, coef), (ref_inp,)
        )
    (grad,) = torch.autograd.grad(reconstruction_loss(w, v, target, coef), (inp,))

    torch.testing.assert_close(
        utils.to_cpu(grad, ref_grad),
        ref_grad,
        atol=1e-6,
        rtol=END_TO_END_RTOL[dtype],
    )
    assert grad.shape == inp.shape and grad.dtype == dtype
    # The backward must produce a symmetric gradient for a symmetric input.
    torch.testing.assert_close(grad, grad.transpose(-2, -1), atol=1e-5, rtol=0)


@pytest.mark.linalg_eigh_backward
@pytest.mark.parametrize("shape", [(8, 8), (32, 32), (48, 48)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_eigh_backward_matches_closed_form_exactly(shape, dtype):
    """The kernel reproduces the ATen closed form at working-dtype precision.

    Feeding the *same* (w, V) to both sides isolates the backward arithmetic
    from the forward's accuracy, so this is a tight check: it must agree to
    rounding, not merely to the forward-limited tolerance above.
    """
    inp = _make_eigh_input(shape, dtype).requires_grad_()
    w, v = flag_gems.linalg_eigh(inp)
    torch.manual_seed(8)
    grad_w = torch.randn_like(w)
    grad_v = torch.randn_like(v)
    (grad,) = torch.autograd.grad((w, v), (inp,), (grad_w, grad_v))
    with ieee_float32_matmul():
        expected = eigh_backward_closed_form(grad_w, grad_v, w.detach(), v.detach())
    # tl.dot with input_precision="ieee" rounds the same way as the reference
    # matmul chain, so only accumulations in a different order differ.
    atol = 1e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(grad, expected, atol=atol, rtol=1e-5)


@pytest.mark.linalg_eigh_backward
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_eigh_backward_tiled_path_matches_fused(dtype):
    """The large-n padded path agrees with the fused single-program path."""

    eigh_module = sys.modules["flag_gems.ops.linalg_eigh"]
    inp = _make_eigh_input((40, 40), dtype).requires_grad_()
    w, v = flag_gems.linalg_eigh(inp)
    torch.manual_seed(9)
    grad_w = torch.randn_like(w)
    grad_v = torch.randn_like(v)
    (grad,) = torch.autograd.grad((w, v), (inp,), (grad_w, grad_v))
    with ieee_float32_matmul():
        expected = eigh_backward_closed_form(grad_w, grad_v, w.detach(), v.detach())
    saved = dict(eigh_module._EIGH_BWD_FUSED_MAX_N)
    try:
        eigh_module._EIGH_BWD_FUSED_MAX_N[dtype] = 10**6
        fused = eigh_module._eigh_backward_launch(
            grad_w, grad_v, w.detach(), v.detach()
        )
        eigh_module._EIGH_BWD_FUSED_MAX_N[dtype] = 0
        tiled = eigh_module._eigh_backward_launch(
            grad_w, grad_v, w.detach(), v.detach()
        )
    finally:
        eigh_module._EIGH_BWD_FUSED_MAX_N.update(saved)
    atol = 1e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(fused, expected, atol=atol, rtol=1e-5)
    torch.testing.assert_close(tiled, expected, atol=atol, rtol=1e-5)


@pytest.mark.linalg_eigh_backward
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_eigh_backward_closed_form(dtype):
    """The Triton backward reproduces the ATen closed form of the same eigenpairs.

    This pins the arithmetic (gap scaling, skew projection, diagonal insertion)
    independently of which eigenvector branch the forward happened to return.
    """
    n = 12
    inp = _make_eigh_input((n, n), dtype)
    inp = inp.clone().requires_grad_()
    w, v = flag_gems.linalg_eigh(inp)
    torch.manual_seed(2)
    grad_w = torch.randn_like(w)
    grad_v = torch.randn_like(v)
    (grad,) = torch.autograd.grad((w, v), (inp,), (grad_w, grad_v))

    with ieee_float32_matmul():
        expected = eigh_backward_closed_form(grad_w, grad_v, w.detach(), v.detach())
    atol = 1e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(grad, expected, atol=atol, rtol=1e-4)
    # The closed form is *not* invariant under a sign flip of one eigenvector,
    # which is exactly why an arbitrary grad_V has no native reference.
    flip = torch.ones(n, dtype=dtype, device=inp.device)
    flip[0] = -1.0
    with ieee_float32_matmul():
        other = eigh_backward_closed_form(grad_w, grad_v, w.detach(), v.detach() * flip)
    assert not torch.allclose(grad, other, atol=1e-4), "test would be vacuous"


@pytest.mark.linalg_eigh_backward
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_eigh_backward_eigenvalue_only(dtype):
    """A pure eigenvalue loss reproduces native autograd exactly."""
    _skip_if_cpu_reference(
        "test_linalg_eigh_backward_eigenvalue_only needs the device reference branch"
    )

    inp = _make_eigh_input((16, 16), dtype)
    ref_inp = utils.to_reference(inp).requires_grad_()
    inp = inp.clone().requires_grad_()
    _, coef = _loss_data((16, 16), dtype, inp.device, 3)

    ref_w, _ = torch.linalg.eigh(ref_inp)
    w, _ = flag_gems.linalg_eigh(inp)
    with ieee_float32_matmul():
        (ref_grad,) = torch.autograd.grad((coef * ref_w).sum(), (ref_inp,))
    (grad,) = torch.autograd.grad((coef * w).sum(), (inp,))
    # V diag(coef) V^T inherits the forward's eigenvector accuracy, so this
    # shares the end-to-end tolerance used for the two-gradient cases.
    torch.testing.assert_close(
        utils.to_cpu(grad, ref_grad),
        ref_grad,
        atol=1e-6,
        rtol=END_TO_END_RTOL[dtype],
    )
    grad = utils.to_cpu(grad, ref_grad)
    torch.testing.assert_close(grad, grad.transpose(-2, -1), atol=1e-5, rtol=0)


@pytest.mark.linalg_eigh_backward
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_eigh_backward_batched(dtype):
    """Batched inputs get an independent per-matrix gradient."""
    _skip_if_cpu_reference(
        "test_linalg_eigh_backward_batched needs the device reference branch"
    )

    shape = (3, 4, 4)
    inp = _make_eigh_input(shape, dtype)
    target, coef = _loss_data(shape, dtype, inp.device, 4)
    ref_inp = utils.to_reference(inp).requires_grad_()
    inp = inp.clone().requires_grad_()
    ref_w, ref_v = torch.linalg.eigh(ref_inp)
    w, v = flag_gems.linalg_eigh(inp)
    with ieee_float32_matmul():
        (ref_grad,) = torch.autograd.grad(
            reconstruction_loss(ref_w, ref_v, target, coef), (ref_inp,)
        )
    (grad,) = torch.autograd.grad(reconstruction_loss(w, v, target, coef), (inp,))
    atol = 2e-4 if dtype == torch.float32 else 1e-8
    torch.testing.assert_close(
        utils.to_cpu(grad, ref_grad), ref_grad, atol=atol, rtol=1e-3
    )
    assert grad.shape == shape


@pytest.mark.linalg_eigh_backward
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("uplo", ["L", "U"])
def test_linalg_eigh_backward_uplo(dtype, uplo):
    """UPLO chooses the forward triangle and leaves the backward Hermitian."""
    _skip_if_cpu_reference(
        "test_linalg_eigh_backward_uplo needs the device reference branch"
    )

    n = 9
    inp = _make_eigh_input((n, n), dtype, asymmetric=True)
    ref_inp = utils.to_reference(inp).requires_grad_()
    inp = inp.clone().requires_grad_()
    ref_w, ref_v = torch.linalg.eigh(ref_inp, UPLO=uplo)
    w, v = flag_gems.linalg_eigh(inp, UPLO=uplo)

    def loss_parts(device):
        # The target is the symmetrised matrix, which is what eigh decomposes;
        # it is rebuilt per side so it lands on that side's device (--ref cpu
        # runs the reference on the host).
        local = inp.detach().to(device)
        tri = torch.tril(local) if uplo == "L" else torch.triu(local)
        target = tri + tri.transpose(-2, -1) - torch.diag_embed(tri.diagonal())
        # Seed locally so both sides get identical coefficients regardless of
        # any RNG the other side's forward consumed.
        generator = torch.Generator(device="cpu").manual_seed(5)
        coef = torch.randn(n, dtype=dtype, generator=generator).to(device)
        return target, coef

    ref_target, ref_coef = loss_parts(ref_inp.device)
    target, coef = loss_parts(inp.device)
    with ieee_float32_matmul():
        (ref_grad,) = torch.autograd.grad(
            reconstruction_loss(ref_w, ref_v, ref_target, ref_coef), (ref_inp,)
        )
    (grad,) = torch.autograd.grad(reconstruction_loss(w, v, target, coef), (inp,))
    atol = 2e-4 if dtype == torch.float32 else 1e-8
    torch.testing.assert_close(
        utils.to_cpu(grad, ref_grad), ref_grad, atol=atol, rtol=1e-3
    )
    grad = utils.to_cpu(grad, ref_grad)
    torch.testing.assert_close(grad, grad.transpose(-2, -1), atol=1e-5, rtol=0)


@pytest.mark.linalg_eigh_backward
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_eigh_backward_uses_autograd_function(dtype):
    """The Gems forward must build the custom node and route to the kernel."""
    inp = _make_eigh_input((8, 8), dtype).requires_grad_()
    w, v = flag_gems.linalg_eigh(inp)
    assert w.requires_grad and v.requires_grad
    assert w.grad_fn is not None and v.grad_fn is not None
    assert type(w.grad_fn).__name__.startswith("_EighBackward")
    assert w.grad_fn is v.grad_fn

    # flag_gems.ops.linalg_eigh is re-exported as a function at package level,
    # so fetch the module object from sys.modules to reach its helper.
    eigh_module = sys.modules["flag_gems.ops.linalg_eigh"]
    original = eigh_module._eigh_backward_launch
    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    eigh_module._eigh_backward_launch = counting
    try:
        (grad,) = torch.autograd.grad(w.sum(), (inp,))
    finally:
        eigh_module._eigh_backward_launch = original
    assert calls == [1], f"backward kernel launched {len(calls)} times"
    assert grad.shape == inp.shape


@pytest.mark.linalg_eigh_backward
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_eigh_backward_requires_grad_mode(dtype):
    """No autograd node without grad mode or without requires_grad."""
    inp = _make_eigh_input((6, 6), dtype)
    w, v = flag_gems.linalg_eigh(inp)
    assert w.grad_fn is None and v.grad_fn is None
    with torch.no_grad():
        w2, v2 = flag_gems.linalg_eigh(inp.clone().requires_grad_())
    assert w2.grad_fn is None and v2.grad_fn is None
    assert w.is_leaf and v.is_leaf
    assert w.requires_grad is False and v.requires_grad is False


@pytest.mark.linalg_eigh_backward
def test_linalg_eigh_backward_complex_not_implemented():
    """The complex eigenvector gradient is phase-ambiguous; both sides refuse."""
    inp = _make_eigh_input((4, 4), torch.complex64)
    ref_inp = utils.to_reference(inp).requires_grad_()
    inp = inp.clone().requires_grad_()
    ref_w, ref_v = torch.linalg.eigh(ref_inp)
    torch.manual_seed(6)
    grad_w = torch.randn_like(ref_w)
    grad_v = torch.randn_like(ref_v)
    with pytest.raises(RuntimeError, match="ill-defined"):
        torch.autograd.grad((ref_w, ref_v), (ref_inp,), (grad_w, grad_v))
    # Gems reports the same ambiguity when the differentiable input is seen,
    # because the forward has to build the node before backward can run.
    with pytest.raises(NotImplementedError, match="complex phase"):
        flag_gems.linalg_eigh(inp)


@pytest.mark.linalg_eigh_backward
def test_linalg_eigh_backward_compute_v_false():
    """compute_v=False has no eigenvector basis; both sides refuse the backward."""
    inp = _make_eigh_input((5, 5), torch.float32)
    ref_inp = utils.to_reference(inp).requires_grad_()
    inp = inp.clone().requires_grad_()
    ref_w, _ = torch.ops.aten._linalg_eigh.default(ref_inp, "L", False)
    w, _ = flag_gems._linalg_eigh(inp, "L", False)
    grad_w = torch.ones_like(w)
    with pytest.raises(RuntimeError):
        torch.autograd.grad(ref_w, (ref_inp,), (grad_w,))
    with pytest.raises(NotImplementedError, match="compute_v"):
        torch.autograd.grad(w, (inp,), (grad_w,))


@pytest.mark.linalg_eigh_backward
@pytest.mark.parametrize("shape", [(0, 0), (1, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_eigh_backward_trivial(shape, dtype):
    """Empty and single-element inputs follow the forward's own contract."""
    inp = _make_eigh_input(shape, dtype).requires_grad_()
    w, v = flag_gems.linalg_eigh(inp)
    (grad,) = torch.autograd.grad(
        (w * 1.0).sum() + (v * 1.0).sum() if v.numel() else (w * 1.0).sum(),
        (inp,),
        allow_unused=True,
    )
    if shape == (0, 0):
        assert grad is None or grad.shape == shape
    else:
        assert grad.shape == shape


@pytest.mark.linalg_eigh_backward
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_eigh_backward_non_contiguous_grad(dtype):
    """A transposed (non-contiguous) upstream gradient is accepted."""
    n = 7
    inp = _make_eigh_input((n, n), dtype)
    target, coef = _loss_data((n, n), dtype, inp.device, 7)

    inp = inp.clone().requires_grad_()
    w, v = flag_gems.linalg_eigh(inp)
    grad_w, grad_v = _loss_grads(w, v, target, coef)
    grad_v = grad_v.transpose(-2, -1).contiguous().transpose(-2, -1)
    assert not grad_v.is_contiguous()
    (grad,) = torch.autograd.grad((w, v), (inp,), (grad_w, grad_v))
    with ieee_float32_matmul():
        expected = eigh_backward_closed_form(grad_w, grad_v, w.detach(), v.detach())
    atol = 1e-5 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(grad, expected, atol=atol, rtol=1e-4)


@pytest.mark.linalg_eigh_backward
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_eigh_backward_expanded_grad(dtype):
    """An expanded upstream eigenvalue gradient contributes once per element."""
    n = 6
    inp = _make_eigh_input((n, n), dtype)
    inp = inp.clone().requires_grad_()
    w, v = flag_gems.linalg_eigh(inp)
    base = torch.randn(1, dtype=dtype, device=inp.device)
    grad_w = base.expand(w.shape)
    grad_v = torch.zeros_like(v)
    (grad,) = torch.autograd.grad((w, v), (inp,), (grad_w, grad_v))
    with ieee_float32_matmul():
        expected = eigh_backward_closed_form(grad_w, grad_v, w.detach(), v.detach())
    atol = 1e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(grad, expected, atol=atol, rtol=1e-4)


@pytest.mark.linalg_eigh_backward
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_eigh_backward_repeated_eigenvalues_non_finite(dtype):
    """Repeated eigenvalues make 1/(w[j]-w[i]) unbounded, as in native ATen."""
    eye = torch.eye(4, dtype=dtype, device=flag_gems.device).requires_grad_()
    w, v = flag_gems.linalg_eigh(eye)
    grad_w = torch.ones_like(w)
    grad_v = torch.ones_like(v)
    (grad,) = torch.autograd.grad((w, v), (eye,), (grad_w, grad_v))
    assert not torch.isfinite(grad).all(), "degenerate spectrum must not be masked"
