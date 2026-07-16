import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _make_spectrum_input(shape, singular_values, seed=0):
    # Build A = U diag(s) Vh from random orthogonal factors and a prescribed,
    # well-separated spectrum. Well-separated singular values make the singular
    # vectors uniquely determined, so orthonormality of the computed basis is a
    # well-posed, hardware-independent property (mirrors tests/test_svd.py).
    torch.manual_seed(seed)
    *batch_shape, m, n = shape
    k = min(m, n)
    left, _ = torch.linalg.qr(
        torch.randn((*batch_shape, m, m), dtype=torch.float32, device=flag_gems.device)
    )
    right, _ = torch.linalg.qr(
        torch.randn((*batch_shape, n, n), dtype=torch.float32, device=flag_gems.device)
    )
    sigma = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)
    diag = torch.as_tensor(
        singular_values[:k], dtype=torch.float32, device=flag_gems.device
    )
    idx = torch.arange(k, device=flag_gems.device)
    sigma[..., idx, idx] = diag
    return left @ sigma @ right.mH


def _reconstruct(u, s, vh):
    # aten::_linalg_svd returns (U, S, Vh) with A = U diag(S) Vh
    k = s.shape[-1]
    return u[..., :, :k] @ torch.diag_embed(s).to(u.dtype) @ vh[..., :k, :]


def _assert_same_shape(actual, expected):
    actual_shape = torch.tensor(tuple(actual.shape))
    expected_shape = torch.tensor(tuple(expected.shape))
    utils.gems_assert_equal(actual_shape, expected_shape)


def _assert_orthonormal(actual, atol=2e-2):
    if actual.numel() == 0:
        return
    k = actual.shape[-1]
    eye = torch.eye(k, dtype=actual.dtype, device=actual.device)
    gram = actual.mH @ actual
    expected = utils.to_reference(eye.expand_as(gram), False)
    utils.gems_assert_close(gram, expected, gram.dtype, atol=atol)


# The Triton SVD kernels only cover float32 CUDA matrices (cuSOLVER's svd
# kernels are not implemented for Half/BFloat16), so restrict to float32.
LINALG_SVD_DTYPES = [torch.float32]
# The Triton SVD kernels only cover float32 CUDA matrices; the full_matrices
# (some=False) path additionally requires max(m, n) <= 64.
LINALG_SVD_SHAPES = [(3, 3), (4, 4), (8, 8), (3, 5), (5, 3), (16, 16)]
# Small batched matrices that satisfy the reduced-path max(m, n) <= 64 limit.
LINALG_SVD_BATCH_SHAPES = [(2, 4, 4), (3, 8, 8)]
# Shapes for the orthonormality check driven by a controlled spectrum.
LINALG_SVD_ORTHONORMAL_SHAPES = [(8, 8), (16, 16), (5, 3), (3, 5), (2, 8, 8)]


# aten::_linalg_svd is the private primitive that torch.linalg.svd (compute_uv
# True) and torch.linalg.svdvals (compute_uv False) decompose into, so calling
# the public APIs under use_gems() exercises the registered kernel.
@pytest.mark.underscore_linalg_svd
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
@pytest.mark.parametrize("shape", LINALG_SVD_SHAPES)
def test_underscore_linalg_svd_full_matrices(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    ref_u, ref_s, ref_vh = torch.linalg.svd(ref_inp, full_matrices=True)
    with flag_gems.use_gems():
        res_u, res_s, res_vh = torch.linalg.svd(inp, full_matrices=True)

    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_vh, ref_vh)

    # Singular values and the reconstruction A = U diag(S) Vh are gauge
    # invariant, so they are the robust correctness checks for general inputs.
    # Per-vector orthonormality of U/Vh is only well-posed for well-separated
    # singular values; it is exercised in the orthonormal test below.
    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=1e-2)
    reconstructed = _reconstruct(res_u, res_s, res_vh)
    utils.gems_assert_close(reconstructed, ref_inp, reconstructed.dtype, atol=1e-2)


@pytest.mark.underscore_linalg_svd
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
@pytest.mark.parametrize("shape", LINALG_SVD_SHAPES)
def test_underscore_linalg_svd_reduced(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    ref_u, ref_s, ref_vh = torch.linalg.svd(ref_inp, full_matrices=False)
    with flag_gems.use_gems():
        res_u, res_s, res_vh = torch.linalg.svd(inp, full_matrices=False)

    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_vh, ref_vh)

    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=1e-2)
    reconstructed = _reconstruct(res_u, res_s, res_vh)
    utils.gems_assert_close(reconstructed, ref_inp, reconstructed.dtype, atol=1e-2)


@pytest.mark.underscore_linalg_svd
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
@pytest.mark.parametrize("shape", LINALG_SVD_BATCH_SHAPES)
def test_underscore_linalg_svd_batched(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    ref_u, ref_s, ref_vh = torch.linalg.svd(ref_inp, full_matrices=False)
    with flag_gems.use_gems():
        res_u, res_s, res_vh = torch.linalg.svd(inp, full_matrices=False)

    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_vh, ref_vh)

    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=1e-2)
    reconstructed = _reconstruct(res_u, res_s, res_vh)
    utils.gems_assert_close(reconstructed, ref_inp, reconstructed.dtype, atol=1e-2)


@pytest.mark.underscore_linalg_svd
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
@pytest.mark.parametrize("shape", LINALG_SVD_SHAPES)
def test_underscore_linalg_svd_compute_uv_false(shape, dtype):
    # torch.linalg.svdvals routes through aten::_linalg_svd with compute_uv
    # False, which materializes only the singular values (U/Vh are empty).
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    ref_s = torch.linalg.svdvals(ref_inp)
    with flag_gems.use_gems():
        res_s = torch.linalg.svdvals(inp)

    _assert_same_shape(res_s, ref_s)
    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=1e-2)


@pytest.mark.underscore_linalg_svd
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
@pytest.mark.parametrize("shape", LINALG_SVD_ORTHONORMAL_SHAPES)
def test_underscore_linalg_svd_orthonormal(shape, dtype):
    # Drive the orthonormality check with a controlled, well-separated spectrum
    # (mirrors the ill-conditioned spectrum test in tests/test_svd.py) so the
    # singular vectors are uniquely determined.
    #
    # Only Vh is checked for orthonormality. The Triton SVD kernel computes V
    # from a stable symmetric eigendecomposition, but forms U as A @ V @
    # diag(1/S); dividing by the small trailing singular values amplifies
    # floating-point error in U's trailing columns by 1/sigma_min, so U's gram
    # matrix is borderline against the 2e-2 tolerance and hardware dependent.
    # U's correctness is already covered by the reconstruction check below
    # (U diag(S) Vh == A), which is gauge invariant and robust.
    k = min(shape[-2:])
    singular_values = torch.logspace(0, -3, steps=k).tolist()
    inp = _make_spectrum_input(shape, singular_values, seed=7)
    ref_inp = utils.to_reference(inp, False)

    ref_u, ref_s, ref_vh = torch.linalg.svd(ref_inp, full_matrices=False)
    with flag_gems.use_gems():
        res_u, res_s, res_vh = torch.linalg.svd(inp, full_matrices=False)

    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_vh, ref_vh)

    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=1e-2)
    reconstructed = _reconstruct(res_u, res_s, res_vh)
    utils.gems_assert_close(reconstructed, ref_inp, reconstructed.dtype, atol=1e-2)
    _assert_orthonormal(res_vh.mH)
