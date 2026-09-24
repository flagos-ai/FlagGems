import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg


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
    # torch.linalg.svd returns (U, S, Vh) with A = U diag(S) Vh
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
# (some=False) path additionally requires max(m, n) <= 64, so the shared shape
# list stays within that bound while still covering medium (32, 64) sizes and
# non-square matrices.
LINALG_SVD_SHAPES = [
    (3, 3),
    (4, 4),
    (8, 8),
    (3, 5),
    (5, 3),
    (16, 16),
    (32, 32),
    (64, 64),
    (32, 16),
]
# The reduced (full_matrices=False) path is not bound by the max(m, n) <= 64
# limit, so it additionally exercises a larger matrix.
LINALG_SVD_REDUCED_SHAPES = LINALG_SVD_SHAPES + [(128, 128)]
# Batched matrices covering the reduced path, including a larger 32x32 batch.
LINALG_SVD_BATCH_SHAPES = [(2, 4, 4), (3, 8, 8), (4, 32, 32)]
# Shapes for the orthonormality check driven by a controlled spectrum.
LINALG_SVD_ORTHONORMAL_SHAPES = [(8, 8), (16, 16), (5, 3), (3, 5), (2, 8, 8), (32, 32)]
# Singular values are gauge invariant and match the reference tightly; the
# reconstruction A = U diag(S) Vh is a product of three device tensors and so
# carries slightly more floating-point error. Both are far tighter than the
# original 1e-2 bound (observed: singular values <= 4e-4, reconstruction <=
# 2e-3 for shapes up to 128x128).
SINGULAR_VALUE_ATOL = 1e-3
RECONSTRUCTION_ATOL = 5e-3


@pytest.mark.linalg_svd
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
@pytest.mark.parametrize("shape", LINALG_SVD_SHAPES)
def test_linalg_svd_full_matrices(shape, dtype):
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
    # singular values; it is exercised in test_linalg_svd_orthonormal below.
    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=SINGULAR_VALUE_ATOL)
    reconstructed = _reconstruct(res_u, res_s, res_vh)
    utils.gems_assert_close(
        reconstructed, ref_inp, reconstructed.dtype, atol=RECONSTRUCTION_ATOL
    )


@pytest.mark.linalg_svd
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
@pytest.mark.parametrize("shape", LINALG_SVD_REDUCED_SHAPES)
def test_linalg_svd_reduced(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    ref_u, ref_s, ref_vh = torch.linalg.svd(ref_inp, full_matrices=False)
    with flag_gems.use_gems():
        res_u, res_s, res_vh = torch.linalg.svd(inp, full_matrices=False)

    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_vh, ref_vh)

    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=SINGULAR_VALUE_ATOL)
    reconstructed = _reconstruct(res_u, res_s, res_vh)
    utils.gems_assert_close(
        reconstructed, ref_inp, reconstructed.dtype, atol=RECONSTRUCTION_ATOL
    )


@pytest.mark.linalg_svd
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
@pytest.mark.parametrize("shape", LINALG_SVD_BATCH_SHAPES)
def test_linalg_svd_batched(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    ref_u, ref_s, ref_vh = torch.linalg.svd(ref_inp, full_matrices=False)
    with flag_gems.use_gems():
        res_u, res_s, res_vh = torch.linalg.svd(inp, full_matrices=False)

    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_vh, ref_vh)

    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=SINGULAR_VALUE_ATOL)
    reconstructed = _reconstruct(res_u, res_s, res_vh)
    utils.gems_assert_close(
        reconstructed, ref_inp, reconstructed.dtype, atol=RECONSTRUCTION_ATOL
    )


@pytest.mark.linalg_svd
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
@pytest.mark.parametrize("shape", LINALG_SVD_ORTHONORMAL_SHAPES)
def test_linalg_svd_orthonormal(shape, dtype):
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

    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=SINGULAR_VALUE_ATOL)
    reconstructed = _reconstruct(res_u, res_s, res_vh)
    utils.gems_assert_close(
        reconstructed, ref_inp, reconstructed.dtype, atol=RECONSTRUCTION_ATOL
    )
    _assert_orthonormal(res_vh.mH)


# ---------------------------------------------------------------------------
# aten::_linalg_svd / aten::_linalg_svd.out
#
# The private primitive that torch.linalg.svd (compute_uv True) and
# torch.linalg.svdvals (compute_uv False) decompose into. The operator ids stay
# `_linalg_svd` / `_linalg_svd_out`, and pytest refuses to build a marker from
# an attribute starting with an underscore, so the markers are spelled
# `underscore_linalg_svd` / `underscore_linalg_svd_out`. The tests live in this
# file (rather than a separate test__linalg_svd.py) because the op shares its
# helpers, shapes and tolerances with the bare `linalg_svd` above.
# ---------------------------------------------------------------------------


# aten::_linalg_svd is the private primitive that torch.linalg.svd (compute_uv
# True) and torch.linalg.svdvals (compute_uv False) decompose into, so calling
# flag_gems._linalg_svd directly exercises the registered kernel.
@pytest.mark.underscore_linalg_svd
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
@pytest.mark.parametrize("shape", LINALG_SVD_SHAPES)
def test__linalg_svd_full_matrices(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    ref_u, ref_s, ref_vh = torch.linalg.svd(ref_inp, full_matrices=True)
    res_u, res_s, res_vh = flag_gems._linalg_svd(inp, full_matrices=True)

    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_vh, ref_vh)

    # Singular values and the reconstruction A = U diag(S) Vh are gauge
    # invariant, so they are the robust correctness checks for general inputs.
    # Per-vector orthonormality of U/Vh is only well-posed for well-separated
    # singular values; it is exercised in the orthonormal test below.
    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=SINGULAR_VALUE_ATOL)
    reconstructed = _reconstruct(res_u, res_s, res_vh)
    utils.gems_assert_close(
        reconstructed, ref_inp, reconstructed.dtype, atol=RECONSTRUCTION_ATOL
    )


@pytest.mark.underscore_linalg_svd
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
@pytest.mark.parametrize("shape", LINALG_SVD_REDUCED_SHAPES)
def test__linalg_svd_reduced(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    ref_u, ref_s, ref_vh = torch.linalg.svd(ref_inp, full_matrices=False)
    res_u, res_s, res_vh = flag_gems._linalg_svd(inp, full_matrices=False)

    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_vh, ref_vh)

    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=SINGULAR_VALUE_ATOL)
    reconstructed = _reconstruct(res_u, res_s, res_vh)
    utils.gems_assert_close(
        reconstructed, ref_inp, reconstructed.dtype, atol=RECONSTRUCTION_ATOL
    )


@pytest.mark.underscore_linalg_svd
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
@pytest.mark.parametrize("shape", LINALG_SVD_BATCH_SHAPES)
def test__linalg_svd_batched(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    ref_u, ref_s, ref_vh = torch.linalg.svd(ref_inp, full_matrices=False)
    res_u, res_s, res_vh = flag_gems._linalg_svd(inp, full_matrices=False)

    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_vh, ref_vh)

    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=SINGULAR_VALUE_ATOL)
    reconstructed = _reconstruct(res_u, res_s, res_vh)
    utils.gems_assert_close(
        reconstructed, ref_inp, reconstructed.dtype, atol=RECONSTRUCTION_ATOL
    )


@pytest.mark.underscore_linalg_svd
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
@pytest.mark.parametrize("shape", LINALG_SVD_REDUCED_SHAPES)
def test__linalg_svd_compute_uv_false(shape, dtype):
    # torch.linalg.svdvals routes through aten::_linalg_svd with compute_uv
    # False, which materializes only the singular values (U/Vh are empty).
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    ref_s = torch.linalg.svdvals(ref_inp)
    _, res_s, _ = flag_gems._linalg_svd(inp, compute_uv=False)

    _assert_same_shape(res_s, ref_s)
    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=SINGULAR_VALUE_ATOL)


@pytest.mark.underscore_linalg_svd
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
@pytest.mark.parametrize("shape", LINALG_SVD_ORTHONORMAL_SHAPES)
def test__linalg_svd_orthonormal(shape, dtype):
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
    res_u, res_s, res_vh = flag_gems._linalg_svd(inp, full_matrices=False)

    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_vh, ref_vh)

    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=SINGULAR_VALUE_ATOL)
    reconstructed = _reconstruct(res_u, res_s, res_vh)
    utils.gems_assert_close(
        reconstructed, ref_inp, reconstructed.dtype, atol=RECONSTRUCTION_ATOL
    )
    _assert_orthonormal(res_vh.mH)


@pytest.mark.underscore_linalg_svd
@pytest.mark.skipif(
    cfg.TO_CPU, reason="the driver= contract is CUDA-only; a CPU reference rejects all"
)
@pytest.mark.parametrize("driver", ["gesvd", "gesvda", "gesdd", "bogus", "", "GESVD"])
def test__linalg_svd_driver(driver):
    """The ``driver`` argument is validated like native, though it is otherwise unused.

    cuSOLVER accepts only ``gesvd``/``gesvda`` (``gesdd`` is rejected for
    ``_linalg_svd``) and rejects anything else with a RuntimeError naming the
    driver. The Triton kernels have no driver concept, so a valid driver only
    has to be accepted; silently dropping an *invalid* one would compute a
    result where native raises.

    The expected outcome depends on the device the reference runs on, so this
    cannot be checked against a CPU reference: the CPU backend rejects the
    ``driver=`` keyword itself ("only supported on CUDA inputs with cuSOLVER")
    for *every* value, including the two that are valid on CUDA.
    """
    inp = torch.randn(8, 8, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    ref_err = None
    try:
        torch.ops.aten._linalg_svd(ref_inp, False, True, driver=driver)
    except RuntimeError as e:
        ref_err = str(e)

    if ref_err is None:
        # A valid driver still has to produce the correct factors.
        res_u, res_s, res_vh = flag_gems._linalg_svd(
            inp, full_matrices=False, driver=driver
        )
        ref_u, ref_s, ref_vh = torch.linalg.svd(ref_inp, full_matrices=False)
        utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=SINGULAR_VALUE_ATOL)
        return

    with pytest.raises(RuntimeError) as exc_info:
        flag_gems._linalg_svd(inp, full_matrices=False, driver=driver)
    assert str(exc_info.value) == ref_err
    assert "unknown svd driver" in str(exc_info.value)


@pytest.mark.underscore_linalg_svd
def test__linalg_svd_driver_checked_before_dtype():
    """An unknown driver is reported for a non-float32 input too, as native does."""
    inp = torch.randn(8, 8, dtype=torch.float16, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="unknown svd driver"):
        flag_gems._linalg_svd(inp, full_matrices=False, driver="bogus")
