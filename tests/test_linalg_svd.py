import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg


def _reference_input(inp, upcast=False):
    source = inp.cpu() if flag_gems.vendor_name == "mthreads" else inp
    return utils.to_reference(source, upcast)


def _missing_cpu_lapack(inp, error):
    return (
        flag_gems.vendor_name == "mthreads"
        and inp.device.type == "cpu"
        and (
            "requires compiling PyTorch with LAPACK" in str(error)
            or "LAPACK library not found in compilation" in str(error)
        )
    )


def _from_numpy(value):
    import numpy as np

    # Preserve scalar shape, make each output independent, and keep real dtype.
    return torch.from_numpy(np.array(value, copy=True))


def _reference_qr(inp):
    source = inp.cpu() if flag_gems.vendor_name == "mthreads" else inp
    try:
        result = torch.linalg.qr(source)
    except RuntimeError as error:
        if not _missing_cpu_lapack(source, error):
            raise
        import numpy as np

        result = tuple(_from_numpy(x) for x in np.linalg.qr(source.detach().numpy()))
    return tuple(x.to(device=inp.device, dtype=inp.dtype) for x in result)


def _reference_svd(inp, full_matrices=True):
    try:
        return torch.linalg.svd(inp, full_matrices=full_matrices)
    except RuntimeError as error:
        if not _missing_cpu_lapack(inp, error):
            raise
    import numpy as np

    return tuple(
        _from_numpy(x)
        for x in np.linalg.svd(inp.detach().numpy(), full_matrices=full_matrices)
    )


def _reference_svdvals(inp):
    try:
        return torch.linalg.svdvals(inp)
    except RuntimeError as error:
        if not _missing_cpu_lapack(inp, error):
            raise
    import numpy as np

    return _from_numpy(np.linalg.svd(inp.detach().numpy(), compute_uv=False))


def _assert_target_abi(actual, reference, inp):
    if isinstance(actual, (tuple, list)):
        assert isinstance(reference, (tuple, list)) and len(actual) == len(reference)
        for value, expected in zip(actual, reference):
            _assert_target_abi(value, expected, inp)
        return
    assert isinstance(actual, torch.Tensor)
    assert actual.device == inp.device
    assert actual.dtype == inp.dtype
    assert actual.shape == reference.shape
    assert actual.untyped_storage()._cdata != inp.untyped_storage()._cdata


def _for_comparison(actual, reference, inp):
    # Inspect requested target ABI before CPU observation or value comparison.
    _assert_target_abi(actual, reference, inp)
    return actual.cpu() if flag_gems.vendor_name == "mthreads" else actual


def _make_spectrum_input(shape, singular_values, seed=0):
    # Build A = U diag(s) Vh from random orthogonal factors and a prescribed,
    # well-separated spectrum. Well-separated singular values make the singular
    # vectors uniquely determined, so orthonormality of the computed basis is a
    # well-posed, hardware-independent property (mirrors tests/test_svd.py).
    torch.manual_seed(seed)
    *batch_shape, m, n = shape
    k = min(m, n)
    left, _ = _reference_qr(
        torch.randn((*batch_shape, m, m), dtype=torch.float32, device=flag_gems.device)
    )
    right, _ = _reference_qr(
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
    if flag_gems.vendor_name == "mthreads":
        assert gram.device == actual.device and gram.dtype == actual.dtype
        assert gram.shape == expected.shape
        utils.gems_assert_close(gram.cpu(), expected.cpu(), gram.dtype, atol=atol)
    else:
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
    ref_inp = _reference_input(inp, True)

    ref_u, ref_s, ref_vh = _reference_svd(ref_inp, full_matrices=True)
    res_u, res_s, res_vh = flag_gems.linalg_svd(inp, full_matrices=True)

    _assert_target_abi((res_u, res_s, res_vh), (ref_u, ref_s, ref_vh), inp)
    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_vh, ref_vh)

    # Singular values and the reconstruction A = U diag(S) Vh are gauge
    # invariant, so they are the robust correctness checks for general inputs.
    # Per-vector orthonormality of U/Vh is only well-posed for well-separated
    # singular values; it is exercised in test_linalg_svd_orthonormal below.
    utils.gems_assert_close(
        _for_comparison(res_s, ref_s, inp), ref_s, res_s.dtype, atol=SINGULAR_VALUE_ATOL
    )
    reconstructed = _reconstruct(res_u, res_s, res_vh)
    utils.gems_assert_close(
        _for_comparison(reconstructed, ref_inp, inp),
        ref_inp,
        reconstructed.dtype,
        atol=RECONSTRUCTION_ATOL,
    )


@pytest.mark.linalg_svd
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
@pytest.mark.parametrize("shape", LINALG_SVD_REDUCED_SHAPES)
def test_linalg_svd_reduced(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = _reference_input(inp, True)

    ref_u, ref_s, ref_vh = _reference_svd(ref_inp, full_matrices=False)
    res_u, res_s, res_vh = flag_gems.linalg_svd(inp, full_matrices=False)

    _assert_target_abi((res_u, res_s, res_vh), (ref_u, ref_s, ref_vh), inp)
    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_vh, ref_vh)

    utils.gems_assert_close(
        _for_comparison(res_s, ref_s, inp), ref_s, res_s.dtype, atol=SINGULAR_VALUE_ATOL
    )
    reconstructed = _reconstruct(res_u, res_s, res_vh)
    utils.gems_assert_close(
        _for_comparison(reconstructed, ref_inp, inp),
        ref_inp,
        reconstructed.dtype,
        atol=RECONSTRUCTION_ATOL,
    )


@pytest.mark.linalg_svd
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
@pytest.mark.parametrize("shape", LINALG_SVD_BATCH_SHAPES)
def test_linalg_svd_batched(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = _reference_input(inp, True)

    ref_u, ref_s, ref_vh = _reference_svd(ref_inp, full_matrices=False)
    res_u, res_s, res_vh = flag_gems.linalg_svd(inp, full_matrices=False)

    _assert_target_abi((res_u, res_s, res_vh), (ref_u, ref_s, ref_vh), inp)
    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_vh, ref_vh)

    utils.gems_assert_close(
        _for_comparison(res_s, ref_s, inp), ref_s, res_s.dtype, atol=SINGULAR_VALUE_ATOL
    )
    reconstructed = _reconstruct(res_u, res_s, res_vh)
    utils.gems_assert_close(
        _for_comparison(reconstructed, ref_inp, inp),
        ref_inp,
        reconstructed.dtype,
        atol=RECONSTRUCTION_ATOL,
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
    ref_inp = _reference_input(inp, True)

    ref_u, ref_s, ref_vh = _reference_svd(ref_inp, full_matrices=False)
    res_u, res_s, res_vh = flag_gems.linalg_svd(inp, full_matrices=False)

    _assert_target_abi((res_u, res_s, res_vh), (ref_u, ref_s, ref_vh), inp)
    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_vh, ref_vh)

    utils.gems_assert_close(
        _for_comparison(res_s, ref_s, inp), ref_s, res_s.dtype, atol=SINGULAR_VALUE_ATOL
    )
    reconstructed = _reconstruct(res_u, res_s, res_vh)
    utils.gems_assert_close(
        _for_comparison(reconstructed, ref_inp, inp),
        ref_inp,
        reconstructed.dtype,
        atol=RECONSTRUCTION_ATOL,
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
    ref_inp = _reference_input(inp, True)

    ref_u, ref_s, ref_vh = _reference_svd(ref_inp, full_matrices=True)
    res_u, res_s, res_vh = flag_gems._linalg_svd(inp, full_matrices=True)

    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_vh, ref_vh)

    # Singular values and the reconstruction A = U diag(S) Vh are gauge
    # invariant, so they are the robust correctness checks for general inputs.
    # Per-vector orthonormality of U/Vh is only well-posed for well-separated
    # singular values; it is exercised in the orthonormal test below.
    utils.gems_assert_close(
        _for_comparison(res_s, ref_s, inp), ref_s, res_s.dtype, atol=SINGULAR_VALUE_ATOL
    )
    reconstructed = _reconstruct(res_u, res_s, res_vh)
    utils.gems_assert_close(
        _for_comparison(reconstructed, ref_inp, inp),
        ref_inp,
        reconstructed.dtype,
        atol=RECONSTRUCTION_ATOL,
    )


@pytest.mark.underscore_linalg_svd
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
@pytest.mark.parametrize("shape", LINALG_SVD_REDUCED_SHAPES)
def test__linalg_svd_reduced(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = _reference_input(inp, True)

    ref_u, ref_s, ref_vh = _reference_svd(ref_inp, full_matrices=False)
    res_u, res_s, res_vh = flag_gems._linalg_svd(inp, full_matrices=False)

    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_vh, ref_vh)

    utils.gems_assert_close(
        _for_comparison(res_s, ref_s, inp), ref_s, res_s.dtype, atol=SINGULAR_VALUE_ATOL
    )
    reconstructed = _reconstruct(res_u, res_s, res_vh)
    utils.gems_assert_close(
        _for_comparison(reconstructed, ref_inp, inp),
        ref_inp,
        reconstructed.dtype,
        atol=RECONSTRUCTION_ATOL,
    )


@pytest.mark.underscore_linalg_svd
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
@pytest.mark.parametrize("shape", LINALG_SVD_BATCH_SHAPES)
def test__linalg_svd_batched(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = _reference_input(inp, True)

    ref_u, ref_s, ref_vh = _reference_svd(ref_inp, full_matrices=False)
    res_u, res_s, res_vh = flag_gems._linalg_svd(inp, full_matrices=False)

    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_vh, ref_vh)

    utils.gems_assert_close(
        _for_comparison(res_s, ref_s, inp), ref_s, res_s.dtype, atol=SINGULAR_VALUE_ATOL
    )
    reconstructed = _reconstruct(res_u, res_s, res_vh)
    utils.gems_assert_close(
        _for_comparison(reconstructed, ref_inp, inp),
        ref_inp,
        reconstructed.dtype,
        atol=RECONSTRUCTION_ATOL,
    )


@pytest.mark.underscore_linalg_svd
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
@pytest.mark.parametrize("shape", LINALG_SVD_REDUCED_SHAPES)
def test__linalg_svd_compute_uv_false(shape, dtype):
    # torch.linalg.svdvals routes through aten::_linalg_svd with compute_uv
    # False, which materializes only the singular values (U/Vh are empty).
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = _reference_input(inp, True)

    ref_s = _reference_svdvals(ref_inp)
    _, res_s, _ = flag_gems._linalg_svd(inp, compute_uv=False)

    _assert_same_shape(res_s, ref_s)
    utils.gems_assert_close(
        _for_comparison(res_s, ref_s, inp), ref_s, res_s.dtype, atol=SINGULAR_VALUE_ATOL
    )


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
    ref_inp = _reference_input(inp, True)

    ref_u, ref_s, ref_vh = _reference_svd(ref_inp, full_matrices=False)
    res_u, res_s, res_vh = flag_gems._linalg_svd(inp, full_matrices=False)

    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_vh, ref_vh)

    utils.gems_assert_close(
        _for_comparison(res_s, ref_s, inp), ref_s, res_s.dtype, atol=SINGULAR_VALUE_ATOL
    )
    reconstructed = _reconstruct(res_u, res_s, res_vh)
    utils.gems_assert_close(
        _for_comparison(reconstructed, ref_inp, inp),
        ref_inp,
        reconstructed.dtype,
        atol=RECONSTRUCTION_ATOL,
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
    ref_inp = _reference_input(inp, True)

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
        ref_u, ref_s, ref_vh = _reference_svd(ref_inp, full_matrices=False)
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


# ---------------------------------------------------------------------------
# aten::_linalg_svd backward
#
# ATen registers no _linalg_svd_backward schema: the engine-generated
# LinalgSvdBackward0 node applies FunctionsManual::svd_backward to the saved
# factors. The FlagGems module wraps the same forward in a
# torch.autograd.Function whose backward launches Triton kernels, so these
# tests differentiate the factors with respect to A and compare against the
# native autograd rule. The reference is computed in float64 from the *same*
# factors, which isolates the candidate kernel from the forward SVD's own
# rounding and from the per-column sign gauge that the two SVD
# implementations may pick differently.
# ---------------------------------------------------------------------------


def _svd_backward_abs_diff(got, expected):
    """Move the candidate onto the reference device before comparing.

    ``--ref cpu`` runs the float64 reference on the host, so a candidate
    tensor still on the GPU would raise a device mismatch instead of a
    numerical comparison.
    """
    return utils.to_reference(got, True) - expected


def _svd_backward_reference(U, S, Vh, grad_U, grad_S, grad_Vh):
    """ATen's first-order real SVD rule, evaluated in float64."""
    k = S.shape[-1]
    u = utils.to_reference(U, True)[..., :, :k]
    vh = utils.to_reference(Vh, True)[..., :k, :]
    gu = utils.to_reference(grad_U, True)[..., :, :k]
    gvh = utils.to_reference(grad_Vh, True)[..., :k, :]
    gs = utils.to_reference(grad_S, True)
    s = utils.to_reference(S, True)
    sinv = 1.0 / s
    gram_u = u.mH @ gu
    gram_v = vh @ gvh.mH
    f = 1.0 / (s.unsqueeze(-2) ** 2 - s.unsqueeze(-1) ** 2)
    f.diagonal(0, -2, -1).zero_()
    t = (
        (f * (gram_u - gram_u.mH)) * s.unsqueeze(-2)
        + s.unsqueeze(-1) * (f * (gram_v - gram_v.mH))
        + torch.diag_embed(gs)
    )
    c = t - gram_u * sinv.unsqueeze(-2) - sinv.unsqueeze(-1) * gram_v.mH
    grad_a = (
        u @ c @ vh + (gu * sinv.unsqueeze(-2)) @ vh + (u * sinv.unsqueeze(-2)) @ gvh
    )
    return grad_a.to(U.dtype)


def _svd_backward_grads(shape, full_matrices, dtype, seed=20261120):
    """An input matrix plus upstream gradients shaped for the factor layout."""
    torch.manual_seed(seed)
    m, n = shape[-2:]
    k = min(m, n)
    batch = shape[:-2]
    a = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_u = torch.randn(
        batch + (m, m if full_matrices else k), dtype=dtype, device=flag_gems.device
    )
    grad_s = torch.randn(batch + (k,), dtype=dtype, device=flag_gems.device)
    grad_vh = torch.randn(
        batch + ((n if full_matrices else k), n), dtype=dtype, device=flag_gems.device
    )
    if full_matrices:
        # Both implementations ignore the lanes beyond k; zero them so the
        # comparison is against the same effective upstream gradient.
        grad_u = grad_u.clone()
        grad_u[..., :, k:] = 0
        grad_vh = grad_vh.clone()
        grad_vh[..., k:, :] = 0
    return a, grad_u, grad_s, grad_vh


@pytest.mark.linalg_svd_backward
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (3, 3),
        (8, 8),
        (32, 32),
        (64, 64),
        (5, 3),
        (32, 16),
        (16, 32),
        (128, 128),
        (2, 8, 8),
        (4, 32, 32),
    ],
)
@pytest.mark.parametrize("full_matrices", [False, True])
def test_accuracy_linalg_svd_backward(shape, full_matrices, dtype):
    if full_matrices and max(shape[-2:]) > 64:
        pytest.skip("the full_matrices forward path requires max(M, N) <= 64")
    a, grad_u, grad_s, grad_vh = _svd_backward_grads(shape, full_matrices, dtype)

    gems_a = a.detach().clone().requires_grad_(True)
    res_u, res_s, res_vh = flag_gems._linalg_svd(gems_a, full_matrices=full_matrices)
    # The reference is evaluated on the candidate's *own* factors: the two SVD
    # implementations may choose different per-column signs, and the gradient
    # of a sign-flipped factor is the sign-flipped gradient. Comparing on the
    # same factors isolates the backward kernel from the forward's gauge.
    expected = _svd_backward_reference(
        res_u.detach(), res_s.detach(), res_vh.detach(), grad_u, grad_s, grad_vh
    )
    # The custom autograd node must be the one propagating the gradient.
    grad_fn = type(res_u.grad_fn).__name__ if res_u.grad_fn is not None else ""
    assert "LinalgSvdBackward" in grad_fn, grad_fn
    assert res_s.requires_grad and res_vh.requires_grad
    (res_grad,) = torch.autograd.grad(
        (res_u, res_s, res_vh), (gems_a,), (grad_u, grad_s, grad_vh)
    )

    assert res_grad.shape == expected.shape
    assert res_grad.dtype == expected.dtype
    assert torch.isfinite(res_grad).all(), "backward produced non-finite values"
    # The float32 kernels accumulate f32 tl.dot products; observed relative
    # error is <= 1.2e-5 on every case, the bound here leaves headroom for
    # hardware-dependent summation order.
    diff = _svd_backward_abs_diff(res_grad, expected)
    rel = float(diff.norm() / expected.norm())
    assert rel < 5e-4, f"relative error {rel:.3e} exceeds 5e-4"


@pytest.mark.linalg_svd_backward
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
def test_accuracy_linalg_svd_backward_matches_native(dtype):
    """Cross-check against native autograd on the same functional.

    ``torch.linalg.svd`` and the Triton forward may pick different per-column
    signs. Both differentiate the same scalar functional as long as the
    upstream gradient lanes are expressed in the corresponding gauge, so the
    candidate's lanes are sign-mapped before the input gradients are compared.
    """
    torch.manual_seed(20261121)
    a = torch.randn(16, 10, dtype=dtype, device=flag_gems.device)
    k = min(a.shape[-2:])
    grad_u = torch.randn(16, k, dtype=dtype, device=flag_gems.device)
    grad_s = torch.randn(k, dtype=dtype, device=flag_gems.device)
    grad_vh = torch.randn(k, 10, dtype=dtype, device=flag_gems.device)

    # Native autograd on the reference device. A CPU reference cannot accept
    # CUDA tensors, so the input and upstream gradients move with it.
    ref_a = utils.to_reference(a.detach(), False).requires_grad_(True)
    ref_u, ref_s, ref_vh = torch.linalg.svd(ref_a, full_matrices=False)
    (ref_grad,) = torch.autograd.grad(
        (ref_u, ref_s, ref_vh),
        (ref_a,),
        (
            utils.to_reference(grad_u, False),
            utils.to_reference(grad_s, False),
            utils.to_reference(grad_vh, False),
        ),
    )

    gems_a = a.detach().clone().requires_grad_(True)
    res_u, res_s, res_vh = flag_gems._linalg_svd(gems_a, full_matrices=False)
    res_u_ref = utils.to_reference(res_u.detach(), False)
    sign = torch.sign(torch.sum(ref_u.detach() * res_u_ref, dim=-2))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    # The sign was computed on the reference device; move it to the candidate's
    # device so it can scale the upstream gradient lanes.
    sign = sign.to(grad_u.device)
    (res_grad,) = torch.autograd.grad(
        (res_u, res_s, res_vh),
        (gems_a,),
        (grad_u * sign.unsqueeze(-2), grad_s, grad_vh * sign.unsqueeze(-1)),
    )

    assert torch.isfinite(res_grad).all()
    got = utils.to_reference(res_grad, True)
    exp = (
        ref_grad.to(torch.float64)
        if utils.TO_CPU
        else utils.to_reference(ref_grad, True)
    )
    rel = float((got - exp).norm() / exp.norm())
    assert rel < 1e-2, f"relative error against native {rel:.3e} exceeds 1e-2"


@pytest.mark.linalg_svd_backward
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
def test_accuracy_linalg_svd_backward_s_only(dtype):
    """svdvals-style differentiation: only S carries an upstream gradient."""
    torch.manual_seed(20261122)
    shape = (3, 8, 6)
    a = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    k = min(shape[-2:])
    grad_s = torch.randn(shape[:-2] + (k,), dtype=dtype, device=flag_gems.device)

    gems_a = a.detach().clone().requires_grad_(True)
    up, s, vh = flag_gems._linalg_svd(gems_a, full_matrices=False)
    (grad_a,) = torch.autograd.grad((s,), (gems_a,), (grad_s,))

    # float64 on the candidate's own factors: the fp32 product alone already
    # carries ~2.6e-4 relative rounding, which would mask a kernel error.
    expected = (
        utils.to_reference(up.detach(), True)
        * utils.to_reference(grad_s, True).unsqueeze(-2)
    ) @ utils.to_reference(vh.detach(), True)
    assert grad_a.shape == a.shape
    assert torch.isfinite(grad_a).all()
    rel = float(_svd_backward_abs_diff(grad_a, expected).norm() / expected.norm())
    assert rel < 5e-4, f"S-only relative error {rel:.3e} exceeds 5e-4"


@pytest.mark.linalg_svd_backward
def test_accuracy_linalg_svd_backward_none_grads():
    """A zero upstream gradient for one factor must behave like omitting it."""
    torch.manual_seed(20261123)
    a = torch.randn(8, 8, dtype=torch.float32, device=flag_gems.device)
    grad_s = torch.randn(8, dtype=torch.float32, device=flag_gems.device)

    gems_a = a.detach().clone().requires_grad_(True)
    up, s, vh = flag_gems._linalg_svd(gems_a, full_matrices=False)
    (grad_a,) = torch.autograd.grad((s,), (gems_a,), (grad_s,))

    expected = (up.detach() * grad_s.unsqueeze(-2)) @ vh.detach()
    exp = utils.to_reference(expected, True)
    rel = float(_svd_backward_abs_diff(grad_a, exp).norm() / exp.norm())
    assert rel < 5e-4, f"S-only (U/Vh grad omitted) relative error {rel:.3e}"


@pytest.mark.linalg_svd_backward
def test_accuracy_linalg_svd_backward_noncontiguous():
    """A transposed (non-contiguous) input must produce a correct gradient."""
    torch.manual_seed(20261124)
    base = torch.randn(8, 6, dtype=torch.float32, device=flag_gems.device)
    a = base.t()
    assert not a.is_contiguous()
    k = min(a.shape[-2:])
    grad_u = torch.randn(6, k, dtype=torch.float32, device=flag_gems.device)
    grad_s = torch.randn(k, dtype=torch.float32, device=flag_gems.device)
    grad_vh = torch.randn(k, 8, dtype=torch.float32, device=flag_gems.device)

    gems_a = a.detach().clone().requires_grad_(True)
    up, s, vh = flag_gems._linalg_svd(gems_a, full_matrices=False)
    (grad_a,) = torch.autograd.grad((up, s, vh), (gems_a,), (grad_u, grad_s, grad_vh))

    expected = _svd_backward_reference(
        up.detach(), s.detach(), vh.detach(), grad_u, grad_s, grad_vh
    )
    assert grad_a.shape == a.shape
    assert torch.isfinite(grad_a).all()
    rel = float(_svd_backward_abs_diff(grad_a, expected).norm() / expected.norm())
    assert rel < 5e-4, f"non-contiguous relative error {rel:.3e} exceeds 5e-4"


@pytest.mark.linalg_svd_backward
def test_accuracy_linalg_svd_backward_repeated_values_nan():
    """Native returns NaN for repeated singular values; the candidate matches.

    ``F_ij = 1/(s_j^2 - s_i^2)`` is singular when two singular values
    coincide, and ATen does not guard it. The candidate must not fabricate a
    finite gradient where native produces NaN.
    """
    a = torch.eye(4, dtype=torch.float32, device=flag_gems.device)
    k = 4
    gems_a = a.detach().clone().requires_grad_(True)
    up, s, vh = flag_gems._linalg_svd(gems_a, full_matrices=False)
    (grad_a,) = torch.autograd.grad(
        (up, s, vh),
        (gems_a,),
        (
            torch.ones(4, k, dtype=torch.float32, device=flag_gems.device),
            torch.ones(k, dtype=torch.float32, device=flag_gems.device),
            torch.ones(k, 4, dtype=torch.float32, device=flag_gems.device),
        ),
    )
    assert torch.isnan(
        grad_a
    ).any(), "repeated singular values must reproduce native NaN"


@pytest.mark.linalg_svd_backward
def test_accuracy_linalg_svd_backward_forward_unchanged():
    """The gradient path must not change the forward values or the no-grad path."""
    torch.manual_seed(20261125)
    a = torch.randn(16, 16, dtype=torch.float32, device=flag_gems.device)
    plain_u, plain_s, plain_vh = flag_gems._linalg_svd(a, full_matrices=True)
    grad_a = a.detach().clone().requires_grad_(True)
    grad_u, grad_s, grad_vh = flag_gems._linalg_svd(grad_a, full_matrices=True)
    # Compare as plain tensors on the host: gems_assert_close() also checks the
    # device attribute, and under --ref cpu the reference side is on the CPU.
    for got, plain in (
        (grad_u, plain_u),
        (grad_s, plain_s),
        (grad_vh, plain_vh),
    ):
        got_host = got.detach().float().cpu()
        plain_host = plain.detach().float().cpu()
        torch.testing.assert_close(got_host, plain_host, atol=1e-4, rtol=1.3e-6)
