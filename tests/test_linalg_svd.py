import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


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
# (some=False) path additionally requires max(m, n) <= 64.
LINALG_SVD_SHAPES = [(3, 3), (4, 4), (8, 8), (3, 5), (5, 3), (16, 16)]
# Small batched matrices that satisfy the reduced-path max(m, n) <= 64 limit.
LINALG_SVD_BATCH_SHAPES = [(2, 4, 4), (3, 8, 8)]


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

    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=1e-2)
    reconstructed = _reconstruct(res_u, res_s, res_vh)
    utils.gems_assert_close(reconstructed, ref_inp, reconstructed.dtype, atol=1e-2)
    _assert_orthonormal(res_u)
    _assert_orthonormal(res_vh.mH)


@pytest.mark.linalg_svd
@pytest.mark.parametrize("dtype", LINALG_SVD_DTYPES)
@pytest.mark.parametrize("shape", LINALG_SVD_SHAPES)
def test_linalg_svd_reduced(shape, dtype):
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
    _assert_orthonormal(res_u)
    _assert_orthonormal(res_vh.mH)


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

    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=1e-2)
    reconstructed = _reconstruct(res_u, res_s, res_vh)
    utils.gems_assert_close(reconstructed, ref_inp, reconstructed.dtype, atol=1e-2)
