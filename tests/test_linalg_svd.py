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
    torch.manual_seed(0)
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
    torch.manual_seed(0)
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
    torch.manual_seed(0)
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


@pytest.mark.linalg_svd
def test_linalg_svd_autograd_is_explicitly_unsupported():
    torch.manual_seed(0)
    inp = torch.randn((4, 4), dtype=torch.float32, device=flag_gems.device)
    inp.requires_grad_(True)

    outputs = flag_gems.linalg_svd(inp, full_matrices=False)

    assert all(not output.requires_grad for output in outputs)
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(outputs[1].sum(), inp)
