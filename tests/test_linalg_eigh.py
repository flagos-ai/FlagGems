import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# linalg_eigh only supports float32/float64.
#
# The operator (src/flag_gems/ops/linalg_eigh.py) has two execution paths:
#   - n == 2  -> Triton `_eig_2x2_kernel` (analytical 2x2 formula, on device).
#   - n > 2   -> CPU fallback via `torch.linalg.eigh` (breaks CUDA graph compat.).
# Shapes below are split so each path is explicitly exercised and labelled.

# Path A: hits the Triton 2x2 kernel.
EIG_2X2_SHAPES = [(2, 2)]

# Path B: hits the n>2 CPU fallback.
EIG_FALLBACK_SHAPES = [(3, 3), (5, 5), (8, 8), (16, 16), (32, 32)]

# Batched variants: (2, 2, 2) hits the 2x2 kernel per batch element; the rest
# hit the CPU fallback.
EIG_BATCH_2X2_SHAPES = [(2, 2, 2)]
EIG_BATCH_FALLBACK_SHAPES = [(4, 3, 3), (1, 8, 8)]


def make_symmetric_matrix(shape, dtype, device):
    """Create a symmetric matrix for eigendecomposition."""
    A = torch.randn(shape, dtype=dtype, device=device)
    A = (A + A.transpose(-2, -1)) / 2
    return A


def _assert_orthonormal(v, atol=1e-2):
    """Columns of v are eigenvectors: Vᵀ V ≈ I.

    Avoids comparing eigenvectors elementwise, since v and -v are both valid
    eigenvectors (sign ambiguity).
    """
    n = v.shape[-1]
    eye = torch.eye(n, dtype=v.dtype, device=v.device)
    gram = v.transpose(-2, -1) @ v
    expected = utils.to_reference(eye.expand_as(gram), False)
    utils.gems_assert_close(gram, expected, gram.dtype, atol=atol)


def _check_eigh_decomposition(A, eigenvalues, eigenvectors, atol=1e-3):
    """Verify the eigendecomposition via the defining relation A = V diag(w) Vᵀ.

    This is sign-ambiguous-free: any valid eigenbasis reconstructs A and is
    orthonormal, regardless of per-vector sign choices.
    """
    reconstructed = (
        eigenvectors
        @ torch.diag_embed(eigenvalues).to(eigenvectors.dtype)
        @ eigenvectors.transpose(-2, -1)
    )
    ref_A = utils.to_reference(A, False)
    utils.gems_assert_close(reconstructed, ref_A, reconstructed.dtype, atol=atol)
    _assert_orthonormal(eigenvectors)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_2X2_SHAPES,
    ids=[f"gpu_kernel_2x2-{s[0]}x{s[1]}" for s in EIG_2X2_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_2x2_kernel(shape, dtype):
    """n == 2: exercised by the Triton `_eig_2x2_kernel` on device."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    # Eigenvalues
    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    # Eigenvectors: reconstruct + orthonormality (sign-ambiguous-free)
    _check_eigh_decomposition(inp, res_out[0], res_out[1])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_FALLBACK_SHAPES,
    ids=[f"cpu_fallback_n{s[0]}" for s in EIG_FALLBACK_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_cpu_fallback(shape, dtype):
    """n > 2: exercised by the `torch.linalg.eigh` CPU fallback."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    # Eigenvalues
    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    # Eigenvectors: reconstruct + orthonormality (sign-ambiguous-free)
    _check_eigh_decomposition(inp, res_out[0], res_out[1])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_BATCH_2X2_SHAPES,
    ids=[f"gpu_kernel_2x2-batch{s[0]}" for s in EIG_BATCH_2X2_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_batch_2x2_kernel(shape, dtype):
    """Batched n == 2: each batch element hits the Triton 2x2 kernel."""
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
    EIG_BATCH_FALLBACK_SHAPES,
    ids=[f"cpu_fallback-batch{s[-1]}" for s in EIG_BATCH_FALLBACK_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_batch_cpu_fallback(shape, dtype):
    """Batched n > 2: hits the `torch.linalg.eigh` CPU fallback."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    _check_eigh_decomposition(inp, res_out[0], res_out[1])
