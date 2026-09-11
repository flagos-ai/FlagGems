import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _make_diagonally_dominant_csr(n, dtype, device, density=0.3):
    """Build a well-conditioned, invertible sparse CSR matrix of shape (n, n).

    A random sparse pattern is made diagonally dominant so the linear system has
    a stable, unique solution across the tested dtypes.
    """
    dense = torch.randn((n, n), dtype=dtype, device=device)
    # Sparsify off-diagonal entries.
    mask = torch.rand((n, n), device=device) > density
    dense = dense.masked_fill(mask, 0.0)
    # Enforce diagonal dominance for invertibility / numerical stability.
    row_abs_sum = dense.abs().sum(dim=1)
    diag = row_abs_sum + 1.0
    dense = dense - torch.diag(torch.diagonal(dense)) + torch.diag(diag)
    return dense.to_sparse_csr(), dense


@pytest.mark.spsolve
@pytest.mark.parametrize("n", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_spsolve_vector_rhs(n, dtype):
    """Solve A @ x = b for a single right-hand-side vector."""
    A_csr, A_dense = _make_diagonally_dominant_csr(n, dtype, flag_gems.device)
    b = torch.randn((n,), dtype=dtype, device=flag_gems.device)

    # Reference: mathematically equivalent dense solve (torch has no runnable
    # native _spsolve in this environment: CPU has no kernel, CUDA needs cuDSS).
    ref_A = utils.to_reference(A_dense, True)
    ref_b = utils.to_reference(b, True)
    ref_out = torch.linalg.solve(ref_A, ref_b)

    res_out = flag_gems._spsolve(A_csr, b)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.spsolve
@pytest.mark.parametrize("n", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("k", [1, 2, 5])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_spsolve_matrix_rhs(n, k, dtype):
    """Solve A @ X = B for a multi-column right-hand side."""
    A_csr, A_dense = _make_diagonally_dominant_csr(n, dtype, flag_gems.device)
    B = torch.randn((n, k), dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A_dense, True)
    ref_B = utils.to_reference(B, True)
    ref_out = torch.linalg.solve(ref_A, ref_B)

    res_out = flag_gems._spsolve(A_csr, B)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.spsolve
@pytest.mark.parametrize("n", [8, 16, 32])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_spsolve_right(n, dtype):
    """Solve X @ A = B (left=False)."""
    A_csr, A_dense = _make_diagonally_dominant_csr(n, dtype, flag_gems.device)
    B = torch.randn((3, n), dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A_dense, True)
    ref_B = utils.to_reference(B, True)
    # X @ A = B  <=>  A^T @ X^T = B^T
    ref_out = torch.linalg.solve(ref_A.transpose(-2, -1), ref_B.transpose(-2, -1))
    ref_out = ref_out.transpose(-2, -1)

    res_out = flag_gems._spsolve(A_csr, B, left=False)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.spsolve
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_spsolve_identity(dtype):
    """A = I means the solution equals the right-hand side."""
    n = 16
    A_dense = torch.eye(n, dtype=dtype, device=flag_gems.device)
    A_csr = A_dense.to_sparse_csr()
    b = torch.randn((n,), dtype=dtype, device=flag_gems.device)

    res_out = flag_gems._spsolve(A_csr, b)
    ref_out = utils.to_reference(b, True)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.spsolve
def test_spsolve_requires_csr():
    """Passing a non-CSR matrix raises a clear error."""
    n = 8
    A_dense = torch.eye(n, device=flag_gems.device)
    b = torch.randn((n,), device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems._spsolve(A_dense, b)


@pytest.mark.spsolve
def test_spsolve_requires_square():
    """Passing a non-square matrix raises a clear error."""
    A_dense = torch.randn((4, 6), device=flag_gems.device)
    A_csr = A_dense.to_sparse_csr()
    b = torch.randn((4,), device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems._spsolve(A_csr, b)
