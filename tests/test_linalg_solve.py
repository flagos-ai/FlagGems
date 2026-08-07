import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# LU decomposition requires high precision; float16/bfloat16 lack mantissa bits for pivoting
SOLVE_DTYPES = [torch.float32, torch.float64]

# Test shapes: (n, nrhs) - cover small to medium matrices for correctness
SOLVE_SHAPES = [(4, 4), (8, 8), (16, 16), (32, 32)]


def _make_solve_inputs(n, k, dtype, device):
    """Create well-conditioned A and random B for solve test."""
    A = torch.randn(n, n, dtype=dtype, device=device)
    A = A @ A.mT + torch.eye(n, dtype=dtype, device=device) * n
    B = torch.randn(n, k, dtype=dtype, device=device)
    return A, B


@pytest.mark.linalg_solve
@pytest.mark.parametrize("shape", SOLVE_SHAPES)
@pytest.mark.parametrize("dtype", SOLVE_DTYPES)
def test_linalg_solve(shape, dtype):
    n, k = shape
    A, B = _make_solve_inputs(n, k, dtype, "cuda")

    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)
    ref_out = torch.linalg.solve(ref_A, ref_B)
    with flag_gems.use_gems():
        res_out = torch.linalg.solve(A, B)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_solve
@pytest.mark.parametrize("shape", SOLVE_SHAPES)
@pytest.mark.parametrize("dtype", SOLVE_DTYPES)
def test_linalg_solve_batched(shape, dtype):
    n, k = shape
    batch = 4
    A = torch.randn(batch, n, n, dtype=dtype, device="cuda")
    eye = torch.eye(n, dtype=dtype, device="cuda").unsqueeze(0).expand(batch, n, n)
    A = A @ A.mT + eye * n
    B = torch.randn(batch, n, k, dtype=dtype, device="cuda")

    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)
    ref_out = torch.linalg.solve(ref_A, ref_B)
    with flag_gems.use_gems():
        res_out = torch.linalg.solve(A, B)

    utils.gems_assert_close(res_out, ref_out, dtype)
