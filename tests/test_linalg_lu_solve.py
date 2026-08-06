import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Test shapes: (n, k) pairs
LU_SOLVE_SHAPES = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (32, 7),
    (16, 1),
]

# LU solve supports float32/float64
LU_SOLVE_DTYPES = [torch.float32]
if utils.fp64_is_supported:
    LU_SOLVE_DTYPES.append(torch.float64)


def _make_lu_inputs(batch_shape, n, k, dtype, device):
    """Create a well-conditioned matrix and compute its LU factorization."""
    A = torch.randn(*batch_shape, n, n, dtype=dtype, device=device)
    # Make A well-conditioned by adding n * I
    A = A @ A.mT + torch.eye(n, dtype=dtype, device=device) * n
    B = torch.randn(*batch_shape, n, k, dtype=dtype, device=device)
    LU, pivots = torch.linalg.lu_factor(A)
    return LU, pivots, B


@pytest.mark.linalg_lu_solve
@pytest.mark.parametrize("shape", LU_SOLVE_SHAPES)
@pytest.mark.parametrize("dtype", LU_SOLVE_DTYPES)
def test_linalg_lu_solve(shape, dtype):
    n, k = shape
    LU, pivots, B = _make_lu_inputs((), n, k, dtype, flag_gems.device)

    ref_LU = utils.to_reference(LU)
    ref_pivots = utils.to_reference(pivots)
    ref_B = utils.to_reference(B)
    ref_out = torch.linalg.lu_solve(ref_LU, ref_pivots, ref_B)

    with flag_gems.use_gems():
        res_out = torch.linalg.lu_solve(LU, pivots, B)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_lu_solve
@pytest.mark.parametrize("shape", LU_SOLVE_SHAPES)
@pytest.mark.parametrize("dtype", LU_SOLVE_DTYPES)
def test_linalg_lu_solve_batched(shape, dtype):
    batch_size = 4
    n, k = shape
    LU, pivots, B = _make_lu_inputs((batch_size,), n, k, dtype, flag_gems.device)

    ref_LU = utils.to_reference(LU)
    ref_pivots = utils.to_reference(pivots)
    ref_B = utils.to_reference(B)
    ref_out = torch.linalg.lu_solve(ref_LU, ref_pivots, ref_B)

    with flag_gems.use_gems():
        res_out = torch.linalg.lu_solve(LU, pivots, B)

    utils.gems_assert_close(res_out, ref_out, dtype)
