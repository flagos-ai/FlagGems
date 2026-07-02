import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Test for linalg_ldl_solve
LDL_SOLVE_SHAPES = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
]


# CUDA ldl_factor_ex used to build LD supports only float32 and float64 here.
LDL_SOLVE_DTYPES = [torch.float32, torch.float64]


@pytest.mark.linalg_ldl_solve
@pytest.mark.parametrize("shape", LDL_SOLVE_SHAPES)
@pytest.mark.parametrize("dtype", LDL_SOLVE_DTYPES)
def test_linalg_ldl_solve(shape, dtype):
    n, k = shape
    # Create a symmetric positive definite matrix
    A = torch.randn(n, n, dtype=dtype, device=flag_gems.device)
    A = A @ A.mT + torch.eye(n, dtype=dtype, device=flag_gems.device) * n

    # Compute LDL factorization
    LD, pivots, info = torch.linalg.ldl_factor_ex(A)

    # Right-hand side
    B = torch.randn(n, k, dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)
    ref_LD, ref_pivots, ref_info = torch.linalg.ldl_factor_ex(ref_A)

    ref_out = torch.linalg.ldl_solve(ref_LD, ref_pivots, ref_B)
    with flag_gems.use_gems():
        res_out = torch.linalg.ldl_solve(LD, pivots, B)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_ldl_solve
@pytest.mark.parametrize("shape", LDL_SOLVE_SHAPES)
@pytest.mark.parametrize("dtype", LDL_SOLVE_DTYPES)
def test_linalg_ldl_solve_batched(shape, dtype):
    batch_size = 4
    n, k = shape
    # Create batched symmetric positive definite matrices
    A = torch.randn(batch_size, n, n, dtype=dtype, device=flag_gems.device)
    for i in range(batch_size):
        A[i] = A[i] @ A[i].mT + torch.eye(n, dtype=dtype, device=flag_gems.device) * n

    # Compute LDL factorization
    LD, pivots, info = torch.linalg.ldl_factor_ex(A)

    # Right-hand side (batched)
    B = torch.randn(batch_size, n, k, dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)
    ref_LD, ref_pivots, ref_info = torch.linalg.ldl_factor_ex(ref_A)

    ref_out = torch.linalg.ldl_solve(ref_LD, ref_pivots, ref_B)
    with flag_gems.use_gems():
        res_out = torch.linalg.ldl_solve(LD, pivots, B)

    utils.gems_assert_close(res_out, ref_out, dtype)
