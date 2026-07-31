import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Worktree test uses square matrices for LDL factorization.
LDLD_SHAPES = [(4, 4), (8, 8), (16, 16), (32, 32)]
# PyTorch linalg_ldl_factor_ex only supports float32 on CUDA.
LDLD_DTYPES = [torch.float32]


@pytest.mark.linalg_ldl_factor_ex
@pytest.mark.parametrize("shape", LDLD_SHAPES)
@pytest.mark.parametrize("dtype", LDLD_DTYPES)
def test_linalg_ldl_factor_ex(shape, dtype):
    # LDL factorization requires symmetric positive definite matrices
    # Create a random matrix and make it symmetric positive definite
    n = shape[-1]
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Make it symmetric: A = A @ A^T + small diagonal for numerical stability
    A = A @ A.mT + torch.eye(n, dtype=dtype, device=flag_gems.device) * 0.1

    ref_A = utils.to_reference(A)

    ref_out = torch.linalg.ldl_factor_ex(ref_A)
    with flag_gems.use_gems():
        res_out = torch.linalg.ldl_factor_ex(A)

    # Compare LD matrices
    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    # Compare pivots
    utils.gems_assert_equal(res_out[1], ref_out[1])
    # Compare info (should be 0 for successful factorization)
    utils.gems_assert_equal(res_out[2], ref_out[2])
