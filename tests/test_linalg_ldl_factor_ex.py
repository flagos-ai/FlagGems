import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Worktree test uses square matrices for LDL factorization.
LDLD_SHAPES = [(4, 4), (8, 8), (16, 16), (32, 32)]
# PyTorch linalg_ldl_factor_ex only supports float32 on CUDA.
LDLD_DTYPES = [torch.float32]


@pytest.mark.linalg_ldl_factor_ex
@pytest.mark.ldl_factor_ex
@pytest.mark.parametrize("shape", LDLD_SHAPES)
@pytest.mark.parametrize("dtype", LDLD_DTYPES)
def test_linalg_ldl_factor_ex(shape, dtype):
    # LDL factorization requires symmetric positive definite matrices
    # Create a random matrix and make it symmetric positive definite
    n = shape[-1]
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Make it symmetric: A = A @ A^T + small diagonal for numerical stability
    A = A @ A.mT + torch.eye(n, dtype=dtype, device=flag_gems.device) * 0.1

    # PPU SYTRF is unavailable. Preserve the original Float64 reference on
    # CPU in both modes, using the same target-generated input payload.
    reference_input = A.cpu() if flag_gems.vendor_name == "thead" else A
    ref_A = utils.to_reference(reference_input, True)
    original_input = A.clone()

    ref_out = torch.linalg.ldl_factor_ex(ref_A)
    res_out = flag_gems.ldl_factor_ex(A)

    # Check the target ABI before numerical helpers can move tensors to CPU.
    assert len(res_out) == 3
    for result, expected_shape, expected_dtype in zip(
        res_out,
        (A.shape, A.shape[:-1], A.shape[:-2]),
        (dtype, torch.int32, torch.int32),
    ):
        assert result.device == A.device
        assert result.dtype == expected_dtype
        assert result.shape == expected_shape
        assert result.untyped_storage()._cdata != A.untyped_storage()._cdata
    assert len({result.untyped_storage()._cdata for result in res_out}) == 3
    assert torch.equal(A, original_input)

    # Only move results after checking their original target ABI above.
    comparison_out = (
        tuple(value.cpu() for value in res_out)
        if flag_gems.vendor_name == "thead"
        else res_out
    )
    # Compare LD matrices
    utils.gems_assert_close(comparison_out[0], ref_out[0], dtype)
    # Compare pivots
    utils.gems_assert_equal(comparison_out[1], ref_out[1])
    # Compare info (should be 0 for successful factorization)
    utils.gems_assert_equal(comparison_out[2], ref_out[2])
