import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shapes for linalg_eig - square matrices
# Note: Currently only 2x2 matrices are implemented in Triton
# Larger matrices would require implementing a full eigenvalue algorithm in Triton
LINALG_EIG_SHAPES = [
    (2, 2),
]


@pytest.mark.linalg_eig
@pytest.mark.parametrize("shape", LINALG_EIG_SHAPES)
# linalg_eig only supports float32 — Half/BFloat16 not supported by torch.linalg.eig
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eig(shape, dtype):
    """Test linalg_eig accuracy for square matrices."""
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    # Create a square matrix
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)

    # Compute eigenvalue decomposition
    ref_out = torch.linalg.eig(ref_A)
    with flag_gems.use_gems():
        res_out = torch.linalg.eig(A)

    # Compare eigenvalues - use higher tolerance for complex comparison
    # Eigenvalue ordering may differ, so we need to sort or compare differently
    ref_eigenvalues = ref_out.eigenvalues
    res_eigenvalues = res_out.eigenvalues

    # Sort eigenvalues for comparison (they may be in different order)
    ref_eig_sorted, _ = torch.sort(ref_eigenvalues.abs())
    res_eig_sorted, _ = torch.sort(res_eigenvalues.abs())

    # Check if the eigenvalue magnitudes match (allow for different ordering)
    utils.gems_assert_close(res_eig_sorted, ref_eig_sorted, dtype, atol=1e-1)


@pytest.mark.linalg_eig
@pytest.mark.parametrize("shape", LINALG_EIG_SHAPES)
# linalg_eig only supports float32 — Half/BFloat16 not supported by torch.linalg.eig
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eig_batched(shape, dtype):
    """Test linalg_eig accuracy for batched square matrices."""
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    # Create a batch of square matrices - only test 2x2
    batch_shape = (4,)
    A = torch.randn(*batch_shape, *shape, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)

    # Compute eigenvalue decomposition
    ref_out = torch.linalg.eig(ref_A)
    with flag_gems.use_gems():
        res_out = torch.linalg.eig(A)

    # Compare eigenvalue magnitudes
    ref_eig_sorted, _ = torch.sort(ref_out.eigenvalues.abs(), dim=-1)
    res_eig_sorted, _ = torch.sort(res_out.eigenvalues.abs(), dim=-1)

    utils.gems_assert_close(res_eig_sorted, ref_eig_sorted, dtype, atol=1e-1)
