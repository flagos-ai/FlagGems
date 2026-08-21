import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

DEVICE = flag_gems.device
VENDOR = flag_gems.vendor_name

if VENDOR == "nvidia":
    _TEST_DTYPES = [torch.float32, torch.float64]
else:
    _TEST_DTYPES = [torch.float32]


def _make_invertible_matrix(shape, dtype, device):
    """Generate a well-conditioned invertible matrix for testing.

    Constructs a diagonally dominant matrix to ensure invertibility
    and numerical stability.
    """
    A = torch.randn(shape, dtype=dtype, device=device)
    n = shape[-1]
    # Make diagonally dominant to ensure invertibility
    A = A + n * torch.eye(n, dtype=dtype, device=device).expand_as(A)
    return A


@pytest.mark.linalg_inv_ex
@pytest.mark.parametrize(
    "shape",
    [
        (2, 2),
        (3, 3),
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (4, 4, 4),
        (2, 8, 8),
        (3, 16, 16),
        (2, 3, 4, 4),
    ],
)
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
def test_accuracy_linalg_inv_ex(shape, dtype):
    A = _make_invertible_matrix(shape, dtype, DEVICE)
    ref_A = utils.to_reference(A)

    ref_result = torch.linalg.inv_ex(ref_A)
    with flag_gems.use_gems():
        res_result = torch.linalg.inv_ex(A)

    n = shape[-1]

    # Check shapes
    assert res_result.inverse.shape == A.shape
    assert res_result.info.shape == A.shape[:-2]
    assert res_result.info.dtype == torch.int32

    # Check info is 0 (successful inversion)
    assert torch.all(res_result.info == 0)

    # Compare inverse directly (LU + solve accumulates O(n^2) error)
    utils.gems_assert_close(
        res_result.inverse, ref_result.inverse, dtype, reduce_dim=n * n
    )


@pytest.mark.linalg_inv_ex
@pytest.mark.parametrize(
    "shape",
    [
        (3, 3),
        (4, 4),
    ],
)
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
def test_accuracy_linalg_inv_ex_singular(shape, dtype):
    """Test that singular matrices produce non-zero info."""
    # Create a singular matrix: zero out one row to guarantee rank deficiency
    A = torch.zeros(shape, dtype=dtype, device=DEVICE)
    n = shape[-1]
    # Fill with identity-like structure but leave last row as zeros
    for i in range(n - 1):
        A[i, i] = 1.0

    with flag_gems.use_gems():
        res_result = torch.linalg.inv_ex(A)

    # info should be non-zero for singular matrix
    assert res_result.info != 0


@pytest.mark.linalg_inv_ex
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
def test_accuracy_linalg_inv_ex_check_errors(dtype):
    """Test that check_errors=True raises LinAlgError for singular matrices."""
    shape = (3, 3)
    n = shape[-1]
    # Create a singular matrix (last row all zeros)
    A = torch.zeros(shape, dtype=dtype, device=DEVICE)
    for i in range(n - 1):
        A[i, i] = 1.0

    with flag_gems.use_gems():
        with pytest.raises(torch.linalg.LinAlgError):
            torch.linalg.inv_ex(A, check_errors=True)
