import pytest
import torch
import flag_gems
from flag_gems.ops.linalg_diagonal import linalg_diagonal


@pytest.mark.parametrize("shape", [(3, 3), (4, 5), (5, 4), (2, 3, 4), (3, 4, 5, 6)])
@pytest.mark.parametrize("offset", [-2, -1, 0, 1, 2])
def test_diagonal_basic(shape, offset):
    """Test basic functionality: different shapes and offsets."""
    device = flag_gems.device
    A = torch.randn(shape, device=device)
    expected = torch.linalg.diagonal(A, offset=offset)
    result = linalg_diagonal(A, offset=offset)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(3, 3), (4, 5)])
@pytest.mark.parametrize("dim1, dim2", [(0, 1), (-2, -1)])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_diagonal_with_dims(shape, dim1, dim2, offset):
    """Test specifying dim1 and dim2."""
    device = flag_gems.device
    A = torch.randn(shape, device=device)
    expected = torch.linalg.diagonal(A, offset=offset, dim1=dim1, dim2=dim2)
    result = linalg_diagonal(A, offset=offset, dim1=dim1, dim2=dim2)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


def test_diagonal_empty():
    """Test empty diagonal (offset out of range)."""
    device = flag_gems.device
    A = torch.randn(2, 3, device=device)
    result = linalg_diagonal(A, offset=3)
    expected = torch.linalg.diagonal(A, offset=3)
    assert result.shape == expected.shape
    torch.testing.assert_close(result, expected)


def test_diagonal_2d_manual():
    """Manual verification of 2D diagonal values."""
    device = flag_gems.device
    A = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]], device=device)
    result = linalg_diagonal(A)
    expected = torch.tensor([1, 5, 9], device=device)
    torch.testing.assert_close(result, expected)

    result = linalg_diagonal(A, offset=1)
    expected = torch.tensor([2, 6], device=device)
    torch.testing.assert_close(result, expected)

    result = linalg_diagonal(A, offset=-1)
    expected = torch.tensor([4, 8], device=device)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("shape", [(3, 3), (4, 5, 6)])
def test_diagonal_non_last_dims(shape):
    """Test taking diagonal on non-last two dimensions."""
    device = flag_gems.device
    A = torch.randn(shape, device=device)
    dim1, dim2 = 0, 1
    expected = torch.linalg.diagonal(A, dim1=dim1, dim2=dim2)
    result = linalg_diagonal(A, dim1=dim1, dim2=dim2)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


def test_diagonal_negative_dim():
    """Test negative dim1/dim2."""
    device = flag_gems.device
    A = torch.randn(2, 3, 4, 5, device=device)
    # Default: last two dimensions using -1 and -2
    expected = torch.linalg.diagonal(A, dim1=-2, dim2=-1)
    result = linalg_diagonal(A, dim1=-2, dim2=-1)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    # Using -3 and -1
    expected = torch.linalg.diagonal(A, dim1=-3, dim2=-1)
    result = linalg_diagonal(A, dim1=-3, dim2=-1)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(4, 4), (5, 6, 7)])
def test_diagonal_large_offset(shape):
    """Test case where offset equals dimension size - 1, diagonal length is 1."""
    device = flag_gems.device
    A = torch.randn(shape, device=device)
    if len(shape) == 2:
        max_offset = min(shape) - 1
        offset = max_offset
    else:
        # For 3D, default last two dimensions
        max_offset = min(shape[-2], shape[-1]) - 1
        offset = max_offset
    expected = torch.linalg.diagonal(A, offset=offset)
    result = linalg_diagonal(A, offset=offset)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
