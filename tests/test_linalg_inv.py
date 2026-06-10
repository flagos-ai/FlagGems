import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# linalg_inv only supports 2x2 and 3x3 matrices in the Triton kernel
MATRIX_SHAPES = [(2, 2), (3, 3)]

# torch.linalg.inv only supports float32 and float64 on CUDA, half precision not supported
MATRIX_DTYPES = [torch.float32]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.linalg_inv
@pytest.mark.parametrize("shape", MATRIX_SHAPES)
@pytest.mark.parametrize("dtype", MATRIX_DTYPES)
def test_linalg_inv(shape, dtype):
    """Test linalg_inv accuracy against torch.linalg.inv"""
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Create invertible matrix
    n = shape[0]
    # Use random matrix and add identity to ensure it's invertible
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    A = A + torch.eye(n, dtype=dtype, device=flag_gems.device) * n

    ref_A = utils.to_reference(A)

    ref_out = torch.linalg.inv(ref_A)
    with flag_gems.use_gems():
        res_out = torch.linalg.inv(A)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.linalg_inv
@pytest.mark.parametrize("shape", MATRIX_SHAPES)
@pytest.mark.parametrize("dtype", MATRIX_DTYPES)
@pytest.mark.skip(reason="Batched matrix inverse not yet supported")
def test_linalg_inv_batched(shape, dtype):
    """Test linalg_inv accuracy for batched matrices"""
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Create batch of invertible matrices
    batch_size = 4
    n = shape[0]
    A = torch.randn(batch_size, n, n, dtype=dtype, device=flag_gems.device)
    # Add identity to ensure invertibility
    A = A + torch.eye(n, dtype=dtype, device=flag_gems.device).unsqueeze(0) * n

    ref_A = utils.to_reference(A)

    ref_out = torch.linalg.inv(ref_A)
    with flag_gems.use_gems():
        res_out = torch.linalg.inv(A)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-4)
