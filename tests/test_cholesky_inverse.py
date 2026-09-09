import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shapes for cholesky_inverse: square matrices from small to medium
CHOLESKY_INVERSE_SHAPES = [
    (2, 2),
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
]

# Batched shapes for cholesky_inverse: (batch, n, n)
CHOLESKY_INVERSE_BATCH_SHAPES = [
    (4, 4, 4),
    (2, 8, 8),
    (3, 16, 16),
]


def _make_positive_definite(shape, dtype, device):
    """Create a positive-definite matrix and return its Cholesky factor."""
    n = shape[-1]
    B = torch.randn(shape, dtype=dtype, device=device)
    A = B @ B.transpose(-2, -1) + torch.eye(n, dtype=dtype, device=device) * n
    L = torch.linalg.cholesky(A)
    return L


@pytest.mark.cholesky_inverse
@pytest.mark.parametrize("shape", CHOLESKY_INVERSE_SHAPES)
# cholesky_inverse only supports float32/float64
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cholesky_inverse(shape, dtype):
    L = _make_positive_definite(shape, dtype, flag_gems.device)
    ref_L = utils.to_reference(L)

    ref_out = torch.cholesky_inverse(ref_L, upper=False)

    with flag_gems.use_gems():
        res_out = torch.cholesky_inverse(L, upper=False)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cholesky_inverse
@pytest.mark.parametrize("shape", CHOLESKY_INVERSE_SHAPES[:3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cholesky_inverse_upper(shape, dtype):
    L = _make_positive_definite(shape, dtype, flag_gems.device)
    U = L.transpose(-2, -1).contiguous()
    ref_U = utils.to_reference(U)

    ref_out = torch.cholesky_inverse(ref_U, upper=True)

    with flag_gems.use_gems():
        res_out = torch.cholesky_inverse(U, upper=True)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cholesky_inverse
@pytest.mark.parametrize("shape", CHOLESKY_INVERSE_BATCH_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cholesky_inverse_batch(shape, dtype):
    L = _make_positive_definite(shape, dtype, flag_gems.device)
    ref_L = utils.to_reference(L)

    ref_out = torch.cholesky_inverse(ref_L, upper=False)

    with flag_gems.use_gems():
        res_out = torch.cholesky_inverse(L, upper=False)

    utils.gems_assert_close(res_out, ref_out, dtype)
