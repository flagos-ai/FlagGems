import pytest
import torch

from flag_gems.ops.linalg_pinv import linalg_pinv

from . import accuracy_utils as utils

# Jacobi SVD kernel operates in float32; float16/bfloat16 lack precision for convergence
PINV_DTYPES = [torch.float32]

# Test shapes: (m, n) - square matrices only
# Rectangular matrices have lower SVD precision with fixed Jacobi sweep count
# n <= 8 due to Triton static_range compile time for larger sizes
PINV_SHAPES = [(4, 4), (8, 8)]


@pytest.mark.linalg_pinv
@pytest.mark.parametrize("shape", PINV_SHAPES)
@pytest.mark.parametrize("dtype", PINV_DTYPES)
def test_linalg_pinv(shape, dtype):
    m, n = shape
    # Use well-conditioned matrix for reliable SVD convergence
    A = torch.randn(m, n, dtype=dtype, device="cuda")
    A = A + torch.eye(m, n, dtype=dtype, device="cuda") * 2.0
    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.pinv(ref_A)
    res_out = linalg_pinv(A)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_pinv
@pytest.mark.parametrize("shape", PINV_SHAPES)
@pytest.mark.parametrize("dtype", PINV_DTYPES)
def test_linalg_pinv_batched(shape, dtype):
    m, n = shape
    batch = 4
    # Use well-conditioned matrices
    A = torch.randn(batch, m, n, dtype=dtype, device="cuda")
    eye = torch.eye(m, n, dtype=dtype, device="cuda").unsqueeze(0).expand(batch, m, n)
    A = A + eye * 2.0
    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.pinv(ref_A)
    res_out = linalg_pinv(A)
    utils.gems_assert_close(res_out, ref_out, dtype)
