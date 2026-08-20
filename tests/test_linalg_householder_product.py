import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# (m, n) shapes where m >= n (required by householder_product)
HOUSEHOLDER_SHAPES = [
    (4, 3),
    (8, 5),
    (16, 8),
    (32, 16),
    (64, 32),
]

# (batch, m, n) shapes for batched householder_product tests
HOUSEHOLDER_BATCH_SHAPES = [
    (2, 4, 3),
    (3, 8, 5),
]

# PyTorch supports float32 and float64 for linalg.householder_product
# float16/bfloat16 are not supported on CUDA
HOUSEHOLDER_DTYPES = [torch.float32, torch.float64]


@pytest.mark.linalg_householder_product
@pytest.mark.parametrize("m, n", HOUSEHOLDER_SHAPES)
@pytest.mark.parametrize("dtype", HOUSEHOLDER_DTYPES)
def test_linalg_householder_product(m, n, dtype):
    # Create input matrix A with Householder vectors in lower triangular part
    # We use geqrf to get a valid (A, tau) pair
    A_gen = torch.randn(m, n, dtype=torch.float32, device=flag_gems.device)
    h_float, tau_float = torch.geqrf(A_gen)

    # Convert to target dtype
    h = h_float.to(dtype)
    tau = tau_float.to(dtype)

    ref_h = utils.to_reference(h)
    ref_tau = utils.to_reference(tau)

    ref_out = torch.linalg.householder_product(ref_h, ref_tau)
    with flag_gems.use_gems():
        res_out = torch.linalg.householder_product(h, tau)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_householder_product
@pytest.mark.parametrize("shape", HOUSEHOLDER_BATCH_SHAPES)
@pytest.mark.parametrize("dtype", HOUSEHOLDER_DTYPES)
def test_linalg_householder_product_batched(shape, dtype):
    # Batched case: shape is (batch, m, n)
    batch, m, n = shape
    A_gen = torch.randn(batch, m, n, dtype=torch.float32, device=flag_gems.device)
    h_float, tau_float = torch.geqrf(A_gen)

    # Convert to target dtype
    h = h_float.to(dtype)
    tau = tau_float.to(dtype)

    ref_h = utils.to_reference(h)
    ref_tau = utils.to_reference(tau)

    ref_out = torch.linalg.householder_product(ref_h, ref_tau)
    with flag_gems.use_gems():
        res_out = torch.linalg.householder_product(h, tau)

    utils.gems_assert_close(res_out, ref_out, dtype)
