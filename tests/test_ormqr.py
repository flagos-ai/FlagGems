import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shapes for ormqr: (M, N) pairs covering square and rectangular matrices
ORMQR_SHAPES = [
    (32, 32),
    (64, 64),
    (128, 64),
    (64, 128),
]

# ormqr only supports float32 and float64 (LAPACK limitation, no half/bfloat16)
ORMQR_DTYPES = [torch.float32, torch.float64]


@pytest.mark.ormqr
@pytest.mark.parametrize("M, N", ORMQR_SHAPES)
@pytest.mark.parametrize("dtype", ORMQR_DTYPES)
@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("transpose", [True, False])
def test_ormqr(M, N, dtype, left, transpose):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skipping fp32 ormqr test on tsingmicro platform")

    # For ormqr, we need valid Householder reflectors
    # Generate them from a QR decomposition (geqrf)
    k = min(M, N)

    # The matrix dimension depends on 'left' parameter
    # If left=True, we need (M, k); if left=False, we need (N, k)
    if left:
        a = torch.randn((M, k), dtype=dtype, device=flag_gems.device)
    else:
        a = torch.randn((N, k), dtype=dtype, device=flag_gems.device)

    # Generate Householder reflectors via QR decomposition
    input_tensor, tau = torch.geqrf(a)

    # The other matrix to multiply with - always has shape (M, N)
    other = torch.randn((M, N), dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input_tensor, True)
    ref_tau = utils.to_reference(tau, True)
    ref_other = utils.to_reference(other, True)

    ref_out = torch.ormqr(ref_input, ref_tau, ref_other, left=left, transpose=transpose)
    with flag_gems.use_gems():
        res_out = torch.ormqr(input_tensor, tau, other, left=left, transpose=transpose)

    utils.gems_assert_close(res_out, ref_out, dtype)
