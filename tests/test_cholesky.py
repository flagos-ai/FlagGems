import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Cholesky decomposition only supports float32 on GPU
# float64 is not supported by many GPU backends
CHOLESKY_DTYPES = [torch.float32]


@pytest.mark.cholesky
@pytest.mark.parametrize("shape", utils.UT_SHAPES_2D)
@pytest.mark.parametrize("dtype", CHOLESKY_DTYPES)
def test_cholesky(shape, dtype):
    if flag_gems.vendor_name == "mthreads":
        pytest.skip("Skipping cholesky test on mthreads platform")

    # Create a symmetric positive-definite matrix
    # A = A @ A.T + I ensures positive definiteness
    n = shape[0]
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Make it symmetric positive definite
    A = (
        A @ A.transpose(-2, -1)
        + torch.eye(n, dtype=dtype, device=flag_gems.device) * 0.1
    )

    ref_inp = utils.to_reference(A)

    # Use torch.linalg.cholesky which supports more dtypes
    ref_out = torch.linalg.cholesky(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.cholesky(A)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cholesky
@pytest.mark.parametrize("shape", utils.UT_SHAPES_2D)
@pytest.mark.parametrize("dtype", CHOLESKY_DTYPES)
def test_cholesky_upper(shape, dtype):
    if flag_gems.vendor_name == "mthreads":
        pytest.skip("Skipping cholesky upper test on mthreads platform")

    # Create a symmetric positive-definite matrix
    n = shape[0]
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    A = (
        A @ A.transpose(-2, -1)
        + torch.eye(n, dtype=dtype, device=flag_gems.device) * 0.1
    )

    ref_inp = utils.to_reference(A)

    ref_out = torch.linalg.cholesky(ref_inp, upper=True)
    with flag_gems.use_gems():
        res_out = torch.cholesky(A, upper=True)

    utils.gems_assert_close(res_out, ref_out, dtype)
