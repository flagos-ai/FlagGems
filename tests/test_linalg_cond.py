import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Matrix shapes for linalg_cond tests
LINALG_COND_SHAPES = [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)]
LINALG_COND_SHAPES_SVD = [(2, 3), (4, 4), (4, 6), (8, 8), (16, 16)]

# Only float32 supported: Triton Jacobi SVD implementation uses float32
LINALG_COND_DTYPES = [torch.float32]


@pytest.mark.linalg_cond
@pytest.mark.parametrize("shape", LINALG_COND_SHAPES)
@pytest.mark.parametrize("dtype", LINALG_COND_DTYPES)
def test_linalg_cond(shape, dtype):
    # Test with p=None (2-norm via Jacobi SVD)
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)

    ref_out = torch.linalg.cond(ref_A)
    with flag_gems.use_gems():
        res_out = torch.linalg.cond(A)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_cond
@pytest.mark.parametrize("shape", LINALG_COND_SHAPES)
@pytest.mark.parametrize("dtype", LINALG_COND_DTYPES)
@pytest.mark.parametrize("p", ["fro", "nuc", float("inf"), 1, -1])
def test_linalg_cond_with_p(shape, dtype, p):
    # Test with various p values that require square matrices
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)

    ref_out = torch.linalg.cond(ref_A, p=p)
    with flag_gems.use_gems():
        res_out = torch.linalg.cond(A, p=p)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_cond
@pytest.mark.parametrize("shape", LINALG_COND_SHAPES_SVD)
@pytest.mark.parametrize("dtype", LINALG_COND_DTYPES)
@pytest.mark.parametrize("p", [2, -2])
def test_linalg_cond_svd(shape, dtype, p):
    # Test with p=2 or p=-2 which can work with non-square matrices
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)

    ref_out = torch.linalg.cond(ref_A, p=p)
    with flag_gems.use_gems():
        res_out = torch.linalg.cond(A, p=p)

    utils.gems_assert_close(res_out, ref_out, dtype)
