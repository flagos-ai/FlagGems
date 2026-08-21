import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.linalg_cholesky_ex
@pytest.mark.parametrize("shape", [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)])
# Cholesky only supports float32/float64; fp16/bf16 not supported by PyTorch
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_cholesky_ex(shape, dtype):
    # Create a positive-definite matrix: A = B @ B^T + I
    n = shape[-1]
    B = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    A = (
        B @ B.transpose(-2, -1)
        + torch.eye(n, dtype=dtype, device=flag_gems.device) * 0.1
    )

    # For reference, convert to CPU and use torch.linalg.cholesky_ex
    ref_A = utils.to_reference(A)
    ref_L, ref_info = torch.linalg.cholesky_ex(ref_A)

    # For gems, use aten.linalg_cholesky_ex with flag_gems
    with flag_gems.use_gems():
        res_L, res_info = torch.ops.aten.linalg_cholesky_ex(A)

    utils.gems_assert_close(res_L, ref_L, dtype)
    utils.gems_assert_equal(res_info, ref_info)


@pytest.mark.linalg_cholesky_ex
@pytest.mark.parametrize("shape", [(2, 2), (4, 4), (8, 8)])
# Cholesky only supports float32/float64; fp16/bf16 not supported by PyTorch
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_cholesky_ex_upper(shape, dtype):
    # Test with upper=True
    n = shape[-1]
    B = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    A = (
        B @ B.transpose(-2, -1)
        + torch.eye(n, dtype=dtype, device=flag_gems.device) * 0.1
    )

    # For reference, convert to CPU and use torch.linalg.cholesky_ex
    ref_A = utils.to_reference(A)
    ref_L, ref_info = torch.linalg.cholesky_ex(ref_A, upper=True)

    # For gems, use aten.linalg_cholesky_ex with flag_gems
    with flag_gems.use_gems():
        res_L, res_info = torch.ops.aten.linalg_cholesky_ex(A, upper=True)

    utils.gems_assert_close(res_L, ref_L, dtype)
    utils.gems_assert_equal(res_info, ref_info)


@pytest.mark.linalg_cholesky_ex
@pytest.mark.parametrize("shape", [(2, 4, 4), (3, 8, 8)])
# Cholesky only supports float32/float64; fp16/bf16 not supported by PyTorch
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_cholesky_ex_batch(shape, dtype):
    # Create positive-definite matrices for batched input: A = B @ B^T + I
    n = shape[-1]
    B = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    A = (
        B @ B.transpose(-2, -1)
        + torch.eye(n, dtype=dtype, device=flag_gems.device) * 0.1
    )

    # For reference, convert to CPU and use torch.linalg.cholesky_ex
    ref_A = utils.to_reference(A)
    ref_L, ref_info = torch.linalg.cholesky_ex(ref_A)

    # For gems, use aten.linalg_cholesky_ex with flag_gems
    with flag_gems.use_gems():
        res_L, res_info = torch.ops.aten.linalg_cholesky_ex(A)

    utils.gems_assert_close(res_L, ref_L, dtype)
    utils.gems_assert_equal(res_info, ref_info)
