import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.matrix_exp
@pytest.mark.parametrize(
    "shape", [(2, 2), (3, 3), (5, 5), (10, 10), (16, 16), (32, 32)]
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linalg_matrix_exp(shape, dtype):
    """Test linalg_matrix_exp accuracy against PyTorch reference."""
    # Create a square matrix - scale down to avoid numerical overflow
    # Matrix exponential can grow very large, so we use smaller input values
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 0.1
    ref_inp = utils.to_reference(inp)

    # PyTorch's matrix_exp has numerical issues with float16/bfloat16
    # Compute reference at higher precision for low-precision types
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.linalg.matrix_exp(ref_inp.to(torch.float32)).to(dtype)
    else:
        ref_out = torch.linalg.matrix_exp(ref_inp)

    res_out = flag_gems.matrix_exp(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.matrix_exp
@pytest.mark.parametrize("shape", [(2, 3, 3), (4, 5, 5), (8, 4, 4)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linalg_matrix_exp_batch(shape, dtype):
    """Test linalg_matrix_exp with batched inputs."""
    # Batch of square matrices
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 0.1
    ref_inp = utils.to_reference(inp)

    # Compute reference at higher precision for low-precision types
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.linalg.matrix_exp(ref_inp.to(torch.float32)).to(dtype)
    else:
        ref_out = torch.linalg.matrix_exp(ref_inp)

    res_out = flag_gems.matrix_exp(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.matrix_exp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linalg_matrix_exp_identity(dtype):
    """Test matrix exponential of identity matrix = e * I."""
    n = 4
    inp = torch.eye(n, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Compute reference at higher precision for low-precision types
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.linalg.matrix_exp(ref_inp.to(torch.float32)).to(dtype)
    else:
        ref_out = torch.linalg.matrix_exp(ref_inp)

    res_out = flag_gems.matrix_exp(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.matrix_exp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linalg_matrix_exp_zero(dtype):
    """Test matrix exponential of zero matrix = I."""
    n = 4
    inp = torch.zeros(n, n, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Compute reference at higher precision for low-precision types
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.linalg.matrix_exp(ref_inp.to(torch.float32)).to(dtype)
    else:
        ref_out = torch.linalg.matrix_exp(ref_inp)

    res_out = flag_gems.matrix_exp(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
