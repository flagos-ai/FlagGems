import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Test shapes for linalg_matmul (2D and batched)
LINALG_MNK_SHAPES = [
    (1, 1, 32),
    (15, 160, 1024),
    (495, 5333, 71),
    (128, 256, 512),
    (64, 128, 256),
]


@pytest.mark.linalg_matmul
@pytest.mark.parametrize("M, N, K", LINALG_MNK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linalg_matmul_2d(M, N, K, dtype):
    """Test 2D matrix multiplication: (M, K) @ (K, N) -> (M, N)"""
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skiping fp32 linalg_matmul test on tsingmicro platform")

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)

    ref_out = torch.linalg.matmul(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.linalg.matmul(mat1, mat2)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.linalg_matmul
@pytest.mark.parametrize("M, N, K", LINALG_MNK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linalg_matmul_3d(M, N, K, dtype):
    """Test 3D (batched) matrix multiplication: (B, M, K) @ (B, K, N) -> (B, M, N)"""
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skiping fp32 linalg_matmul test on tsingmicro platform")

    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)

    ref_out = torch.linalg.matmul(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.linalg.matmul(mat1, mat2)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)
