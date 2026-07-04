import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# The generated BLAS worktree uses these representative M/N/K cases for accuracy.
MNK_SHAPES = [
    (1, 1, 32),
    (15, 160, 1024),
    (495, 5333, 71),
]


@pytest.mark.matmul_bias_residual
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_matmul_bias_residual(M, N, K, dtype):
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    residual = torch.randn((M, N), dtype=dtype, device=flag_gems.device)

    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_bias = utils.to_reference(bias, True)
    ref_residual = utils.to_reference(residual, True)

    alpha = 1.0
    beta = 1.0

    # Reference: output = alpha * (mat1 @ mat2) + beta * bias + residual
    ref_out = torch.addmm(ref_bias, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    ref_out = ref_out + ref_residual

    with flag_gems.use_gems():
        res_out = flag_gems.matmul_bias_residual(
            mat1, mat2, bias, residual, alpha=alpha, beta=beta
        )

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)
