import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

if QUICK_MODE:
    # Reduced shapes for CI quick mode
    MNK_SHAPES = [
        (1, 1, 32),
        (16, 16, 64),
    ]
else:
    # Standard test shapes covering small, medium, large and non-aligned dimensions
    MNK_SHAPES = [
        (1, 1, 32),
        (16, 16, 64),
        (15, 160, 1024),
        (128, 256, 512),
        (256, 256, 256),
        (495, 5333, 71),
        (1024, 1024, 128),
    ]


@pytest.mark.addbmm_
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_addbmm_(M, N, K, scalar, dtype):
    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_bias = utils.to_reference(bias, True)

    alpha = beta = scalar

    ref_out = torch.addbmm(ref_bias, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    with flag_gems.use_gems():
        res_out = bias.addbmm_(mat1, mat2, beta=beta, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)
    # Verify inplace: result is the same object as input
    assert res_out is bias
