import random

import numpy as np
import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES as ORIG_FLOAT_DTYPES
from .accuracy_utils import utils
from .conftest import QUICK_MODE

if QUICK_MODE:
    MN_SHAPES = [
        (1, 32),
    ]
    MNK_SHAPES = [
        (1, 1, 32),
    ]
    FLOAT_DTYPES = [torch.float32]
else:
    MN_SHAPES = [
        (1, 32),
        (160, 1024),
        (5333, 497),
    ]
    MNK_SHAPES = [
        (1, 1, 32),
        (15, 160, 1024),
        (495, 5333, 71),
    ]
    FLOAT_DTYPES = ORIG_FLOAT_DTYPES

GNK_SHAPES = [(16, 512, 2048), (16, 2560, 2048), (64, 2048, 128)]


@pytest.mark.gemm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("b_column_major", [True, False])
def test_accuracy_gemm(M, N, K, dtype, b_column_major):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skiping fp32 mm test on tsingmicro platform")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    alpha = 2.0
    beta = 0
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    if b_column_major:
        mat2 = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    else:
        mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)

    ref_out = torch.mm(ref_mat1, ref_mat2) * alpha
    with flag_gems.use_gems():
        res_out = flag_gems.ops.gemm(mat1, mat2, beta, alpha)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)
