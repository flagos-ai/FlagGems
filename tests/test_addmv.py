import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    MN_SHAPES = [
        (1, 32),
    ]
    FLOAT_DTYPES = [torch.float32]
else:
    MN_SHAPES = [
        (1, 32),
        (160, 1024),
        (5333, 497),
    ]
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.addmv
@pytest.mark.parametrize("M, N", MN_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_addmv(M, N, scalar, dtype):
    mat = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    vec = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    bias1 = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    ref_mat = utils.to_reference(mat, True)
    ref_vec = utils.to_reference(vec, True)
    ref_bias1 = utils.to_reference(bias1, True)

    alpha = beta = scalar

    ref_out1 = torch.addmv(ref_bias1, ref_mat, ref_vec, alpha=alpha, beta=beta)
    with flag_gems.use_gems():
        res_out1 = torch.addmv(bias1, mat, vec, alpha=alpha, beta=beta)

    utils.gems_assert_close(res_out1, ref_out1, dtype, reduce_dim=N)

    # broadcast bias scalar
    bias2 = torch.randn((), dtype=dtype, device=flag_gems.device)
    if flag_gems.vendor_name == "kunlunxin":
        ref_bias2 = utils.to_reference(bias2, True)
    else:
        ref_bias2 = utils.to_reference(bias2)

    ref_out2 = torch.addmv(ref_bias2, ref_mat, ref_vec, alpha=alpha, beta=beta)
    with flag_gems.use_gems():
        res_out2 = torch.addmv(bias2, mat, vec, alpha=alpha, beta=beta)

    utils.gems_assert_close(res_out2, ref_out2, dtype, reduce_dim=N)


@pytest.mark.addmv_out
@pytest.mark.parametrize("M, N", MN_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_addmv_out(M, N, scalar, dtype):
    mat = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    vec = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    out = torch.empty((M,), dtype=dtype, device=flag_gems.device)
    ref_mat = utils.to_reference(mat, True)
    ref_vec = utils.to_reference(vec, True)
    ref_bias = utils.to_reference(bias, True)
    ref_out = utils.to_reference(out, True)

    alpha = beta = scalar

    torch.addmv(ref_bias, ref_mat, ref_vec, alpha=alpha, beta=beta, out=ref_out)
    with flag_gems.use_gems():
        torch.addmv(bias, mat, vec, alpha=alpha, beta=beta, out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=N)
