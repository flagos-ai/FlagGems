import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

if QUICK_MODE:
    MNK_SHAPES = [
        (1, 1, 32),
    ]
    FLOAT_DTYPES = [torch.float32]
else:
    MNK_SHAPES = [
        (1, 1, 32),
        (15, 160, 1024),
        (495, 5333, 71),
    ]
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES


@pytest.mark.addmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("b_column_major", [True, False])
def test_addmm(monkeypatch, M, N, K, scalar, dtype, b_column_major):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skiping fp32 addmm test on tsingmicro platform")
    if dtype == torch.float64 and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("tl.dot does not support fp64 on compute capability < 9.0")

    if flag_gems.vendor_name == "mthreads":
        monkeypatch.env("MUSA_ENABLE_SQMMA", "1")

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    if b_column_major:
        mat2 = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    else:
        mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias1 = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_bias1 = utils.to_reference(bias1, True)

    alpha = beta = scalar

    ref_out1 = torch.addmm(ref_bias1, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    with flag_gems.use_gems():
        res_out1 = torch.addmm(bias1, mat1, mat2, alpha=alpha, beta=beta)

    utils.gems_assert_close(res_out1, ref_out1, dtype, reduce_dim=K)

    bias2 = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_bias2 = utils.to_reference(bias2, True)

    ref_out2 = torch.addmm(ref_bias2, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    with flag_gems.use_gems():
        res_out2 = torch.addmm(bias2, mat1, mat2, alpha=alpha, beta=beta)

    utils.gems_assert_close(res_out2, ref_out2, dtype, reduce_dim=K)


@pytest.mark.addmm_out
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_addmm_out(M, N, K, scalar, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skiping fp32 addmm_out test on tsingmicro platform")
    if dtype == torch.float64 and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("tl.dot does not support fp64 on compute capability < 9.0")

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias1 = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_bias1 = utils.to_reference(bias1, True)
    ref_out = utils.to_reference(out, True)

    alpha = beta = scalar

    torch.addmm(ref_bias1, ref_mat1, ref_mat2, alpha=alpha, beta=beta, out=ref_out)
    with flag_gems.use_gems():
        torch.addmm(bias1, mat1, mat2, alpha=alpha, beta=beta, out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K)

    bias2 = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_bias2 = utils.to_reference(bias2, True)

    torch.addmm(ref_bias2, ref_mat1, ref_mat2, alpha=alpha, beta=beta, out=ref_out)
    with flag_gems.use_gems():
        torch.addmm(bias2, mat1, mat2, alpha=alpha, beta=beta, out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K)


MIXED_DTYPE_PAIRS = [
    (torch.float16, torch.float32),
    (torch.float32, torch.float16),
]


@pytest.mark.addmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype_a, dtype_b", MIXED_DTYPE_PAIRS)
def test_addmm_mixed_dtype(M, N, K, dtype_a, dtype_b):
    mat1 = torch.randn((M, K), dtype=dtype_a, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype_b, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype_a, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.addmm(ref_bias, ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.addmm(bias, mat1, mat2)

    utils.gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=K)
