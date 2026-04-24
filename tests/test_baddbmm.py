import pytest
import torch

import flag_gems

from .accuracy_utils import ALL_FLOAT_DTYPES as ORIG_ALL_FLOAT_DTYPES
from .accuracy_utils import SCALARS, gems_assert_close, to_reference
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
    FLOAT_DTYPES = ORIG_ALL_FLOAT_DTYPES

GNK_SHAPES = [(16, 512, 2048), (16, 2560, 2048), (64, 2048, 128)]


FP8_MNK_SHAPES = [
    (128, 256, 512),
    (64, 128, 128),
    (256, 256, 256),
    (83, 7748, 3884),
    (84, 7168, 3884),
]


@pytest.mark.baddbmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_baddbmm(monkeypatch, M, N, K, scalar, dtype):
    if dtype == torch.float64 and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("tl.dot does not support fp64 on compute capability < 9.0")
    if flag_gems.vendor_name == "mthreads" and dtype in [torch.float16, torch.bfloat16]:
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")
    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)
    ref_bias = to_reference(bias, True)

    alpha = beta = scalar

    ref_out = torch.baddbmm(ref_bias, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    res_out = flag_gems.baddbmm(bias, mat1, mat2, alpha=alpha, beta=beta)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.baddbmm_backward
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_baddbmm_backward(M, N, K, scalar, dtype):
    if dtype == torch.float64 and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("tl.dot does not support fp64 on compute capability < 9.0")
    batch = 2
    mat1 = torch.randn(
        (batch, M, K), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    mat2 = torch.randn(
        (batch, K, N), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    bias = torch.randn(
        (batch, M, N), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)
    ref_bias = to_reference(bias, True)
    alpha = beta = scalar

    ref_out = torch.baddbmm(ref_bias, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    res_out = flag_gems.baddbmm(bias, mat1, mat2, alpha=alpha, beta=beta)

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)

    (ref_in_bias, ref_in_grad1, ref_in_grad2) = torch.autograd.grad(
        ref_out, (ref_bias, ref_mat1, ref_mat2), ref_grad
    )
    (res_in_bias, res_in_grad1, res_in_grad2) = torch.autograd.grad(
        res_out, (bias, mat1, mat2), out_grad
    )

    gems_assert_close(res_in_bias, ref_in_bias, dtype, reduce_dim=K)
    gems_assert_close(res_in_grad1, ref_in_grad1, dtype, reduce_dim=N)
    gems_assert_close(res_in_grad2, ref_in_grad2, dtype, reduce_dim=M)


MIXED_DTYPE_PAIRS = [
    (torch.float16, torch.float32),
    (torch.float32, torch.float16),
]


@pytest.mark.baddbmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype_a, dtype_b", MIXED_DTYPE_PAIRS)
def test_baddbmm_mixed_dtype(M, N, K, dtype_a, dtype_b):
    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype_a, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype_b, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype_a, device=flag_gems.device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)
    ref_bias = to_reference(bias, True)

    ref_out = torch.baddbmm(ref_bias, ref_mat1, ref_mat2)
    res_out = flag_gems.baddbmm(bias, mat1, mat2)

    gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=K)
