import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    MNK_SHAPES,
    SCALARS,
    gems_assert_close,
    to_reference,
)


@pytest.mark.parametrize("M", MNK_SHAPES)
@pytest.mark.parametrize("N", MNK_SHAPES)
@pytest.mark.parametrize("K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_addmm(M, N, K, scalar, dtype):
    mat1 = torch.randn((M, K), dtype=dtype, device="cuda")
    mat2 = torch.randn((K, N), dtype=dtype, device="cuda")
    bias = torch.randn((N,), dtype=dtype, device="cuda")
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)
    ref_bias = to_reference(bias, True)

    alpha = beta = scalar

    ref_out = torch.addmm(ref_bias, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    with flag_gems.use_gems():
        res_out = torch.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.parametrize("M", MNK_SHAPES)
@pytest.mark.parametrize("N", MNK_SHAPES)
@pytest.mark.parametrize("K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_bmm(M, N, K, dtype):
    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device="cuda")
    mat2 = torch.randn((batch, K, N), dtype=dtype, device="cuda")
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)

    ref_out = torch.bmm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.bmm(mat1, mat2)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.parametrize("M", MNK_SHAPES)
@pytest.mark.parametrize("N", MNK_SHAPES)
@pytest.mark.parametrize("K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mm(M, N, K, dtype):
    mat1 = torch.randn((M, K), dtype=dtype, device="cuda")
    mat2 = torch.randn((K, N), dtype=dtype, device="cuda")
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)

    ref_out = torch.mm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.mm(mat1, mat2)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.parametrize("M", MNK_SHAPES)
@pytest.mark.parametrize("N", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mv(M, N, dtype):
    matrix = torch.randn((N, M), dtype=dtype, device="cuda")
    vector = torch.randn((M,), dtype=dtype, device="cuda")
    ref_matrix = to_reference(matrix, True)
    ref_vector = to_reference(vector, True)

    ref_out = torch.mv(ref_matrix, ref_vector)
    with flag_gems.use_gems():
        res_out = torch.mv(matrix, vector)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("M", MNK_SHAPES)
@pytest.mark.parametrize("N", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_outer(M, N, dtype):
    inp1 = torch.randn(M, dtype=dtype, device="cuda", requires_grad=True)
    inp2 = torch.randn(N, dtype=dtype, device="cuda", requires_grad=True)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.outer(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.outer(inp1, inp2)
    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)

    ref_in1_grad, ref_in2_grad = torch.autograd.grad(
        ref_out, (ref_inp1, ref_inp2), ref_grad
    )
    res_in1_grad, res_in2_grad = torch.autograd.grad(res_out, (inp1, inp2), out_grad)
    gems_assert_close(res_in1_grad, ref_in1_grad, dtype)
    gems_assert_close(res_in2_grad, ref_in2_grad, dtype)
