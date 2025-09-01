import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    SCALARS,
    UT_SHAPES_1D,
    gems_assert_close,
    to_reference,
)
from .conftest import QUICK_MODE

MN_SHAPES = [(1, 32)] if QUICK_MODE else [(1, 32), (160, 1024), (5333, 497)]
MNK_SHAPES = (
    [(1, 1, 32)] if QUICK_MODE else [(1, 1, 32), (15, 160, 1024), (495, 5333, 71)]
)
FLOAT_DTYPES = [torch.float32] if QUICK_MODE else FLOAT_DTYPES


@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="temp disable for updating",
)
@pytest.mark.addmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_addmm(M, N, K, scalar, dtype):
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias1 = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)
    ref_bias1 = to_reference(bias1, True)

    alpha = beta = scalar

    ref_out1 = torch.addmm(ref_bias1, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    with flag_gems.use_gems():
        res_out1 = torch.addmm(bias1, mat1, mat2, alpha=alpha, beta=beta)

    gems_assert_close(res_out1, ref_out1, dtype, reduce_dim=K)

    bias2 = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_bias2 = to_reference(bias2, True)

    ref_out2 = torch.addmm(ref_bias2, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    with flag_gems.use_gems():
        res_out2 = torch.addmm(bias2, mat1, mat2, alpha=alpha, beta=beta)

    gems_assert_close(res_out2, ref_out2, dtype, reduce_dim=K)


@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="temp disable for updating",
)
@pytest.mark.bmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_bmm(M, N, K, dtype):
    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)

    ref_out = torch.bmm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.bmm(mat1, mat2)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="temp disable for updating",
)
# TODO: failed at (1, 1, 2)
@pytest.mark.mm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mm(M, N, K, dtype):
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)

    ref_out = torch.mm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.mm(mat1, mat2)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="temp disable for updating",
)
@pytest.mark.mv
@pytest.mark.parametrize("M, N", MN_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mv(M, N, dtype):
    matrix = torch.randn((N, M), dtype=dtype, device=flag_gems.device)
    vector = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    ref_matrix = to_reference(matrix, True)
    ref_vector = to_reference(vector, True)

    ref_out = torch.mv(ref_matrix, ref_vector)
    with flag_gems.use_gems():
        res_out = torch.mv(matrix, vector)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=M)


@pytest.mark.outer
@pytest.mark.parametrize(
    "M, N", MN_SHAPES + ([(32, 131072)] if flag_gems.vendor_name == "cambricon" else [])
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_outer(M, N, dtype):
    inp1 = torch.randn(M, dtype=dtype, device=flag_gems.device, requires_grad=True)
    inp2 = torch.randn(N, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.outer(ref_inp1, ref_inp2)
    res_out = flag_gems.outer(inp1, inp2)
    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)

    ref_in1_grad, ref_in2_grad = torch.autograd.grad(
        ref_out, (ref_inp1, ref_inp2), ref_grad
    )
    res_in1_grad, res_in2_grad = torch.autograd.grad(res_out, (inp1, inp2), out_grad)
    gems_assert_close(res_in1_grad, ref_in1_grad, dtype, reduce_dim=N)
    gems_assert_close(res_in2_grad, ref_in2_grad, dtype, reduce_dim=M)


@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.vdot
@pytest.mark.parametrize("M", UT_SHAPES_1D)
@pytest.mark.parametrize(
    "is_conj", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.cfloat])
@pytest.mark.parametrize("stride", [1, 2])
def test_accuracy_vdot(M, is_conj, dtype, stride):
    inp1_is_conj, inp2_is_conj = is_conj

    if flag_gems.vendor_name == "mthreads":
        inp1 = torch.randn(M, dtype=dtype, device="cpu")
        inp2 = torch.randn(M, dtype=dtype, device="cpu")
    else:
        inp1 = torch.randn(M, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(M, dtype=dtype, device=flag_gems.device)

    inp1 = inp1[::stride]
    inp2 = inp2[::stride]

    if inp1_is_conj:
        inp1 = inp1.conj()
    if inp2_is_conj:
        inp2 = inp2.conj()

    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    with flag_gems.use_gems():
        if flag_gems.vendor_name == "mthreads":
            res_out = torch.vdot(
                inp1.to(device=flag_gems.device), inp2.to(device=flag_gems.device)
            )
        else:
            res_out = torch.vdot(inp1, inp2)
    ref_out = torch.vdot(ref_inp1, ref_inp2)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.dot
@pytest.mark.parametrize("shape", UT_SHAPES_1D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_dot_tensor_tensor(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.dot(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.dot(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
