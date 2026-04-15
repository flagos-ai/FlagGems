import os
import random

import numpy as np
import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES as ORIG_FLOAT_DTYPES
from .accuracy_utils import (
    SCALARS,
    UT_SHAPES_1D,
    SkipVersion,
    gems_assert_close,
    to_reference,
)
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
MIXED_DTYPE_PAIRS = [
    (torch.float16, torch.float32),
    (torch.float32, torch.float16),
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.bfloat16),
]


@pytest.mark.addmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("b_column_major", [True, False])
def test_addmm(M, N, K, scalar, dtype, b_column_major):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skiping fp32 addmm test on tsingmicro platform")

    if flag_gems.vendor_name == "mthreads":
        os.environ["MUSA_ENABLE_SQMMA"] = "1"
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    if b_column_major:
        mat2 = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    else:
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

    if flag_gems.vendor_name == "mthreads":
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.addmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype_a, dtype_b", MIXED_DTYPE_PAIRS)
def test_addmm_mixed_dtype(M, N, K, scalar, dtype_a, dtype_b):
    mat1 = torch.randn((M, K), dtype=dtype_a, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype_b, device=flag_gems.device)
    out_dtype = torch.promote_types(dtype_a, dtype_b)
    bias1 = torch.randn((N,), dtype=out_dtype, device=flag_gems.device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)
    ref_bias1 = to_reference(bias1, True)

    alpha = beta = scalar

    ref_out = torch.addmm(ref_bias1, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    with flag_gems.use_gems():
        res_out = torch.addmm(bias1, mat1, mat2, alpha=alpha, beta=beta)

    gems_assert_close(res_out, ref_out, out_dtype, reduce_dim=K)


@pytest.mark.addmm_out
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_addmm_out(M, N, K, scalar, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skiping fp32 addmm_out test on tsingmicro platform")

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias1 = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)
    ref_bias1 = to_reference(bias1, True)
    ref_out = to_reference(out, True)

    alpha = beta = scalar

    torch.addmm(ref_bias1, ref_mat1, ref_mat2, alpha=alpha, beta=beta, out=ref_out)
    with flag_gems.use_gems():
        torch.addmm(bias1, mat1, mat2, alpha=alpha, beta=beta, out=out)

    gems_assert_close(out, ref_out, dtype, reduce_dim=K)

    bias2 = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_bias2 = to_reference(bias2, True)

    torch.addmm(ref_bias2, ref_mat1, ref_mat2, alpha=alpha, beta=beta, out=ref_out)
    with flag_gems.use_gems():
        torch.addmm(bias2, mat1, mat2, alpha=alpha, beta=beta, out=out)

    gems_assert_close(out, ref_out, dtype, reduce_dim=K)


@pytest.mark.bmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_bmm(M, N, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skiping fp32 bmm test on tsingmicro platform")

    if flag_gems.vendor_name == "mthreads":
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        random.seed(0)

    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)

    ref_out = torch.bmm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.bmm(mat1, mat2)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)

    if flag_gems.vendor_name == "mthreads":
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.bmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_bmm_out(M, N, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skiping fp32 bmm test on tsingmicro platform")

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        random.seed(0)

    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    out = torch.empty((batch, M, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)

    ref_out = torch.bmm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        torch.bmm(mat1, mat2, out=out)

    gems_assert_close(out, ref_out, dtype, reduce_dim=K)


@pytest.mark.bmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_bmm_non_contiguous(M, N, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skiping fp32 bmm test on tsingmicro platform")

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        random.seed(0)

    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2_raw = torch.randn((batch, N, K), dtype=dtype, device=flag_gems.device)
    # make mat2 non-contiguous
    mat2 = mat2_raw.transpose(1, 2)

    if N > 1 and K > 1:
        assert not mat2.is_contiguous()
    else:
        pytest.skip("Skipping non-contiguous test for small N or K")

    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)
    ref_out = torch.bmm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.bmm(mat1, mat2)
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


FP8_MNK_SHAPES = [
    (128, 256, 512),
    (64, 128, 128),
    (256, 256, 256),
    (83, 7748, 3884),
    (84, 7168, 3884),
]


@pytest.mark.w8a8_block_fp8_matmul
@pytest.mark.parametrize("M,N,K", FP8_MNK_SHAPES)
def test_w8a8_block_fp8_matmul(M, N, K):
    if flag_gems.vendor_name == "mthreads":
        if hasattr(torch, "float8_e4m3fn"):
            dtype = torch.float8_e4m3fn
        else:
            dtype = torch.float32
    else:
        if not torch.cuda.is_available():
            pytest.skip("w8a8_block_fp8_matmul test requires CUDA or mthreads")
        major, _ = torch.cuda.get_device_capability()
        if major > 8:
            dtype = torch.float8_e4m3fn
        elif major == 8:
            dtype = torch.float8_e5m2
        else:
            dtype = torch.float32
    device = flag_gems.device
    block_n = 128
    block_k = 128
    block_size = [block_n, block_k]

    A = torch.randn((M, K), device=device).to(dtype)
    B = torch.randn((N, K), device=device).to(dtype)

    num_k_groups = (K + block_k - 1) // block_k
    num_n_groups = (N + block_n - 1) // block_n

    As = (0.01 * torch.rand(M, num_k_groups, device=device) + 0.005).to(torch.float32)
    Bs = (0.01 * torch.rand(num_n_groups, num_k_groups, device=device) + 0.005).to(
        torch.float32
    )

    A_ref = A.to(torch.float32)
    B_ref = B.to(torch.float32)
    As_ref = As.to(torch.float32)
    Bs_ref = Bs.to(torch.float32)

    A_scaled = torch.zeros_like(A_ref)
    for k_group in range(num_k_groups):
        k_start = k_group * block_k
        k_end = min(k_start + block_k, K)
        scale = As_ref[:, k_group : k_group + 1]  # [M, 1]
        A_scaled[:, k_start:k_end] = A_ref[:, k_start:k_end] * scale

    B_scaled = torch.zeros_like(B_ref)
    for n_group in range(num_n_groups):
        n_start = n_group * block_n
        n_end = min(n_start + block_n, N)
        for k_group in range(num_k_groups):
            k_start = k_group * block_k
            k_end = min(k_start + block_k, K)
            scale = Bs_ref[n_group, k_group]  # scalar
            B_scaled[n_start:n_end, k_start:k_end] = (
                B_ref[n_start:n_end, k_start:k_end] * scale
            )

    ref_out = torch.matmul(A_scaled, B_scaled.T)
    ref_out = to_reference(ref_out, True)
    with flag_gems.use_gems():
        res_out = flag_gems.w8a8_block_fp8_matmul(
            A, B, As, Bs, block_size, output_dtype=torch.float16
        )
    ref_out_fp16 = ref_out.to(torch.float16)
    gems_assert_close(res_out, ref_out_fp16, dtype=torch.float16, reduce_dim=K)


@pytest.mark.baddbmm
@pytest.mark.linear
@pytest.mark.matmul
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_baddbmm(M, N, K, scalar, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype in [torch.float16, torch.bfloat16]:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"
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

    if flag_gems.vendor_name == "mthreads" and dtype in [torch.float16, torch.bfloat16]:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.skipif(
    flag_gems.vendor_name == "spacemit",
    reason="TODO: backward not supported on spacemit",
)
@pytest.mark.baddbmm_backward
@pytest.mark.linear
@pytest.mark.matmul
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_baddbmm_backward(M, N, K, scalar, dtype):
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


# TODO: failed at (1, 1, 2)
@pytest.mark.mm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("b_column_major", [True, False])
def test_mm(M, N, K, dtype, b_column_major):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skiping fp32 mm test on tsingmicro platform")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    if b_column_major:
        mat2 = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    else:
        mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)

    ref_out = torch.mm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.mm(mat1, mat2)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.grouped_mm
@pytest.mark.skipif(
    SkipVersion("torch", "<2.8"),
    reason="torch._grouped_mm requires PyTorch >= 2.8.0.",
)
@pytest.mark.parametrize("groups, N, K", GNK_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_grouped_mm(groups, N, K, dtype):
    assert dtype == torch.bfloat16
    group_A_list = []
    group_B_list = []
    M_list = []
    A_offs = 0
    B_offs = 0
    for i in range(groups):
        M_g = random.randint(1, 16384)
        N_g = N
        K_g = K
        A_g = torch.rand([M_g, K_g], device="cuda", dtype=dtype)
        B_g = torch.rand([K_g, N_g], device="cuda", dtype=dtype)
        group_A_list.append(A_g)
        group_B_list.append(B_g)
        M_list.append(M_g)
        A_offs += M_g
        B_offs += K_g

    mat_a = torch.cat([x for x in group_A_list], dim=0)
    mat_b = torch.stack([x for x in group_B_list], dim=0)
    offs = torch.tensor(
        [sum(M_list[: i + 1]) for i in range(groups)],
        dtype=torch.int32,
        device="cuda",
    )

    ref_out = torch._grouped_mm(mat_a, mat_b, offs)
    with flag_gems.use_gems():
        res_out = torch._grouped_mm(mat_a, mat_b, offs)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype_a, dtype_b", MIXED_DTYPE_PAIRS)
@pytest.mark.parametrize("b_column_major", [True, False])
def test_mm_mixed_dtype(M, N, K, dtype_a, dtype_b, b_column_major):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    mat1 = torch.randn((M, K), dtype=dtype_a, device=flag_gems.device)
    if b_column_major:
        mat2 = torch.randn((N, K), dtype=dtype_b, device=flag_gems.device).t()
    else:
        mat2 = torch.randn((K, N), dtype=dtype_b, device=flag_gems.device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)

    out_dtype = torch.promote_types(dtype_a, dtype_b)

    ref_out = torch.mm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.mm(mat1, mat2)

    gems_assert_close(res_out, ref_out, out_dtype, reduce_dim=K)


@pytest.mark.mv
@pytest.mark.parametrize("M, N", MN_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mv(M, N, dtype):
    matrix = torch.randn((N, M), dtype=dtype, device=flag_gems.device)
    vector = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    ref_matrix = to_reference(matrix, True)
    ref_vector = to_reference(vector, True)

    ref_out = torch.mv(ref_matrix, ref_vector)
    with flag_gems.use_gems():
        res_out = torch.mv(matrix, vector)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=M)


@pytest.mark.addmv
@pytest.mark.parametrize("M, N", MN_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_addmv(M, N, scalar, dtype):
    mat = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    vec = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    bias1 = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    ref_mat = to_reference(mat, True)
    ref_vec = to_reference(vec, True)
    ref_bias1 = to_reference(bias1, True)

    alpha = beta = scalar

    ref_out1 = torch.addmv(ref_bias1, ref_mat, ref_vec, alpha=alpha, beta=beta)
    with flag_gems.use_gems():
        res_out1 = torch.addmv(bias1, mat, vec, alpha=alpha, beta=beta)

    gems_assert_close(res_out1, ref_out1, dtype, reduce_dim=N)

    # broadcast bias scalar
    bias2 = torch.randn((), dtype=dtype, device=flag_gems.device)
    if flag_gems.vendor_name == "kunlunxin":
        ref_bias2 = to_reference(bias2, True)
    else:
        ref_bias2 = to_reference(bias2)

    ref_out2 = torch.addmv(ref_bias2, ref_mat, ref_vec, alpha=alpha, beta=beta)
    with flag_gems.use_gems():
        res_out2 = torch.addmv(bias2, mat, vec, alpha=alpha, beta=beta)

    gems_assert_close(res_out2, ref_out2, dtype, reduce_dim=N)


@pytest.mark.addmv_out
@pytest.mark.parametrize("M, N", MN_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_addmv_out(M, N, scalar, dtype):
    mat = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    vec = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    out = torch.empty((M,), dtype=dtype, device=flag_gems.device)
    ref_mat = to_reference(mat, True)
    ref_vec = to_reference(vec, True)
    ref_bias = to_reference(bias, True)
    ref_out = to_reference(out, True)

    alpha = beta = scalar

    torch.addmv(ref_bias, ref_mat, ref_vec, alpha=alpha, beta=beta, out=ref_out)
    with flag_gems.use_gems():
        torch.addmv(bias, mat, vec, alpha=alpha, beta=beta, out=out)

    gems_assert_close(out, ref_out, dtype, reduce_dim=N)


@pytest.mark.outer
@pytest.mark.parametrize(
    "M, N", MN_SHAPES + ([(32, 131072)] if flag_gems.vendor_name == "cambricon" else [])
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_outer(M, N, dtype):
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


@pytest.mark.skipif(flag_gems.vendor_name == "spacemit", reason="TODO")
@pytest.mark.vdot
@pytest.mark.parametrize("M", UT_SHAPES_1D)
@pytest.mark.parametrize(
    "is_conj", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.cfloat])
@pytest.mark.parametrize("stride", [1, 2])
def test_vdot(M, is_conj, dtype, stride):
    if flag_gems.vendor_name == "kunlunxin" or flag_gems.vendor_name == "tsingmicro":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    inp1_is_conj, inp2_is_conj = is_conj

    if flag_gems.vendor_name == "mthreads" or flag_gems.vendor_name == "tsingmicro":
        inp1 = torch.randn(M, dtype=dtype, device="cpu")
        inp2 = torch.randn(M, dtype=dtype, device="cpu")
    elif flag_gems.vendor_name == "ascend" and dtype == torch.cfloat:
        pytest.skip("Skipping torch.cfloat tests on Ascend platform")
    elif flag_gems.vendor_name == "tsingmicro" and dtype == torch.cfloat:
        pytest.skip("Skipping torch.cfloa tests on tsingmicro platform")
    elif flag_gems.vendor_name == "kunlunxin" and dtype == torch.cfloat:
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
        elif flag_gems.vendor_name == "tsingmicro":
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
def test_dot_tensor_tensor(shape, dtype):
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


@pytest.mark.addr
@pytest.mark.parametrize("M, N", MN_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_addr(M, N, dtype):
    input_tensor = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    vec1 = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    vec2 = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    alpha = torch.randn((), dtype=dtype, device=flag_gems.device)
    beta = torch.randn((), dtype=dtype, device=flag_gems.device)

    ref_input = to_reference(input_tensor, True)
    ref_vec1 = to_reference(vec1, True)
    ref_vec2 = to_reference(vec2, True)

    ref_out = torch.addr(ref_input, ref_vec1, ref_vec2, alpha=alpha, beta=beta)
    with flag_gems.use_gems():
        res_out = torch.addr(input_tensor, vec1, vec2, alpha=alpha, beta=beta)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
