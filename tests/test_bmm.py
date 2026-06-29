import random

import numpy as np
import pytest
import torch

import flag_gems
from flag_gems.ops.bmm import bmm_fp8_w8a8, bmm_fp8_w8a8_block_scale, bmm_fp8_w8a16

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
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
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.bmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_bmm(monkeypatch, M, N, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("#2799: Skipping fp32 bmm test on tsingmicro platform.")

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        random.seed(0)

    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)

    ref_out = torch.bmm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.bmm(mat1, mat2)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.bmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_bmm_non_contiguous(M, N, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2799: Skipping fp32 bmm test on tsingmicro.")

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
        # Skipping non-contiguous test for small N or K
        return

    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_out = torch.bmm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.bmm(mat1, mat2)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.bmm_out
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_bmm_out(M, N, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2799: Skipping fp32 bmm test on tsingmicro.")

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        random.seed(0)

    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    out = torch.empty((batch, M, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)

    ref_out = torch.bmm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        torch.bmm(mat1, mat2, out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K)


FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
FP8_BMM_SHAPES = [
    (2, 16, 32, 64),
    (4, 64, 64, 128),
    (2, 128, 128, 256),
]
FP8_W8A8_BMM_SHAPES = [
    (2, 16, 32, 64),
    (4, 64, 64, 128),
    (2, 128, 128, 256),
    (4, 256, 256, 512),
]
FP8_W8A8_BLOCK_SCALE_BMM_SHAPES = [
    (2, 16, 32, 64),
    (4, 64, 64, 128),
    (2, 128, 128, 256),
]


def _is_fp8e4nv_supported():
    if flag_gems.device != "cuda" or FP8_DTYPE is None:
        return False
    major, minor = torch.cuda.get_device_capability()
    return major + minor / 10 >= 8.9


def _quantize_b_fp8_per_k_block(B, block_size=128):
    fp8_info = torch.finfo(FP8_DTYPE)
    batch, K, N = B.shape
    num_blocks = (K + block_size - 1) // block_size
    padded_k = num_blocks * block_size
    if padded_k != K:
        B_for_scale = torch.cat(
            [
                B,
                torch.zeros((batch, padded_k - K, N), dtype=B.dtype, device=B.device),
            ],
            dim=1,
        )
    else:
        B_for_scale = B
    B_blocked = B_for_scale.reshape(batch, num_blocks, block_size, N).float()
    scale = (B_blocked.abs().amax(dim=2) / fp8_info.max).clamp(min=1e-8)
    B_q = (
        (B_blocked / scale[:, :, None, :])
        .clamp(fp8_info.min, fp8_info.max)
        .to(FP8_DTYPE)
    )
    B_q = B_q.reshape(batch, padded_k, N)[:, :K, :].contiguous()
    return B_q, scale.to(B.dtype).contiguous()


def _dequant_b_fp8_per_k_block(B_fp8, B_scale, block_size=128):
    K = B_fp8.shape[1]
    block_ids = torch.arange(K, device=B_fp8.device) // block_size
    return B_fp8.float() * B_scale.index_select(1, block_ids).float()


def _quantize_a_fp8_per_k_block(A, block_k=128):
    fp8_info = torch.finfo(FP8_DTYPE)
    batch, M, K = A.shape
    num_k_blocks = (K + block_k - 1) // block_k
    padded_k = num_k_blocks * block_k
    if padded_k != K:
        A_for_scale = torch.cat(
            [
                A,
                torch.zeros((batch, M, padded_k - K), dtype=A.dtype, device=A.device),
            ],
            dim=2,
        )
    else:
        A_for_scale = A
    A_blocked = A_for_scale.reshape(batch, M, num_k_blocks, block_k).float()
    scale = (A_blocked.abs().amax(dim=3) / fp8_info.max).clamp(min=1e-8)
    A_q = (
        (A_blocked / scale[:, :, :, None])
        .clamp(fp8_info.min, fp8_info.max)
        .to(FP8_DTYPE)
    )
    A_q = A_q.reshape(batch, M, padded_k)[:, :, :K].contiguous()
    return A_q, scale.float().contiguous()


def _dequant_a_fp8_per_k_block(A_fp8, A_scale, block_k=128):
    K = A_fp8.shape[2]
    block_ids = torch.arange(K, device=A_fp8.device) // block_k
    return A_fp8.float() * A_scale.index_select(2, block_ids).float()


def _quantize_a_fp8_per_mk_block(A, block_m=128, block_k=128):
    fp8_info = torch.finfo(FP8_DTYPE)
    batch, M, K = A.shape
    num_m_blocks = (M + block_m - 1) // block_m
    num_k_blocks = (K + block_k - 1) // block_k
    padded_m = num_m_blocks * block_m
    padded_k = num_k_blocks * block_k
    A_for_scale = A
    if padded_m != M:
        A_for_scale = torch.cat(
            [
                A_for_scale,
                torch.zeros((batch, padded_m - M, K), dtype=A.dtype, device=A.device),
            ],
            dim=1,
        )
    if padded_k != K:
        A_for_scale = torch.cat(
            [
                A_for_scale,
                torch.zeros(
                    (batch, padded_m, padded_k - K),
                    dtype=A.dtype,
                    device=A.device,
                ),
            ],
            dim=2,
        )
    A_blocked = A_for_scale.reshape(
        batch, num_m_blocks, block_m, num_k_blocks, block_k
    ).float()
    scale = (A_blocked.abs().amax(dim=(2, 4)) / fp8_info.max).clamp(min=1e-8)
    A_q = (
        (A_blocked / scale[:, :, None, :, None])
        .clamp(fp8_info.min, fp8_info.max)
        .to(FP8_DTYPE)
    )
    A_q = A_q.reshape(batch, padded_m, padded_k)[:, :M, :K].contiguous()
    return A_q, scale.float().contiguous()


def _dequant_a_fp8_per_mk_block(A_fp8, A_scale, block_m=128, block_k=128):
    M = A_fp8.shape[1]
    K = A_fp8.shape[2]
    m_ids = torch.arange(M, device=A_fp8.device) // block_m
    k_ids = torch.arange(K, device=A_fp8.device) // block_k
    scale = A_scale.index_select(1, m_ids).index_select(2, k_ids)
    return A_fp8.float() * scale.float()


def _quantize_b_fp8_per_nk_block(B, block_n=128, block_k=128):
    fp8_info = torch.finfo(FP8_DTYPE)
    batch, K, N = B.shape
    num_k_blocks = (K + block_k - 1) // block_k
    num_n_blocks = (N + block_n - 1) // block_n
    padded_k = num_k_blocks * block_k
    padded_n = num_n_blocks * block_n
    B_for_scale = B
    if padded_k != K:
        B_for_scale = torch.cat(
            [
                B_for_scale,
                torch.zeros((batch, padded_k - K, N), dtype=B.dtype, device=B.device),
            ],
            dim=1,
        )
    if padded_n != N:
        B_for_scale = torch.cat(
            [
                B_for_scale,
                torch.zeros(
                    (batch, padded_k, padded_n - N),
                    dtype=B.dtype,
                    device=B.device,
                ),
            ],
            dim=2,
        )
    B_blocked = B_for_scale.reshape(
        batch, num_k_blocks, block_k, num_n_blocks, block_n
    ).float()
    scale = (B_blocked.abs().amax(dim=(2, 4)) / fp8_info.max).clamp(min=1e-8)
    B_q = (
        (B_blocked / scale[:, :, None, :, None])
        .clamp(fp8_info.min, fp8_info.max)
        .to(FP8_DTYPE)
    )
    B_q = B_q.reshape(batch, padded_k, padded_n)[:, :K, :N].contiguous()
    return B_q, scale.float().contiguous()


def _dequant_b_fp8_per_nk_block(B_fp8, B_scale, block_n=128, block_k=128):
    K = B_fp8.shape[1]
    N = B_fp8.shape[2]
    k_ids = torch.arange(K, device=B_fp8.device) // block_k
    n_ids = torch.arange(N, device=B_fp8.device) // block_n
    scale = B_scale.index_select(1, k_ids).index_select(2, n_ids)
    return B_fp8.float() * scale.float()


@pytest.mark.bmm
@pytest.mark.skipif(
    not _is_fp8e4nv_supported(),
    reason="FP8 BMM W8A16 requires CUDA fp8e4nv support",
)
@pytest.mark.parametrize("batch, M, N, K", FP8_BMM_SHAPES)
def test_bmm_fp8_w8a16(batch, M, N, K):
    torch.manual_seed(0)
    A = torch.randn((batch, M, K), dtype=torch.bfloat16, device=flag_gems.device)
    B = torch.randn((batch, K, N), dtype=torch.bfloat16, device=flag_gems.device)
    B_fp8, B_scale = _quantize_b_fp8_per_k_block(B)
    B_dequant = _dequant_b_fp8_per_k_block(B_fp8, B_scale).to(torch.bfloat16)

    ref_out = torch.bmm(A, B_dequant)
    res_out = bmm_fp8_w8a16(A, B_fp8, B_scale)

    utils.gems_assert_close(res_out, ref_out, torch.bfloat16, reduce_dim=K)


@pytest.mark.bmm
@pytest.mark.skipif(
    not _is_fp8e4nv_supported(),
    reason="FP8 BMM W8A8 requires CUDA fp8e4nv support",
)
@pytest.mark.parametrize("batch, M, N, K", FP8_W8A8_BMM_SHAPES)
def test_bmm_fp8_w8a8(batch, M, N, K):
    torch.manual_seed(0)
    A = torch.randn((batch, M, K), dtype=torch.bfloat16, device=flag_gems.device)
    B = torch.randn((batch, K, N), dtype=torch.bfloat16, device=flag_gems.device)
    A_fp8, A_scale = _quantize_a_fp8_per_k_block(A)
    B_fp8, B_scale = _quantize_b_fp8_per_nk_block(B)
    A_dequant = _dequant_a_fp8_per_k_block(A_fp8, A_scale)
    B_dequant = _dequant_b_fp8_per_nk_block(B_fp8, B_scale)

    ref_out = torch.bmm(A_dequant, B_dequant).to(torch.bfloat16)
    res_out = bmm_fp8_w8a8(A_fp8, B_fp8, A_scale, B_scale)

    utils.gems_assert_close(res_out, ref_out, torch.bfloat16, reduce_dim=K)


@pytest.mark.bmm
@pytest.mark.skipif(
    not _is_fp8e4nv_supported(),
    reason="FP8 BMM W8A8 block-scale requires CUDA fp8e4nv support",
)
@pytest.mark.parametrize("batch, M, N, K", FP8_W8A8_BLOCK_SCALE_BMM_SHAPES)
def test_bmm_fp8_w8a8_block_scale(batch, M, N, K):
    torch.manual_seed(0)
    A = torch.randn((batch, M, K), dtype=torch.bfloat16, device=flag_gems.device)
    B = torch.randn((batch, K, N), dtype=torch.bfloat16, device=flag_gems.device)
    A_fp8, A_scale = _quantize_a_fp8_per_mk_block(A)
    B_fp8, B_scale = _quantize_b_fp8_per_nk_block(B)
    A_dequant = _dequant_a_fp8_per_mk_block(A_fp8, A_scale)
    B_dequant = _dequant_b_fp8_per_nk_block(B_fp8, B_scale)

    ref_out = torch.bmm(A_dequant, B_dequant).to(torch.bfloat16)
    res_out = bmm_fp8_w8a8_block_scale(A_fp8, B_fp8, A_scale, B_scale)

    utils.gems_assert_close(res_out, ref_out, torch.bfloat16, reduce_dim=K)
