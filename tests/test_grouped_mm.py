# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import pytest
import torch
import triton

import flag_gems
from flag_gems.ops.group_gemm_w8a8_fp8 import _int32_addressing_is_safe

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

GNK_SHAPES = [(16, 512, 2048), (16, 2560, 2048), (64, 2048, 128)]
FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
FP8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    )
    if dtype is not None
)


@pytest.mark.grouped_mm
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.8"),
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

    for _ in range(groups):
        M_g = random.randint(1, 16384)
        N_g = N
        K_g = K
        A_g = torch.rand([M_g, K_g], device=flag_gems.device, dtype=dtype)
        B_g = torch.rand([K_g, N_g], device=flag_gems.device, dtype=dtype)
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
        device=flag_gems.device,
    )

    if utils.TO_CPU:
        ref_out = torch._grouped_mm(mat_a.cpu(), mat_b.cpu(), offs.cpu())
    else:
        ref_out = torch._grouped_mm(mat_a, mat_b, offs)
    res_out = flag_gems.group_mm(mat_a, mat_b, offs)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.group_gemm_w8a8_fp8
def test_group_gemm_w8a8_fp8_int32_address_boundaries():
    int32_max = torch.iinfo(torch.int32).max
    common = {
        "m": 2,
        "n": 1,
        "k": 1,
        "num_groups": 1,
        "block_m": 1,
        "block_n": 1,
        "block_k": 1,
        "scale_block_m": 1,
        "scale_block_n": 1,
        "a_scale_per_row": False,
        "a_strides": (int32_max, 1),
        "b_strides": (1, 1, 1),
        "a_scale_strides": (1, 1),
        "b_scale_strides": (1, 1, 1),
        "out_strides": (1, 1),
        "offs_stride": 1,
    }

    assert _int32_addressing_is_safe(**common)
    assert not _int32_addressing_is_safe(**{**common, "a_strides": (int32_max + 1, 1)})
    assert not _int32_addressing_is_safe(
        **{
            **common,
            "num_groups": 2,
            "b_strides": (int32_max + 1, 1, 1),
        }
    )
    assert not _int32_addressing_is_safe(
        **{
            **common,
            "m": 1,
            "block_m": 16,
            "a_strides": (int32_max // 15 + 1, 1),
        }
    )


def _cuda_fp8_available():
    if FP8_DTYPE is None or flag_gems.device != "cuda" or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 9


def _quantize_a(A, block_m, block_k, per_row):
    M, K = A.shape
    padded_m = triton.cdiv(M, block_m) * block_m
    padded_k = triton.cdiv(K, block_k) * block_k
    padded = torch.zeros((padded_m, padded_k), dtype=A.dtype, device=A.device)
    padded[:M, :K] = A
    fp8_info = torch.finfo(FP8_DTYPE)

    if per_row:
        grouped = padded[:M].reshape(M, padded_k // block_k, block_k).float()
        scale = (grouped.abs().amax(dim=2) / fp8_info.max).clamp(min=1e-8)
        quantized = (
            (grouped / scale[:, :, None])
            .clamp(fp8_info.min, fp8_info.max)
            .to(FP8_DTYPE)
            .reshape(M, padded_k)[:, :K]
            .contiguous()
        )
        return quantized, scale.float().contiguous()

    grouped = padded.reshape(
        padded_m // block_m, block_m, padded_k // block_k, block_k
    ).float()
    scale = (grouped.abs().amax(dim=(1, 3)) / fp8_info.max).clamp(min=1e-8)
    quantized = (
        (grouped / scale[:, None, :, None])
        .clamp(fp8_info.min, fp8_info.max)
        .to(FP8_DTYPE)
        .reshape(padded_m, padded_k)[:M, :K]
        .contiguous()
    )
    return quantized, scale.float().contiguous()


def _quantize_b(B_nk, block_n, block_k, k_major):
    groups, N, K = B_nk.shape
    padded_n = triton.cdiv(N, block_n) * block_n
    padded_k = triton.cdiv(K, block_k) * block_k
    padded = torch.zeros(
        (groups, padded_n, padded_k), dtype=B_nk.dtype, device=B_nk.device
    )
    padded[:, :N, :K] = B_nk
    grouped = padded.reshape(
        groups, padded_n // block_n, block_n, padded_k // block_k, block_k
    ).float()
    fp8_info = torch.finfo(FP8_DTYPE)
    scale_nk = (grouped.abs().amax(dim=(2, 4)) / fp8_info.max).clamp(min=1e-8)
    quantized_nk = (
        (grouped / scale_nk[:, :, None, :, None])
        .clamp(fp8_info.min, fp8_info.max)
        .to(FP8_DTYPE)
        .reshape(groups, padded_n, padded_k)[:, :N, :K]
        .contiguous()
    )
    B_scale = scale_nk.permute(0, 2, 1).float().contiguous()
    if k_major:
        return quantized_nk.transpose(-1, -2), B_scale
    return quantized_nk.transpose(-1, -2).contiguous(), B_scale


def _dequantize_a(A, A_scale, block_m, block_k, per_row):
    M, K = A.shape
    k_blocks = torch.arange(K, device=A.device) // block_k
    if per_row:
        scale = A_scale[:, k_blocks]
    else:
        m_blocks = torch.arange(M, device=A.device) // block_m
        scale = A_scale[m_blocks[:, None], k_blocks[None, :]]
    return A.float() * scale.float()


def _dequantize_b(B, B_scale, block_n, block_k):
    _, K, N = B.shape
    k_blocks = torch.arange(K, device=B.device) // block_k
    n_blocks = torch.arange(N, device=B.device) // block_n
    scale = B_scale[:, k_blocks[:, None], n_blocks[None, :]]
    return B.float() * scale.float()


def _reference(A, B, A_scale, B_scale, offs, block_size, per_row):
    block_m, block_n, block_k = block_size
    A = _dequantize_a(A, A_scale, block_m, block_k, per_row)
    B = _dequantize_b(B, B_scale, block_n, block_k)
    starts = [0] + offs.cpu().tolist()
    chunks = [A[starts[g] : starts[g + 1]].mm(B[g]) for g in range(offs.numel())]
    return torch.cat(chunks, dim=0).to(torch.bfloat16)


@pytest.mark.group_gemm_w8a8_fp8
@pytest.mark.skipif(
    not _cuda_fp8_available(),
    reason="group GEMM W8A8 FP8 requires CUDA SM90+ float8_e4m3fn support",
)
@pytest.mark.parametrize(
    "sizes,N,K,block_size,per_row,k_major,strided",
    [
        ([3, 5, 1], 192, 256, (128, 128, 128), False, True, False),
        ([0, 1, 17, 0, 33], 96, 384, (128, 128, 128), True, False, False),
        ([33, 17], 192, 256, (128, 128, 128), False, False, False),
        ([65, 80], 96, 256, (128, 128, 128), False, True, False),
        ([1, 2], 48, 80, (32, 32, 32), False, False, True),
        ([4] * 64, 256, 128, (128, 128, 128), False, True, False),
        ([256] + [0] * 63, 192, 128, (128, 128, 128), True, True, False),
        ([193] + [1] * 63, 192, 128, (128, 128, 128), False, False, False),
        ([33] * 16, 130, 129, (128, 128, 128), False, True, False),
    ],
)
def test_group_gemm_w8a8_fp8(sizes, N, K, block_size, per_row, k_major, strided):
    torch.manual_seed(0)
    M = sum(sizes)
    A = torch.randn((M, K), dtype=torch.bfloat16, device=flag_gems.device) * 0.25
    B_nk = (
        torch.randn((len(sizes), N, K), dtype=torch.bfloat16, device=flag_gems.device)
        * 0.25
    )
    block_m, block_n, block_k = block_size
    A_fp8, A_scale = _quantize_a(A, block_m, block_k, per_row)
    B_fp8, B_scale = _quantize_b(B_nk, block_n, block_k, k_major)
    if strided:
        A_storage = torch.empty((M, K * 2 + 1), dtype=A_fp8.dtype, device=A_fp8.device)
        A_storage[:, 1::2] = A_fp8
        A_fp8 = A_storage[:, 1::2]
        B_storage = torch.empty(
            (len(sizes), K, N * 2 + 1), dtype=B_fp8.dtype, device=B_fp8.device
        )
        B_storage[:, :, 1::2] = B_fp8
        B_fp8 = B_storage[:, :, 1::2]
    offs = torch.tensor(
        [sum(sizes[: g + 1]) for g in range(len(sizes))],
        dtype=torch.int32,
        device=flag_gems.device,
    )
    if strided:
        offs_storage = torch.empty(
            (offs.numel() * 2 + 1,), dtype=offs.dtype, device=offs.device
        )
        offs_storage[1::2] = offs
        offs = offs_storage[1::2]

    ref = _reference(A_fp8, B_fp8, A_scale, B_scale, offs, block_size, per_row)
    result = flag_gems.group_gemm_w8a8_fp8(
        A_fp8,
        B_fp8,
        A_scale,
        B_scale,
        offs,
        block_size=block_size,
    )
    cached_result = torch.empty_like(result)
    flag_gems.group_gemm_w8a8_fp8(
        A_fp8,
        B_fp8,
        A_scale,
        B_scale,
        offs,
        block_size=block_size,
        out=cached_result,
    )
    ref = ref.cpu() if utils.TO_CPU else ref
    utils.gems_assert_close(result, ref, torch.bfloat16, reduce_dim=K)
    utils.gems_assert_close(cached_result, ref, torch.bfloat16, reduce_dim=K)


@pytest.mark.group_gemm_w8a8_fp8
@pytest.mark.skipif(
    not _cuda_fp8_available(),
    reason="group GEMM W8A8 FP8 requires CUDA SM90+ float8_e4m3fn support",
)
def test_group_gemm_w8a8_fp8_rejects_small_block_k():
    A = torch.empty((1, 32), dtype=FP8_DTYPE, device=flag_gems.device)
    B = torch.empty((1, 32, 16), dtype=FP8_DTYPE, device=flag_gems.device)
    A_scale = torch.empty((1, 2), dtype=torch.float32, device=flag_gems.device)
    B_scale = torch.empty((1, 2, 1), dtype=torch.float32, device=flag_gems.device)
    offs = torch.ones((1,), dtype=torch.int32, device=flag_gems.device)

    with pytest.raises(RuntimeError, match=r"block_k must be in \[32, 256\]"):
        flag_gems.group_gemm_w8a8_fp8(
            A,
            B,
            A_scale,
            B_scale,
            offs,
            block_size=(16, 16, 16),
        )


@pytest.mark.group_gemm_w8a8_fp8
@pytest.mark.skipif(
    not _cuda_fp8_available(),
    reason="group GEMM W8A8 FP8 requires CUDA SM90+ FP8 support",
)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_group_gemm_w8a8_fp8_dtypes(fp8_dtype, out_dtype):
    torch.manual_seed(1)
    sizes = [4, 3]
    M, N, K = sum(sizes), 32, 32
    A = (torch.randn((M, K), dtype=torch.bfloat16, device=flag_gems.device) * 0.1).to(
        fp8_dtype
    )
    B = (
        torch.randn(
            (len(sizes), N, K),
            dtype=torch.bfloat16,
            device=flag_gems.device,
        )
        * 0.1
    ).to(fp8_dtype)
    B = B.transpose(-1, -2)
    A_scale = torch.ones((1, 1), dtype=torch.float32, device=flag_gems.device)
    B_scale = torch.ones(
        (len(sizes), 1, 1), dtype=torch.float32, device=flag_gems.device
    )
    offs = torch.tensor([4, 7], dtype=torch.int32, device=flag_gems.device)
    ref = torch.cat((A[:4].float().mm(B[0].float()), A[4:].float().mm(B[1].float())))
    result = flag_gems.group_gemm_w8a8_fp8(
        A,
        B,
        A_scale,
        B_scale,
        offs,
        block_size=(32, 32, 32),
        out_dtype=out_dtype,
    )
    torch.testing.assert_close(result.float(), ref, atol=2e-2, rtol=2e-2)
