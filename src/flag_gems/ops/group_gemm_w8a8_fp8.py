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

import functools
import logging
from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.device_info import get_sm_count

logger = logging.getLogger(__name__)


@triton.jit
def _grouped_tile_id(
    tile_id,
    m,
    n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    num_m_tiles = tl.cdiv(m, BLOCK_M)
    num_n_tiles = tl.cdiv(n, BLOCK_N)
    tiles_per_group = GROUP_M * num_n_tiles
    group_id = tile_id // tiles_per_group
    first_m_tile = group_id * GROUP_M
    group_m = tl.minimum(num_m_tiles - first_m_tile, GROUP_M)
    tile_in_group = tile_id % tiles_per_group
    tile_m = first_m_tile + tile_in_group % group_m
    tile_n = tile_in_group // group_m
    return tile_m, tile_n


@libentry()
@triton.jit
def group_gemm_w8a8_fp8_kernel(
    A,
    B,
    A_SCALE,
    B_SCALE,
    C,
    OFFS,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bg: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_ask: tl.constexpr,
    stride_bsg: tl.constexpr,
    stride_bsk: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    SCALE_BLOCK_M: tl.constexpr,
    SCALE_BLOCK_N: tl.constexpr,
    SCALE_BLOCK_K: tl.constexpr,
    A_SCALE_PER_ROW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    tile_id = tl.program_id(0).to(tl.int64)
    grid_size = tl.num_programs(0).to(tl.int64)
    problem_start = tl.full((), 0, tl.int64)
    group_start = tl.full((), 0, tl.int64)

    for group_id in tl.range(NUM_GROUPS):
        group_end = tl.load(OFFS + group_id).to(tl.int64)
        group_m = group_end - group_start
        num_tiles = tl.cdiv(group_m, BLOCK_M) * tl.cdiv(N, BLOCK_N)
        problem_end = problem_start + num_tiles

        if tile_id >= problem_start and tile_id < problem_end:
            loop_count = (problem_end - tile_id + grid_size - 1) // grid_size
            for _ in tl.range(loop_count):
                local_tile = tile_id - problem_start
                tile_m, tile_n = _grouped_tile_id(
                    local_tile, group_m, N, BLOCK_M, BLOCK_N, GROUP_M
                )
                m_offset = group_start + tile_m * BLOCK_M
                n_offset = tile_n * BLOCK_N
                rows = m_offset + tl.arange(0, BLOCK_M)
                cols = n_offset + tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)

                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for k_block in range(0, tl.cdiv(K, BLOCK_K)):
                    k_offset = k_block * BLOCK_K
                    ks = k_offset + offs_k
                    a_ptrs = A + rows[:, None] * stride_am + ks[None, :] * stride_ak
                    b_ptrs = (
                        B
                        + group_id * stride_bg
                        + ks[:, None] * stride_bk
                        + cols[None, :] * stride_bn
                    )
                    a_mask = (
                        (rows[:, None] >= 0)
                        & (rows[:, None] < group_end)
                        & (rows[:, None] < M)
                        & (ks[None, :] < K)
                    )
                    b_mask = (ks[:, None] < K) & (cols[None, :] < N)
                    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
                    b = tl.load(b_ptrs, mask=b_mask, other=0.0)
                    partial = tl.dot(a, b, allow_tf32=False)

                    if A_SCALE_PER_ROW:
                        a_scale_rows = rows
                    else:
                        a_scale_rows = rows // SCALE_BLOCK_M
                    scale_k = k_offset // SCALE_BLOCK_K
                    a_scale = tl.load(
                        A_SCALE + a_scale_rows * stride_asm + scale_k * stride_ask,
                        mask=(rows >= 0) & (rows < group_end) & (rows < M),
                        other=0.0,
                    ).to(tl.float32)
                    scale_n = n_offset // SCALE_BLOCK_N
                    b_scale = tl.load(
                        B_SCALE
                        + group_id * stride_bsg
                        + scale_k * stride_bsk
                        + scale_n * stride_bsn
                    ).to(tl.float32)
                    accumulator += partial * a_scale[:, None] * b_scale

                c_ptrs = C + rows[:, None] * stride_cm + cols[None, :] * stride_cn
                c_mask = (
                    (rows[:, None] >= 0)
                    & (rows[:, None] < group_end)
                    & (rows[:, None] < M)
                    & (cols[None, :] < N)
                )
                tl.store(c_ptrs, accumulator, mask=c_mask)
                tile_id += grid_size

        problem_start = problem_end
        group_start = group_end


_FLOAT8_DTYPES = tuple(
    getattr(torch, name)
    for name in (
        "float8_e4m3fn",
        "float8_e5m2",
    )
    if hasattr(torch, name)
)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


@functools.lru_cache
def _get_group_gemm_w8a8_fp8_config(
    m: int,
    n: int,
    k: int,
    num_groups: int,
    block_n: int,
    block_k: int,
):
    average_m = triton.cdiv(m, num_groups)
    if average_m <= 16:
        block_m, group_m = 16, 1
    else:
        block_m, group_m = 64, 8

    if average_m > 16:
        tile_n = min(block_n, 64)
    elif n <= 64:
        tile_n = min(block_n, 64)
    elif average_m <= 32 and n < 1024 and k > 128:
        tile_n = min(block_n, 64)
    else:
        tile_n = min(block_n, 128)
    tile_n = min(tile_n, max(16, triton.next_power_of_2(n)))
    return {
        "BLOCK_M": block_m,
        "BLOCK_N": tile_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
        "num_warps": 4,
        "num_stages": 3,
    }


def group_gemm_w8a8_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    offs: torch.Tensor,
    block_size: Sequence[int] = (128, 128, 128),
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Grouped W8A8 FP8 GEMM with block-wise dequantization.

    ``A`` has shape ``[M, K]``, ``B`` has shape ``[G, K, N]``, and ``offs``
    contains the nondecreasing cumulative row count for every group and must
    end at ``M``. ``B_scale`` has shape
    ``[G, ceil(K / block_k), ceil(N / block_n)]``. ``A_scale`` can use either
    symmetric blocks, ``[ceil(M / block_m), ceil(K / block_k)]``, or per-row
    activation blocks, ``[M, ceil(K / block_k)]``. Both common ``B`` strides
    are supported; on H20, a logical ``[G, K, N]`` transpose view backed by a
    contiguous ``[G, N, K]`` tensor gives the highest throughput.
    """
    logger.debug("GEMS GROUP GEMM W8A8 FP8")
    if A.ndim != 2 or B.ndim != 3:
        raise RuntimeError("A must be 2D and B must be 3D")
    if A.dtype != B.dtype or A.dtype not in _FLOAT8_DTYPES:
        raise RuntimeError("A and B must have the same FP8 dtype")
    if A.device != B.device:
        raise RuntimeError("A and B must be on the same device")
    if len(block_size) != 3:
        raise RuntimeError("block_size must contain (block_m, block_n, block_k)")

    block_m, block_n, block_k = (int(value) for value in block_size)
    if not all(_is_power_of_two(value) for value in (block_m, block_n, block_k)):
        raise RuntimeError("all block sizes must be positive powers of two")
    if min(block_m, block_n) < 16 or not 32 <= block_k <= 256:
        raise RuntimeError(
            "block_m and block_n must be at least 16; block_k must be in [32, 256]"
        )

    M, K = A.shape
    num_groups, BK, N = B.shape
    if num_groups == 0:
        raise RuntimeError("B must contain at least one group")
    if K != BK:
        raise RuntimeError(f"K dimension mismatch: {K} and {BK}")
    if offs.ndim != 1 or offs.dtype != torch.int32 or offs.numel() != num_groups:
        raise RuntimeError("offs must be a 1D int32 tensor with one entry per group")
    if offs.device != A.device:
        raise RuntimeError("offs must be on the same device as A")
    offs = offs.contiguous()

    num_k_blocks = triton.cdiv(K, block_k)
    num_n_blocks = triton.cdiv(N, block_n)
    block_a_shape = (triton.cdiv(M, block_m), num_k_blocks)
    row_a_shape = (M, num_k_blocks)
    if tuple(A_scale.shape) == row_a_shape:
        a_scale_per_row = True
    elif tuple(A_scale.shape) == block_a_shape:
        a_scale_per_row = False
    else:
        raise RuntimeError(f"A_scale must have shape {block_a_shape} or {row_a_shape}")
    expected_b_scale_shape = (num_groups, num_k_blocks, num_n_blocks)
    if tuple(B_scale.shape) != expected_b_scale_shape:
        raise RuntimeError(f"B_scale must have shape {expected_b_scale_shape}")
    if A_scale.dtype != torch.float32 or B_scale.dtype != torch.float32:
        raise RuntimeError("A_scale and B_scale must have dtype torch.float32")
    if A_scale.device != A.device or B_scale.device != A.device:
        raise RuntimeError("scales must be on the same device as A")
    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise RuntimeError("out_dtype must be bfloat16, float16, or float32")

    if out is None:
        out = torch.empty((M, N), dtype=out_dtype, device=A.device)
    elif out.shape != (M, N) or out.dtype != out_dtype or out.device != A.device:
        raise RuntimeError("out must match the expected shape, dtype, and device")
    if out.numel() == 0:
        return out

    config = _get_group_gemm_w8a8_fp8_config(M, N, K, num_groups, block_n, block_k)
    if block_n % config["BLOCK_N"] != 0:
        raise RuntimeError("block_n must be divisible by the selected BLOCK_N")

    grid_multiplier = 4 if K <= 256 or config["BLOCK_N"] <= 64 else 2
    launch_args = dict(
        M=M,
        N=N,
        K=K,
        NUM_GROUPS=num_groups,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_bg=B.stride(0),
        stride_bk=B.stride(1),
        stride_bn=B.stride(2),
        stride_asm=A_scale.stride(0),
        stride_ask=A_scale.stride(1),
        stride_bsg=B_scale.stride(0),
        stride_bsk=B_scale.stride(1),
        stride_bsn=B_scale.stride(2),
        stride_cm=out.stride(0),
        stride_cn=out.stride(1),
        SCALE_BLOCK_M=block_m,
        SCALE_BLOCK_N=block_n,
        SCALE_BLOCK_K=block_k,
        A_SCALE_PER_ROW=a_scale_per_row,
        **config,
    )
    with torch_device_fn.device(A.device):
        grid = (get_sm_count() * grid_multiplier,)
        group_gemm_w8a8_fp8_kernel.fn[grid](
            A, B, A_scale, B_scale, out, offs, **launch_args
        )
    return out


group_mm_w8a8_fp8 = group_gemm_w8a8_fp8
