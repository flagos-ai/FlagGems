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

"""Ascend channel-scaled INT8 batched matrix multiplication."""

import torch
import triton
import triton.language as tl
from triton.experimental.tle import language as tle_async

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .mm_w8a8_int8 import (
    direct_mm_kernel,
    mm_int8_vector_full,
    mm_int8_vector_loop,
    mm_w8a8_int8_out,
)

direct_scaled_mm = direct_mm_kernel.fn
vector_loop = mm_int8_vector_loop.fn
vector_full = mm_int8_vector_full.fn


@libentry()
@triton.jit
def bmm_int8_vector_loop(
    A,
    B,
    SA,
    SB,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
):
    batch = tl.program_id(1).to(tl.int64)
    vector_loop(
        A + batch * M * K,
        B + batch * N * K,
        SA + batch * M,
        SB + batch * N,
        OUT + batch * M * N,
        M,
        N,
        K,
        K,
        1,
        1,
        K,
        N,
        1,
        BN,
        1024,
        True,
    )


@libentry()
@triton.jit
def bmm_int8_vector_full(
    A,
    B,
    SA,
    SB,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    batch = tl.program_id(1).to(tl.int64)
    vector_full(
        A + batch * M * K,
        B + batch * N * K,
        SA + batch * M,
        SB + batch * N,
        OUT + batch * M * N,
        M,
        N,
        K,
        K,
        1,
        1,
        K,
        N,
        1,
        1,
        BK,
    )


@libentry()
@triton.jit
def bmm_direct_scaled_mm(
    A,
    B,
    SA,
    SB,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PADDED_M: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    CORES: tl.constexpr,
):
    batch = tl.program_id(1).to(tl.int64)
    direct_scaled_mm(
        A + batch * PADDED_M * K,
        B + batch * N * K,
        SA + batch * M,
        SB + batch * N,
        OUT + batch * M * N,
        M,
        N,
        K,
        PADDED_M,
        K,
        1,
        1,
        K,
        N,
        1,
        BM,
        BN,
        BK,
        CORES,
        True,
    )


@libentry()
@triton.jit
def bmm_split_k_kernel(
    A,
    B,
    PARTIAL,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BATCH: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    CORES: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    grid_m: tl.constexpr = tl.cdiv(M, BM)
    grid_n: tl.constexpr = tl.cdiv(N, BN)
    tiles: tl.constexpr = grid_m * grid_n
    work_items: tl.constexpr = tiles * SPLIT_K
    quotient = work_items // CORES
    remainder = work_items % CORES
    start = tl.where(
        pid < remainder,
        pid * (quotient + 1),
        remainder * (quotient + 1) + (pid - remainder) * quotient,
    )
    count = quotient + tl.where(pid < remainder, 1, 0)
    for work_offset in tl.range(0, count):
        work = start + work_offset
        tile = work // SPLIT_K
        part = work % SPLIT_K
        row_start = (tile // grid_n * BM).to(tl.int32)
        col_start = (tile % grid_n * BN).to(tl.int32)
        ap = tl.make_block_ptr(
            A + batch * M * K,
            (M, K),
            (K, 1),
            (row_start, part * (K // SPLIT_K)),
            (BM, BK),
            (1, 0),
        )
        bp = tl.make_block_ptr(
            B + batch * N * K,
            (K, N),
            (1, K),
            (part * (K // SPLIT_K), col_start),
            (BK, BN),
            (0, 1),
        )
        acc = tl.zeros((BM, BN), tl.int32)
        for _ in tl.range(0, tl.cdiv(K, BK * SPLIT_K)):
            if M % BM == 0 and K % BK == 0:
                av = tl.load(ap)
            else:
                av = tl.load(ap, boundary_check=(0, 1), padding_option="zero")
            if N % BN == 0 and K % BK == 0:
                bv = tl.load(bp)
            else:
                bv = tl.load(bp, boundary_check=(0, 1), padding_option="zero")
            acc = tl.dot(av, bv, acc, out_dtype=tl.int32)
            ap = tl.advance(ap, (0, BK))
            bp = tl.advance(bp, (BK, 0))
        rows = row_start + tl.arange(0, BM)
        columns = col_start + tl.arange(0, BN)
        partial = PARTIAL + (part * BATCH + batch) * M * N
        tl.store(
            partial + rows[:, None] * N + columns[None, :],
            acc,
            (rows[:, None] < M) & (columns[None, :] < N),
        )


@libentry()
@triton.jit
def bmm_split_k_finish(
    PARTIAL,
    SA,
    SB,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    BATCH: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0) * BM
    column = tl.program_id(1) * BN
    batch = tl.program_id(2).to(tl.int64)
    acc = tl.zeros((BM, BN), tl.int32)
    for part in range(SPLIT_K):
        ptr = tl.make_block_ptr(
            PARTIAL + (part * BATCH + batch) * M * N,
            (M, N),
            (N, 1),
            (row, column),
            (BM, BN),
            (1, 0),
        )
        acc += tle_async.load(
            ptr, boundary_check=(0, 1), padding_option="zero", is_async=True
        )
    rows = row + tl.arange(0, BM)
    columns = column + tl.arange(0, BN)
    scale_a = tl.load(SA + batch * M + rows, rows < M, other=0)
    scale_b = tl.load(SB + batch * N + columns, columns < N, other=0)
    value = acc.to(tl.float32) * scale_a[:, None] * scale_b[None, :]
    tl.store(
        OUT + batch * M * N + rows[:, None] * N + columns[None, :],
        value,
        (rows[:, None] < M) & (columns[None, :] < N),
    )


def bmm_w8a8_int8(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute channel-scaled INT8 A[B,M,K] @ W[B,K,N] on Ascend."""
    if activation.ndim != 3 or weight.ndim != 3:
        raise ValueError("expected activation[B,M,K] and weight[B,K,N]")
    batch, rows, reduction = activation.shape
    columns = weight.shape[2]
    if weight.shape[:2] != (batch, reduction):
        raise ValueError("batch and reduction dimensions must match")
    if activation.dtype != torch.int8 or weight.dtype != torch.int8:
        raise TypeError("activation and weight must be signed INT8 tensors")
    if activation.device.type != "npu" or any(
        tensor.device != activation.device
        for tensor in (weight, activation_scale, weight_scale)
    ):
        raise ValueError("inputs and scales must be on the same Ascend NPU")
    if not activation.is_contiguous() or not weight.transpose(1, 2).is_contiguous():
        raise ValueError("expected contiguous activation and column-major weight")
    if activation_scale.shape not in ((batch, rows), (batch, rows, 1)):
        raise ValueError("activation_scale must have shape [B,M] or [B,M,1]")
    if weight_scale.shape not in ((batch, columns), (batch, 1, columns)):
        raise ValueError("weight_scale must have shape [B,N] or [B,1,N]")
    if any(
        scale.dtype != torch.float32 or not scale.is_contiguous()
        for scale in (activation_scale, weight_scale)
    ):
        raise TypeError("scales must be contiguous FP32 tensors")
    if any(scale.requires_grad for scale in (activation_scale, weight_scale)):
        raise NotImplementedError("bmm_w8a8_int8 is forward-only")
    if reduction >= 2**31:
        raise ValueError("reduction must be smaller than 2**31")
    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("output dtype must be bfloat16, float16 or float32")
    if out is None:
        out = torch.empty(
            (batch, rows, columns), dtype=out_dtype, device=activation.device
        )
    elif (
        out.shape != (batch, rows, columns)
        or out.dtype != out_dtype
        or out.device != activation.device
        or not out.is_contiguous()
        or out.requires_grad
    ):
        raise ValueError(
            "out must have matching shape, dtype and device, be contiguous "
            "and not require gradients"
        )
    if out.numel() and any(
        out.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
        for tensor in (activation, weight, activation_scale, weight_scale)
    ):
        raise ValueError("out must not share storage with an input")
    if out.numel() == 0:
        return out
    if reduction == 0 or reduction > 65536:
        for index in range(batch):
            mm_w8a8_int8_out(
                activation[index],
                weight[index],
                activation_scale[index],
                weight_scale[index],
                out=out[index],
            )
        return out
    with torch_device_fn.device(activation.device):
        if rows in (1, 2) and columns == 64 and reduction == 4096:
            block_n = 4 if rows == 1 else 8
            bmm_int8_vector_loop[(rows * triton.cdiv(columns, block_n), batch)](
                activation,
                weight,
                activation_scale,
                weight_scale,
                out,
                rows,
                columns,
                reduction,
                block_n,
                num_warps=1,
                multibuffer=False,
                enable_fp_fusion=False,
            )
            return out
        if (
            rows in (1, 2)
            and columns in (1, 4)
            and 1024 <= reduction <= 16384
            and reduction % 1024 == 0
        ):
            bmm_int8_vector_full[(rows * columns, batch)](
                activation,
                weight,
                activation_scale,
                weight_scale,
                out,
                rows,
                columns,
                reduction,
                triton.next_power_of_2(reduction),
                num_warps=1,
                multibuffer=False,
                enable_fp_fusion=False,
            )
            return out
        if columns <= 256:
            block_m = 16 if rows <= 128 else 128
            block_n = 32 if rows <= 128 else (64 if columns < 256 else 256)
            block_k = min(
                1024 if rows <= 128 else 512,
                max(32, triton.next_power_of_2(reduction)),
            )
        else:
            if rows <= 16:
                block_m = 16
            elif rows <= 32:
                block_m = 32
            elif rows <= 256:
                block_m = 64
            else:
                block_m = 128
            block_n = 64 if rows <= 64 else (128 if rows <= 256 else 256)
            if rows <= 64:
                max_block_k = 2048
            else:
                max_block_k = 1024 if rows <= 256 else 512
            block_k = min(max_block_k, max(32, triton.next_power_of_2(reduction)))
        if columns != 1024 or reduction != 4096:
            split_k = 0
        elif batch == 1 and (rows <= 256 or 384 < rows < 512):
            # A separate epilogue avoids per-tile Cube/Vector synchronization.
            if rows <= 32:
                block_m = 16 if rows <= 16 else 32
                block_n, block_k, split_k = 128, 1024, 2
            elif rows <= 64:
                block_m, block_n, block_k, split_k = 64, 128, 1024, 2
            elif rows <= 128:
                block_m, block_n, block_k, split_k = 128, 256, 512, 4
            elif rows <= 256:
                block_m, block_n, block_k = 64, 128, 1024
                split_k = 2 if rows <= 192 else 1
            else:
                block_m, block_n, block_k, split_k = 128, 256, 512, 2
        elif 128 < rows <= 224 or 256 < rows <= 384:
            split_k = 4
        else:
            split_k = 0
        if split_k:
            tiles = triton.cdiv(rows, block_m) * triton.cdiv(columns, block_n)
            cores = min(20, tiles * split_k)
            partials = torch.empty(
                (split_k, batch, rows, columns),
                dtype=torch.int32,
                device=activation.device,
            )
            bmm_split_k_kernel[(cores, batch)](
                activation,
                weight,
                partials,
                rows,
                columns,
                reduction,
                batch,
                block_m,
                block_n,
                block_k,
                cores,
                split_k,
                num_warps=1,
                num_stages=2,
                optimize_dynamic_offset=True,
                unit_flag=False,
                limit_auto_multi_buffer_of_local_buffer="no-l0c",
                set_workspace_multibuffer=1,
            )
            bmm_split_k_finish[
                (triton.cdiv(rows, 32), triton.cdiv(columns, 256), batch)
            ](
                partials,
                activation_scale,
                weight_scale,
                out,
                rows,
                columns,
                batch,
                split_k,
                32,
                256,
                num_warps=1,
                multibuffer=False,
                enable_fp_fusion=False,
            )
        else:
            cores = min(20, triton.cdiv(rows, block_m) * triton.cdiv(columns, block_n))
            bmm_direct_scaled_mm[(cores, batch)](
                activation,
                weight,
                activation_scale,
                weight_scale,
                out,
                rows,
                columns,
                reduction,
                rows,
                block_m,
                block_n,
                block_k,
                cores,
                num_warps=1,
                num_stages=2,
                optimize_dynamic_offset=True,
                unit_flag=False,
                limit_auto_multi_buffer_of_local_buffer="no-l0c",
                set_workspace_multibuffer=1,
            )
    return out
