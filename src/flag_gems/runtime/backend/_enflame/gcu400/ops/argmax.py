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

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

from ..utils.config_utils import MAX_GRID_DIM

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def argmax_kernel_global_1(
    X,
    Mid_val,
    Mid_idx,
    M,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)

    dtype = X.type.element_ty
    min_value = get_dtype_min(dtype)

    if tl.constexpr(dtype == tl.float16) or tl.constexpr(dtype == tl.bfloat16):
        acc_dtype = tl.float32
    else:
        acc_dtype = dtype

    best_val = get_dtype_min(acc_dtype)
    best_idx = tl.zeros([], dtype=tl.int64)

    for tile_start in tl.range(
        pid * BLOCK_SIZE, M, num_prog * BLOCK_SIZE, num_stages=num_stages
    ):
        offset = tile_start + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        x = tl.load(X + offset, mask=mask, other=min_value).to(acc_dtype)
        local_max, local_argmax = tl.max(
            x, axis=0, return_indices=True, return_indices_tie_break_left=True
        )
        update = local_max > best_val
        best_val = tl.where(update, local_max, best_val)
        best_idx = tl.where(update, (tile_start + local_argmax).to(tl.int64), best_idx)

    tl.store(Mid_val + pid, best_val.to(dtype))
    tl.store(Mid_idx + pid, best_idx)


@libentry()
@triton.jit
def argmax_kernel_global_2(
    Mid_val,
    Mid_idx,
    Out,
    MID_SIZE,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    min_value = get_dtype_min(Mid_val.type.element_ty)
    vals = tl.load(Mid_val + offset, mask=mask, other=min_value)
    idx_val = tl.argmax(vals, axis=0)
    out_idx = tl.load(Mid_idx + idx_val)
    tl.store(Out, out_idx)


@libentry()
@triton.jit
def argmax_kernel_inner_2d(
    X,
    Out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (M + BLOCK_M - 1) // BLOCK_M

    dtype = X.type.element_ty
    min_value = get_dtype_min(dtype)

    if tl.constexpr(dtype == tl.float16) or tl.constexpr(dtype == tl.bfloat16):
        acc_dtype = tl.float32
    else:
        acc_dtype = dtype

    for tile_id in tl.range(pid, num_tile, step, num_stages=num_stages):
        m_offset = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = m_offset < M

        max_vals = tl.full([BLOCK_M], dtype=acc_dtype, value=get_dtype_min(acc_dtype))
        max_idxs = tl.zeros([BLOCK_M], dtype=tl.int32)

        for col_off in tl.range(0, N, BLOCK_N):
            n_offset = col_off + tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N + n_offset[None, :]
            mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
            x = tl.load(X + offset, mask=mask, other=min_value).to(acc_dtype)

            local_max, local_argmax = tl.max(
                x, axis=1, return_indices=True, return_indices_tie_break_left=True
            )

            update = local_max > max_vals
            max_vals = tl.where(update, local_max, max_vals)
            max_idxs = tl.where(update, col_off + local_argmax, max_idxs)

        tl.store(Out + m_offset, max_idxs, row_mask)


@libentry()
@triton.jit
def argmax_kernel_inner_1d(
    X,
    Out,
    M,
    N,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)

    dtype = X.type.element_ty
    min_value = get_dtype_min(dtype)

    if tl.constexpr(dtype == tl.float16) or tl.constexpr(dtype == tl.bfloat16):
        acc_dtype = tl.float32
    else:
        acc_dtype = dtype

    for row_id in tl.range(pid, M, step, num_stages=num_stages):
        max_val = get_dtype_min(acc_dtype)
        max_idx = tl.zeros([], dtype=tl.int32)

        base = row_id * N
        for col_off in tl.range(0, N, BLOCK_N):
            n_offset = col_off + tl.arange(0, BLOCK_N)
            mask = n_offset < N
            x = tl.load(X + base + n_offset, mask=mask, other=min_value).to(acc_dtype)

            local_max, local_argmax = tl.max(
                x, axis=0, return_indices=True, return_indices_tie_break_left=True
            )

            update = local_max > max_val
            max_val = tl.where(update, local_max, max_val)
            max_idx = tl.where(update, col_off + local_argmax, max_idx)

        tl.store(Out + row_id, max_idx)


@libentry()
@triton.jit
def argmax_kernel_non_inner(
    X,
    Out,
    M,
    N,
    K,
    num_k_tiles,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    total_work = M * num_k_tiles

    dtype = X.type.element_ty
    if tl.constexpr(dtype == tl.float16) or tl.constexpr(dtype == tl.bfloat16):
        acc_dtype = tl.float32
    else:
        acc_dtype = dtype
    min_value = get_dtype_min(acc_dtype)
    orig_min = get_dtype_min(dtype)

    for work_id in tl.range(pid, total_work, num_prog, num_stages=num_stages):
        m = work_id // num_k_tiles
        k_tile = work_id % num_k_tiles
        k_offset = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offset < K

        max_vals = tl.full([BLOCK_K], dtype=acc_dtype, value=min_value)
        max_idxs = tl.zeros([BLOCK_K], dtype=tl.int32)

        base = m * N * K
        for n_off in tl.range(0, N, BLOCK_N):
            n_idx = n_off + tl.arange(0, BLOCK_N)
            offset = base + n_idx[:, None] * K + k_offset[None, :]
            mask = (n_idx[:, None] < N) & k_mask[None, :]
            vals = tl.load(X + offset, mask=mask, other=orig_min).to(acc_dtype)

            local_max, local_argmax = tl.max(
                vals, axis=0, return_indices=True, return_indices_tie_break_left=True
            )

            update = local_max > max_vals
            max_vals = tl.where(update, local_max, max_vals)
            max_idxs = tl.where(update, n_off + local_argmax, max_idxs)

        out_offset = m * K + k_offset
        tl.store(Out + out_offset, max_idxs, k_mask)


def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_ENFLAME ARGMAX")

    if dim is None:
        M = inp.numel()
        if dtype is None:
            dtype = inp.dtype

        block_size = 32 * 64
        if M < 24 * 16 * 1024:
            block_size = 16 * 1024
        elif M >= 24 * 32 * 1024 and M < 24 * 64 * 1024:
            block_size = 32 * 1024
        elif M >= 24 * 64 * 1024:
            block_size = 64 * 1024

        num_tiles = triton.cdiv(M, block_size)
        grid_size = min(num_tiles, 24)

        num_stages = 1
        tiles_per_prog = triton.cdiv(num_tiles, grid_size)
        if tiles_per_prog > 4:
            num_stages = 3

        block_mid = triton.next_power_of_2(grid_size)

        mid_value = torch.empty([grid_size], dtype=dtype, device=inp.device)
        mid_index = torch.empty([grid_size], dtype=torch.int64, device=inp.device)

        if keepdim:
            shape = [1] * inp.ndim
            out = torch.empty(shape, dtype=torch.int64, device=inp.device)
        else:
            out = torch.empty([], dtype=torch.int64, device=inp.device)

        with torch_device_fn.device(inp.device):
            argmax_kernel_global_1[(grid_size, 1, 1)](
                inp,
                mid_value,
                mid_index,
                M,
                BLOCK_SIZE=block_size,
                num_stages=num_stages,
                num_warps=1,
            )
            argmax_kernel_global_2[(1, 1, 1)](
                mid_value,
                mid_index,
                out,
                grid_size,
                BLOCK_MID=block_mid,
                num_warps=1,
            )
        return out.to(torch.int32).to(torch.int64)

    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        shape = inp.shape
        dim = dim % inp.ndim

        if inp.numel() == 0:
            out_shape = list(shape)
            if keepdim:
                out_shape[dim] = 1
            else:
                del out_shape[dim]
            return torch.zeros(out_shape, dtype=torch.int32, device=inp.device).to(
                torch.int64
            )

        N = shape[dim]
        M = math.prod(shape[:dim])
        K = inp.numel() // M // N

        inp = inp.contiguous()

        shape_list = list(shape)
        shape_list[dim] = 1
        out_index = torch.empty(shape_list, dtype=torch.int32, device=inp.device)

        if K == 1:
            if N <= 1024:
                BLOCK_N = min(triton.next_power_of_2(N), 4096)
                BLOCK_M = max(1, min(128, 32768 // BLOCK_N))
                grid_m = min(triton.cdiv(M, BLOCK_M), MAX_GRID_DIM)
                num_stages = 1
                with torch_device_fn.device(inp.device):
                    argmax_kernel_inner_2d[(grid_m,)](
                        inp,
                        out_index,
                        M,
                        N,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        num_stages=num_stages,
                        num_warps=1,
                    )
            else:
                BLOCK_N = min(triton.next_power_of_2(N), 4096)
                grid_m = min(M, MAX_GRID_DIM)
                num_stages = 1
                if N > BLOCK_N:
                    num_stages = 3
                with torch_device_fn.device(inp.device):
                    argmax_kernel_inner_1d[(grid_m,)](
                        inp,
                        out_index,
                        M,
                        N,
                        BLOCK_N=BLOCK_N,
                        num_stages=num_stages,
                        num_warps=1,
                    )
        else:
            BLOCK_K = min(triton.next_power_of_2(K), 128)
            BLOCK_N = min(triton.next_power_of_2(N), max(4, 32768 // BLOCK_K))

            num_k_tiles = triton.cdiv(K, BLOCK_K)
            total_work = M * num_k_tiles
            grid_size = min(total_work, MAX_GRID_DIM)

            num_stages = 1
            if N > BLOCK_N:
                num_stages = 3

            with torch_device_fn.device(inp.device):
                argmax_kernel_non_inner[(grid_size,)](
                    inp,
                    out_index,
                    M,
                    N,
                    K,
                    num_k_tiles,
                    BLOCK_K=BLOCK_K,
                    BLOCK_N=BLOCK_N,
                    num_stages=num_stages,
                    num_warps=1,
                )

        if not keepdim:
            out_index = torch.squeeze(out_index, dim)
        return out_index.to(torch.int32).to(torch.int64)
