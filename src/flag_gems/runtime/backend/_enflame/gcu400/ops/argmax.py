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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def argmax_kernel_flat(
    inp,
    mid_value,
    mid_index,
    M,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # Fixed Grid (GCU400: 48 CTAs) + grid-stride single-pass scan:
    # each CTA strides over the whole input, keeping a local (max, index),
    # and writes only GRID_SIZE partials, reduced by argmax_kernel_2.
    pid = ext.program_id(0)
    if tl.constexpr(inp.dtype.element_ty == tl.float16) | tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty
    min_value = get_dtype_min(cdtype)
    max_value = tl.full([1], min_value, dtype=cdtype)
    max_index = tl.full([1], 0, dtype=tl.int64)
    num_tile = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, GRID_SIZE, num_stages=NUM_STAGES):
        offset = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        inp_val = tl.load(inp + offset, mask=mask, other=min_value)
        tile_max, tile_argmax = tl.max(
            inp_val, axis=0, return_indices=True, return_indices_tie_break_left=True
        )
        update = tile_max > max_value
        max_value = tl.where(update, tile_max, max_value)
        max_index = tl.where(
            update, (tile_id * BLOCK_SIZE + tile_argmax).to(tl.int64), max_index
        )
    tl.store(mid_value + pid + tl.arange(0, 1), max_value.to(mid_value.type.element_ty))
    tl.store(mid_index + pid + tl.arange(0, 1), max_index)


@libentry()
@triton.jit
def argmax_kernel_2(mid_value, mid_index, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid_value + offset
    mask = offset < mid_size
    min_value = get_dtype_min(mid_value.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=min_value)
    index_val = tl.argmax(mid_val, axis=0)
    mid_index_ptrs = mid_index + index_val
    out_val = tl.load(mid_index_ptrs)
    tl.store(out, out_val)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("argmax_non_inner"))
@triton.jit
def argmax_kernel_non_inner(
    inp,
    out_index,
    M,
    N,
    K,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # Fixed Grid (48) + grid-stride: each CTA strides over the (m, k) tile
    # space; every tile is [TILE_N rows x TILE_K cols], reduced along N.
    pid = ext.program_id(0)
    num_k_tile = (K + TILE_K - 1) // TILE_K
    num_tile = M * num_k_tile
    for tile_id in tl.range(pid, num_tile, GRID_SIZE, num_stages=NUM_STAGES):
        pid_m = tile_id // num_k_tile
        pid_k = tile_id % num_k_tile
        k_offset = pid_k * TILE_K + tl.arange(0, TILE_K)

        if tl.constexpr(inp.dtype.element_ty == tl.float16) | tl.constexpr(
            inp.dtype.element_ty == tl.bfloat16
        ):
            cdtype = tl.float32
        else:
            cdtype = inp.dtype.element_ty

        min_value = get_dtype_min(cdtype)

        if ONE_TILE_PER_CTA:
            n_offset = tl.arange(0, TILE_N)
            offset = pid_m * N * K + n_offset[:, None] * K + k_offset
            mask = (k_offset < K) & (n_offset[:, None] < N)
            inp_ptrs = inp + offset
            inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
            local_max, local_argmax = tl.max(
                inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
            )
            offset_index = pid_m * K + k_offset
            out_index_ptrs = out_index + offset_index
            mask1 = k_offset < K
            tl.store(out_index_ptrs, local_argmax, mask=mask1)
        else:
            max_values = tl.full([TILE_K], dtype=cdtype, value=min_value)
            argmax_values = tl.full([TILE_K], dtype=tl.int64, value=0)

            for start_n in range(0, N, TILE_N):
                n_offset = start_n + tl.arange(0, TILE_N)
                offset = pid_m * N * K + n_offset[:, None] * K + k_offset
                mask = (k_offset < K) & (n_offset[:, None] < N)
                inp_ptrs = inp + offset
                inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
                local_max, local_argmax = tl.max(
                    inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
                )
                update = local_max > max_values
                max_values = tl.where(update, local_max, max_values)
                argmax_values = tl.where(update, start_n + local_argmax, argmax_values)
            offset_index = pid_m * K + k_offset
            out_index_ptrs = out_index + offset_index
            mask1 = k_offset < K
            tl.store(out_index_ptrs, argmax_values, mask=mask1)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("argmax_inner"))
@triton.jit
def argmax_kernel_inner(
    inp,
    out_index,
    M,
    N,
    TILE_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # 2D tile (BLOCK_M rows x TILE_N cols) + Fixed Grid (48) + grid-stride:
    # each CTA strides over multiple row blocks; rows are reduced along N in
    # TILE_N chunks. tl.max with tie_break_left matches torch.argmax's
    # first-maximum semantics.
    pid = ext.program_id(0)
    if tl.constexpr(inp.dtype.element_ty == tl.float16) | tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty
    min_value = get_dtype_min(cdtype)
    num_row_tile = (M + BLOCK_M - 1) // BLOCK_M
    for row_tile in tl.range(pid, num_row_tile, GRID_SIZE, num_stages=NUM_STAGES):
        row = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = row < M
        max_values = tl.full([BLOCK_M], min_value, dtype=cdtype)
        argmax_values = tl.full([BLOCK_M], 0, dtype=tl.int64)
        for n_tile in tl.range(0, N, TILE_N, num_stages=NUM_STAGES):
            n_offset = n_tile + tl.arange(0, TILE_N)
            offset = row[:, None] * N + n_offset[None, :]
            mask = row_mask[:, None] & (n_offset[None, :] < N)
            inp_vals = tl.load(inp + offset, mask=mask, other=min_value)
            tile_max, tile_argmax = tl.max(
                inp_vals, 1, return_indices=True, return_indices_tie_break_left=True
            )
            update = tile_max > max_values
            max_values = tl.where(update, tile_max, max_values)
            argmax_values = tl.where(
                update, (n_tile + tile_argmax).to(tl.int64), argmax_values
            )
        tl.store(out_index + row, argmax_values, mask=row_mask)


def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_ENFLAME ARGMAX")
    if dim is None:
        M = inp.numel()
        if dtype is None:
            dtype = inp.dtype
        # Fixed Grid: 48 CTAs on GCU400, grid-stride single-pass scan
        # (replaces the two-stage mid_size-wide grid that reads data twice).
        # Measured optimum: BLOCK=min(32768, npo2(M)) + num_stages=4 + num_warps=2.
        GRID_SIZE = 48
        block_size = min(32768, triton.next_power_of_2(M))
        num_stages = 4
        num_warps = 2
        num_tile = triton.cdiv(M, block_size)
        grid_size = min(GRID_SIZE, num_tile)
        block_mid = triton.next_power_of_2(grid_size)

        mid_value = torch.empty((grid_size,), dtype=dtype, device=inp.device)
        mid_index = torch.empty((grid_size,), dtype=torch.int64, device=inp.device)
        if keepdim:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=torch.int64, device=inp.device)
        else:
            out = torch.empty([], dtype=torch.int64, device=inp.device)

        with torch_device_fn.device(inp.device):
            argmax_kernel_flat[(grid_size, 1, 1)](
                inp,
                mid_value,
                mid_index,
                M,
                block_size,
                grid_size,
                num_stages,
                num_warps=num_warps,
            )
            argmax_kernel_2[(1, 1, 1)](mid_value, mid_index, out, grid_size, block_mid)
        return out
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
            return torch.zeros(out_shape, dtype=torch.int64, device=inp.device)
        N = shape[dim]
        M = math.prod(shape[:dim])
        K = inp.numel() // M // N

        inp = inp.contiguous()

        shape_list = list(shape)
        shape_list[dim] = 1
        out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)
        if not keepdim:
            out_index = torch.squeeze(out_index, dim)

        with torch_device_fn.device(inp.device):
            if K > 1:
                grid = lambda meta: (
                    min(48, M * triton.cdiv(K, meta["TILE_K"])),
                    1,
                    1,
                )
                argmax_kernel_non_inner[grid](
                    inp,
                    out_index,
                    M,
                    N,
                    K,
                )
            else:
                grid = lambda meta: (
                    min(48, triton.cdiv(M, meta["BLOCK_M"])),
                    1,
                    1,
                )
                argmax_kernel_inner[grid](
                    inp,
                    out_index,
                    M,
                    N,
                )
        return out_index
