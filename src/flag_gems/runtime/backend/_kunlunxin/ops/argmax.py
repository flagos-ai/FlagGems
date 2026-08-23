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

torch_dtype_to_tl_dtype_and_min_value = {
    torch.int16: (tl.int16, torch.iinfo(torch.int16).min),
    torch.int32: (tl.int32, torch.iinfo(torch.int32).min),
    torch.float16: (tl.float16, torch.finfo(torch.float16).min),
    torch.float32: (tl.float32, torch.finfo(torch.float32).min),
    torch.bfloat16: (tl.float32, torch.finfo(torch.float32).min),
}
logger = logging.getLogger(__name__)

# N above this width cannot be reduced (with return_indices) in a single XPU
# tile: both the single-load kernel and the loop-accumulator kernel fail to
# compile ("out of resource: uni_sram"). Such N is handled by the two-stage
# reduction below.
MAX_TILE_N = 8192
# Chunk width / row-tile for the two-stage large-N path (measured fastest on XPU).
STAGE_BLOCK_N = 2048
STAGE_BLOCK_M = 32
# For a contiguous inner reduce (K == 1), route N at or above this width to the
# constexpr two-stage path even when it would fit a single tile: the runtime-N/K
# single-tile kernel degrades to discrete access and is far slower here. Below
# this, single-tile launch overhead wins and the discrete penalty is negligible.
TWO_STAGE_MIN_N = 256


@libentry()
@triton.jit
def argmax_kernel_1(
    inp,
    mid_value,
    mid_index,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    min_value = get_dtype_min(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=min_value)
    max_val, max_index = tl.max(inp_val, axis=0, return_indices=True)
    max_index = max_index + pid * BLOCK_SIZE
    mid_value_ptr = mid_value + pid
    max_index_ptr = mid_index + pid
    tl.store(mid_value_ptr, max_val)
    tl.store(max_index_ptr, max_index)


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
@triton.jit
def argmax_stage1(
    inp,
    part_val,
    part_idx,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Stage 1 of the large-N path. Each program reduces one BLOCK_N-wide chunk
    # of a BLOCK_M row block and emits the chunk-local max value plus its
    # *global* argmax index. Output is [M, NUM_CHUNKS, K].
    pid_m = ext.program_id(0)
    pid_c = ext.program_id(1)
    pid_k = ext.program_id(2)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    chunk_start = pid_c * BLOCK_N
    n_offset = chunk_start + tl.arange(0, BLOCK_N)
    dtype = inp.type.element_ty
    min_value = get_dtype_min(dtype)
    offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    vals = tl.load(inp + offset, mask=mask, other=min_value)
    lmax, largmax = tl.max(
        vals, axis=1, return_indices=True, return_indices_tie_break_left=True
    )
    gidx = chunk_start + largmax
    part_offset = m_offset * NUM_CHUNKS * K + pid_c * K + pid_k
    pmask = m_offset < M
    tl.store(part_val + part_offset, lmax, mask=pmask)
    tl.store(part_idx + part_offset, gidx, mask=pmask)


@libentry()
@triton.jit
def argmax_stage2(
    part_val,
    part_idx,
    out_index,
    M,
    K: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Stage 2: reduce the NUM_CHUNKS per-row partial maxes, then gather the
    # global argmax index of the winning chunk. tie_break_left keeps the
    # earliest chunk on ties, matching torch semantics.
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_offset = tl.arange(0, BLOCK_C)
    dtype = part_val.type.element_ty
    min_value = get_dtype_min(dtype)
    offset = m_offset[:, None] * NUM_CHUNKS * K + c_offset[None, :] * K + pid_k
    mask = m_offset[:, None] < M and c_offset[None, :] < NUM_CHUNKS
    vals = tl.load(part_val + offset, mask=mask, other=min_value)
    _, best_c = tl.max(
        vals, axis=1, return_indices=True, return_indices_tie_break_left=True
    )
    gather = m_offset * NUM_CHUNKS * K + best_c * K + pid_k
    pmask = m_offset < M
    res = tl.load(part_idx + gather, mask=pmask)
    tl.store(out_index + m_offset * K + pid_k, res, mask=pmask)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("argmax"))
@triton.jit
def argmax_kernel_small_n(
    inp,
    out_index,
    M,
    N,
    K,
    tl_dtype: tl.constexpr,
    dtype_min_value: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Single-tile path (N <= MAX_TILE_N so a single load covers all of N).
    # Runtime N/K, matching the proven original form.
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    if tl_dtype is tl.int16:
        tl_dtype = tl.int32
    n_offset = tl.arange(0, BLOCK_N)
    offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    offset_index = m_offset * K + pid_k
    mask1 = m_offset < M
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask, other=dtype_min_value)
    _, result_index = tl.max(inp_vals, axis=1, return_indices=True)

    out_index_ptrs = out_index + offset_index

    tl.store(out_index_ptrs, result_index, mask=mask1)


def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN ARGMAX")
    if dim is None:
        M = inp.numel()
        if dtype is None:
            dtype = inp.dtype
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)

        mid_value = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        mid_index = torch.empty((mid_size,), dtype=torch.int64, device=inp.device)
        if keepdim:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=torch.int64, device=inp.device)
        else:
            out = torch.empty([], dtype=torch.int64, device=inp.device)

        with torch_device_fn.device(inp.device):
            argmax_kernel_1[(mid_size, 1, 1)](
                inp,
                mid_value,
                mid_index,
                M,
                block_size,
            )
            argmax_kernel_2[(1, 1, 1)](mid_value, mid_index, out, mid_size, block_mid)
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

        # argmax along a size-1 dim is trivially index 0.
        if N == 1:
            out_index.zero_()
            return out_index

        tl_dtype, dtype_min_value = torch_dtype_to_tl_dtype_and_min_value[inp.dtype]

        # Routing (see argmin.py for the full rationale):
        #   * N > MAX_TILE_N : reduce axis cannot fit one XPU tile, two-stage
        #     is mandatory.
        #   * K == 1 and N >= TWO_STAGE_MIN_N : contiguous inner reduce. The
        #     single-tile kernel uses *runtime* N/K -> discrete access that is
        #     catastrophically slow as N grows. The two-stage kernels take N/K
        #     as constexpr -> provable stride-1 block DMA (~13x faster).
        #   * otherwise (small N, or K > 1 with N <= MAX_TILE_N) : the proven
        #     single-tile kernel.
        use_two_stage = N > MAX_TILE_N or (K == 1 and N >= TWO_STAGE_MIN_N)
        if not use_two_stage:
            grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), K)  # noqa: E731
            with torch_device_fn.device(inp.device):
                argmax_kernel_small_n[grid](
                    inp,
                    out_index,
                    M,
                    N,
                    K,
                    tl_dtype,
                    dtype_min_value,
                )
            return out_index

        # Two-stage per-row reduction.
        block_n = STAGE_BLOCK_N
        block_m = STAGE_BLOCK_M
        num_chunks = triton.cdiv(N, block_n)
        part_val = torch.empty(
            (M * num_chunks * K,), dtype=inp.dtype, device=inp.device
        )
        part_idx = torch.empty(
            (M * num_chunks * K,), dtype=torch.int64, device=inp.device
        )
        grid1 = (triton.cdiv(M, block_m), num_chunks, K)
        with torch_device_fn.device(inp.device):
            argmax_stage1[grid1](
                inp,
                part_val,
                part_idx,
                M,
                N,
                K,
                num_chunks,
                block_m,
                block_n,
            )
            # A single chunk already holds the whole reduce axis: stage 1 wrote
            # the final global argmax into part_idx (layout matches out_index),
            # so skip stage 2 entirely.
            if num_chunks == 1:
                out_index.view(-1).copy_(part_idx)
                return out_index
            block_c = triton.next_power_of_2(num_chunks)
            grid2 = (triton.cdiv(M, block_m), K)
            argmax_stage2[grid2](
                part_val,
                part_idx,
                out_index,
                M,
                K,
                num_chunks,
                block_m,
                block_c,
            )
        return out_index
