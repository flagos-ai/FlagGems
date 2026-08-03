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
from flag_gems.utils.limits import get_dtype_max

logger = logging.getLogger(__name__)
torch_dtype_to_tl_dtype_and_max_value = {
    torch.int16: (tl.int16, torch.iinfo(torch.int16).max),
    torch.int32: (tl.int32, torch.iinfo(torch.int32).max),
    torch.float16: (tl.float16, torch.finfo(torch.float16).max),
    torch.float32: (tl.float32, torch.finfo(torch.float32).max),
    torch.bfloat16: (tl.float32, torch.finfo(torch.float32).max),
}

# N above this width cannot be reduced (with return_indices) in a single XPU
# tile: both the single-load kernel and the loop-accumulator kernel fail to
# compile ("out of resource: uni_sram" / "2D Shape[-1] <= core_num *
# buffer_size Limit"). Such N is handled by the two-stage reduction below.
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
def argmin_kernel_1(
    inp,
    mid_value,
    mid_index,
    M,
    BLOCK_SIZE: tl.constexpr,
    dtype_max_value: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=dtype_max_value)
    min_val, min_index = tl.min(inp_val, axis=0, return_indices=True)
    min_index = min_index + pid * BLOCK_SIZE
    mid_value_ptr = mid_value + pid
    min_index_ptr = mid_index + pid
    tl.store(mid_value_ptr, min_val)
    tl.store(min_index_ptr, min_index)


@libentry()
@triton.jit
def argmin_kernel_2(
    mid_value,
    mid_index,
    out,
    mid_size,
    BLOCK_MID: tl.constexpr,
    dtype_max_value: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid_value + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=dtype_max_value)
    index_val = tl.argmin(mid_val, axis=0)
    mid_index_ptrs = mid_index + index_val
    out_val = tl.load(mid_index_ptrs)
    tl.store(out, out_val)


@libentry()
@triton.jit
def argmin_stage1(
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
    # Stage 1 of the large-N (N > MAX_TILE_N) path. Each program reduces one
    # BLOCK_N-wide chunk of a BLOCK_M row block and emits the chunk-local min
    # value plus its *global* argmin index. Output is [M, NUM_CHUNKS, K].
    pid_m = ext.program_id(0)
    pid_c = ext.program_id(1)
    pid_k = ext.program_id(2)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    chunk_start = pid_c * BLOCK_N
    n_offset = chunk_start + tl.arange(0, BLOCK_N)
    dtype = inp.type.element_ty
    max_value = get_dtype_max(dtype)
    offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    vals = tl.load(inp + offset, mask=mask, other=max_value)
    lmin, largmin = tl.min(vals, axis=1, return_indices=True)
    gidx = chunk_start + largmin
    part_offset = m_offset * NUM_CHUNKS * K + pid_c * K + pid_k
    pmask = m_offset < M
    tl.store(part_val + part_offset, lmin, mask=pmask)
    tl.store(part_idx + part_offset, gidx, mask=pmask)


@libentry()
@triton.jit
def argmin_stage2(
    part_val,
    part_idx,
    out_index,
    M,
    K: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Stage 2: reduce the NUM_CHUNKS per-row partial mins, then gather the
    # global argmin index of the winning chunk. The reduce keeps the earliest
    # chunk on ties (default first-occurrence), matching torch semantics.
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_offset = tl.arange(0, BLOCK_C)
    dtype = part_val.type.element_ty
    max_value = get_dtype_max(dtype)
    offset = m_offset[:, None] * NUM_CHUNKS * K + c_offset[None, :] * K + pid_k
    mask = m_offset[:, None] < M and c_offset[None, :] < NUM_CHUNKS
    vals = tl.load(part_val + offset, mask=mask, other=max_value)
    _, best_c = tl.min(vals, axis=1, return_indices=True)
    gather = m_offset * NUM_CHUNKS * K + best_c * K + pid_k
    pmask = m_offset < M
    res = tl.load(part_idx + gather, mask=pmask)
    tl.store(out_index + m_offset * K + pid_k, res, mask=pmask)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("argmin"))
@triton.jit
def argmin_kernel_small_n(
    inp,
    out_index,
    M,
    N,
    K,
    tl_dtype: tl.constexpr,
    dtype_max_value: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Single-tile path (N <= MAX_TILE_N so a single load covers all of N),
    # preserved verbatim from the original kernel (runtime N/K, no tie-break
    # flag). A constexpr-N/K variant compiles ~13x faster for large contiguous
    # tiles but retriggers an XPU 2D-reduce layout-inference failure for narrow
    # tiles (small BLOCK_N) and for int16 -- so keep the proven original form.
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
    inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    _, result_index = tl.min(inp_vals, axis=1, return_indices=True)

    out_index_ptrs = out_index + offset_index

    tl.store(out_index_ptrs, result_index, mask=mask1)


def argmin(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN ARGMIN")
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

        tl_dtype, dtype_max_value = torch_dtype_to_tl_dtype_and_max_value[inp.dtype]
        with torch_device_fn.device(inp.device):
            argmin_kernel_1[(mid_size, 1, 1)](
                inp,
                mid_value,
                mid_index,
                M,
                block_size,
                dtype_max_value,
            )
            argmin_kernel_2[(1, 1, 1)](
                mid_value,
                mid_index,
                out,
                mid_size,
                block_mid,
                dtype_max_value,
            )
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

        # argmin along a size-1 dim is trivially index 0. A return_indices
        # reduce over a size-1 axis fails XPU layout inference for BLOCK_M > 1,
        # so skip the kernel entirely.
        if N == 1:
            out_index.zero_()
            return out_index

        tl_dtype, dtype_max_value = torch_dtype_to_tl_dtype_and_max_value[inp.dtype]

        # Routing:
        #   * N > MAX_TILE_N : the reduce axis cannot fit one XPU tile at all,
        #     so the two-stage path is mandatory.
        #   * K == 1 and N >= TWO_STAGE_MIN_N : contiguous inner reduce. The
        #     original single-tile kernel uses *runtime* N/K, so the compiler
        #     cannot prove the inner stride is 1 -> discrete/strided access that
        #     is catastrophically slow as N grows (e.g. [4096,4096] dim=1 ran at
        #     ~14ms). The two-stage kernels take N/K as constexpr, so the inner
        #     dim is a provable stride-1 block DMA (~13x faster). Its BLOCK_N is
        #     a fixed wide tile (STAGE_BLOCK_N), which also sidesteps the XPU
        #     narrow-tile / int16 2D-reduce-with-return_indices layout bug that
        #     a constexpr single-tile kernel would hit for small N.
        #   * otherwise (small N, or K > 1 with N <= MAX_TILE_N) : the proven
        #     original single-tile kernel. For K > 1 the reduce dim is strided
        #     anyway (elements K apart), so the two-stage constexpr trick gives
        #     no contiguity win and only adds launch/alloc overhead; small N is
        #     already launch-bound, not access-bound.
        use_two_stage = N > MAX_TILE_N or (K == 1 and N >= TWO_STAGE_MIN_N)
        if not use_two_stage:
            # Whole reduce axis fits a single tile.
            grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), K)  # noqa: E731
            with torch_device_fn.device(inp.device):
                argmin_kernel_small_n[grid](
                    inp,
                    out_index,
                    M,
                    N,
                    K,
                    tl_dtype,
                    dtype_max_value,
                )
            return out_index

        # Two-stage per-row reduction. Stage 1 reduces BLOCK_N-wide chunks to
        # per-chunk (min, global-argmin); stage 2 reduces the per-chunk mins and
        # gathers the winning global index.
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
            argmin_stage1[grid1](
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
            # the final global argmin into part_idx (layout matches out_index),
            # so skip stage 2 entirely.
            if num_chunks == 1:
                out_index.view(-1).copy_(part_idx)
                return out_index
            block_c = triton.next_power_of_2(num_chunks)
            grid2 = (triton.cdiv(M, block_m), K)
            argmin_stage2[grid2](
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
