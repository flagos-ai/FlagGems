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

# Sentinel for "no index chosen yet". Must be a tl.constexpr: a plain int global
# raises NameError inside a @triton.jit function.
INT32_MAX = tl.constexpr(2147483647)
_INT32_MAX_PY = 2147483647

# The XPU has 12 clusters. The flat (dim=None) reduce is launch/latency bound,
# not bandwidth bound, so one program per cluster with a wide inner loop beats
# one program per tile: a loop-accumulator reaches ~375 GB/s where a grid of
# narrow one-shot tiles manages 39-158 GB/s. Measured best on 1M elements:
# 12 programs x 8192-wide steps (sp 1.47-1.65 vs 0.30-0.35 for the old form).
FLAT_NUM_PROGRAMS = 12
FLAT_BLOCK = 8192
# Widest row tile we will materialise for the fused per-row kernel.
ROW_MAX_BLOCK_N = 8192
# Row tiles are capped so BLOCK_M * BLOCK_N stays within uni_sram.
ROW_MAX_TILE = 131072
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
    MIN_VALUE: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Stage 1 of the flat (dim=None) reduce. One program per cluster walks a
    # CHUNK-wide slice in BLOCK_SIZE steps, keeping a per-lane running max and
    # the offset that produced it. `return_indices` is avoided: it costs ~2.4x
    # over a plain tl.max here, and the explicit epilogue below is exact.
    pid = ext.program_id(0)
    base = pid * CHUNK
    # The accumulator dtype follows the mid_value buffer the caller allocated:
    # bfloat16 input promotes to fp32 (the bf16 accumulator exhausts uni_sram),
    # every other dtype accumulates natively so integer inputs keep full
    # precision. Selecting the dtype here rather than via an `if` on a
    # constexpr flag matters -- that branch form fails to compile on this XPU.
    acc_dtype = mid_value.type.element_ty
    acc = tl.full((BLOCK_SIZE,), MIN_VALUE, dtype=acc_dtype)
    acc_index = tl.zeros((BLOCK_SIZE,), dtype=tl.int32) + INT32_MAX
    for i in range(0, CHUNK, BLOCK_SIZE):
        offset = base + i + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        val = tl.load(inp + offset, mask=mask, other=MIN_VALUE).to(acc_dtype)
        index = tl.where(mask, offset, INT32_MAX).to(tl.int32)
        # Strict > keeps the earliest offset within a lane, since i ascends.
        better = val > acc
        acc = tl.where(better, val, acc)
        acc_index = tl.where(better, index, acc_index)
    max_val = tl.max(acc, axis=0)
    # Earliest offset across lanes that holds the max, matching torch's
    # tie-break.
    max_index = tl.min(tl.where(acc == max_val, acc_index, INT32_MAX), axis=0)
    tl.store(mid_value + pid, max_val)
    tl.store(mid_index + pid, max_index)


@libentry()
@triton.jit
def argmax_kernel_2(
    mid_value,
    mid_index,
    out,
    mid_size,
    MIN_VALUE: tl.constexpr,
    BLOCK_MID: tl.constexpr,
):
    # Stage 2: pick the earliest index among the per-program partials. Masked
    # loads supply the identity, so the caller need not pre-fill the buffers.
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    mid_val = tl.load(mid_value + offset, mask=mask, other=MIN_VALUE)
    mid_idx = tl.load(mid_index + offset, mask=mask, other=INT32_MAX)
    max_val = tl.max(mid_val, axis=0)
    res = tl.min(tl.where(mid_val == max_val, mid_idx, INT32_MAX), axis=0)
    tl.store(out, res.to(tl.int64))


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
@triton.jit
def argmax_row_kernel(
    inp,
    out_index,
    M,
    N: tl.constexpr,
    MIN_VALUE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Contiguous inner reduce (K == 1) whose row fits one tile of width BLOCK_N
    # (>= N). N is constexpr so the whole row is a provable stride-1 block DMA;
    # a runtime N degrades to discrete access and is far slower here.
    pid_m = ext.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offset < M
    n_offset = tl.arange(0, BLOCK_N)
    n_mask = n_offset < N
    vals = tl.load(
        inp + m_offset[:, None] * N + n_offset[None, :],
        mask=m_mask[:, None] & n_mask[None, :],
        other=MIN_VALUE,
    )
    max_val = tl.max(vals, axis=1)
    # min-where instead of return_indices: ~2.4x cheaper and gives torch's
    # earliest-index tie-break exactly.
    index = tl.min(
        tl.where(
            (vals == max_val[:, None]) & n_mask[None, :],
            n_offset[None, :].to(tl.int32),
            INT32_MAX,
        ),
        axis=1,
    )
    tl.store(out_index + m_offset, index.to(tl.int64), mask=m_mask)


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
        inp = inp.contiguous()
        # One program per cluster, each looping over a FLAT_BLOCK-wide window.
        # CHUNK is rounded up to a whole number of blocks so the inner loop
        # bound is constexpr; programs whose slice starts past M fall out via
        # the load mask.
        block_size = FLAT_BLOCK
        mid_size = min(FLAT_NUM_PROGRAMS, triton.cdiv(M, block_size))
        chunk = triton.cdiv(triton.cdiv(M, mid_size), block_size) * block_size
        block_mid = triton.next_power_of_2(mid_size)

        # bfloat16 accumulates in fp32 (a bf16 accumulator exhausts uni_sram);
        # all other dtypes accumulate natively so that integer inputs near the
        # top of their range are not rounded into each other.
        acc_dtype = torch.float32 if inp.dtype is torch.bfloat16 else inp.dtype
        mid_value = torch.empty((mid_size,), dtype=acc_dtype, device=inp.device)
        mid_index = torch.empty((mid_size,), dtype=torch.int32, device=inp.device)
        if keepdim:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=torch.int64, device=inp.device)
        else:
            out = torch.empty([], dtype=torch.int64, device=inp.device)

        if acc_dtype.is_floating_point:
            min_value = float(torch.finfo(acc_dtype).min)
        else:
            min_value = int(torch.iinfo(acc_dtype).min)
        with torch_device_fn.device(inp.device):
            argmax_kernel_1[(mid_size, 1, 1)](
                inp,
                mid_value,
                mid_index,
                M,
                min_value,
                chunk,
                block_size,
            )
            argmax_kernel_2[(1, 1, 1)](
                mid_value, mid_index, out, mid_size, min_value, block_mid
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
        # A contiguous inner reduce whose row fits one tile is handled by the
        # fused row kernel: constexpr N gives stride-1 block DMA, and resolving
        # the index with min-where avoids the ~2.4x `return_indices` penalty.
        # It subsumes both the two-stage path (no partial buffers, no second
        # launch) and the runtime-N single-tile kernel for this shape class.
        block_n = triton.next_power_of_2(N)
        if K == 1 and block_n <= ROW_MAX_BLOCK_N:
            block_m = max(1, min(STAGE_BLOCK_M, ROW_MAX_TILE // block_n))
            grid = (triton.cdiv(M, block_m),)
            with torch_device_fn.device(inp.device):
                argmax_row_kernel[grid](
                    inp,
                    out_index,
                    M,
                    N,
                    dtype_min_value,
                    block_m,
                    block_n,
                )
            return out_index

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
