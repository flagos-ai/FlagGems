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
import os

import torch
import triton
import triton.language as tl

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
    max_index = tl.full([1], 0, dtype=tl.int32)
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
            update, (tile_id * BLOCK_SIZE + tile_argmax).to(tl.int32), max_index
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
@triton.jit
def argmax_kernel_non_inner(
    inp,
    out_index,
    M,
    N,
    K,
    TILE_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # One (m, k) tile per CTA, scanning in blocks of BLOCK_N rows along the N direction.
    #
    # Key point: the reduction dimension N has stride exactly K in memory, so "BLOCK_N adjacent n × all
    # K" is a fully contiguous BLOCK_N*K chunk. Using it for a 2D load means each transfer moves a contiguous
    # BLOCK_N*K*itemsize bytes (padded to 16K elements = 32~64KB on this machine), allowing DTE to form large
    # bursts. The old row-by-row load (TILE_K elements each) achieved only
    # 142us / 117 GB/s on (64,512,512), vs torch 63.9us / 263 GB/s.
    #
    # Do not use a 2D return_indices reduction (that form misaligns on GCU400: when K needs multiple
    # k tiles, out[k+1] gets the result of out[k] and odd columns are all 0; with grid-stride the error rate
    # approaches 100%). Here the final cross-n step uses two index-free reductions: tl.max for the column maximum,
    # tl.min for the smallest n among ties, consistent with torch's first-maximum semantics.
    #
    # Use int32 for indices: on GCU400, a "computed int64 vector" (result of tl.where / tl.full)
    # is stored with int32 stride, so the kernel emits only int32 and the host side widens it to int64.
    # Keep element offsets in int32 as well (N*K <= 2^31) to avoid int64 vector arithmetic.
    pid = ext.program_id(0)
    num_k_tile = (K + TILE_K - 1) // TILE_K
    pid_m = pid // num_k_tile
    pid_k = pid % num_k_tile
    k_offset = pid_k * TILE_K + tl.arange(0, TILE_K)
    k_mask = k_offset < K

    if tl.constexpr(inp.dtype.element_ty == tl.float16) | tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty
    min_value = get_dtype_min(cdtype)

    base = inp + pid_m.to(tl.int64) * (N * K)
    max_values = tl.full([TILE_K], min_value, dtype=cdtype)
    argmax_values = tl.zeros([TILE_K], dtype=tl.int32)
    for n0 in tl.range(0, N, BLOCK_N, num_stages=NUM_STAGES):
        n_offset = n0 + tl.arange(0, BLOCK_N)
        mask = (n_offset < N)[:, None] & k_mask[None, :]
        inp_vals = tl.load(
            base + n_offset[:, None] * K + k_offset[None, :], mask=mask, other=min_value
        ).to(cdtype)
        chunk_max = tl.max(inp_vals, 0)
        chunk_index = tl.min(
            tl.where(inp_vals == chunk_max[None, :], n_offset[:, None], N), 0
        )
        update = chunk_max > max_values
        max_values = tl.where(update, chunk_max, max_values)
        argmax_values = tl.where(update, chunk_index, argmax_values)

    tl.store(
        out_index + pid_m.to(tl.int64) * K + k_offset.to(tl.int64),
        argmax_values,
        mask=k_mask,
    )


@libentry()
@triton.jit
def argmax_kernel_inner(
    inp,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr,
    TILE_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # One CTA handles BLOCK_M rows, scanning within each row in TILE_N segments (K == 1: the reduction dimension is
    # contiguous).
    #
    # Why switch from "one CTA per row" to multi-row tiles (batch measurements on this machine, 2026-11):
    #  On (4096,4096)/dim=1 the old form took 307us (104 GB/s, torch 84us), because each CTA
    #  moved only 8KB before wrapping up, with CTA launch/teardown completely dominating memory access; the same code on
    # (64,64) is
    #  64 CTAs each scanning 64 elements, a pure launch floor (7.7us). With BLOCK_M rows, each CTA's
    #  contiguous access is amplified BLOCK_M-fold and the grid shrinks to 1/BLOCK_M, benefiting both shapes.
    #
    # Deliberately avoid a 2D return_indices reduction (that form misaligns on GCU400, see
    # the notes on argmax_kernel_non_inner); the in-row final step is still two "index-free" reductions
    # (tl.max for the maximum, tl.min for the smallest n among ties), consistent with torch's first-maximum semantics.
    pid = ext.program_id(0)
    if tl.constexpr(inp.dtype.element_ty == tl.float16) | tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty
    min_value = get_dtype_min(cdtype)

    row = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row < M
    base = inp + row[:, None].to(tl.int64) * N
    max_values = tl.full([BLOCK_M, TILE_N], min_value, dtype=cdtype)
    argmax_values = tl.zeros([BLOCK_M, TILE_N], dtype=tl.int32)
    for n0 in tl.range(0, N, TILE_N, num_stages=NUM_STAGES):
        n_offset = n0 + tl.arange(0, TILE_N)
        mask = row_mask[:, None] & (n_offset < N)[None, :]
        inp_vals = tl.load(base + n_offset[None, :], mask=mask, other=min_value).to(
            cdtype
        )
        update = inp_vals > max_values
        max_values = tl.where(update, inp_vals, max_values)
        argmax_values = tl.where(update, n_offset[None, :].to(tl.int32), argmax_values)

    row_max = tl.max(max_values, 1)
    first_index = tl.min(tl.where(max_values == row_max[:, None], argmax_values, N), 1)
    tl.store(out_index + row, first_index.to(tl.int32), mask=row_mask)


@libentry()
@triton.jit
def argmax_kernel_inner_row(
    inp,
    out_index,
    M,
    N,
    TILE_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # A true 1D version with one CTA per row, used only when M is very small (M <= 64).
    #
    # Why not reuse the 2D kernel above with BLOCK_M=1: at BLOCK_M=1 the tile is [1, TILE_N],
    # and codegen still follows a 2D layout; measured 19.9us on (64,64), 44% slower than the true 1D's 13.8us
    # (torch 5.6us). Small shapes are already at the launch floor, so those 6us eat 40% of the speedup.
    pid = ext.program_id(0)
    if tl.constexpr(inp.dtype.element_ty == tl.float16) | tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty
    min_value = get_dtype_min(cdtype)

    base = inp + pid.to(tl.int64) * N
    max_values = tl.full([TILE_N], min_value, dtype=cdtype)
    argmax_values = tl.zeros([TILE_N], dtype=tl.int32)
    for n0 in tl.range(0, N, TILE_N, num_stages=NUM_STAGES):
        n_offset = n0 + tl.arange(0, TILE_N)
        inp_vals = tl.load(base + n_offset, mask=n_offset < N, other=min_value).to(
            cdtype
        )
        update = inp_vals > max_values
        max_values = tl.where(update, inp_vals, max_values)
        argmax_values = tl.where(update, n_offset.to(tl.int32), argmax_values)

    row_max = tl.max(max_values, 0)
    first_index = tl.min(tl.where(max_values == row_max, argmax_values, N), 0)
    tl.store(out_index + pid, first_index.to(tl.int32))


def _non_inner_config(k, n):
    """(TILE_K, BLOCK_N, NUM_STAGES, num_warps) for the block-row non-contiguous path.

    Two constraints jointly decide the values (local GCU400 sweep, 2026-11):
      * per-CTA transfer of [BLOCK_N, TILE_K] is locked at 32768 elements -- one step up
        (512x128 / 512x256) hits the dsm cap (Required 1.05MB > 917KB).
      * at the same 32768 elements, halving TILE_K raises num_k_tile from 1 to 2, i.e. doubles
        the grid: on (64,512,512) TILE_K=512/BN=64 is 74.2us while TILE_K=256/BN=128 is 59.9us
        (1.06x); on (1024,1024,512) 1506us -> 1392us (1.27x).
    num_stages 1/2/3 and num_warps 1/2 measured no difference.
    """
    override = os.environ.get("FG_ARGMAX_NON_INNER_TILE")
    if override:
        tile_k, block_n, num_stages, num_warps = (
            int(value) for value in override.split(",")
        )
        return tile_k, block_n, num_stages, num_warps
    tile_k = min(2048, max(64, triton.next_power_of_2(k) // 2))
    block_n = min(32768 // tile_k, triton.next_power_of_2(n))
    block_n = max(1, block_n)
    return tile_k, block_n, 2, 2


def _inner_config(n, m):
    """(BLOCK_M, TILE_N, NUM_STAGES, num_warps) for the K == 1 contiguous path.

    Target 16384 elements per CTA (64KB of values plus 64KB of indices under fp32, still inside
    the register budget); the segment width is half the reduction length (clamped to [64, 512])
    and the remaining freedom goes to BLOCK_M. Local sweep:
      n=64   -> BM=64/TN=64    13.8us  (M=64 is a single CTA, launch floor)
      n=256  -> BM=128/TN=128  16.6us  (torch 48.5us, 2.92x)
      n=4096 -> BM=32/TN=512   74.4us  (torch 83.6us, 1.12x; the old one-CTA-per-row took 307us)
    num_warps=1 and 2 show no difference (one warp suffices for wide tiles); num_stages is always
    1 -- fp16/bf16 2-D loads misplace 47% at stages>=4 with warps=4, while stages=1 is all correct.
    """
    override = os.environ.get("FG_ARGMAX_INNER_TILE")
    if override:
        block_m, tile_n, num_stages, num_warps = (
            int(value) for value in override.split(",")
        )
        return block_m, tile_n, num_stages, num_warps
    n_pow2 = triton.next_power_of_2(n)
    if m <= 64:
        # When M is too small (e.g. (64,64)), multi-row tiles yield only 1~4 CTAs and a single warp cannot finish
        # scanning
        # so it is slower instead (measured 20.6us vs 14.9us). Fall back to "one CTA per row + 4 warps",
        # so grid == M fills all 48 CTA slots; the gain comes from CTA-level parallelism rather than tile width.
        return 1, min(4096, max(64, n_pow2)), 2, 4
    tile_n = min(512, max(64, n_pow2 // 2))
    block_m = min(256, max(1, 16384 // tile_n))
    block_m = min(block_m, max(1, triton.next_power_of_2(m) // 4))
    return block_m, tile_n, 1, 1


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
        # int32 intermediate indices (see notes on argmax_kernel_non_inner), widened to int64 at the end.
        mid_index = torch.empty((grid_size,), dtype=torch.int32, device=inp.device)
        if keepdim:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=torch.int32, device=inp.device)
        else:
            out = torch.empty([], dtype=torch.int32, device=inp.device)

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
        return out.to(torch.int64)
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
        # Both kernels write only int32 indices (see in-kernel notes: on GCU400 a "computed
        # int64 vector" is stored with int32 stride); the host side widens all of them to int64.
        out_index = torch.empty(shape_list, dtype=torch.int32, device=inp.device)

        with torch_device_fn.device(inp.device):
            if K > 1:
                tile_k, block_n, num_stages_non_inner, num_warps_non_inner = (
                    _non_inner_config(K, N)
                )
                num_k_tile = triton.cdiv(K, tile_k)
                # One (m, k) tile per CTA: this is both a correctness requirement (see kernel comments)
                # and also avoids redundant scans under grid-stride.
                grid = (M * num_k_tile, 1, 1)
                argmax_kernel_non_inner[grid](
                    inp,
                    out_index,
                    M,
                    N,
                    K,
                    tile_k,
                    block_n,
                    num_stages_non_inner,
                    num_warps=num_warps_non_inner,
                )
            else:
                block_m, tile_n, num_stages_inner, num_warps_inner = _inner_config(N, M)
                if block_m == 1:
                    # M very small: use the true 1D kernel (see argmax_kernel_inner_row comments)
                    argmax_kernel_inner_row[(M, 1, 1)](
                        inp,
                        out_index,
                        M,
                        N,
                        tile_n,
                        num_stages_inner,
                        num_warps=num_warps_inner,
                    )
                else:
                    argmax_kernel_inner[(triton.cdiv(M, block_m), 1, 1)](
                        inp,
                        out_index,
                        M,
                        N,
                        block_m,
                        tile_n,
                        num_stages_inner,
                        num_warps=num_warps_inner,
                    )
        out_index = out_index.to(torch.int64)
        if not keepdim:
            out_index = torch.squeeze(out_index, dim)
        return out_index
