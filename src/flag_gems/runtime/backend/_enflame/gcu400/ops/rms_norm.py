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

from ..utils.config_utils import MAX_GRID_DIM

logger = logging.getLogger(__name__)

MAX_BLOCK_SIZE = 32768
_FORWARD_WARPS = (
    4  # Number of warps for the forward launch, fixed after sweeping shapes
)

# ---- Row-batch (2D [ROWS_PER_CTA, BLOCK] tile) budget --------------------
# Ported from the verified layernorm-family row-batch kernels in this backend (layernorm.py /
# fused/gcu400/skip_layernorm.py, compiler_issues/
# vllm_fused_q_kv_rmsnorm_rowbatch_bcast_miscompile.md and its 225-config validation set):
# - tile element budget: the fp32 live set (x tile + squared products + [1,BLOCK] weights) stays within each
#   program's DSM budget;
# - R >= 32 with a 32768-element tile is a triton_gcu silent-wrong-result region; tile <= 16384 with
#   R <= 32 has been verified correct across 225 (shape, dtype, R, num_warps) combinations.
# Extended region (16384 < tile <= dtype budget): calibrated from measured gcu.dsm_memory_size in compiler output -
# fp16 R=16/B=4096 (tile 65536) dsm=282KB, fp32 R=8/B=4096 (tile
# 32768) dsm=299KB, both < 448KB dual-program line, keeping 48-way; fp32 tile 65536 is measured
# at dsm=561KB, over the line, locking to 24-way and actually slower. In the extended region R is always capped at 16
# (far from the R>=32 wrong-result trigger axis).
# Measured benefit of the large-tile band ((4096,4096) do_bench median): fp16 R=4 68.6us ->
# R=8 49.5us -> R=16 40.9us (1.64TB/s); fp32 R=4 88.3 -> R=8 71.6us.
_ROW_BATCH_MAX_BLOCK = 16384
_ROW_TILE_ELEM_MAX = 16384
_ROW_TILE_ELEM_MAX_F16 = 65536
_ROW_TILE_ELEM_MAX_F32 = 32768
_ROWS_PER_CTA_CAP = 32
_ROWS_PER_CTA_CAP_EXT = 16  # Upper limit of R for the extended region (tile > 16384)
_ROWS_PER_CTA_MIN = 4
_TARGET_CTAS = 48
# When a row-batch tile has fewer lanes than this, the per-program fixed cost outweighs the parallelism gain.
_ROW_TILE_MIN_LANES = 4096
# Row-batch is used only when the row count fills the device (>= 48 tiles at R=4); smaller inputs keep the row-by-row
# legacy kernel.
_ROW_BATCH_MIN_ROWS = _TARGET_CTAS * _ROWS_PER_CTA_MIN
# When the 1-D tail branch is compiled into the kernel, num_warps=1 is required (nw>=2 makes the tail degrade the whole
# 2-D hot path by
# 1.3-1.9x, measured on 2026-11-08 with vllm skip_layer_norm); with the tail branch compiled out entirely
# (HAS_TAIL=False) and no [R] secondary-output store, [1,BLOCK] 2-D shaped weights +
# a software-pipelined loop + num_warps=2 has been verified correct in the fused_add_rms_norm row-batch kernel.
# With an [R] secondary-output store (SAVE_INV_RMS), conservatively lock nw=1 (same as layernorm.py).
_TAIL_NUM_WARPS = 1
_ROWS_NUM_WARPS = 2

# ---- wide row (N > MAX_BLOCK_SIZE) chunk width ------------------------------
# A wide row needs two passes to normalize and is limited by stream bandwidth.
# For 2-byte dtypes, dropping the chunk from 32768 to 16384 and using a persistent
# grid of _TARGET_CTAS programs takes the (1024,65536) do_bench median from
# 0.378ms to 0.237ms (-37%, ~1.7TB/s measured stream-bandwidth ceiling); for
# 4-byte dtypes a small chunk is 10-18% slower, so MAX_BLOCK_SIZE is kept.
_WIDE_ROW_BLOCK_F16 = 16384


def _row_tile_elem_max(dtype):
    if dtype == torch.float16 or dtype == torch.bfloat16:
        return _ROW_TILE_ELEM_MAX_F16
    if dtype == torch.float32:
        return _ROW_TILE_ELEM_MAX_F32
    return _ROW_TILE_ELEM_MAX


def _rows_per_cta(m, n, dtype):
    """Choose the row-batch factor R for a [R, BLOCK] 2-D tile launch.

    Return 1 (legacy per-row kernel) when the row width already fills the tile budget or the input is too small to fill
    the device;
    otherwise take the largest power of two keeping about _TARGET_CTAS tiles in flight, bounded by the dtype-calibrated
    DSM tile budget and verified-safe R caps (the 225-config validated region R <= 32 for tile <= 16384;
    R <= 16 in the extension region), plus a minimum lane count for the tile to amortize
    per-program fixed cost.
    """
    block = triton.next_power_of_2(n)
    if block > _ROW_BATCH_MAX_BLOCK:
        return 1
    tile_max = _row_tile_elem_max(dtype)
    tile_cap = tile_max // max(block, 1)
    if tile_cap < _ROWS_PER_CTA_MIN or m < _ROW_BATCH_MIN_ROWS:
        return 1
    rows = m // _TARGET_CTAS
    rows = 1 << max(0, rows.bit_length() - 1) if rows >= 1 else 1
    rows = max(rows, _ROW_TILE_MIN_LANES // max(block, 1))
    rows = max(rows, _ROWS_PER_CTA_MIN)
    rows = max(1, min(_ROWS_PER_CTA_CAP, tile_cap, rows))
    if rows * block > _ROW_TILE_ELEM_MAX:
        # Extension region: large tile tiers push R back under 16 (R>=32 with >=32768 tile is a
        # known silent wrong-value trigger axis).
        rows = min(rows, _ROWS_PER_CTA_CAP_EXT)
    return rows


@triton.jit
def _rms_norm_rows(
    Y,
    INV_RMS,
    X,
    W,
    pid,
    y_stride_r: tl.constexpr,
    y_stride_c: tl.constexpr,
    x_stride_r: tl.constexpr,
    x_stride_c: tl.constexpr,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    SINGLE_BLOCK: tl.constexpr,
    SAVE_INV_RMS: tl.constexpr,
):
    """One program handles ROWS rows: in the single-block path w is loaded once and reused by all rows.

    Each row still reduces with scalar tl.sum(axis=0) (fully equivalent to the 1D kernel). This legacy
    form is kept for small M / wide N / row-batch-disabled tiers; large M uses the row-batch 2-D tile of
    rms_norm_rows_kernel
    (its axis=1 reduction needs [1,BLOCK] 2-D shaped weights + nw discipline; the early
    1-D w broadcast + nw>=2 combination without the immune recipe is the wrong-value trigger window).
    """
    cols = tl.arange(0, BLOCK_SIZE)
    cmask = cols < N
    if SINGLE_BLOCK:
        w_shared = tl.load(W + cols, mask=cmask, other=0.0)
    else:
        w_shared = tl.full((BLOCK_SIZE,), 0.0, tl.float32)

    for r in range(ROWS):
        row = pid * ROWS + r
        if row < M:
            Y_cur = row * y_stride_r + Y
            X_cur = row * x_stride_r + X
            if SINGLE_BLOCK:
                x = tl.load(X_cur + cols * x_stride_c, mask=cmask, other=0.0).to(
                    tl.float32
                )
                var = tl.sum(x * x, axis=0) / N
                rrms = tl.math.rsqrt(var + eps)
                y = (x * rrms).to(Y_cur.dtype.element_ty) * w_shared
                tl.store(Y_cur + cols * y_stride_c, y, mask=cmask)
                if SAVE_INV_RMS:
                    tl.store(INV_RMS + row, rrms)
            else:
                # Row wider than one tile: accumulate variance in chunks, then write results in chunks
                _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
                for off in range(0, N, BLOCK_SIZE):
                    c = off + tl.arange(0, BLOCK_SIZE)
                    m = c < N
                    t = tl.load(X_cur + c * x_stride_c, mask=m, other=0.0).to(
                        tl.float32
                    )
                    _var += t * t
                var = tl.sum(_var, axis=0) / N
                rrms = tl.math.rsqrt(var + eps)
                for off in range(0, N, BLOCK_SIZE):
                    c = off + tl.arange(0, BLOCK_SIZE)
                    m = c < N
                    t = tl.load(X_cur + c * x_stride_c, mask=m, other=0.0).to(
                        tl.float32
                    )
                    w = tl.load(W + c, mask=m, other=0.0)
                    y = (t * rrms).to(Y_cur.dtype.element_ty) * w
                    tl.store(Y_cur + c * y_stride_c, y, mask=m)
                if SAVE_INV_RMS:
                    tl.store(INV_RMS + row, rrms)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_kernel(
    Y,
    INV_RMS,
    X,
    W,
    y_stride_r: tl.constexpr,
    y_stride_c: tl.constexpr,
    x_stride_r: tl.constexpr,
    x_stride_c: tl.constexpr,
    M,
    N,
    eps,
    total_programs,
    BLOCK_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    SINGLE_BLOCK: tl.constexpr,
    SAVE_INV_RMS: tl.constexpr,
):
    for pid in range(tl.program_id(0), total_programs, tl.num_programs(0)):
        _rms_norm_rows(
            Y,
            INV_RMS,
            X,
            W,
            pid,
            y_stride_r,
            y_stride_c,
            x_stride_r,
            x_stride_c,
            M,
            N,
            eps,
            BLOCK_SIZE,
            ROWS,
            SINGLE_BLOCK,
            SAVE_INV_RMS,
        )


@triton.jit
def _rms_norm_row(
    Y,
    INV_RMS,
    X,
    W,
    row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
    SAVE_INV_RMS: tl.constexpr,
):
    """1-D single-row body (tail-block fallback, correct at any BLOCK width)."""
    Y += row * N
    X += row * N
    cols = tl.arange(0, BLOCK_SIZE)
    if EXACT:
        x = tl.load(X + cols).to(tl.float32)
        w = tl.load(W + cols)
        var = tl.sum(x * x, axis=0) / N
        rrms = tl.math.rsqrt(var + eps)
        y = (x * rrms).to(Y.dtype.element_ty) * w
        tl.store(Y + cols, y)
        if SAVE_INV_RMS:
            tl.store(INV_RMS + row, rrms)
    else:
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0)
        var = tl.sum(x * x, axis=0) / N
        rrms = tl.math.rsqrt(var + eps)
        y = (x * rrms).to(Y.dtype.element_ty) * w
        tl.store(Y + cols, y, mask=mask)
        if SAVE_INV_RMS:
            tl.store(INV_RMS + row, rrms)


@triton.jit
def _rms_norm_rows_tile(
    Y,
    INV_RMS,
    X,
    w,
    row0,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    EXACT: tl.constexpr,
    SAVE_INV_RMS: tl.constexpr,
):
    """Whole [ROWS, BLOCK_SIZE] tile body; w has [1, BLOCK_SIZE] 2-D shape."""
    rows = row0 + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_SIZE)
    x_ptrs = X + rows[:, None] * N + cols[None, :]
    y_ptrs = Y + rows[:, None] * N + cols[None, :]
    if EXACT:
        x = tl.load(x_ptrs).to(tl.float32)
        var = tl.sum(x * x, axis=1) / N
        rrms = tl.math.rsqrt(var + eps)
        if SAVE_INV_RMS:
            tl.store(INV_RMS + rows, rrms)
        y = (x * rrms[:, None]).to(Y.dtype.element_ty) * w
        tl.store(y_ptrs, y)
    else:
        col_mask = cols < N
        x = tl.load(x_ptrs, mask=col_mask[None, :], other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=1) / N
        rrms = tl.math.rsqrt(var + eps)
        if SAVE_INV_RMS:
            tl.store(INV_RMS + rows, rrms)
        y = (x * rrms[:, None]).to(Y.dtype.element_ty) * w
        tl.store(y_ptrs, y, mask=col_mask[None, :])


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_rows_kernel(
    Y,
    INV_RMS,
    X,
    W,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    EXACT: tl.constexpr,  # N == BLOCK_SIZE: no column mask
    HAS_TAIL: tl.constexpr,  # M % ROWS != 0: compile the 1-D tail branch into the kernel
    SAVE_INV_RMS: tl.constexpr,
):
    """Row-batch kernel: each program normalizes several rows through a [ROWS, BLOCK_SIZE] 2-D tile,
    amortizing the per-program fixed cost that dominates one-row-per-program at large M (load
    dispatch + reduction tree + weight load). A persistent tile loop with grid <= 48 removes the
    inter-wave CTA retire/relaunch gap and keeps the launch grid 1-D, never touching the GCU
    65535 grid.x cap. Residual tail rows use the 1-D row body (the 2-D masked tail variant is a
    silent wrong-value window on this backend). Weights are loaded once outside the loop as a
    [1, BLOCK_SIZE] 2-D shape: 1-D vector broadcast operands are a known wrong-value window at
    BLOCK 512/1024 with num_warps >= 2, while the 2-D form is verified correct across every
    trigger window of the fused_add_rms_norm row-batch kernel.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(M, ROWS)
    cols = tl.arange(0, BLOCK_SIZE)
    # 1-D tail arm compiled in: launch must use num_warps=1.
    if EXACT:
        w = tl.load(W + cols[None, :])
    else:
        col_mask0 = cols < N
        w = tl.load(W + cols[None, :], mask=col_mask0[None, :], other=0.0)
    if HAS_TAIL:
        # Tail block (residual rows): handled per row in 1-D form.
        for tile in range(pid, num_tiles, num_programs):
            row0 = tile * ROWS
            if row0 + ROWS <= M:
                _rms_norm_rows_tile(
                    Y,
                    INV_RMS,
                    X,
                    w,
                    row0,
                    N,
                    eps,
                    BLOCK_SIZE,
                    ROWS,
                    EXACT,
                    SAVE_INV_RMS,
                )
            else:
                # Host guarantees M % ROWS == 0: every tile is full, the loop body is branch-free pure 2-D;
                for i in tl.static_range(ROWS):
                    row = row0 + i
                    if row < M:
                        _rms_norm_row(
                            Y,
                            INV_RMS,
                            X,
                            W,
                            row,
                            N,
                            eps,
                            BLOCK_SIZE,
                            EXACT,
                            SAVE_INV_RMS,
                        )
    else:
        # software-pipeline that loop.
        # Save inv_rms only when backward is needed; benchmark / inference paths neither allocate nor write it
        for tile in tl.range(pid, num_tiles, num_programs, num_stages=2):
            row0 = tile * ROWS
            _rms_norm_rows_tile(
                Y,
                INV_RMS,
                X,
                w,
                row0,
                N,
                eps,
                BLOCK_SIZE,
                ROWS,
                EXACT,
                SAVE_INV_RMS,
            )


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_grad_dx_kernel(
    X,  # pointer to the input
    DY,
    INV_RMS,  # pointer to inverse rms
    DX,  # pointer to the output
    W,  # pointer to the weights
    dx_stride_r,
    dx_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    for i in range(BLOCK_SIZE_M):
        pid = tl.program_id(0) * BLOCK_SIZE_M + i
        DX_cur = pid * dx_stride_r + DX
        X_cur = pid * x_stride_r + X
        DY_cur = pid * x_stride_r + DY
        INV_RMS_cur = pid + INV_RMS

        inv_rms = tl.load(INV_RMS_cur).to(tl.float32)

        _row_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X_cur + cols * x_stride_c, mask, other=0.0).to(tl.float32)
            dy = tl.load(DY_cur + cols * x_stride_c, mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0)
            dy_w = dy * w
            normalized_buf = x * inv_rms
            _row_sum += normalized_buf * dy_w
        row_sum_stats = tl.sum(_row_sum, axis=0)

        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X_cur + cols * x_stride_c, mask, other=0.0).to(tl.float32)
            dy = tl.load(DY_cur + cols * x_stride_c, mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0)
            dy_w = dy * w
            normalized_buf = x * inv_rms
            norm_val = normalized_buf / N
            dx = (dy_w - norm_val * row_sum_stats) * inv_rms
            tl.store(DX_cur + cols * dx_stride_c, dx, mask=mask)


@libentry()
@triton.jit
def rms_norm_grad_dw_kernel(
    X,  # pointer to the input
    DY,
    INV_RMS,  # pointer to inverse rms
    DW,  # pointer to the output
    dx_stride_r,
    dx_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    M,  # number of rows in X
    N,  # number of columns in X
    ROW_BLOCK_SIZE: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)

    row_start = row_pid * ROW_BLOCK_SIZE
    col_start = col_pid * COL_BLOCK_SIZE

    offset = row_start * x_stride_r + col_start * x_stride_c
    X += offset
    DY += offset
    INV_RMS += row_start

    rows = tl.arange(0, ROW_BLOCK_SIZE)
    cols = tl.arange(0, COL_BLOCK_SIZE)

    row_mask = (row_start + rows) < M
    col_mask = (col_start + cols) < N

    x = tl.load(
        X + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
        row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    inv_rms = tl.load(INV_RMS + rows, row_mask, other=0.0).to(tl.float32)
    dy = tl.load(
        DY + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
        row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    d_weight = x * dy * inv_rms[:, None]
    partial_dweight_sum = tl.sum(d_weight, axis=0)

    tl.store(
        DW + row_pid * N + col_start + cols,
        partial_dweight_sum,
        mask=col_mask,
    )


class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-5):
        logger.debug("GEMS_ENFLAME RMSNORM_FORWARD")
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        BLOCK_SIZE = min(triton.next_power_of_2(N), MAX_BLOCK_SIZE)
        single_block = triton.next_power_of_2(N) <= MAX_BLOCK_SIZE
        # Row-batch 2-D tile fast path: amortizes the per-row fixed cost of the per-row kernel.
        save_inv_rms = x.requires_grad or weight.requires_grad
        x = x.contiguous()
        weight = weight.contiguous()
        y = torch.empty_like(x)
        inv_rms = (
            torch.empty((M,), device=x.device, dtype=torch.float32)
            if save_inv_rms
            else x
        )

        with torch_device_fn.device(x.device):
            # With a [R] side-output store, conservatively lock nw=1 and keep the tail arm as a runtime
            rows_per_cta = _rows_per_cta(M, N, x.dtype)
            if rows_per_cta > 1:
                num_tiles = triton.cdiv(M, rows_per_cta)
                has_tail = (M % rows_per_cta) != 0
                grid_m = min(num_tiles, _TARGET_CTAS)
                if save_inv_rms:
                    # branch (verified form from layernorm.py).
                    # nw=2 + software pipelining is allowed for pure 2-D without a tail arm
                    kernel_has_tail = True
                    num_warps = _TAIL_NUM_WARPS
                else:
                    kernel_has_tail = has_tail
                    # (verified form from fused_add_rms_norm).
                    # Each program handles several rows: w is loaded once and program count drops with M
                    num_warps = _TAIL_NUM_WARPS if has_tail else _ROWS_NUM_WARPS
                rms_norm_rows_kernel[grid_m,](
                    y,
                    inv_rms,
                    x,
                    weight,
                    M,
                    N,
                    eps,
                    BLOCK_SIZE=BLOCK_SIZE,
                    ROWS=rows_per_cta,
                    EXACT=(N == BLOCK_SIZE),
                    HAS_TAIL=kernel_has_tail,
                    SAVE_INV_RMS=save_inv_rms,
                    num_warps=num_warps,
                )
            else:
                # Each program handles multiple rows: w is loaded only once and the program count decreases with M
                wide_block = BLOCK_SIZE
                wide_persistent = False
                if single_block:
                    if M >= 512:
                        rows = 8
                    elif M >= 128:
                        rows = 4
                    else:
                        rows = 2
                else:
                    # Wide row (N > MAX_BLOCK_SIZE, does not fit one tile): two chunk
                    # passes per row, limited by stream bandwidth rather than chunk
                    # width. For 2-byte dtypes, shrinking the chunk to
                    # _WIDE_ROW_BLOCK_F16 and switching to a persistent grid
                    # (<= _TARGET_CTAS programs) takes (1024,65536) do_bench from
                    # 0.378ms to 0.237ms (-37%); 4-byte dtypes stay optimal with a
                    # large chunk and the full grid (a small chunk is 10-18% slower).
                    rows = 1
                    if x.element_size() <= 2:
                        wide_block = min(triton.next_power_of_2(N), _WIDE_ROW_BLOCK_F16)
                        wide_persistent = wide_block < BLOCK_SIZE
                total_programs = triton.cdiv(M, rows)
                grid_m = (
                    min(total_programs, _TARGET_CTAS)
                    if wide_persistent
                    else min(total_programs, MAX_GRID_DIM)
                )

                rms_norm_kernel[grid_m,](
                    y,
                    inv_rms,
                    x,
                    weight,
                    N,
                    1,
                    N,
                    1,
                    M,
                    N,
                    eps,
                    total_programs,
                    wide_block,
                    rows,
                    single_block,
                    save_inv_rms,
                    num_warps=_FORWARD_WARPS,
                )

        ctx.save_for_backward(x, inv_rms, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        logger.debug("GEMS_ENFLAME RMSNORM_BACKWARD")
        x, inv_rms, weight = ctx.saved_tensors
        normalized_shape = ctx.normalized_shape
        eps = ctx.eps

        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        BLOCK_SIZE = min(triton.next_power_of_2(N), MAX_BLOCK_SIZE)
        x = x.contiguous()
        weight = weight.contiguous()
        dx = torch.empty_like(x)

        grid_m = M
        BLOCK_SIZE_M = 1
        if M > 65535:
            grid_m = MAX_GRID_DIM
            BLOCK_SIZE_M = (M + MAX_GRID_DIM - 1) // MAX_GRID_DIM

        with torch_device_fn.device(x.device):
            rms_norm_grad_dx_kernel[grid_m,](
                x, dy, inv_rms, dx, weight, N, 1, N, 1, N, eps, BLOCK_SIZE, BLOCK_SIZE_M
            )

        ROW_BLOCK_SIZE = 16
        COL_BLOCK_SIZE = 256
        row_block_num = triton.cdiv(M, ROW_BLOCK_SIZE)
        col_block_num = triton.cdiv(N, COL_BLOCK_SIZE)

        partial_buffer = torch.empty(
            (row_block_num, N), dtype=torch.float32, device=x.device
        )

        with torch_device_fn.device(x.device):
            rms_norm_grad_dw_kernel[row_block_num, col_block_num](
                x,
                dy,
                inv_rms,
                partial_buffer,
                N,
                1,
                N,
                1,
                M,
                N,
                ROW_BLOCK_SIZE,
                COL_BLOCK_SIZE,
            )
            # TODO(haizhu.shao TBD) after support dim=0 in triton_gcu backend,
            # we can directly use flag_gems.sum(partial_buffer, dim=0, dtype=x.dtype)
            dw = torch.sum(partial_buffer, dim=0).to(x.dtype).reshape(-1)

        return dx, None, dw, None


def rms_norm(x, normalized_shape, weight, eps=1e-5):
    logger.debug("GEMS_ENFLAME RMS_NORM")
    return RmsNorm.apply(x, normalized_shape, weight, eps)
