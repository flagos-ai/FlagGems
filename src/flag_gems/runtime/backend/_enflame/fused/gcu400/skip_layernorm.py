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
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

MAX_BLOCK_SIZE = 4096
MAX_GRID_SIZE = 65535
# gcu400 has 24 SIPs; 48 programs (2 waves) measured best for the row-wise
# kernel on L600.  The old grid=min(M, 65535) launched up to 65535 programs
# for M=1M rows and paid scheduling overhead without extra parallelism.
OPTIMAL_GRID_SIZE = 48

# Row-batch (2D [ROWS_PER_CTA, BLOCK_SIZE] tile) budget, transplanted from the
# proven row-batch layernorm-family kernels on this backend (see
# compiler_issues/vllm_fused_q_kv_rmsnorm_rowbatch_bcast_miscompile.md):
# - tile element budget: keeps the fp32 live set (value tile + deviation tile
#   + a transient product) inside the per-program DSM budget together with the
#   [BLOCK_SIZE] weight/bias vectors (~224KB worst case, under the 448KB cap);
# - rows cap: R >= 32 combined with a 32768-element tile is a triton_gcu
#   silent-miscompile zone; tile <= 16384 with R <= 32 was verified correct
#   across 225 (shape, dtype, R, num_warps) configurations.
_ROW_TILE_ELEM_MAX = 16384
_ROWS_PER_CTA_CAP = 32
_ROWS_PER_CTA_MIN = 4
_ROW_TILE_MIN_LANES = 4096
_ROW_BATCH_MIN_ROWS = OPTIMAL_GRID_SIZE * _ROWS_PER_CTA_MIN
# The row-batch kernel must launch with num_warps=1 whenever the 1-D tail
# body is compiled in (M % ROWS_PER_CTA != 0): at num_warps >= 2 the tail
# branch degrades the whole 2-D hot path by 1.3-1.9x (measured 2026-11-08,
# vllm skip_layer_norm / rmsnorm_hf, same backend).  When the host guarantees
# M % ROWS_PER_CTA == 0 the tail body is compiled out entirely (HAS_TAIL=False)
# and the 2-D-shaped weight/bias broadcast form is used, which was verified
# correct with num_warps=2 in the fused_add_rms_norm row-batch kernel.
_TAIL_NUM_WARPS = 1
_ROWS_NUM_WARPS = 2


def _rows_per_cta(m, n):
    """Pick the row-batch factor R for the [R, BLOCK_SIZE] 2D-tile launch.

    Returns 1 (one row per program, the legacy kernel) when the row width
    already fills the tile budget or the input is too small to fill the
    device, and otherwise the largest power of two that keeps roughly
    OPTIMAL_GRID_SIZE tiles in flight, bounded by the verified-safe
    tile/rows caps and floored so a tile still spans enough lanes to
    amortize the per-program fixed cost.
    """
    block = min(triton.next_power_of_2(n), MAX_BLOCK_SIZE)
    tile_cap = _ROW_TILE_ELEM_MAX // max(block, 1)
    if tile_cap < _ROWS_PER_CTA_MIN or m < _ROW_BATCH_MIN_ROWS:
        return 1
    rows = m // OPTIMAL_GRID_SIZE
    rows = 1 << max(0, rows.bit_length() - 1) if rows >= 1 else 1
    rows = max(rows, _ROW_TILE_MIN_LANES // max(block, 1))
    rows = max(rows, _ROWS_PER_CTA_MIN)
    return max(1, min(_ROWS_PER_CTA_CAP, tile_cap, rows))


@libentry()
@triton.jit(do_not_specialize=["eps"])
def skip_layer_norm_kernel_generic(
    Y,  # pointer to the output
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weights
    B,  # pointer to the biases
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    M,  # total number of rows
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    num_programs = ext.num_programs(0)

    for row_idx in range(pid, M, num_programs):
        row_Y = Y + row_idx * y_stride_r
        row_X = X + row_idx * x_stride_r
        row_R = R + row_idx * r_stride_r

        # Pass 1: accumulate sum for mean
        sum_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(row_X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
            r = tl.load(row_R + cols * r_stride_c, mask, other=0.0).to(tl.float32)
            sum_acc += tl.where(mask, x + r, 0.0)

        mean = tl.sum(sum_acc, axis=0) / N

        # Pass 2: compute variance
        var_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(row_X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
            r = tl.load(row_R + cols * r_stride_c, mask, other=0.0).to(tl.float32)
            val = tl.where(mask, x + r - mean, 0.0)
            var_acc += val * val

        var = tl.sum(var_acc, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)

        # Pass 3: normalize and apply weight/bias, write output
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(row_X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
            r = tl.load(row_R + cols * r_stride_c, mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
            x_hat = (x + r - mean) * rstd
            y = w * x_hat + b
            y = y.to(row_Y.dtype.element_ty)
            tl.store(row_Y + cols * y_stride_c, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def skip_layer_norm_kernel_fused(
    Y,
    X,
    R,
    W,
    B,
    y_stride_r,
    y_stride_c,
    x_stride_r,
    x_stride_c,
    r_stride_r,
    r_stride_c,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-pass row LayerNorm for N <= BLOCK_SIZE.

    The generic kernel walks each row three times and reloads X / R on every
    pass (9 memory ops per row).  Keeping x+residual in registers lets one
    load of each serve mean, variance and normalization (5 ops per row).
    Measured on gcu400 L600: 1024x1024 108us -> 44us, 4096x4096 493us ->
    193us.  Relative error vs torch.layer_norm is <= 4e-4 for fp16.
    """
    pid = ext.program_id(0)
    num_programs = ext.num_programs(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    for row_idx in range(pid, M, num_programs):
        row_X = X + row_idx * x_stride_r
        row_R = R + row_idx * r_stride_r
        row_Y = Y + row_idx * y_stride_r
        x = tl.load(row_X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
        r = tl.load(row_R + cols * r_stride_c, mask, other=0.0).to(tl.float32)
        val = tl.where(mask, x + r, 0.0)
        mean = tl.sum(val, axis=0) / N
        diff = tl.where(mask, val - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / N
        rstd = tl.rsqrt(var + eps)
        w = tl.load(W + cols, mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask, other=0.0).to(tl.float32)
        y = w * (diff * rstd) + b
        tl.store(row_Y + cols * y_stride_c, y, mask)


@triton.jit
def _skip_layer_norm_row(
    Y,
    X,
    R,
    W,
    B,
    y_stride_r,
    y_stride_c,
    x_stride_r,
    x_stride_c,
    r_stride_r,
    r_stride_c,
    row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
):
    """Normalize a single row (1-D form, correct at every BLOCK width)."""
    Y += row * y_stride_r
    X += row * x_stride_r
    R += row * r_stride_r
    cols = tl.arange(0, BLOCK_SIZE)
    if EXACT:
        x = tl.load(X + cols * x_stride_c).to(tl.float32)
        r = tl.load(R + cols * r_stride_c).to(tl.float32)
        x += r
        mean = tl.sum(x, axis=0) / N
        _var = x - mean
        _var = _var * _var
        var = tl.sum(_var, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        w = tl.load(W + cols).to(tl.float32)
        b = tl.load(B + cols).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = w * x_hat + b
        tl.store(Y + cols * y_stride_c, y.to(Y.dtype.element_ty))
    else:
        mask = cols < N
        x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)
        x += r
        mean = tl.sum(x, axis=0) / N
        _var = tl.where(mask, x - mean, 0.0)
        _var = _var * _var
        var = tl.sum(_var, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = w * x_hat + b
        tl.store(Y + cols * y_stride_c, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _skip_layer_norm_rows_tile(
    Y,
    X,
    R,
    w,
    b,
    y_stride_r,
    y_stride_c,
    x_stride_r,
    x_stride_c,
    r_stride_r,
    r_stride_c,
    row0,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
    EXACT: tl.constexpr,
):
    """Full [ROWS_PER_CTA, BLOCK_SIZE] tile body; w/b are [1, BLOCK_SIZE]."""
    rows = row0 + tl.arange(0, ROWS_PER_CTA)
    cols = tl.arange(0, BLOCK_SIZE)
    x_ptrs = X + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c
    r_ptrs = R + rows[:, None] * r_stride_r + cols[None, :] * r_stride_c
    y_ptrs = Y + rows[:, None] * y_stride_r + cols[None, :] * y_stride_c
    if EXACT:
        x = tl.load(x_ptrs).to(tl.float32)
        r = tl.load(r_ptrs).to(tl.float32)
        v = x + r
        mean = tl.sum(v, axis=1) / N
        d = v - mean[:, None]
        var = tl.sum(d * d, axis=1) / N
        rstd = 1 / tl.sqrt(var + eps)
        y = w * (d * rstd[:, None]) + b
        tl.store(y_ptrs, y.to(Y.dtype.element_ty))
    else:
        col_mask = cols < N
        x = tl.load(x_ptrs, mask=col_mask[None, :], other=0.0).to(tl.float32)
        r = tl.load(r_ptrs, mask=col_mask[None, :], other=0.0).to(tl.float32)
        v = x + r
        mean = tl.sum(v, axis=1) / N
        d = tl.where(col_mask[None, :], v - mean[:, None], 0.0)
        var = tl.sum(d * d, axis=1) / N
        rstd = 1 / tl.sqrt(var + eps)
        y = w * (d * rstd[:, None]) + b
        tl.store(y_ptrs, y.to(Y.dtype.element_ty), mask=col_mask[None, :])


@libentry()
@triton.jit(do_not_specialize=["eps"])
def skip_layer_norm_rows_kernel(
    Y,
    X,
    R,
    W,
    B,
    y_stride_r,
    y_stride_c,
    x_stride_r,
    x_stride_c,
    r_stride_r,
    r_stride_c,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
    EXACT: tl.constexpr,  # N == BLOCK_SIZE: no column masking needed
    HAS_TAIL: tl.constexpr,  # M % ROWS_PER_CTA != 0: 1-D tail body compiled in
):
    """Row-batch kernel: each program normalizes ROWS_PER_CTA rows via a 2D
    [ROWS_PER_CTA, BLOCK_SIZE] tile, amortizing the per-program fixed cost
    (load dispatch + reduction trees + weight/bias loads) that dominates the
    one-row-per-program form at large M.  A persistent tile loop over
    grid <= OPTIMAL_GRID_SIZE programs removes CTA retire/re-launch gaps
    between waves and keeps the launch grid 1-D, so the GCU 65535 grid.x
    limit cannot be reached.  Partial tail tiles are handled by the 1-D row
    body (2-D masked tail variants are a silent-miscompile zone on this
    backend).  Weight/bias are loaded once per program as [1, BLOCK_SIZE]
    2-D tiles: 1-D-vector broadcast operands are a verified miscompile
    window in this kernel family at BLOCK 512/1024 with num_warps >= 2,
    while the 2-D-shaped form (loop-invariant, software-pipelined loop,
    num_warps=2) was verified correct in the fused_add_rms_norm row-batch
    kernel across every trigger window of that family.
    """
    pid = ext.program_id(0)
    num_programs = ext.num_programs(0)
    num_tiles = tl.cdiv(M, ROWS_PER_CTA)
    cols = tl.arange(0, BLOCK_SIZE)
    # Weight/bias are loop-invariant: load them once per program.
    if EXACT:
        w = tl.load(W + cols[None, :]).to(tl.float32)
        b = tl.load(B + cols[None, :]).to(tl.float32)
    else:
        col_mask0 = cols < N
        w = tl.load(W + cols[None, :], mask=col_mask0[None, :], other=0.0).to(
            tl.float32
        )
        b = tl.load(B + cols[None, :], mask=col_mask0[None, :], other=0.0).to(
            tl.float32
        )

    if HAS_TAIL:
        # 1-D tail body compiled in: must launch with num_warps=1 (at
        # num_warps >= 2 the tail branch degrades the 2-D hot path).
        for tile in range(pid, num_tiles, num_programs):
            row0 = tile * ROWS_PER_CTA
            if row0 + ROWS_PER_CTA <= M:
                _skip_layer_norm_rows_tile(
                    Y,
                    X,
                    R,
                    w,
                    b,
                    y_stride_r,
                    y_stride_c,
                    x_stride_r,
                    x_stride_c,
                    r_stride_r,
                    r_stride_c,
                    row0,
                    N,
                    eps,
                    BLOCK_SIZE,
                    ROWS_PER_CTA,
                    EXACT,
                )
            else:
                # Tail tile (partial rows): normalize the leftover rows one
                # at a time with the 1-D form (correct at every BLOCK width).
                for i in tl.static_range(ROWS_PER_CTA):
                    row = row0 + i
                    if row < M:
                        _skip_layer_norm_row(
                            Y,
                            X,
                            R,
                            W,
                            B,
                            y_stride_r,
                            y_stride_c,
                            x_stride_r,
                            x_stride_c,
                            r_stride_r,
                            r_stride_c,
                            row,
                            N,
                            eps,
                            BLOCK_SIZE,
                            EXACT,
                        )
    else:
        # Host guarantees M % ROWS_PER_CTA == 0: every tile is full and the
        # loop body is branch-free pure 2-D; software-pipeline the loop.
        for tile in tl.range(pid, num_tiles, num_programs, num_stages=2):
            row0 = tile * ROWS_PER_CTA
            _skip_layer_norm_rows_tile(
                Y,
                X,
                R,
                w,
                b,
                y_stride_r,
                y_stride_c,
                x_stride_r,
                x_stride_c,
                r_stride_r,
                r_stride_c,
                row0,
                N,
                eps,
                BLOCK_SIZE,
                ROWS_PER_CTA,
                EXACT,
            )


class SkipLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, normalized_shape, weight, bias, eps=1e-5):
        logger.debug("GEMS_ENFLAME SKIP_LAYERNORM_FORWARD")
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        BLOCK_SIZE = min(triton.next_power_of_2(N), MAX_BLOCK_SIZE)
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        y = torch.empty_like(x)

        strides = (N, 1, N, 1, N, 1)
        # A single warp is fastest for narrow rows (measured 1024 columns
        # 44.2us -> 34.9us, 512 columns with M=32768 1480us -> 1015us);
        # wide rows need 4 warps to cover the 4096-element reduction.
        num_warps_fused = 1 if N <= 1024 else 4
        with torch_device_fn.device(x.device):
            if N <= BLOCK_SIZE:
                rows_per_cta = _rows_per_cta(M, N)
                if rows_per_cta > 1:
                    num_tiles = triton.cdiv(M, rows_per_cta)
                    has_tail = (M % rows_per_cta) != 0
                    grid_size = min(num_tiles, OPTIMAL_GRID_SIZE)
                    # With the 1-D tail body compiled in, num_warps >= 2
                    # degrades the whole 2-D hot path (measured 1.3-1.9x);
                    # with the tail compiled out, wider warps are safe.
                    num_warps = _TAIL_NUM_WARPS if has_tail else _ROWS_NUM_WARPS
                    args = (
                        y,
                        x,
                        residual,
                        weight,
                        bias,
                        *strides,
                        M,
                        N,
                        eps,
                    )
                    skip_layer_norm_rows_kernel[grid_size,](
                        *args,
                        BLOCK_SIZE=BLOCK_SIZE,
                        ROWS_PER_CTA=rows_per_cta,
                        EXACT=(N == BLOCK_SIZE),
                        HAS_TAIL=has_tail,
                        num_warps=num_warps,
                    )
                else:
                    # One row per program: the legacy persistent kernel is
                    # already at-or-above torch parity for these sizes.
                    grid_size = min(M, OPTIMAL_GRID_SIZE)
                    args = (
                        y,
                        x,
                        residual,
                        weight,
                        bias,
                        *strides,
                        M,
                        N,
                        eps,
                        BLOCK_SIZE,
                    )
                    skip_layer_norm_kernel_fused[grid_size,](
                        *args, num_warps=num_warps_fused
                    )
            else:
                grid_size = min(M, OPTIMAL_GRID_SIZE)
                args = (
                    y,
                    x,
                    residual,
                    weight,
                    bias,
                    *strides,
                    M,
                    N,
                    eps,
                    BLOCK_SIZE,
                )
                skip_layer_norm_kernel_generic[grid_size,](*args, num_warps=4)
        return y


def skip_layer_norm(x, residual, normalized_shape, weight, bias, eps=1e-5):
    return SkipLayerNorm.apply(x, residual, normalized_shape, weight, bias, eps)
