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
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

MAX_GRID_X = 65535

# Row-batch (2D [ROWS_PER_CTA, BLOCK] tile) budget, transplanted from the
# proven skip_layer_norm / rmsnorm_hf row-batch kernels (same backend,
# compiler_issues/vllm_fused_q_kv_rmsnorm_rowbatch_bcast_miscompile.md):
# - tile element budget: keeps the fp32 live set (x tile + deviation tile +
#   weight/bias vectors) inside the per-program DSM budget;
# - rows cap: R >= 32 combined with a 32768-element tile is a triton_gcu
#   silent-miscompile zone; tile <= 16384 with R <= 32 was verified correct
#   across 225 (shape, dtype, R, num_warps) configurations.
# num_warps must be 1: the 1-D tail body below degrades the whole 2-D hot
# path by 1.3-1.9x at num_warps >= 2.
_ROW_TILE_ELEM_MAX = 16384
_ROWS_PER_CTA_CAP = 32
_ROWS_PER_CTA_MIN = 4
_TARGET_CTAS = 48
# Rows-batch tiles below this many lanes pay too much per-CTA fixed cost
# (measured (256,512): R=8/tile 4096 lanes 5.77us vs R=4/tile 2048 8.86us).
_ROW_TILE_MIN_LANES = 4096
_ROWS_NUM_WARPS = 1
# Only row-batch when the row count fills the device (>= 48 CTAs at R=4);
# smaller inputs keep the legacy one-row-per-CTA kernel, which is already
# at-or-above torch parity for those sizes.
_ROW_BATCH_MIN_ROWS = _TARGET_CTAS * _ROWS_PER_CTA_MIN


def _rows_per_cta(m, n):
    """Pick the row-batch factor R for the [R, BLOCK] 2D-tile launch.

    Returns 1 (one row per CTA, the legacy kernel) when the row width
    already fills the tile budget or the input is too small to fill the
    device, and otherwise the largest power of two that keeps roughly
    _TARGET_CTAS tiles in flight, bounded by the verified-safe caps and
    floored so a tile still spans at least _ROW_TILE_MIN_LANES lanes.
    """
    block = triton.next_power_of_2(n)
    tile_cap = _ROW_TILE_ELEM_MAX // max(block, 1)
    if tile_cap < _ROWS_PER_CTA_MIN or m < _ROW_BATCH_MIN_ROWS:
        return 1
    rows = m // _TARGET_CTAS
    rows = 1 << max(0, rows.bit_length() - 1) if rows >= 1 else 1
    rows = max(rows, _ROW_TILE_MIN_LANES // max(block, 1))
    rows = max(rows, _ROWS_PER_CTA_MIN)
    return max(1, min(_ROWS_PER_CTA_CAP, tile_cap, rows))


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("layer_norm_persistent"),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["eps"])
def layer_norm_persistent_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,  # pointer to the mean
    out_rstd_ptr,  # pointer to the 1/std
    M,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    # using 1d tile makes code clean
    # Map the program id to the row of X and Y it should compute.
    pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    if pid >= M:
        return

    n_offsets = tl.arange(0, TILE_N)
    mask = n_offsets < N

    x = tl.load(in_ptr + pid * N + n_offsets, mask, other=0.0).to(tl.float32)
    m = tl.sum(x) / N
    d = x - m  # deviation
    s = tl.where(mask, d * d, 0)
    sum_square = tl.sum(s)  # sum of square of deviation
    var = sum_square / N
    rstd = tl.math.rsqrt(var + eps)

    tl.store(out_mean_ptr + pid, m)
    tl.store(out_rstd_ptr + pid, rstd)

    if weight_ptr is None:
        w = 1
    else:
        w = tl.load(weight_ptr + n_offsets, mask=mask)
    if bias_ptr is None:
        b = 0
    else:
        b = tl.load(bias_ptr + n_offsets, mask=mask)
    out = (x - m) * rstd * w + b

    tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)


@triton.jit
def _layer_norm_row(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,
    out_rstd_ptr,
    row,
    N,
    eps,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,  # N == BLOCK: no column masking needed
):
    """Normalize a single row (1-D form, correct at every BLOCK width)."""
    in_ptr += row * N
    out_ptr += row * N
    cols = tl.arange(0, BLOCK)
    if EXACT:
        x = tl.load(in_ptr + cols).to(tl.float32)
        mean = tl.sum(x, axis=0) / N
        d = x - mean
        var = tl.sum(d * d, axis=0) / N
        rstd = tl.math.rsqrt(var + eps)
        tl.store(out_mean_ptr + row, mean)
        tl.store(out_rstd_ptr + row, rstd)
        if weight_ptr is None:
            w = 1
        else:
            w = tl.load(weight_ptr + cols)
        if bias_ptr is None:
            b = 0
        else:
            b = tl.load(bias_ptr + cols)
        out = d * rstd * w + b
        tl.store(out_ptr + cols, out)
    else:
        mask = cols < N
        x = tl.load(in_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / N
        d = tl.where(mask, x - mean, 0.0)
        var = tl.sum(d * d, axis=0) / N
        rstd = tl.math.rsqrt(var + eps)
        tl.store(out_mean_ptr + row, mean)
        tl.store(out_rstd_ptr + row, rstd)
        if weight_ptr is None:
            w = 1
        else:
            w = tl.load(weight_ptr + cols, mask=mask)
        if bias_ptr is None:
            b = 0
        else:
            b = tl.load(bias_ptr + cols, mask=mask)
        out = (x - mean) * rstd * w + b
        tl.store(out_ptr + cols, out, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def layer_norm_rows_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,  # pointer to the mean
    out_rstd_ptr,  # pointer to the 1/std
    M,
    N,
    eps,
    BLOCK: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
    EXACT: tl.constexpr,  # N == BLOCK: no column masking needed
):
    # Row-batch kernel: each program normalizes ROWS_PER_CTA rows via a 2D
    # [ROWS_PER_CTA, BLOCK] tile, amortizing the per-program fixed cost
    # (load dispatch + reduction trees) that dominates the one-row-per-CTA
    # form at large M.  A persistent tile loop over grid <= 48 programs
    # (the L600 SPU count) removes CTA retire/re-launch gaps between waves
    # (measured -6..-12% vs a flat grid of the same tiles) and keeps the
    # launch grid 1-D, so the GCU 65535 grid.x limit cannot be reached.
    # Partial tail tiles fall back to the 1-D row body, which is correct
    # at every BLOCK width (2-D masked tail variants are a
    # silent-miscompile zone on this backend).
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(M, ROWS_PER_CTA)
    for tile in range(pid, num_tiles, num_programs):
        row0 = tile * ROWS_PER_CTA
        if row0 + ROWS_PER_CTA <= M:
            rows = row0 + tl.arange(0, ROWS_PER_CTA)
            cols = tl.arange(0, BLOCK)
            x_ptrs = in_ptr + rows[:, None] * N + cols[None, :]
            y_ptrs = out_ptr + rows[:, None] * N + cols[None, :]
            if EXACT:
                x = tl.load(x_ptrs).to(tl.float32)
                mean = tl.sum(x, axis=1) / N
                d = x - mean[:, None]
                var = tl.sum(d * d, axis=1) / N
                rstd = tl.math.rsqrt(var + eps)
                tl.store(out_mean_ptr + rows, mean)
                tl.store(out_rstd_ptr + rows, rstd)
                if weight_ptr is None:
                    w = 1
                else:
                    w = tl.load(weight_ptr + cols)
                if bias_ptr is None:
                    b = 0
                else:
                    b = tl.load(bias_ptr + cols)
                out = d * rstd[:, None] * w + b
                tl.store(y_ptrs, out)
            else:
                col_mask = cols < N
                x = tl.load(x_ptrs, mask=col_mask[None, :], other=0.0).to(tl.float32)
                mean = tl.sum(x, axis=1) / N
                d = tl.where(col_mask[None, :], x - mean[:, None], 0.0)
                var = tl.sum(d * d, axis=1) / N
                rstd = tl.math.rsqrt(var + eps)
                tl.store(out_mean_ptr + rows, mean)
                tl.store(out_rstd_ptr + rows, rstd)
                if weight_ptr is None:
                    w = 1
                else:
                    w = tl.load(weight_ptr + cols, mask=col_mask)
                if bias_ptr is None:
                    b = 0
                else:
                    b = tl.load(bias_ptr + cols, mask=col_mask)
                out = (x - mean[:, None]) * rstd[:, None] * w + b
                tl.store(y_ptrs, out, mask=col_mask[None, :])
        else:
            # Tail tile (partial rows): normalize the leftover rows one
            # at a time with the 1-D form.
            for i in tl.static_range(ROWS_PER_CTA):
                row = row0 + i
                if row < M:
                    _layer_norm_row(
                        in_ptr,
                        out_ptr,
                        weight_ptr,
                        bias_ptr,
                        out_mean_ptr,
                        out_rstd_ptr,
                        row,
                        N,
                        eps,
                        BLOCK,
                        EXACT,
                    )


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("layer_norm_persistent"),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["eps"])
def layer_norm_persistent_2tree_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,  # pointer to the mean
    out_rstd_ptr,  # pointer to the 1/std
    M,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    # Parallel dual-tree form of the persistent single-tile kernel, for wide
    # masked 2-byte rows (N in (4096, 16384) non-power-of-2, TILE_N in
    # {8192, 16384}).  The baseline serial chain (tree1 -> sub -> mul -> tree2)
    # exposes the reduction-tree latency there; shifting by the row's first
    # element lets both trees issue from the same operand and overlap
    # (interleaved A/B: -6~-9% on (16, 6144) and (M, 11264)).  Same shift
    # identity as the loop kernel: var = E[(x-x0)^2] - E[x-x0]^2, exact in
    # real arithmetic; x0 ~ sigma keeps the cancellation mild for large-mean
    # inputs.  fp32 wide rows keep the serial form: the extra shifted live
    # tile loses more DSM/ALU than the tree overlap gains (probe: 8.29 ->
    # 8.66us).
    pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    if pid >= M:
        return

    n_offsets = tl.arange(0, TILE_N)
    mask = n_offsets < N

    base = pid * N
    x = tl.load(in_ptr + base + n_offsets, mask=mask, other=0.0).to(tl.float32)
    x0 = tl.load(in_ptr + base)  # scalar shift center
    xs = tl.where(mask, x - x0, 0.0)
    s1 = tl.sum(xs)  # sum of shifted values
    s2 = tl.sum(xs * xs)  # sum of squared shifted values
    dmean = s1 / N
    mean = dmean + x0
    var = s2 / N - dmean * dmean
    rstd = tl.math.rsqrt(var + eps)

    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_rstd_ptr + pid, rstd)

    if weight_ptr is None:
        w = 1
    else:
        w = tl.load(weight_ptr + n_offsets, mask=mask)
    if bias_ptr is None:
        b = 0
    else:
        b = tl.load(bias_ptr + n_offsets, mask=mask)
    out = (x - mean) * rstd * w + b

    tl.store(out_ptr + base + n_offsets, out, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("layer_norm_persistent"),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["eps"])
def layer_norm_persistent_kernel_multiline(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,  # pointer to the mean
    out_rstd_ptr,  # pointer to the 1/std
    M,
    N,
    eps,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid = tl.program_id(0)
    m_offsets = pid * TILE_M + tl.arange(0, TILE_M)
    m_mask = m_offsets < M

    n_offsets = tl.arange(0, TILE_N)[None, :]
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask

    x = tl.load(in_ptr + m_offsets[:, None] * N + n_offsets, mask, other=0.0).to(
        tl.float32
    )
    m = tl.sum(x, axis=1) / N
    d = x - m[:, None]  # deviation
    s = tl.where(mask, d * d, 0)
    sum_square = tl.sum(s, axis=1)  # sum of square of deviation
    var = sum_square / N
    rstd = tl.math.rsqrt(var + eps)

    tl.store(out_mean_ptr + m_offsets, m, mask=m_mask)
    tl.store(out_rstd_ptr + m_offsets, rstd, mask=m_mask)

    if weight_ptr is None:
        w = 1
    else:
        w = tl.load(weight_ptr + n_offsets, mask=n_mask)
    if bias_ptr is None:
        b = 0
    else:
        b = tl.load(bias_ptr + n_offsets, mask=n_mask)
    out = (x - m[:, None]) * rstd[:, None] * w + b

    tl.store(out_ptr + m_offsets[:, None] * N + n_offsets, out, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def layer_norm_loop_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,  # pointer to the mean
    out_rstd_ptr,  # pointer to the 1/std
    M,
    N,
    eps,
    TILE_N: tl.constexpr,
    UNROLL: tl.constexpr,
):
    # Few-rows / very-wide-rows form (one CTA per row, N > 16384 or fp32
    # N > 8192). The upstream Welford body carries a loop-serial dependency
    # (new_m from m, plus a vector division per step) that pinned the
    # (20, 6, 65536) core shape at ~80/97us (fp16/fp32); this body computes
    # the statistics in a single pass with shifted sum/sum-sq vector
    # accumulators (no per-step division, no serial chain) and applies the
    # affine transform as out = x * (w*rstd) + (b - mean*w*rstd).
    # Measured on GCU400: 79.9 -> 43.4us (fp16), 97.1 -> 69.4us (fp32).
    # The coarse center c (first-tile mean) keeps sum(x-c)^2 free of the
    # E[x^2] - E[x]^2 cancellation for large-mean inputs; the masked tail
    # must zero the shifted lanes explicitly (other=0 - c would pollute).
    pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    if pid >= M:
        return

    base = pid.to(tl.int64) * N
    cols0 = tl.arange(0, TILE_N)
    mask0 = cols0 < N
    x0 = tl.load(in_ptr + base + cols0, mask=mask0, other=0.0).to(tl.float32)
    cnt0 = tl.minimum(TILE_N, N)
    c = tl.sum(x0, axis=0) / cnt0

    num_full = N // TILE_N
    num_groups = num_full // UNROLL
    acc1 = tl.zeros((TILE_N,), dtype=tl.float32)
    acc2 = tl.zeros((TILE_N,), dtype=tl.float32)
    # Hot loop: full chunks only, no masks, no branches; UNROLL chunks in
    # flight via static_range (runtime `if` inside this loop breaks the
    # load pipeline on this backend: 71.9 -> 104.9us when present).
    for g in range(0, num_groups):
        for u in tl.static_range(UNROLL):
            cols = (g * UNROLL + u) * TILE_N + cols0
            x = tl.load(in_ptr + base + cols).to(tl.float32) - c
            acc1 += x
            acc2 += x * x
    # Peeled remainder chunks (UNROLL does not divide num_full).
    for step in range(num_groups * UNROLL, num_full):
        cols = step * TILE_N + cols0
        x = tl.load(in_ptr + base + cols).to(tl.float32) - c
        acc1 += x
        acc2 += x * x
    s1 = tl.sum(acc1, axis=0)
    s2 = tl.sum(acc2, axis=0)
    tail_start = num_full * TILE_N
    if tail_start < N:
        cols = tail_start + cols0
        mask = cols < N
        x = tl.load(in_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x - c, 0.0)
        s1 += tl.sum(x, axis=0)
        s2 += tl.sum(x * x, axis=0)

    dmean = s1 / N
    mean = dmean + c
    var = s2 / N - dmean * dmean
    rstd = tl.math.rsqrt(var + eps)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_rstd_ptr + pid, rstd)

    # Apply pass, reverse order: the statistics pass ends at the row tail,
    # so starting from prev_multiple keeps the first applied chunks L2-hot.
    prev_multiple = prev_multiple_of(N, TILE_N)
    # the first step, masking is needed
    for start_n in range(0, TILE_N, TILE_N):
        cols = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = cols < N
        x = tl.load(
            in_ptr + base + cols,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        if weight_ptr is None:
            if bias_ptr is None:
                out = (x - mean) * rstd
            else:
                b = tl.load(bias_ptr + cols, mask=mask)
                out = (x - mean) * rstd + b
        else:
            w = tl.load(weight_ptr + cols, mask=mask)
            rw = w.to(tl.float32) * rstd
            if bias_ptr is None:
                out = x * rw - mean * rw
            else:
                b = tl.load(bias_ptr + cols, mask=mask)
                out = x * rw + (b.to(tl.float32) - mean * rw)
        tl.store(out_ptr + base + cols, out, mask=mask)

    for start_n in range(TILE_N, N, TILE_N):
        cols = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + base + cols, eviction_policy="evict_first").to(tl.float32)
        if weight_ptr is None:
            if bias_ptr is None:
                out = (x - mean) * rstd
            else:
                b = tl.load(bias_ptr + cols)
                out = (x - mean) * rstd + b
        else:
            w = tl.load(weight_ptr + cols)
            rw = w.to(tl.float32) * rstd
            if bias_ptr is None:
                out = x * rw - mean * rw
            else:
                b = tl.load(bias_ptr + cols)
                out = x * rw + (b.to(tl.float32) - mean * rw)
        tl.store(out_ptr + base + cols, out)


@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK_ROW_SIZE": 1, "BLOCK_COL_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_ROW_SIZE": 2, "BLOCK_COL_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_ROW_SIZE": 4, "BLOCK_COL_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_ROW_SIZE": 8, "BLOCK_COL_SIZE": 1024}, num_warps=4),
    ],
    key=["M", "N"],
    # FlagTree AABS shrinks BLOCK_COL_SIZE toward next_power_of_2(N); the
    # shrunk fp32 [R, 256]/[R, 512] tiles with a W row-vector load are a
    # triton_gcu silent-miscompile zone — keep the configured 2048 tile.
    disable_aabs=True,
)
@triton.jit
def layer_norm_backward_kernel(
    dY,
    X,
    W,
    Mean,
    Rstd,
    dX,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    row_mask = pid < M
    dY += pid * N
    X += pid * N
    dX += pid * N
    Mean += pid
    Rstd += pid

    mean = tl.load(Mean, mask=row_mask).to(tl.float32)
    rstd = tl.load(Rstd, mask=row_mask).to(tl.float32)

    dx_part2 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    dx_part3 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)
        col_mask = cols[None, :] < N
        mask = row_mask & col_mask
        dy = tl.load(dY + cols[None, :], mask).to(tl.float32)
        x = tl.load(X + cols[None, :], mask).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd
        if W is None:
            w = 1
        else:
            w = tl.load(W + cols, mask=cols < N).to(tl.float32)
        dx_hat = dy * w
        dx_part2 += dx_hat
        dx_part3 += dx_hat * x_hat

    dx_2 = tl.sum(dx_part2, axis=1)[:, None]
    dx_3 = tl.sum(dx_part3, axis=1)[:, None]

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)
        col_mask = cols[None, :] < N
        mask = row_mask & col_mask
        dy = tl.load(dY + cols[None, :], mask).to(tl.float32)
        x = tl.load(X + cols[None, :], mask).to(tl.float32)
        if W is None:
            w = 1
        else:
            w = tl.load(W + cols, mask=cols < N).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd
        dx_hat = dy * w
        dx = rstd * (dx_hat - (dx_2 + x_hat * dx_3) / N)
        tl.store(dX + cols, dx, mask=mask)


@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK_ROW_SIZE": 16, "BLOCK_COL_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_ROW_SIZE": 32, "BLOCK_COL_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_ROW_SIZE": 64, "BLOCK_COL_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_ROW_SIZE": 16, "BLOCK_COL_SIZE": 1024}, num_warps=4),
    ],
    key=["M", "N"],
    disable_aabs=True,
)
@triton.jit
def weight_bias_backward_kernel(
    dY,
    X,
    Mean,
    Rstd,
    dW,
    dB,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    pid = ext.program_id(0) * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)
    col_mask = pid < N
    dY += pid[None, :]
    X += pid[None, :]
    accW = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    accB = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for off in range(0, M, BLOCK_ROW_SIZE):
        rows = off + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
        row_mask = rows < M
        mask = row_mask & col_mask[None, :]
        dy = tl.load(dY + rows * N, mask).to(tl.float32)
        x = tl.load(X + rows * N, mask).to(tl.float32)
        mean = tl.load(Mean + rows, mask=rows < M).to(tl.float32)
        rstd = tl.load(Rstd + rows, mask=rows < M).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        accW += dy * x * rstd
        accB += dy
    if dW:
        dw = tl.sum(accW, axis=0)
        tl.store(dW + pid, dw, mask=col_mask)
    if dB:
        db = tl.sum(accB, axis=0)
        tl.store(dB + pid, db, mask=col_mask)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    logger.debug("GEMS_ENFLAME LAYERNORM_FORWARD")

    N = math.prod(normalized_shape)
    M = input.numel() // N

    input = input.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()
    y = torch.empty_like(input)

    # NOTE: when the input is half-precision(either float16 or bfloat16)
    # these statistical data saved for backward is in single precision
    # Allocate a 2*M statistics buffer in one shot and split it, saving one device allocation (allocation overhead is
    # significant
    # on small shapes; each split slice remains contiguous)
    stats = torch.empty(2 * M, dtype=input.dtype, device=input.device)
    mean = stats[:M]
    rstd = stats[M:]

    with torch_device_fn.device(input.device):
        if N <= 128:
            TILE_N = triton.next_power_of_2(N)
            TILE_M = triton.cdiv(1024, TILE_N)
            grid = (triton.cdiv(M, TILE_M), 1, 1)
            layer_norm_persistent_kernel_multiline[grid](
                input,
                y,
                weight,
                bias,
                mean,
                rstd,
                M,
                N,
                eps,
                TILE_M,
                TILE_N,
            )
        elif N <= 4096:
            TILE_N = triton.next_power_of_2(N)
            rows_per_cta = _rows_per_cta(M, N)
            if rows_per_cta > 1:
                num_tiles = triton.cdiv(M, rows_per_cta)
                grid = (min(num_tiles, _TARGET_CTAS), 1, 1)
                layer_norm_rows_kernel[grid](
                    input,
                    y,
                    weight,
                    bias,
                    mean,
                    rstd,
                    M,
                    N,
                    eps,
                    BLOCK=TILE_N,
                    ROWS_PER_CTA=rows_per_cta,
                    EXACT=(N == TILE_N),
                    num_warps=_ROWS_NUM_WARPS,
                )
            else:
                grid_x = min(M, MAX_GRID_X)
                grid_y = triton.cdiv(M, MAX_GRID_X)
                layer_norm_persistent_kernel[(grid_x, grid_y, 1)](
                    input,
                    y,
                    weight,
                    bias,
                    mean,
                    rstd,
                    M,
                    N,
                    eps,
                    TILE_N,
                )
        elif N <= 16384:
            # Single-tile band: one [TILE_N] row tile is register/DSM
            # resident, so the persistent kernel normalizes in a single load
            # pass. Measured vs the previous loop-kernel routing:
            # (16, 6144) fp16 6.37 -> 5.21us, (16, 11264) fp16 7.81 -> 6.33us.
            # fp32 above 8192 keeps the loop form with a 4096-wide tile (see
            # the interleaved A/B note in the branch below).
            if N <= 8192 or input.dtype != torch.float32:
                TILE_N = triton.next_power_of_2(N)
                grid_x = min(M, MAX_GRID_X)
                grid_y = triton.cdiv(M, MAX_GRID_X)
                # Wide masked 2-byte rows: the dual-tree form overlaps the two
                # reduction trees (interleaved A/B: -6~-9% on (16, 6144) and
                # (M, 11264); small exact tiles and fp32 keep the serial form).
                if (
                    input.dtype in (torch.float16, torch.bfloat16)
                    and N > 4096
                    and N < TILE_N
                ):
                    layer_norm_persistent_2tree_kernel[(grid_x, grid_y, 1)](
                        input,
                        y,
                        weight,
                        bias,
                        mean,
                        rstd,
                        M,
                        N,
                        eps,
                        TILE_N,
                    )
                else:
                    layer_norm_persistent_kernel[(grid_x, grid_y, 1)](
                        input,
                        y,
                        weight,
                        bias,
                        mean,
                        rstd,
                        M,
                        N,
                        eps,
                        TILE_N,
                    )
            else:
                # fp32 above 8192: the 16384-wide fp32 tile and the 8192-wide
                # v3 tile both lose to a 4096-wide loop here (interleaved A/B
                # on (16, 11264) fp32: 7.94 welford / 8.04 t4096 / 8.13
                # persistent / 8.63 t8192 us); t4096 keeps v3 within noise.
                grid_x = min(M, MAX_GRID_X)
                grid_y = triton.cdiv(M, MAX_GRID_X)
                layer_norm_loop_kernel[(grid_x, grid_y, 1)](
                    input,
                    y,
                    weight,
                    bias,
                    mean,
                    rstd,
                    M,
                    N,
                    eps,
                    TILE_N=4096,
                    UNROLL=2,
                    num_warps=4,
                )
        else:
            grid_x = min(M, MAX_GRID_X)
            grid_y = triton.cdiv(M, MAX_GRID_X)
            two_byte = input.dtype in (torch.float16, torch.bfloat16)
            TILE_N = 16384 if two_byte else 8192
            UNROLL = 4 if N // TILE_N >= 16 else 2
            layer_norm_loop_kernel[(grid_x, grid_y, 1)](
                input,
                y,
                weight,
                bias,
                mean,
                rstd,
                M,
                N,
                eps,
                TILE_N=TILE_N,
                UNROLL=UNROLL,
                num_warps=4,
            )
    return y, mean, rstd


def layer_norm_backward(
    grad_out,
    input,
    normalized_shape,
    mean,
    rstd,
    weight=None,
    bias=None,
    output_mask=None,
):
    logger.debug("GEMS_ENFLAME LAYERNORM_BACKWARD")

    grad_out = grad_out.contiguous()
    input = input.contiguous()
    mean = mean.contiguous()
    rstd = rstd.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()

    # Rows are flattened over every leading axis, not just dim 0: for a
    # multi-axis normalized_shape (e.g. (16,) over (64, 16, 64, 16)) using
    # input.shape[0] as M shrinks M and inflates N, so the kernels then index
    # the [N]-element weight/bias out of bounds and write NaN.
    N = math.prod(normalized_shape)
    M = input.numel() // N

    if output_mask[0]:
        in_grad = torch.empty_like(input)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_ROW_SIZE"]), 1, 1)
        with torch_device_fn.device(input.device):
            layer_norm_backward_kernel[grid](
                grad_out, input, weight, mean, rstd, in_grad, M, N
            )
    else:
        in_grad = None

    if output_mask[1] is False and output_mask[2] is False:
        return in_grad, None, None

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_COL_SIZE"]), 1, 1)
    weight_grad = torch.empty_like(weight) if output_mask[1] else None
    bias_grad = torch.empty_like(bias) if output_mask[2] else None
    with torch_device_fn.device(input.device):
        weight_bias_backward_kernel[grid](
            grad_out, input, mean, rstd, weight_grad, bias_grad, M, N
        )
    return in_grad, weight_grad, bias_grad
