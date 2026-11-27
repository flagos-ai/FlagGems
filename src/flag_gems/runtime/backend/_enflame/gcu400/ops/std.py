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
import time

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.shape_utils import can_use_int32_index

logger = logging.getLogger(__name__)

# Failed compiles are not cached, so re-attempting the launch recompiles
# and usually succeeds; once a kernel is compiled the retry is a no-op.
_COMPILE_RETRY_ATTEMPTS = 6
_COMPILE_RETRY_DELAY_S = 1.0

_GLOBAL_SINGLE_CTA_LIMIT = 65536
_MAX_GLOBAL_PROGRAMS = 512
_MAX_GLOBAL_WIDE_PROGRAMS = 256
_GLOBAL_WIDE_GRID_MIN_ELEMS = 1 << 21
_GLOBAL_BLOCK_SIZE = 4096
_ROW_VEC_BLOCK_N_CAP = 1024
_ROW_VEC_MIN_BLOCK_N = 128
_ROW_VEC_MIN_N = 8
_ROW_VEC_MIN_M = 16
_ROW_VEC_MIN_ROWS = 8
_ROW_VEC_LANE_BUDGET = 16384
_ROW_VEC_MAX_PROGRAMS = 48
_ROW_VEC_MIN_PROGRAMS = 8


def _launch_retry(launch):
    for attempt in range(_COMPILE_RETRY_ATTEMPTS):
        try:
            return launch()
        except Exception:
            if attempt == _COMPILE_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_COMPILE_RETRY_DELAY_S)


@triton.jit
def _stable_mean_from_shift(
    shift,
    delta_total,
    value_total,
    count,
):
    """Select the better-conditioned of shifted and direct pairwise sums."""
    shifted_mean = shift + delta_total / count
    direct_mean = value_total / count
    # A shifted sum is accurate when the anchor represents the data. If the
    # correction nearly cancels the anchor, as for a first-element outlier,
    # the direct pairwise sum has the better condition number.
    use_shifted = tl.abs(shifted_mean) * 8.0 >= tl.abs(shift)
    return tl.where(use_shifted, shifted_mean, direct_mean)


@triton.jit
def _std_map_kernel(X, Tmp_sum, Tmp_sum_sq, N, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offset < N
    x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
    sum_val = tl.sum(x, axis=0)
    sum_sq_val = tl.sum(x * x, axis=0)
    tl.store(Tmp_sum + pid, sum_val)
    tl.store(Tmp_sum_sq + pid, sum_sq_val)


@triton.jit
def _std_reduce_kernel(
    Tmp_sum, Tmp_sum_sq, Out, N, correction, BLOCK_NUM, BLOCK_SIZE: tl.constexpr
):
    total_sum_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    total_sum_sq_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for off in range(0, BLOCK_NUM, BLOCK_SIZE):
        offset = off + tl.arange(0, BLOCK_SIZE)
        mask = offset < BLOCK_NUM
        tmp_sum_vals = tl.load(Tmp_sum + offset, mask=mask, other=0.0).to(tl.float32)
        tmp_sum_sq_vals = tl.load(Tmp_sum_sq + offset, mask=mask, other=0.0).to(
            tl.float32
        )
        total_sum_acc += tmp_sum_vals
        total_sum_sq_acc += tmp_sum_sq_vals
    total_sum = tl.sum(total_sum_acc, axis=0)
    total_sum_sq = tl.sum(total_sum_sq_acc, axis=0)
    mean = total_sum / N
    var = (total_sum_sq / N) - (mean * mean)
    var = var * N / tl.maximum(N - correction, 1.0)
    safe_var = tl.maximum(var, 0.0)
    std_dev = tl.sqrt(safe_var)
    tl.store(Out, std_dev.to(Out.dtype.element_ty))


@libentry()
@triton.jit
def _std_global_row_kernel(X, Out, N, denominator, BLOCK_SIZE: tl.constexpr):
    """Single-CTA global std with a stable shifted two-pass central moment.

    Mirrors the std_mean single-row recipe (shifted anchor + stable mean
    selection + two-pass moment) but only stores the standard deviation.
    """
    shift = tl.load(X).to(tl.float32)

    delta_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    value_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        values = tl.load(X + offsets, mask=mask, other=shift).to(tl.float32)
        delta_sum += tl.where(mask, values - shift, 0.0)
        value_sum += tl.where(mask, values, 0.0)

    delta_total = tl.sum(delta_sum)
    mean = _stable_mean_from_shift(shift, delta_total, tl.sum(value_sum), N)
    squared_deviation_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        values = tl.load(X + offsets, mask=mask, other=mean).to(tl.float32)
        deviation = values - mean
        squared_deviation_sum += tl.where(mask, deviation * deviation, 0.0)
    m2 = tl.sum(squared_deviation_sum)
    m2 = tl.where(m2 < 0.0, 0.0, m2)
    std_dev = tl.sqrt(m2 / denominator)
    tl.store(Out, std_dev.to(Out.dtype.element_ty))


@libentry()
@triton.jit
def _std_global_map_kernel(X, Scratch, N, BLOCK_SIZE: tl.constexpr):
    """Build mergeable shifted central moments for a flat real tensor.

    Bounded grid-stride walk keeps the launch grid far below the GCU grid.x
    hardware limit for any input size (P19), unlike a block-per-chunk map.
    """
    pid = ext.program_id(0)
    num_programs = tl.num_programs(0)
    lane_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    shift = tl.load(X + pid * BLOCK_SIZE).to(tl.float32)
    grid_stride = num_programs * BLOCK_SIZE

    delta_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    value_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    count = tl.zeros((), dtype=tl.float32)
    for start in range(0, N, grid_stride):
        offsets = start + lane_offsets
        mask = offsets < N
        values = tl.load(X + offsets, mask=mask, other=shift).to(tl.float32)
        delta_sum += tl.where(mask, values - shift, 0.0)
        value_sum += tl.where(mask, values, 0.0)
        count += tl.sum(mask.to(tl.float32))

    delta_total = tl.sum(delta_sum)
    mean = _stable_mean_from_shift(shift, delta_total, tl.sum(value_sum), count)
    squared_deviation_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, N, grid_stride):
        offsets = start + lane_offsets
        mask = offsets < N
        values = tl.load(X + offsets, mask=mask, other=mean).to(tl.float32)
        deviation = values - mean
        squared_deviation_sum += tl.where(mask, deviation * deviation, 0.0)
    m2 = tl.sum(squared_deviation_sum)
    m2 = tl.where(m2 < 0.0, 0.0, m2)
    tl.store(Scratch + pid, mean)
    tl.store(Scratch + num_programs + pid, m2)
    tl.store(Scratch + 2 * num_programs + pid, count)


@libentry()
@triton.jit
def _std_global_finish_kernel(
    Scratch,
    Out,
    N,
    denominator,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Merge real partial moments using another shifted-data reduction."""
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_PARTIALS
    means = tl.load(Scratch + offsets, mask=mask, other=0.0).to(tl.float32)
    partial_m2s = tl.load(Scratch + NUM_PARTIALS + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    counts = tl.load(Scratch + 2 * NUM_PARTIALS + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    reference = tl.load(Scratch).to(tl.float32)
    deltas = means - reference
    weighted_delta = tl.where(mask, counts * deltas, 0.0)
    weighted_values = tl.where(mask, counts * means, 0.0)
    delta_total = tl.sum(weighted_delta)
    mean = _stable_mean_from_shift(reference, delta_total, tl.sum(weighted_values), N)
    m2 = tl.sum(
        tl.where(
            mask,
            partial_m2s + counts * (means - mean) * (means - mean),
            0.0,
        )
    )
    m2 = tl.where(m2 < 0.0, 0.0, m2)
    std_dev = tl.sqrt(m2 / denominator)
    tl.store(Out, std_dev.to(Out.dtype.element_ty))


@libentry()
@triton.jit
def _std_row_vec_kernel(
    X,
    Out,
    M,
    N,
    denominator,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Persistent row-tile reduction for contiguous innermost rows on GCU400.

    Each program walks [ROWS, BLOCK_N] tiles with grid stride, replacing the
    one-program-per-row layout whose CTA dispatch floor dominates large-row
    counts. The numerical recipe (shifted first-element anchor, stable mean
    selection, two-pass central moment) is identical to the std_mean row
    kernels; only the work layout changes and the mean is not stored. When N
    fits in one BLOCK_N tile the values stay register/DSM resident so the
    second pass never re-reads memory.
    """
    pid = ext.program_id(0)
    num_programs = tl.num_programs(0)
    denominator = denominator.to(tl.float32)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    for row_start in range(pid * ROWS, M, num_programs * ROWS):
        rows = row_start + tl.arange(0, ROWS)
        row_mask = rows < M
        mask = row_mask[:, None] & col_mask[None, :]
        shift = tl.load(X + rows * N, mask=row_mask, other=0.0).to(tl.float32)
        if N <= BLOCK_N:
            values = tl.load(
                X + rows[:, None] * N + cols[None, :], mask=mask, other=0.0
            ).to(tl.float32)
            delta_total = tl.sum(
                tl.where(col_mask[None, :], values - shift[:, None], 0.0), axis=1
            )
            value_total = tl.sum(tl.where(col_mask[None, :], values, 0.0), axis=1)
            mean = _stable_mean_from_shift(shift, delta_total, value_total, N)
            deviation = values - mean[:, None]
            m2 = tl.sum(tl.where(col_mask[None, :], deviation * deviation, 0.0), axis=1)
        else:
            delta_acc = tl.zeros((ROWS, BLOCK_N), dtype=tl.float32)
            value_acc = tl.zeros((ROWS, BLOCK_N), dtype=tl.float32)
            for start in range(0, N, BLOCK_N):
                block_cols = start + cols
                block_mask = row_mask[:, None] & (block_cols < N)[None, :]
                values = tl.load(
                    X + rows[:, None] * N + block_cols[None, :],
                    mask=block_mask,
                    other=0.0,
                ).to(tl.float32)
                delta_acc += tl.where(block_mask, values - shift[:, None], 0.0)
                value_acc += tl.where(block_mask, values, 0.0)
            delta_total = tl.sum(delta_acc, axis=1)
            value_total = tl.sum(value_acc, axis=1)
            mean = _stable_mean_from_shift(shift, delta_total, value_total, N)
            m2_acc = tl.zeros((ROWS, BLOCK_N), dtype=tl.float32)
            for start in range(0, N, BLOCK_N):
                block_cols = start + cols
                block_mask = row_mask[:, None] & (block_cols < N)[None, :]
                values = tl.load(
                    X + rows[:, None] * N + block_cols[None, :],
                    mask=block_mask,
                    other=0.0,
                ).to(tl.float32)
                deviation = values - mean[:, None]
                m2_acc += tl.where(block_mask, deviation * deviation, 0.0)
            m2 = tl.sum(m2_acc, axis=1)
        m2 = tl.where(m2 < 0.0, 0.0, m2)
        std_dev = tl.sqrt(m2 / denominator)
        tl.store(Out + rows, std_dev.to(Out.dtype.element_ty), mask=row_mask)


def _keep_config(conf):
    bm = conf.kwargs["BLOCK_M"]
    bn = conf.kwargs["BLOCK_N"]
    if bm * bn >= 128 * 512:
        return False
    return True


_NON_INNER_WELFORD_TILE_N = 16
_NON_INNER_WELFORD_TILE_K = 512
_NON_INNER_WELFORD_MIN_N = 64
_INT32_MAX_ELEMENTS = 2**31 - 1


@libentry()
@triton.jit
def _std_dim_kernel_non_inner_welford(
    Out,
    X,
    M,
    N,
    K,
    correction,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    """Single-pass non-inner reduction via per-tile Welford/Chan combine.

    Each [TILE_N, TILE_K] tile is locally centered before merging, so the
    running moments stay cancellation-free for outlier anchors, and the
    input is read exactly once (the two-pass kernel re-reads it). On
    offset-dominated rows the result stays within native-ATen error class;
    on ordinary data it matches the two-pass recipe to the last ulp of the
    fp32 accumulation.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_off = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]
    k_mask = k_off < K
    n_full = (N // TILE_N) * TILE_N

    count_acc = tl.zeros((TILE_K,), dtype=tl.float32)
    mean_acc = tl.zeros((TILE_K,), dtype=tl.float32)
    m2_acc = tl.zeros((TILE_K,), dtype=tl.float32)

    for n0 in range(0, n_full, TILE_N):
        n_off = n0 + tl.arange(0, TILE_N)[:, None]
        mask = (n_off < N) & k_mask
        x = tl.load(X + pid_m * N * K + n_off * K + k_off, mask=mask, other=0.0).to(
            tl.float32
        )
        t_sum = tl.sum(x, axis=0)
        t_mean = t_sum / TILE_N
        t_m2 = tl.sum(
            tl.where(mask, (x - t_mean[None, :]) * (x - t_mean[None, :]), 0.0), axis=0
        )
        # Chan et al. pairwise combine; a zero running count makes the first
        # merge an exact pick-up of the tile moments.
        delta = t_mean - mean_acc
        total = count_acc + TILE_N
        m2_acc += t_m2 + delta * delta * count_acc * (TILE_N / total)
        mean_acc += delta * (TILE_N / total)
        count_acc = total

    if n_full < N:
        n_off = n_full + tl.arange(0, TILE_N)[:, None]
        mask = (n_off < N) & k_mask
        x = tl.load(X + pid_m * N * K + n_off * K + k_off, mask=mask, other=0.0).to(
            tl.float32
        )
        t_count = tl.sum(mask.to(tl.float32), axis=0)
        t_sum = tl.sum(x, axis=0)
        safe = t_count > 0.0
        t_mean = tl.where(safe, t_sum / t_count, 0.0)
        t_m2 = tl.sum(
            tl.where(mask, (x - t_mean[None, :]) * (x - t_mean[None, :]), 0.0), axis=0
        )
        delta = t_mean - mean_acc
        total = count_acc + t_count
        safe_total = total > 0.0
        frac = tl.where(safe_total, t_count / tl.where(safe_total, total, 1.0), 0.0)
        m2_acc += t_m2 + delta * delta * count_acc * frac
        mean_acc += delta * frac
        count_acc = total

    var = m2_acc / tl.maximum(N - correction, 1e-12)
    std_dev = tl.sqrt(tl.maximum(var, 0.0))
    out_off = pid_m * K + k_off
    tl.store(Out + out_off, std_dev.to(Out.dtype.element_ty), mask=k_mask)


@triton.autotune(
    configs=list(filter(_keep_config, runtime.get_tuned_config("naive_reduction"))),
    key=["M", "N"],
)
@triton.jit
def _std_fused_dim_kernel(
    X,
    Out,
    stride_x_row,
    stride_x_col,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_group = tl.program_id(axis=0)
    start_row = pid_group * BLOCK_M
    row_offsets = start_row + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < M

    mean_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    x_row_ptrs = X + row_offsets[:, None] * stride_x_row

    for off in range(0, N, BLOCK_N):
        col_offsets = off + tl.arange(0, BLOCK_N)
        col_mask = col_offsets < N
        x_ptrs = x_row_ptrs + col_offsets[None, :] * stride_x_col
        final_mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(x_ptrs, mask=final_mask, other=0.0)
        mean_acc += x.to(tl.float32)

    mean = tl.sum(mean_acc, axis=1) / N

    var_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        col_offsets = off + tl.arange(0, BLOCK_N)
        col_mask = col_offsets < N
        x_ptrs = x_row_ptrs + col_offsets[None, :] * stride_x_col
        final_mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(x_ptrs, mask=final_mask, other=0.0)
        diff = x.to(tl.float32) - mean[:, None]
        var_acc += tl.where(final_mask, diff * diff, 0.0)

    var = tl.sum(var_acc, axis=1)

    denom = N - correction
    var = var / tl.maximum(denom, 1e-12)
    safe_var = tl.maximum(var, 0.0)
    std_dev = tl.sqrt(safe_var)

    out_ptrs = Out + row_offsets
    tl.store(out_ptrs, std_dev.to(Out.dtype.element_ty), mask=row_mask)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit(do_not_specialize=["correction"])
def _std_dim_kernel_inner(
    Out,
    X,
    M,
    N,
    correction,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0)

    # Pass 1: compute mean
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(X + pid_m * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / N
    else:
        sum_acc = tl.zeros((TILE_N,), dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            x = tl.load(X + pid_m * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
            sum_acc += x
        mean = tl.sum(sum_acc, axis=0) / N

    # Pass 2: compute sum of squared deviations
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(X + pid_m * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        sq_sum = tl.sum(tl.where(mask, diff * diff, 0.0), axis=0)
    else:
        sq_acc = tl.zeros((TILE_N,), dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            x = tl.load(X + pid_m * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
            diff = x - mean
            sq_acc += tl.where(mask, diff * diff, 0.0)
        sq_sum = tl.sum(sq_acc, axis=0)

    denom = N - correction
    var = sq_sum / tl.maximum(denom, 1e-12)
    std_dev = tl.sqrt(tl.maximum(var, 0.0))
    tl.store(Out + pid_m, std_dev.to(Out.dtype.element_ty), mask=pid_m < M)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit(do_not_specialize=["correction"])
def _std_dim_kernel_non_inner(
    Out,
    X,
    M,
    N,
    K,
    correction,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

    # Pass 1: compute mean
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        mask = (n_offsets < N) & (k_offsets < K)
        offsets = pid_m * N * K + n_offsets * K + k_offsets
        x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0, keep_dims=True) / N
    else:
        sum_acc = tl.zeros((TILE_N, TILE_K), dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            mask = (n_offsets < N) & (k_offsets < K)
            offsets = pid_m * N * K + n_offsets * K + k_offsets
            x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
            sum_acc += x
        mean = tl.sum(sum_acc, axis=0, keep_dims=True) / N

    # Pass 2: compute sum of squared deviations
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        mask = (n_offsets < N) & (k_offsets < K)
        offsets = pid_m * N * K + n_offsets * K + k_offsets
        x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        sq_sum = tl.sum(tl.where(mask, diff * diff, 0.0), axis=0, keep_dims=True)
    else:
        sq_acc = tl.zeros((TILE_N, TILE_K), dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            mask = (n_offsets < N) & (k_offsets < K)
            offsets = pid_m * N * K + n_offsets * K + k_offsets
            x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
            diff = x - mean
            sq_acc += tl.where(mask, diff * diff, 0.0)
        sq_sum = tl.sum(sq_acc, axis=0, keep_dims=True)

    denom = N - correction
    var = sq_sum / tl.maximum(denom, 1e-12)
    std_dev = tl.sqrt(tl.maximum(var, 0.0))
    out_offsets = pid_m * K + k_offsets
    tl.store(Out + out_offsets, std_dev.to(Out.dtype.element_ty), mask=k_offsets < K)


def std(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_ENFLAME STD")
    effective_correction = 1.0 if correction is None else float(correction)
    original_shape = x.shape
    input_ndim = x.ndim

    if dim is None:
        N = x.numel()
        if N == 0 or N - effective_correction <= 0:
            return torch.full([], float("nan"), device=x.device, dtype=x.dtype)
        if N == 1 and effective_correction == 0.0:
            out = torch.zeros([], device=x.device, dtype=x.dtype)
            return out.view([1] * input_ndim) if keepdim else out

        out = torch.empty([], device=x.device, dtype=x.dtype)
        denominator = N - effective_correction
        with torch_device_fn.device(x.device):
            if x.is_complex():
                BLOCK_N_MAP = 1024
                BLOCK_NUM = triton.cdiv(N, BLOCK_N_MAP)
                tmp_sum = torch.empty(
                    (BLOCK_NUM,), dtype=torch.float32, device=x.device
                )
                tmp_sum_sq = torch.empty(
                    (BLOCK_NUM,), dtype=torch.float32, device=x.device
                )
                BLOCK_SIZE_REDUCE = 1024
                _std_map_kernel[(BLOCK_NUM,)](
                    x.contiguous(), tmp_sum, tmp_sum_sq, N, BLOCK_N_MAP
                )
                _std_reduce_kernel[(1,)](
                    tmp_sum,
                    tmp_sum_sq,
                    out,
                    N,
                    effective_correction,
                    BLOCK_NUM,
                    BLOCK_SIZE_REDUCE,
                )
            elif N <= _GLOBAL_SINGLE_CTA_LIMIT:
                block_size = min(triton.next_power_of_2(N), _GLOBAL_BLOCK_SIZE)
                _std_global_row_kernel[(1,)](
                    x.contiguous(), out, N, denominator, BLOCK_SIZE=block_size
                )
            else:
                # Bounded grid-stride map keeps grid.x <= 512 for any N (P19).
                max_programs = (
                    _MAX_GLOBAL_WIDE_PROGRAMS
                    if N >= _GLOBAL_WIDE_GRID_MIN_ELEMS
                    else _MAX_GLOBAL_PROGRAMS
                )
                num_programs = min(triton.cdiv(N, _GLOBAL_BLOCK_SIZE), max_programs)
                scratch = torch.empty(
                    (num_programs * 3,), dtype=torch.float32, device=x.device
                )
                _std_global_map_kernel[(num_programs,)](
                    x.contiguous(), scratch, N, BLOCK_SIZE=_GLOBAL_BLOCK_SIZE
                )
                _std_global_finish_kernel[(1,)](
                    scratch,
                    out,
                    N,
                    denominator,
                    NUM_PARTIALS=num_programs,
                    BLOCK_SIZE=triton.next_power_of_2(num_programs),
                )
        return out.view([1] * input_ndim) if keepdim else out

    else:
        if isinstance(dim, int):
            dim_list = [dim]
        else:
            dim_list = list(dim)
        dim_list_normalized = [d % input_ndim for d in dim_list]

        if len(dim_list_normalized) == 1:
            dim0 = dim_list_normalized[0]
            shape = list(original_shape)
            N = shape[dim0]
            M = 1
            for size in shape[:dim0]:
                M *= size
            K = 1
            for size in shape[dim0 + 1 :]:
                K *= size
            shape[dim0] = 1

            if M * N * K > 0 and (N - effective_correction <= 0):
                final_shape = shape if keepdim else shape[:dim0] + shape[dim0 + 1 :]
                return torch.full(
                    final_shape,
                    float("nan"),
                    device=x.device,
                    dtype=x.dtype,
                )

            if N == 1 and effective_correction == 0.0:
                final_shape = shape if keepdim else shape[:dim0] + shape[dim0 + 1 :]
                return torch.zeros(final_shape, device=x.device, dtype=x.dtype)

            out = torch.empty(shape, device=x.device, dtype=x.dtype)
            if M * N * K == 0:
                return out.squeeze(dim=dim0) if not keepdim else out

            x_contiguous = x.contiguous()
            with torch_device_fn.device(x.device):
                if K > 1:
                    tile_k_welford = min(
                        _NON_INNER_WELFORD_TILE_K, triton.next_power_of_2(K)
                    )
                    # The welford kernel addresses elements with int32 offsets,
                    # so its safety domain is the element count; this matches
                    # the offset math of the two-pass kernel it replaces (which
                    # runs unguarded), not the byte-size heuristic.
                    use_welford = (
                        x.dtype in (torch.float16, torch.bfloat16, torch.float32)
                        and N >= _NON_INNER_WELFORD_MIN_N
                        and M <= 65535
                        and x_contiguous.numel() <= _INT32_MAX_ELEMENTS
                        and triton.cdiv(K, tile_k_welford) <= 255
                    )
                    if use_welford:
                        welford_grid = (M, triton.cdiv(K, tile_k_welford), 1)
                        _launch_retry(
                            lambda: _std_dim_kernel_non_inner_welford[welford_grid](
                                out,
                                x_contiguous,
                                M,
                                N,
                                K,
                                effective_correction,
                                TILE_N=_NON_INNER_WELFORD_TILE_N,
                                TILE_K=tile_k_welford,
                                num_warps=4,
                            )
                        )
                    else:
                        grid = lambda META: (M, triton.cdiv(K, META["TILE_K"]), 1)
                        _std_dim_kernel_non_inner[grid](
                            out,
                            x_contiguous,
                            M,
                            N,
                            K,
                            effective_correction,
                        )
                else:
                    use_row_vec = (
                        x.dtype in (torch.float16, torch.bfloat16, torch.float32)
                        and N >= _ROW_VEC_MIN_N
                        and M >= _ROW_VEC_MIN_M
                        and can_use_int32_index(x_contiguous)
                    )
                    if use_row_vec:
                        # Persistent [ROWS, BLOCK_N] row tiles: the per-row-program
                        # layout pays the CTA dispatch floor once per row, which
                        # dominates when M reaches the tens of thousands. Tile
                        # budget ~16K lanes keeps the peak DSM liveness (three
                        # live tiles at fp32) inside the 448KB dual-program
                        # window. BLOCK_N is padded up to one hardware lane
                        # width so narrow rows still fill a vector, and ROWS
                        # shrinks only when M cannot fill the program count.
                        block_n = min(
                            max(triton.next_power_of_2(N), _ROW_VEC_MIN_BLOCK_N),
                            _ROW_VEC_BLOCK_N_CAP,
                        )
                        vec_rows = max(
                            _ROW_VEC_MIN_ROWS,
                            min(
                                _ROW_VEC_LANE_BUDGET // block_n,
                                triton.next_power_of_2(
                                    triton.cdiv(M, _ROW_VEC_MIN_PROGRAMS)
                                ),
                            ),
                        )
                        vec_grid = (
                            min(triton.cdiv(M, vec_rows), _ROW_VEC_MAX_PROGRAMS),
                        )
                        _launch_retry(
                            lambda: _std_row_vec_kernel[vec_grid](
                                x_contiguous,
                                out,
                                M,
                                N,
                                N - effective_correction,
                                ROWS=vec_rows,
                                BLOCK_N=block_n,
                                num_warps=4,
                            )
                        )
                    else:
                        grid = (M, 1, 1)
                        _std_dim_kernel_inner[grid](
                            out,
                            x_contiguous,
                            M,
                            N,
                            effective_correction,
                        )

            return out.squeeze(dim=dim0) if not keepdim else out

        x_view = dim_compress(x, dim_list_normalized)

        N = 1
        for d in dim_list_normalized:
            N *= original_shape[d]
        M = x.numel() // N

        stride_x_row, stride_x_col = N, 1

        output_shape_kept = list(original_shape)
        for d in dim_list_normalized:
            output_shape_kept[d] = 1

        if M * N > 0 and (N - effective_correction <= 0):
            final_shape = [
                s for i, s in enumerate(original_shape) if i not in dim_list_normalized
            ]
            return torch.full(
                final_shape if not keepdim else output_shape_kept,
                float("nan"),
                device=x.device,
                dtype=x.dtype,
            )

        out = torch.empty(output_shape_kept, device=x.device, dtype=x.dtype)
        if M * N == 0:
            return out.squeeze(dim=tuple(dim_list_normalized)) if not keepdim else out

        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

        _std_fused_dim_kernel[grid](
            x_view, out.view(M), stride_x_row, stride_x_col, M, N, effective_correction
        )

        return out.squeeze(dim=tuple(dim_list_normalized)) if not keepdim else out
