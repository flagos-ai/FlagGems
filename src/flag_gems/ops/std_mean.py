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
import struct
import warnings

import torch
import triton
import triton.language as tl
from packaging.version import Version

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.shape_utils import can_use_int32_index

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 1024
_NON_INNER_BLOCK_SIZE = 64
_GLOBAL_SINGLE_CTA_LIMIT = 32768
_MAX_GLOBAL_PROGRAMS = 512
_MAX_STRIDED_NDIM = 8
_ASCEND_TRITON_BEFORE_3_5 = runtime_device.vendor_name == "ascend" and Version(
    str(triton.__version__).split("+", 1)[0]
) < Version("3.5")
_DOF_WARNING = (
    "std_mean(): degrees of freedom is <= 0. Correction should be strictly "
    "less than the reduction factor (input numel divided by output numel)."
)


@triton.jit
def _decode_denominator(
    denominator,
    denominator_low_bits,
    denominator_high_bits,
    COMPUTE_DTYPE: tl.constexpr,
    USE_FP64_DENOMINATOR: tl.constexpr,
):
    """Preserve a host double on JITs that bind Python floats as fp32."""
    if USE_FP64_DENOMINATOR:
        low = denominator_low_bits.to(tl.uint32, bitcast=True).to(tl.uint64)
        high = denominator_high_bits.to(tl.uint32, bitcast=True).to(tl.uint64)
        bits = low | (high << 32)
        return bits.to(tl.float64, bitcast=True).to(COMPUTE_DTYPE)
    return denominator.to(COMPUTE_DTYPE)


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


@libentry()
@triton.jit
def _std_mean_row_kernel(
    inp,
    out_std,
    out_mean,
    M,
    N,
    denominator,
    denominator_low_bits,
    denominator_high_bits,
    USE_FP64_DENOMINATOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce contiguous rows with a stable two-pass central moment."""
    pid = ext.program_id(0)
    compute_dtype = tl.float64 if inp.type.element_ty == tl.float64 else tl.float32
    denominator = _decode_denominator(
        denominator,
        denominator_low_bits,
        denominator_high_bits,
        compute_dtype,
        USE_FP64_DENOMINATOR,
    )
    row_start = pid * N
    shift = tl.load(inp + row_start).to(compute_dtype)

    delta_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    value_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for start in range(0, N, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        values = tl.load(inp + row_start + offsets, mask=mask, other=0.0).to(
            compute_dtype
        )
        delta = values - shift
        delta_sum += tl.where(mask, delta, 0.0)
        value_sum += tl.where(mask, values, 0.0)

    delta_total = tl.sum(delta_sum)
    mean = _stable_mean_from_shift(shift, delta_total, tl.sum(value_sum), N)
    squared_deviation_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for start in range(0, N, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        values = tl.load(inp + row_start + offsets, mask=mask, other=0.0).to(
            compute_dtype
        )
        deviation = values - mean
        squared_deviation_sum += tl.where(mask, deviation * deviation, 0.0)
    m2 = tl.sum(squared_deviation_sum)
    m2 = tl.where(m2 < 0.0, 0.0, m2)
    std = tl.sqrt(m2 / denominator)
    tl.store(out_std + pid, std)
    tl.store(out_mean + pid, mean)


@libentry()
@triton.jit
def _std_mean_ascend_legacy_rows_mean_kernel(
    inp,
    partial_mean,
    out_mean,
    M: tl.constexpr,
    N: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_INT64_INDEX: tl.constexpr,
):
    """Keep one reduction per pass on the legacy Ascend compiler."""
    index_dtype = tl.int64 if USE_INT64_INDEX else tl.int32
    first_row = ext.program_id(0).to(index_dtype) * ROWS_PER_PROGRAM
    last_row = tl.minimum(first_row + ROWS_PER_PROGRAM, M)
    columns = tl.arange(0, BLOCK_N).to(index_dtype)
    for row in range(first_row, last_row):
        total = tl.zeros((BLOCK_N,), tl.float32)
        for start in range(0, N, BLOCK_N):
            offsets = start + columns
            values = tl.load(inp + row * N + offsets, offsets < N, other=0.0)
            total += values.to(tl.float32)
        mean = tl.sum(total) / N
        tl.store(partial_mean + row, mean)
        tl.store(out_mean + row, mean)


@libentry()
@triton.jit
def _std_mean_ascend_legacy_rows_std_kernel(
    inp,
    partial_mean,
    out_std,
    M: tl.constexpr,
    N: tl.constexpr,
    denominator,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_INT64_INDEX: tl.constexpr,
):
    """Accumulate a centered second moment using the unrounded fp32 means."""
    index_dtype = tl.int64 if USE_INT64_INDEX else tl.int32
    first_row = ext.program_id(0).to(index_dtype) * ROWS_PER_PROGRAM
    last_row = tl.minimum(first_row + ROWS_PER_PROGRAM, M)
    columns = tl.arange(0, BLOCK_N).to(index_dtype)
    for row in range(first_row, last_row):
        mean = tl.load(partial_mean + row)
        total = tl.zeros((BLOCK_N,), tl.float32)
        for start in range(0, N, BLOCK_N):
            offsets = start + columns
            mask = offsets < N
            values = tl.load(inp + row * N + offsets, mask, other=0.0).to(tl.float32)
            deviation = values - mean
            total += tl.where(mask, deviation * deviation, 0.0)
        std = tl.sqrt(tl.sum(total) / denominator)
        tl.store(out_std + row, std)


@libentry()
@triton.jit
def _std_mean_ascend_cached_rows_kernel(
    inp,
    out_std,
    out_mean,
    M: tl.constexpr,
    N: tl.constexpr,
    denominator,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    """Reduce row tiles with padding masks only in the statistical arithmetic."""
    program_start = ext.program_id(0).to(tl.int32) * ROWS_PER_PROGRAM
    columns = tl.arange(0, BLOCK_N)
    column_mask = columns[None, :] < N
    for row_offset in range(0, ROWS_PER_PROGRAM, BLOCK_M):
        rows = program_start + row_offset + tl.arange(0, BLOCK_M)
        # Integer bounds let the compiler derive safe rectangular DMA slices.
        row_mask = rows < M
        mask = row_mask[:, None] & column_mask
        values = tl.load(
            inp + rows[:, None] * N + columns[None, :], mask=mask, other=0.0
        ).to(tl.float32)
        # Invalid rows are already zero-filled and their outputs are masked.
        # Keeping only column padding here avoids redundant row-slice copies.
        shift = tl.min(tl.where(column_mask, values, float("inf")), axis=1)
        delta_total = tl.sum(
            tl.where(column_mask, values - shift[:, None], 0.0), axis=1
        )
        mean = _stable_mean_from_shift(shift, delta_total, tl.sum(values, axis=1), N)
        if inp.type.element_ty == tl.float32:
            # Keep the central moment independent of reused reduction buffers.
            values = tl.load(
                inp + rows[:, None] * N + columns[None, :],
                mask=mask,
                other=0.0,
                volatile=True,
            ).to(tl.float32)
        deviation = values - mean[:, None]
        m2 = tl.sum(tl.where(column_mask, deviation * deviation, 0.0), axis=1)
        # A sum of squares is nonnegative; do not turn a NaN into zero.
        std = tl.sqrt(m2 / denominator)
        tl.store(out_std + rows, std, mask=row_mask)
        tl.store(out_mean + rows, mean, mask=row_mask)


@libentry()
@triton.jit
def _std_mean_row_complex_kernel(
    inp,
    out_std,
    out_mean,
    M,
    N,
    denominator,
    denominator_low_bits,
    denominator_high_bits,
    USE_FP64_DENOMINATOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Complex row reduction over an interleaved real/imaginary view."""
    pid = ext.program_id(0)
    compute_dtype = tl.float64 if inp.type.element_ty == tl.float64 else tl.float32
    denominator = _decode_denominator(
        denominator,
        denominator_low_bits,
        denominator_high_bits,
        compute_dtype,
        USE_FP64_DENOMINATOR,
    )
    row_start = pid * N
    shift_real = tl.load(inp + 2 * row_start).to(compute_dtype)
    shift_imag = tl.load(inp + 2 * row_start + 1).to(compute_dtype)

    delta_real_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    delta_imag_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    real_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    imag_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for start in range(0, N, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        storage_offsets = 2 * (row_start + offsets)
        real = tl.load(inp + storage_offsets, mask=mask, other=0.0).to(compute_dtype)
        imag = tl.load(inp + storage_offsets + 1, mask=mask, other=0.0).to(
            compute_dtype
        )
        delta_real = real - shift_real
        delta_imag = imag - shift_imag
        delta_real_sum += tl.where(mask, delta_real, 0.0)
        delta_imag_sum += tl.where(mask, delta_imag, 0.0)
        real_sum += tl.where(mask, real, 0.0)
        imag_sum += tl.where(mask, imag, 0.0)

    delta_real_total = tl.sum(delta_real_sum)
    delta_imag_total = tl.sum(delta_imag_sum)
    mean_real = _stable_mean_from_shift(
        shift_real, delta_real_total, tl.sum(real_sum), N
    )
    mean_imag = _stable_mean_from_shift(
        shift_imag, delta_imag_total, tl.sum(imag_sum), N
    )
    real_m2_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    imag_m2_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for start in range(0, N, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        storage_offsets = 2 * (row_start + offsets)
        real = tl.load(inp + storage_offsets, mask=mask, other=0.0).to(compute_dtype)
        imag = tl.load(inp + storage_offsets + 1, mask=mask, other=0.0).to(
            compute_dtype
        )
        real_deviation = real - mean_real
        imag_deviation = imag - mean_imag
        real_m2_sum += tl.where(mask, real_deviation * real_deviation, 0.0)
        imag_m2_sum += tl.where(mask, imag_deviation * imag_deviation, 0.0)
    real_m2 = tl.sum(real_m2_sum)
    imag_m2 = tl.sum(imag_m2_sum)
    real_m2 = tl.where(real_m2 < 0.0, 0.0, real_m2)
    imag_m2 = tl.where(imag_m2 < 0.0, 0.0, imag_m2)
    std = tl.sqrt(real_m2 / denominator + imag_m2 / denominator)
    tl.store(out_std + pid, std)
    tl.store(out_mean + 2 * pid, mean_real)
    tl.store(out_mean + 2 * pid + 1, mean_imag)


@libentry()
@triton.jit
def _std_mean_non_inner_kernel(
    inp,
    out_std,
    out_mean,
    M,
    N,
    K,
    denominator,
    denominator_low_bits,
    denominator_high_bits,
    USE_FP64_DENOMINATOR: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Reduce one non-inner dimension without materializing a transpose."""
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    compute_dtype = tl.float64 if inp.type.element_ty == tl.float64 else tl.float32
    denominator = _decode_denominator(
        denominator,
        denominator_low_bits,
        denominator_high_bits,
        compute_dtype,
        USE_FP64_DENOMINATOR,
    )
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K
    base = pid_m * N * K + k_offsets
    shift = tl.load(inp + base, mask=k_mask, other=0.0).to(compute_dtype)

    delta_sum = tl.zeros((BLOCK_K,), dtype=compute_dtype)
    value_sum = tl.zeros((BLOCK_K,), dtype=compute_dtype)
    for n in range(0, N):
        values = tl.load(inp + base + n * K, mask=k_mask, other=0.0).to(compute_dtype)
        delta = values - shift
        delta_sum += tl.where(k_mask, delta, 0.0)
        value_sum += tl.where(k_mask, values, 0.0)

    mean = _stable_mean_from_shift(shift, delta_sum, value_sum, N)
    m2 = tl.zeros((BLOCK_K,), dtype=compute_dtype)
    for n in range(0, N):
        values = tl.load(inp + base + n * K, mask=k_mask, other=0.0).to(compute_dtype)
        deviation = values - mean
        m2 += tl.where(k_mask, deviation * deviation, 0.0)
    m2 = tl.where(m2 < 0.0, 0.0, m2)
    std = tl.sqrt(m2 / denominator)
    out_offsets = pid_m * K + k_offsets
    tl.store(out_std + out_offsets, std, mask=k_mask)
    tl.store(out_mean + out_offsets, mean, mask=k_mask)


@libentry()
@triton.jit
def _std_mean_non_inner_complex_kernel(
    inp,
    out_std,
    out_mean,
    M,
    N,
    K,
    denominator,
    denominator_low_bits,
    denominator_high_bits,
    USE_FP64_DENOMINATOR: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Complex non-inner reduction over an interleaved storage view."""
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    compute_dtype = tl.float64 if inp.type.element_ty == tl.float64 else tl.float32
    denominator = _decode_denominator(
        denominator,
        denominator_low_bits,
        denominator_high_bits,
        compute_dtype,
        USE_FP64_DENOMINATOR,
    )
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K
    base = pid_m * N * K + k_offsets
    shift_real = tl.load(inp + 2 * base, mask=k_mask, other=0.0).to(compute_dtype)
    shift_imag = tl.load(inp + 2 * base + 1, mask=k_mask, other=0.0).to(compute_dtype)

    delta_real_sum = tl.zeros((BLOCK_K,), dtype=compute_dtype)
    delta_imag_sum = tl.zeros((BLOCK_K,), dtype=compute_dtype)
    real_sum = tl.zeros((BLOCK_K,), dtype=compute_dtype)
    imag_sum = tl.zeros((BLOCK_K,), dtype=compute_dtype)
    for n in range(0, N):
        offsets = 2 * (base + n * K)
        real = tl.load(inp + offsets, mask=k_mask, other=0.0).to(compute_dtype)
        imag = tl.load(inp + offsets + 1, mask=k_mask, other=0.0).to(compute_dtype)
        delta_real = real - shift_real
        delta_imag = imag - shift_imag
        delta_real_sum += tl.where(k_mask, delta_real, 0.0)
        delta_imag_sum += tl.where(k_mask, delta_imag, 0.0)
        real_sum += tl.where(k_mask, real, 0.0)
        imag_sum += tl.where(k_mask, imag, 0.0)

    mean_real = _stable_mean_from_shift(shift_real, delta_real_sum, real_sum, N)
    mean_imag = _stable_mean_from_shift(shift_imag, delta_imag_sum, imag_sum, N)
    real_m2 = tl.zeros((BLOCK_K,), dtype=compute_dtype)
    imag_m2 = tl.zeros((BLOCK_K,), dtype=compute_dtype)
    for n in range(0, N):
        offsets = 2 * (base + n * K)
        real = tl.load(inp + offsets, mask=k_mask, other=0.0).to(compute_dtype)
        imag = tl.load(inp + offsets + 1, mask=k_mask, other=0.0).to(compute_dtype)
        real_deviation = real - mean_real
        imag_deviation = imag - mean_imag
        real_m2 += tl.where(k_mask, real_deviation * real_deviation, 0.0)
        imag_m2 += tl.where(k_mask, imag_deviation * imag_deviation, 0.0)
    real_m2 = tl.where(real_m2 < 0.0, 0.0, real_m2)
    imag_m2 = tl.where(imag_m2 < 0.0, 0.0, imag_m2)
    std = tl.sqrt(real_m2 / denominator + imag_m2 / denominator)
    out_offsets = pid_m * K + k_offsets
    tl.store(out_std + out_offsets, std, mask=k_mask)
    tl.store(out_mean + 2 * out_offsets, mean_real, mask=k_mask)
    tl.store(
        out_mean + 2 * out_offsets + 1,
        mean_imag,
        mask=k_mask,
    )


@triton.jit
def _unravel_strided_offset(
    linear,
    D0_SIZE: tl.constexpr,
    D1_SIZE: tl.constexpr,
    D2_SIZE: tl.constexpr,
    D3_SIZE: tl.constexpr,
    D4_SIZE: tl.constexpr,
    D5_SIZE: tl.constexpr,
    D6_SIZE: tl.constexpr,
    D7_SIZE: tl.constexpr,
    D0_STRIDE: tl.constexpr,
    D1_STRIDE: tl.constexpr,
    D2_STRIDE: tl.constexpr,
    D3_STRIDE: tl.constexpr,
    D4_STRIDE: tl.constexpr,
    D5_STRIDE: tl.constexpr,
    D6_STRIDE: tl.constexpr,
    D7_STRIDE: tl.constexpr,
    NDIM: tl.constexpr,
    USE_INT64_INDEX: tl.constexpr,
):
    """Unravel an index without constexpr-tuple indexing (unsupported on Hygon)."""
    if USE_INT64_INDEX:
        remaining = linear.to(tl.int64)
    else:
        remaining = linear.to(tl.int32)
    offset = remaining * 0
    if NDIM >= 1:
        offset += (remaining % D0_SIZE) * D0_STRIDE
        remaining //= D0_SIZE
    if NDIM >= 2:
        offset += (remaining % D1_SIZE) * D1_STRIDE
        remaining //= D1_SIZE
    if NDIM >= 3:
        offset += (remaining % D2_SIZE) * D2_STRIDE
        remaining //= D2_SIZE
    if NDIM >= 4:
        offset += (remaining % D3_SIZE) * D3_STRIDE
        remaining //= D3_SIZE
    if NDIM >= 5:
        offset += (remaining % D4_SIZE) * D4_STRIDE
        remaining //= D4_SIZE
    if NDIM >= 6:
        offset += (remaining % D5_SIZE) * D5_STRIDE
        remaining //= D5_SIZE
    if NDIM >= 7:
        offset += (remaining % D6_SIZE) * D6_STRIDE
        remaining //= D6_SIZE
    if NDIM >= 8:
        offset += (remaining % D7_SIZE) * D7_STRIDE
    return offset


@libentry()
@triton.jit
def _std_mean_strided_kernel(
    inp,
    out_std,
    out_mean,
    N,
    denominator,
    denominator_low_bits,
    denominator_high_bits,
    USE_FP64_DENOMINATOR: tl.constexpr,
    USE_INT64_INDEX: tl.constexpr,
    B0_SIZE: tl.constexpr,
    B1_SIZE: tl.constexpr,
    B2_SIZE: tl.constexpr,
    B3_SIZE: tl.constexpr,
    B4_SIZE: tl.constexpr,
    B5_SIZE: tl.constexpr,
    B6_SIZE: tl.constexpr,
    B7_SIZE: tl.constexpr,
    B0_STRIDE: tl.constexpr,
    B1_STRIDE: tl.constexpr,
    B2_STRIDE: tl.constexpr,
    B3_STRIDE: tl.constexpr,
    B4_STRIDE: tl.constexpr,
    B5_STRIDE: tl.constexpr,
    B6_STRIDE: tl.constexpr,
    B7_STRIDE: tl.constexpr,
    BATCH_NDIM: tl.constexpr,
    R0_SIZE: tl.constexpr,
    R1_SIZE: tl.constexpr,
    R2_SIZE: tl.constexpr,
    R3_SIZE: tl.constexpr,
    R4_SIZE: tl.constexpr,
    R5_SIZE: tl.constexpr,
    R6_SIZE: tl.constexpr,
    R7_SIZE: tl.constexpr,
    R0_STRIDE: tl.constexpr,
    R1_STRIDE: tl.constexpr,
    R2_STRIDE: tl.constexpr,
    R3_STRIDE: tl.constexpr,
    R4_STRIDE: tl.constexpr,
    R5_STRIDE: tl.constexpr,
    R6_STRIDE: tl.constexpr,
    R7_STRIDE: tl.constexpr,
    REDUCE_NDIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce arbitrary dimensions directly from the input's strided layout."""
    pid = ext.program_id(0)
    compute_dtype = tl.float64 if inp.type.element_ty == tl.float64 else tl.float32
    denominator = _decode_denominator(
        denominator,
        denominator_low_bits,
        denominator_high_bits,
        compute_dtype,
        USE_FP64_DENOMINATOR,
    )
    base_offset = _unravel_strided_offset(
        pid,
        B0_SIZE,
        B1_SIZE,
        B2_SIZE,
        B3_SIZE,
        B4_SIZE,
        B5_SIZE,
        B6_SIZE,
        B7_SIZE,
        B0_STRIDE,
        B1_STRIDE,
        B2_STRIDE,
        B3_STRIDE,
        B4_STRIDE,
        B5_STRIDE,
        B6_STRIDE,
        B7_STRIDE,
        BATCH_NDIM,
        USE_INT64_INDEX,
    )

    shift = tl.load(inp + base_offset).to(compute_dtype)
    delta_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    value_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for start in range(0, N, BLOCK_SIZE):
        reduction_linear = start + tl.arange(0, BLOCK_SIZE)
        mask = reduction_linear < N
        reduction_offsets = _unravel_strided_offset(
            reduction_linear,
            R0_SIZE,
            R1_SIZE,
            R2_SIZE,
            R3_SIZE,
            R4_SIZE,
            R5_SIZE,
            R6_SIZE,
            R7_SIZE,
            R0_STRIDE,
            R1_STRIDE,
            R2_STRIDE,
            R3_STRIDE,
            R4_STRIDE,
            R5_STRIDE,
            R6_STRIDE,
            R7_STRIDE,
            REDUCE_NDIM,
            USE_INT64_INDEX,
        )
        values = tl.load(
            inp + base_offset + reduction_offsets,
            mask=mask,
            other=shift,
        ).to(compute_dtype)
        delta = values - shift
        delta_sum += tl.where(mask, delta, 0.0)
        value_sum += tl.where(mask, values, 0.0)

    delta_total = tl.sum(delta_sum)
    mean = _stable_mean_from_shift(shift, delta_total, tl.sum(value_sum), N)
    squared_deviation_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for start in range(0, N, BLOCK_SIZE):
        reduction_linear = start + tl.arange(0, BLOCK_SIZE)
        mask = reduction_linear < N
        reduction_offsets = _unravel_strided_offset(
            reduction_linear,
            R0_SIZE,
            R1_SIZE,
            R2_SIZE,
            R3_SIZE,
            R4_SIZE,
            R5_SIZE,
            R6_SIZE,
            R7_SIZE,
            R0_STRIDE,
            R1_STRIDE,
            R2_STRIDE,
            R3_STRIDE,
            R4_STRIDE,
            R5_STRIDE,
            R6_STRIDE,
            R7_STRIDE,
            REDUCE_NDIM,
            USE_INT64_INDEX,
        )
        values = tl.load(
            inp + base_offset + reduction_offsets,
            mask=mask,
            other=mean,
        ).to(compute_dtype)
        deviation = values - mean
        squared_deviation_sum += tl.where(mask, deviation * deviation, 0.0)
    m2 = tl.sum(squared_deviation_sum)
    m2 = tl.where(m2 < 0.0, 0.0, m2)
    tl.store(out_std + pid, tl.sqrt(m2 / denominator))
    tl.store(out_mean + pid, mean)


@libentry()
@triton.jit
def _std_mean_strided_complex_kernel(
    inp,
    out_std,
    out_mean,
    N,
    denominator,
    denominator_low_bits,
    denominator_high_bits,
    USE_FP64_DENOMINATOR: tl.constexpr,
    USE_INT64_INDEX: tl.constexpr,
    B0_SIZE: tl.constexpr,
    B1_SIZE: tl.constexpr,
    B2_SIZE: tl.constexpr,
    B3_SIZE: tl.constexpr,
    B4_SIZE: tl.constexpr,
    B5_SIZE: tl.constexpr,
    B6_SIZE: tl.constexpr,
    B7_SIZE: tl.constexpr,
    B0_STRIDE: tl.constexpr,
    B1_STRIDE: tl.constexpr,
    B2_STRIDE: tl.constexpr,
    B3_STRIDE: tl.constexpr,
    B4_STRIDE: tl.constexpr,
    B5_STRIDE: tl.constexpr,
    B6_STRIDE: tl.constexpr,
    B7_STRIDE: tl.constexpr,
    BATCH_NDIM: tl.constexpr,
    R0_SIZE: tl.constexpr,
    R1_SIZE: tl.constexpr,
    R2_SIZE: tl.constexpr,
    R3_SIZE: tl.constexpr,
    R4_SIZE: tl.constexpr,
    R5_SIZE: tl.constexpr,
    R6_SIZE: tl.constexpr,
    R7_SIZE: tl.constexpr,
    R0_STRIDE: tl.constexpr,
    R1_STRIDE: tl.constexpr,
    R2_STRIDE: tl.constexpr,
    R3_STRIDE: tl.constexpr,
    R4_STRIDE: tl.constexpr,
    R5_STRIDE: tl.constexpr,
    R6_STRIDE: tl.constexpr,
    R7_STRIDE: tl.constexpr,
    REDUCE_NDIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Complex counterpart of :func:`_std_mean_strided_kernel`."""
    pid = ext.program_id(0)
    compute_dtype = tl.float64 if inp.type.element_ty == tl.float64 else tl.float32
    denominator = _decode_denominator(
        denominator,
        denominator_low_bits,
        denominator_high_bits,
        compute_dtype,
        USE_FP64_DENOMINATOR,
    )
    base_offset = _unravel_strided_offset(
        pid,
        B0_SIZE,
        B1_SIZE,
        B2_SIZE,
        B3_SIZE,
        B4_SIZE,
        B5_SIZE,
        B6_SIZE,
        B7_SIZE,
        B0_STRIDE,
        B1_STRIDE,
        B2_STRIDE,
        B3_STRIDE,
        B4_STRIDE,
        B5_STRIDE,
        B6_STRIDE,
        B7_STRIDE,
        BATCH_NDIM,
        USE_INT64_INDEX,
    )

    shift_real = tl.load(inp + 2 * base_offset).to(compute_dtype)
    shift_imag = tl.load(inp + 2 * base_offset + 1).to(compute_dtype)
    delta_real_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    delta_imag_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    real_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    imag_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for start in range(0, N, BLOCK_SIZE):
        reduction_linear = start + tl.arange(0, BLOCK_SIZE)
        mask = reduction_linear < N
        reduction_offsets = _unravel_strided_offset(
            reduction_linear,
            R0_SIZE,
            R1_SIZE,
            R2_SIZE,
            R3_SIZE,
            R4_SIZE,
            R5_SIZE,
            R6_SIZE,
            R7_SIZE,
            R0_STRIDE,
            R1_STRIDE,
            R2_STRIDE,
            R3_STRIDE,
            R4_STRIDE,
            R5_STRIDE,
            R6_STRIDE,
            R7_STRIDE,
            REDUCE_NDIM,
            USE_INT64_INDEX,
        )
        storage_offsets = 2 * (base_offset + reduction_offsets)
        real = tl.load(
            inp + storage_offsets,
            mask=mask,
            other=shift_real,
        ).to(compute_dtype)
        imag = tl.load(
            inp + storage_offsets + 1,
            mask=mask,
            other=shift_imag,
        ).to(compute_dtype)
        delta_real = real - shift_real
        delta_imag = imag - shift_imag
        delta_real_sum += tl.where(mask, delta_real, 0.0)
        delta_imag_sum += tl.where(mask, delta_imag, 0.0)
        real_sum += tl.where(mask, real, 0.0)
        imag_sum += tl.where(mask, imag, 0.0)

    delta_real_total = tl.sum(delta_real_sum)
    delta_imag_total = tl.sum(delta_imag_sum)
    mean_real = _stable_mean_from_shift(
        shift_real, delta_real_total, tl.sum(real_sum), N
    )
    mean_imag = _stable_mean_from_shift(
        shift_imag, delta_imag_total, tl.sum(imag_sum), N
    )
    real_m2_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    imag_m2_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for start in range(0, N, BLOCK_SIZE):
        reduction_linear = start + tl.arange(0, BLOCK_SIZE)
        mask = reduction_linear < N
        reduction_offsets = _unravel_strided_offset(
            reduction_linear,
            R0_SIZE,
            R1_SIZE,
            R2_SIZE,
            R3_SIZE,
            R4_SIZE,
            R5_SIZE,
            R6_SIZE,
            R7_SIZE,
            R0_STRIDE,
            R1_STRIDE,
            R2_STRIDE,
            R3_STRIDE,
            R4_STRIDE,
            R5_STRIDE,
            R6_STRIDE,
            R7_STRIDE,
            REDUCE_NDIM,
            USE_INT64_INDEX,
        )
        storage_offsets = 2 * (base_offset + reduction_offsets)
        real = tl.load(
            inp + storage_offsets,
            mask=mask,
            other=mean_real,
        ).to(compute_dtype)
        imag = tl.load(
            inp + storage_offsets + 1,
            mask=mask,
            other=mean_imag,
        ).to(compute_dtype)
        real_deviation = real - mean_real
        imag_deviation = imag - mean_imag
        real_m2_sum += tl.where(mask, real_deviation * real_deviation, 0.0)
        imag_m2_sum += tl.where(mask, imag_deviation * imag_deviation, 0.0)
    real_m2 = tl.sum(real_m2_sum)
    imag_m2 = tl.sum(imag_m2_sum)
    real_m2 = tl.where(real_m2 < 0.0, 0.0, real_m2)
    imag_m2 = tl.where(imag_m2 < 0.0, 0.0, imag_m2)
    tl.store(
        out_std + pid,
        tl.sqrt(real_m2 / denominator + imag_m2 / denominator),
    )
    tl.store(out_mean + 2 * pid, mean_real)
    tl.store(out_mean + 2 * pid + 1, mean_imag)


@triton.jit
def _welford_combine(mean_x, count_x, m2_x, mean_y, count_y, m2_y):
    count = count_x + count_y
    safe_count = tl.maximum(count, 1.0)
    delta = mean_y - mean_x
    mean = mean_x + delta * count_y / safe_count
    m2 = m2_x + m2_y + delta * delta * count_x * count_y / safe_count
    return mean, count, m2


@libentry()
@triton.jit
def _std_mean_global_welford_map_kernel(
    inp,
    scratch,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Build one-pass Welford partials with the input opmath precision."""
    pid = ext.program_id(0)
    num_programs = tl.num_programs(0)
    compute_dtype = tl.float64 if inp.type.element_ty == tl.float64 else tl.float32
    lane_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    grid_stride = num_programs * BLOCK_SIZE
    mean = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    m2 = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    count = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for start in range(0, N, grid_stride):
        offsets = start + lane_offsets
        mask = offsets < N
        value = tl.load(inp + offsets, mask=mask, other=0.0).to(compute_dtype)
        new_count = count + mask.to(compute_dtype)
        delta = value - mean
        new_mean = mean + tl.where(
            mask,
            delta / tl.maximum(new_count, 1.0),
            0.0,
        )
        m2 += tl.where(mask, delta * (value - new_mean), 0.0)
        mean = new_mean
        count = new_count

    mean, count, m2 = tl.reduce(
        (mean, count, m2),
        axis=0,
        combine_fn=_welford_combine,
    )
    tl.store(scratch + pid, mean)
    tl.store(scratch + num_programs + pid, tl.maximum(m2, 0.0))
    tl.store(scratch + 2 * num_programs + pid, count)


@libentry()
@triton.jit
def _std_mean_global_map_kernel(
    inp,
    scratch,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Build mergeable shifted central moments for a flat real tensor."""
    pid = ext.program_id(0)
    num_programs = tl.num_programs(0)
    compute_dtype = tl.float64 if inp.type.element_ty == tl.float64 else tl.float32
    lane_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    shift = tl.load(inp + pid * BLOCK_SIZE).to(compute_dtype)
    grid_stride = num_programs * BLOCK_SIZE

    delta_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    value_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    count = tl.zeros((), dtype=tl.float32)
    for start in range(0, N, grid_stride):
        offsets = start + lane_offsets
        mask = offsets < N
        values = tl.load(inp + offsets, mask=mask, other=shift).to(compute_dtype)
        delta = values - shift
        delta_sum += tl.where(mask, delta, 0.0)
        value_sum += tl.where(mask, values, 0.0)
        count += tl.sum(mask.to(tl.float32))

    delta_total = tl.sum(delta_sum)
    mean = _stable_mean_from_shift(shift, delta_total, tl.sum(value_sum), count)
    squared_deviation_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for start in range(0, N, grid_stride):
        offsets = start + lane_offsets
        mask = offsets < N
        values = tl.load(inp + offsets, mask=mask, other=mean).to(compute_dtype)
        deviation = values - mean
        squared_deviation_sum += tl.where(mask, deviation * deviation, 0.0)
    m2 = tl.sum(squared_deviation_sum)
    m2 = tl.where(m2 < 0.0, 0.0, m2)
    tl.store(scratch + pid, mean)
    tl.store(scratch + num_programs + pid, m2)
    tl.store(scratch + 2 * num_programs + pid, count)


@libentry()
@triton.jit
def _std_mean_global_ascend_legacy_mean_map_kernel(
    inp,
    partial_means,
    partial_counts,
    N,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Build real partial means with one reduction on old Ascend JITs."""
    pid = ext.program_id(0)
    base = pid * CHUNK_SIZE
    lane_offsets = tl.arange(0, BLOCK_SIZE)
    value_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, CHUNK_SIZE, BLOCK_SIZE):
        local_offsets = start + lane_offsets
        offsets = base + local_offsets
        mask = (local_offsets < CHUNK_SIZE) & (offsets < N)
        values = tl.load(inp + offsets, mask=mask, other=0.0).to(tl.float32)
        value_sum += tl.where(mask, values, 0.0)
    count = tl.minimum(tl.maximum(N - base, 0), CHUNK_SIZE).to(tl.float32)
    mean = tl.sum(value_sum) / count
    tl.store(partial_means + pid, mean)
    tl.store(partial_counts + pid, count)


@libentry()
@triton.jit
def _std_mean_global_ascend_legacy_m2_map_kernel(
    inp,
    partial_means,
    partial_m2s,
    N,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Build centered real partial M2 with one reduction on old Ascend JITs."""
    pid = ext.program_id(0)
    base = pid * CHUNK_SIZE
    lane_offsets = tl.arange(0, BLOCK_SIZE)
    mean = tl.load(partial_means + pid).to(tl.float32)
    squared_deviation_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, CHUNK_SIZE, BLOCK_SIZE):
        local_offsets = start + lane_offsets
        offsets = base + local_offsets
        mask = (local_offsets < CHUNK_SIZE) & (offsets < N)
        values = tl.load(inp + offsets, mask=mask, other=0.0).to(tl.float32)
        deviation = values - mean
        squared_deviation_sum += tl.where(mask, deviation * deviation, 0.0)
    m2 = tl.sum(squared_deviation_sum)
    tl.store(partial_m2s + pid, tl.maximum(m2, 0.0))


@libentry()
@triton.jit
def _std_mean_global_map_complex_kernel(
    inp,
    scratch,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Build mergeable moments for a flat complex tensor."""
    pid = ext.program_id(0)
    num_programs = tl.num_programs(0)
    compute_dtype = tl.float64 if inp.type.element_ty == tl.float64 else tl.float32
    lane_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    first = 2 * pid * BLOCK_SIZE
    shift_real = tl.load(inp + first).to(compute_dtype)
    shift_imag = tl.load(inp + first + 1).to(compute_dtype)
    grid_stride = num_programs * BLOCK_SIZE

    delta_real_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    delta_imag_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    real_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    imag_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    count = tl.zeros((), dtype=tl.float32)
    for start in range(0, N, grid_stride):
        offsets = start + lane_offsets
        mask = offsets < N
        storage_offsets = 2 * offsets
        real = tl.load(inp + storage_offsets, mask=mask, other=shift_real).to(
            compute_dtype
        )
        imag = tl.load(inp + storage_offsets + 1, mask=mask, other=shift_imag).to(
            compute_dtype
        )
        delta_real = real - shift_real
        delta_imag = imag - shift_imag
        delta_real_sum += tl.where(mask, delta_real, 0.0)
        delta_imag_sum += tl.where(mask, delta_imag, 0.0)
        real_sum += tl.where(mask, real, 0.0)
        imag_sum += tl.where(mask, imag, 0.0)
        count += tl.sum(mask.to(tl.float32))

    delta_real_total = tl.sum(delta_real_sum)
    delta_imag_total = tl.sum(delta_imag_sum)
    mean_real = _stable_mean_from_shift(
        shift_real, delta_real_total, tl.sum(real_sum), count
    )
    mean_imag = _stable_mean_from_shift(
        shift_imag, delta_imag_total, tl.sum(imag_sum), count
    )
    real_m2_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    imag_m2_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for start in range(0, N, grid_stride):
        offsets = start + lane_offsets
        mask = offsets < N
        storage_offsets = 2 * offsets
        real = tl.load(inp + storage_offsets, mask=mask, other=mean_real).to(
            compute_dtype
        )
        imag = tl.load(inp + storage_offsets + 1, mask=mask, other=mean_imag).to(
            compute_dtype
        )
        real_deviation = real - mean_real
        imag_deviation = imag - mean_imag
        real_m2_sum += tl.where(mask, real_deviation * real_deviation, 0.0)
        imag_m2_sum += tl.where(mask, imag_deviation * imag_deviation, 0.0)
    real_m2 = tl.sum(real_m2_sum)
    imag_m2 = tl.sum(imag_m2_sum)
    real_m2 = tl.where(real_m2 < 0.0, 0.0, real_m2)
    imag_m2 = tl.where(imag_m2 < 0.0, 0.0, imag_m2)
    tl.store(scratch + pid, mean_real)
    tl.store(scratch + num_programs + pid, mean_imag)
    tl.store(scratch + 2 * num_programs + pid, real_m2)
    tl.store(scratch + 3 * num_programs + pid, imag_m2)
    tl.store(scratch + 4 * num_programs + pid, count)


@libentry()
@triton.jit
def _std_mean_global_finish_kernel(
    scratch,
    out_std,
    out_mean,
    N,
    denominator,
    denominator_low_bits,
    denominator_high_bits,
    USE_FP64_DENOMINATOR: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Merge real partial moments using another shifted-data reduction."""
    compute_dtype = tl.float64 if scratch.type.element_ty == tl.float64 else tl.float32
    denominator = _decode_denominator(
        denominator,
        denominator_low_bits,
        denominator_high_bits,
        compute_dtype,
        USE_FP64_DENOMINATOR,
    )
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_PARTIALS
    means = tl.load(scratch + offsets, mask=mask, other=0.0).to(compute_dtype)
    partial_m2s = tl.load(scratch + NUM_PARTIALS + offsets, mask=mask, other=0.0).to(
        compute_dtype
    )
    counts = tl.load(scratch + 2 * NUM_PARTIALS + offsets, mask=mask, other=0.0).to(
        compute_dtype
    )
    reference = tl.load(scratch).to(compute_dtype)
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
    tl.store(out_std, tl.sqrt(m2 / denominator))
    tl.store(out_mean, mean)


@libentry()
@triton.jit
def _std_mean_global_ascend_legacy_finish_kernel(
    partial_means,
    partial_m2s,
    partial_counts,
    out_std,
    out_mean,
    N,
    denominator,
    denominator_low_bits,
    denominator_high_bits,
    USE_FP64_DENOMINATOR: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Merge real partial moments using another shifted-data reduction."""
    compute_dtype = (
        tl.float64 if partial_means.type.element_ty == tl.float64 else tl.float32
    )
    denominator = _decode_denominator(
        denominator,
        denominator_low_bits,
        denominator_high_bits,
        compute_dtype,
        USE_FP64_DENOMINATOR,
    )
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_PARTIALS
    means = tl.load(partial_means + offsets, mask=mask, other=0.0).to(compute_dtype)
    m2s = tl.load(partial_m2s + offsets, mask=mask, other=0.0).to(compute_dtype)
    counts = tl.load(partial_counts + offsets, mask=mask, other=0.0).to(compute_dtype)
    reference = tl.load(partial_means).to(compute_dtype)
    deltas = means - reference
    weighted_delta = tl.where(mask, counts * deltas, 0.0)
    weighted_values = tl.where(mask, counts * means, 0.0)
    delta_total = tl.sum(weighted_delta)
    mean = _stable_mean_from_shift(reference, delta_total, tl.sum(weighted_values), N)
    m2 = tl.sum(
        tl.where(
            mask,
            m2s + counts * (means - mean) * (means - mean),
            0.0,
        )
    )
    m2 = tl.where(m2 < 0.0, 0.0, m2)
    tl.store(out_std, tl.sqrt(m2 / denominator))
    tl.store(out_mean, mean)


@libentry()
@triton.jit
def _std_mean_global_finish_complex_kernel(
    scratch,
    out_std,
    out_mean,
    N,
    denominator,
    denominator_low_bits,
    denominator_high_bits,
    USE_FP64_DENOMINATOR: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Merge complex partial moments and retain a real-valued standard deviation."""
    compute_dtype = tl.float64 if scratch.type.element_ty == tl.float64 else tl.float32
    denominator = _decode_denominator(
        denominator,
        denominator_low_bits,
        denominator_high_bits,
        compute_dtype,
        USE_FP64_DENOMINATOR,
    )
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_PARTIALS
    means_real = tl.load(scratch + offsets, mask=mask, other=0.0).to(compute_dtype)
    means_imag = tl.load(scratch + NUM_PARTIALS + offsets, mask=mask, other=0.0).to(
        compute_dtype
    )
    partial_real_m2s = tl.load(
        scratch + 2 * NUM_PARTIALS + offsets, mask=mask, other=0.0
    ).to(compute_dtype)
    partial_imag_m2s = tl.load(
        scratch + 3 * NUM_PARTIALS + offsets, mask=mask, other=0.0
    ).to(compute_dtype)
    counts = tl.load(scratch + 4 * NUM_PARTIALS + offsets, mask=mask, other=0.0).to(
        compute_dtype
    )
    reference_real = tl.load(scratch).to(compute_dtype)
    reference_imag = tl.load(scratch + NUM_PARTIALS).to(compute_dtype)
    delta_real = means_real - reference_real
    delta_imag = means_imag - reference_imag
    weighted_delta_real = tl.where(mask, counts * delta_real, 0.0)
    weighted_delta_imag = tl.where(mask, counts * delta_imag, 0.0)
    weighted_real = tl.where(mask, counts * means_real, 0.0)
    weighted_imag = tl.where(mask, counts * means_imag, 0.0)
    delta_real_total = tl.sum(weighted_delta_real)
    delta_imag_total = tl.sum(weighted_delta_imag)
    mean_real = _stable_mean_from_shift(
        reference_real, delta_real_total, tl.sum(weighted_real), N
    )
    mean_imag = _stable_mean_from_shift(
        reference_imag, delta_imag_total, tl.sum(weighted_imag), N
    )
    real_m2 = tl.sum(
        tl.where(
            mask,
            partial_real_m2s
            + counts * (means_real - mean_real) * (means_real - mean_real),
            0.0,
        )
    )
    imag_m2 = tl.sum(
        tl.where(
            mask,
            partial_imag_m2s
            + counts * (means_imag - mean_imag) * (means_imag - mean_imag),
            0.0,
        )
    )
    real_m2 = tl.where(real_m2 < 0.0, 0.0, real_m2)
    imag_m2 = tl.where(imag_m2 < 0.0, 0.0, imag_m2)
    tl.store(
        out_std,
        tl.sqrt(real_m2 / denominator + imag_m2 / denominator),
    )
    tl.store(out_mean, mean_real)
    tl.store(out_mean + 1, mean_imag)


@libentry()
@triton.jit
def _std_mean_nan_kernel(
    out,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    nan = tl.full((BLOCK_SIZE,), float("nan"), tl.float32)
    tl.store(out + offsets, nan, mask=mask)


def _supported_dtypes():
    dtypes = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    }
    complex32 = getattr(torch, "complex32", None)
    if complex32 is not None:
        dtypes.add(complex32)
    return dtypes


def _real_dtype(dtype):
    complex32 = getattr(torch, "complex32", None)
    if complex32 is not None and dtype == complex32:
        return torch.float16
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    return dtype


def _has_names(inp):
    return any(name is not None for name in inp.names)


def _canonical_dim(ndim, dim):
    lower = -1 if ndim == 0 else -ndim
    upper = 0 if ndim == 0 else ndim - 1
    if dim < lower or dim > upper:
        raise IndexError(
            "Dimension out of range (expected to be in range of "
            f"[{lower}, {upper}], but got {dim})"
        )
    return 0 if ndim == 0 else dim % ndim


def _normalize_dims(dim, ndim):
    if dim is None:
        return list(range(ndim))
    raw_dims = [dim] if isinstance(dim, int) else list(dim)
    if not raw_dims:
        return list(range(ndim))
    dims = []
    seen = set()
    for raw_dim in raw_dims:
        normalized = _canonical_dim(ndim, raw_dim)
        if normalized in seen:
            raise RuntimeError(
                f"dim {normalized} appears multiple times in the list of dims"
            )
        seen.add(normalized)
        dims.append(normalized)
    return dims


def _normalize_named_dims(inp, dim):
    raw_dims = [dim] if isinstance(dim, str) else list(dim)
    if not raw_dims:
        return list(range(inp.ndim))
    numeric_dims = []
    for name in raw_dims:
        if name not in inp.names:
            raise RuntimeError(f"Name '{name}' not found in Tensor{list(inp.names)}.")
        numeric_dims.append(inp.names.index(name))
    return _normalize_dims(numeric_dims, inp.ndim)


def _output_shape(shape, dims, keepdim):
    if not shape:
        return ()
    dim_set = set(dims)
    if keepdim:
        return tuple(
            1 if index in dim_set else size for index, size in enumerate(shape)
        )
    return tuple(size for index, size in enumerate(shape) if index not in dim_set)


def _output_names(names, dims, keepdim):
    if names is None:
        return None
    if not names:
        return ()
    if keepdim:
        return tuple(names)
    dim_set = set(dims)
    return tuple(name for index, name in enumerate(names) if index not in dim_set)


def _dtype_name(dtype):
    return str(dtype).removeprefix("torch.")


def _float64_as_int32_bits(value):
    bits = struct.unpack("=Q", struct.pack("=d", value))[0]
    low = bits & 0xFFFFFFFF
    high = bits >> 32
    if low >= 1 << 31:
        low -= 1 << 32
    if high >= 1 << 31:
        high -= 1 << 32
    return low, high


def _prepare_out(out, shape, dtype, device):
    if out.dtype != dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {_dtype_name(dtype)}, "
            f"but got {_dtype_name(out.dtype)} instead"
        )
    if out.device != device:
        raise RuntimeError(
            f"Expected out tensor to have device {device}, but got {out.device} instead"
        )
    if tuple(out.shape) != tuple(shape):
        if out.numel() != 0:
            warnings.warn(
                "An output with one or more elements was resized since it had shape "
                f"{list(out.shape)}, which does not match the required output shape "
                f"{list(shape)}. This behavior is deprecated, and in a future PyTorch "
                "release outputs will not be resized unless they have zero elements. "
                "You can explicitly reuse an out tensor t by resizing it, inplace, to "
                "zero elements with t.resize_(0).",
                UserWarning,
                stacklevel=3,
            )
        out.resize_(shape)
    return out


def _apply_names(tensor, names, is_out):
    if names is None:
        return tensor
    if is_out:
        tensor.rename_(*names)
        return tensor
    return tensor.refine_names(*names)


def _run_global(
    inp,
    out_std,
    out_mean,
    N,
    denominator,
    denominator_low_bits,
    denominator_high_bits,
    use_fp64_denominator,
    is_complex,
):
    if is_complex:
        inp_ptr = torch.view_as_real(inp).reshape(-1)
        mean_ptr = torch.view_as_real(out_mean).reshape(-1)
    else:
        inp_ptr = inp
        mean_ptr = out_mean

    use_ascend_real_split = (
        runtime_device.vendor_name == "ascend"
        and not is_complex
        and inp_ptr.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )
    single_cta_limit = 4096 if use_ascend_real_split else _GLOBAL_SINGLE_CTA_LIMIT
    if N <= single_cta_limit:
        block_size = min(triton.next_power_of_2(N), _BLOCK_SIZE)
        kernel = _std_mean_row_complex_kernel if is_complex else _std_mean_row_kernel
        kernel[(1,)](
            inp_ptr,
            out_std,
            mean_ptr,
            1,
            N,
            denominator,
            denominator_low_bits,
            denominator_high_bits,
            USE_FP64_DENOMINATOR=use_fp64_denominator,
            BLOCK_SIZE=block_size,
        )
        return

    num_programs = min(triton.cdiv(N, _BLOCK_SIZE), _MAX_GLOBAL_PROGRAMS)
    if use_ascend_real_split:
        ascend_block_size = 4096
        num_programs = min(triton.cdiv(N, ascend_block_size), 40)
        chunk_size = (
            triton.cdiv(triton.cdiv(N, num_programs), ascend_block_size)
            * ascend_block_size
        )
        num_programs = triton.cdiv(N, chunk_size)
        partial_means = torch.empty(
            (num_programs,), dtype=torch.float32, device=inp.device
        )
        partial_m2s = torch.empty(
            (num_programs,), dtype=torch.float32, device=inp.device
        )
        partial_counts = torch.empty(
            (num_programs,), dtype=torch.float32, device=inp.device
        )
        _std_mean_global_ascend_legacy_mean_map_kernel[(num_programs,)](
            inp_ptr,
            partial_means,
            partial_counts,
            N,
            CHUNK_SIZE=chunk_size,
            BLOCK_SIZE=ascend_block_size,
        )
        _std_mean_global_ascend_legacy_m2_map_kernel[(num_programs,)](
            inp_ptr,
            partial_means,
            partial_m2s,
            N,
            CHUNK_SIZE=chunk_size,
            BLOCK_SIZE=ascend_block_size,
        )
        _std_mean_global_ascend_legacy_finish_kernel[(1,)](
            partial_means,
            partial_m2s,
            partial_counts,
            out_std,
            mean_ptr,
            N,
            denominator,
            denominator_low_bits,
            denominator_high_bits,
            USE_FP64_DENOMINATOR=use_fp64_denominator,
            NUM_PARTIALS=num_programs,
            BLOCK_SIZE=triton.next_power_of_2(num_programs),
        )
        return

    partial_dtype = torch.float64 if inp_ptr.dtype == torch.float64 else torch.float32
    scratch_fields = 5 if is_complex else 3
    scratch = torch.empty(
        (num_programs * scratch_fields,),
        dtype=partial_dtype,
        device=inp.device,
    )
    if is_complex:
        map_kernel = _std_mean_global_map_complex_kernel
    elif (
        runtime_device.vendor_name == "nvidia" and inp_ptr.dtype == torch.float32
    ) or (runtime_device.vendor_name == "hygon" and inp_ptr.dtype == torch.float64):
        map_kernel = _std_mean_global_welford_map_kernel
    else:
        map_kernel = _std_mean_global_map_kernel
    finish_kernel = (
        _std_mean_global_finish_complex_kernel
        if is_complex
        else _std_mean_global_finish_kernel
    )
    map_kernel[(num_programs,)](
        inp_ptr,
        scratch,
        N,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    finish_kernel[(1,)](
        scratch,
        out_std,
        mean_ptr,
        N,
        denominator,
        denominator_low_bits,
        denominator_high_bits,
        USE_FP64_DENOMINATOR=use_fp64_denominator,
        NUM_PARTIALS=num_programs,
        BLOCK_SIZE=triton.next_power_of_2(num_programs),
    )


def _fixed_strided_metadata(shape, strides):
    """Pad reversed shape/stride metadata for tuple-free Triton specialization."""
    sizes = list(reversed(shape))
    fixed_strides = list(reversed(strides))
    padding = _MAX_STRIDED_NDIM - len(sizes)
    sizes.extend([1] * padding)
    fixed_strides.extend([0] * padding)
    return (*sizes, *fixed_strides)


def _run_reduction(
    inp,
    out_std,
    out_mean,
    dims,
    N,
    denominator,
    denominator_low_bits,
    denominator_high_bits,
    use_fp64_denominator,
    is_complex,
):
    if is_complex:
        mean_ptr = torch.view_as_real(out_mean).reshape(-1)
    else:
        mean_ptr = out_mean

    use_single_dim_fast_path = len(dims) == 1 and inp.ndim > 0
    if use_single_dim_fast_path and is_complex and _ASCEND_TRITON_BEFORE_3_5:
        inner_size = math.prod(tuple(inp.shape)[dims[0] + 1 :])
        use_single_dim_fast_path = inner_size <= 1

    if use_single_dim_fast_path:
        dim = dims[0]
        shape = tuple(inp.shape)
        M = math.prod(shape[:dim])
        K = math.prod(shape[dim + 1 :])
        contiguous = inp.contiguous()
        inp_ptr = (
            torch.view_as_real(contiguous).reshape(-1) if is_complex else contiguous
        )
        if K > 1:
            block_k = min(triton.next_power_of_2(K), _NON_INNER_BLOCK_SIZE)
            kernel = (
                _std_mean_non_inner_complex_kernel
                if is_complex
                else _std_mean_non_inner_kernel
            )
            kernel[(M, triton.cdiv(K, block_k))](
                inp_ptr,
                out_std,
                mean_ptr,
                M,
                N,
                K,
                denominator,
                denominator_low_bits,
                denominator_high_bits,
                USE_FP64_DENOMINATOR=use_fp64_denominator,
                BLOCK_K=block_k,
            )
            return

        use_ascend_legacy_rows = (
            _ASCEND_TRITON_BEFORE_3_5
            and not is_complex
            and inp_ptr.dtype in (torch.float16, torch.bfloat16, torch.float32)
        )
        if use_ascend_legacy_rows:
            properties = triton.runtime.driver.active.utils.get_device_properties(
                inp.device.index
            )
            rows_per_program = triton.cdiv(M, properties["num_vectorcore"])
            programs = triton.cdiv(M, rows_per_program)
            block_n = min(triton.next_power_of_2(N), _BLOCK_SIZE)
            use_int64_index = not can_use_int32_index(inp_ptr)
            partial_mean = torch.empty((M,), dtype=torch.float32, device=inp.device)
            _std_mean_ascend_legacy_rows_mean_kernel[(programs,)](
                inp_ptr,
                partial_mean,
                mean_ptr,
                M,
                N,
                ROWS_PER_PROGRAM=rows_per_program,
                BLOCK_N=block_n,
                USE_INT64_INDEX=use_int64_index,
            )
            _std_mean_ascend_legacy_rows_std_kernel[(programs,)](
                inp_ptr,
                partial_mean,
                out_std,
                M,
                N,
                denominator,
                ROWS_PER_PROGRAM=rows_per_program,
                BLOCK_N=block_n,
                USE_INT64_INDEX=use_int64_index,
            )
            return

        use_ascend_cached_rows = (
            runtime_device.vendor_name == "ascend"
            and not _ASCEND_TRITON_BEFORE_3_5
            and not is_complex
            and inp_ptr.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and N <= 4096
            and can_use_int32_index(inp_ptr)
        )
        if use_ascend_cached_rows:
            block_n = triton.next_power_of_2(N)
            tile_elements = 4096 if inp_ptr.dtype == torch.float32 else 8192
            block_m = min(16, max(1, tile_elements // block_n))
            properties = triton.runtime.driver.active.utils.get_device_properties(
                inp.device.index
            )
            vector_cores = properties["num_vectorcore"]
            rows_per_program = triton.cdiv(M, vector_cores * block_m) * block_m
            programs = triton.cdiv(M, rows_per_program)
            _std_mean_ascend_cached_rows_kernel[(programs,)](
                inp_ptr,
                out_std,
                mean_ptr,
                M,
                N,
                denominator,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                ROWS_PER_PROGRAM=rows_per_program,
            )
            return

        block_size = min(triton.next_power_of_2(N), _BLOCK_SIZE)
        kernel = _std_mean_row_complex_kernel if is_complex else _std_mean_row_kernel
        kernel[(M,)](
            inp_ptr,
            out_std,
            mean_ptr,
            M,
            N,
            denominator,
            denominator_low_bits,
            denominator_high_bits,
            USE_FP64_DENOMINATOR=use_fp64_denominator,
            BLOCK_SIZE=block_size,
        )
        return

    dim_set = set(dims)
    batch_dims = [index for index in range(inp.ndim) if index not in dim_set]
    batch_shape = tuple(inp.shape[index] for index in batch_dims)
    batch_strides = tuple(inp.stride(index) for index in batch_dims)
    reduce_shape = tuple(inp.shape[index] for index in dims)
    reduce_strides = tuple(inp.stride(index) for index in dims)
    M = out_std.numel()
    block_size = min(triton.next_power_of_2(N), _BLOCK_SIZE)

    # Older vendor Triton forks cannot index constexpr tuples. The fixed-slot
    # kernel below is portable through rank 8 per group; retain the established
    # materialization path only for unusually high-rank reductions.
    if len(batch_dims) > _MAX_STRIDED_NDIM or len(dims) > _MAX_STRIDED_NDIM:
        compressed = dim_compress(inp, dims)
        inp_ptr = (
            torch.view_as_real(compressed).reshape(-1) if is_complex else compressed
        )
        kernel = _std_mean_row_complex_kernel if is_complex else _std_mean_row_kernel
        kernel[(M,)](
            inp_ptr,
            out_std,
            mean_ptr,
            M,
            N,
            denominator,
            denominator_low_bits,
            denominator_high_bits,
            USE_FP64_DENOMINATOR=use_fp64_denominator,
            BLOCK_SIZE=block_size,
        )
        return

    # Keep the original storage view here: reshaping a non-contiguous complex
    # tensor would materialize a copy and invalidate the strides passed below.
    inp_ptr = torch.view_as_real(inp) if is_complex else inp
    batch_metadata = _fixed_strided_metadata(batch_shape, batch_strides)
    reduce_metadata = _fixed_strided_metadata(reduce_shape, reduce_strides)
    use_int64_index = not can_use_int32_index(inp)
    kernel = (
        _std_mean_strided_complex_kernel if is_complex else _std_mean_strided_kernel
    )
    kernel[(M,)](
        inp_ptr,
        out_std,
        mean_ptr,
        N,
        denominator,
        denominator_low_bits,
        denominator_high_bits,
        use_fp64_denominator,
        use_int64_index,
        *batch_metadata,
        len(batch_dims),
        *reduce_metadata,
        len(dims),
        BLOCK_SIZE=block_size,
    )


def _std_mean_impl(
    inp,
    dim,
    correction,
    keepdim,
    *,
    named_dim=False,
    out0=None,
    out1=None,
):
    if inp.dtype not in _supported_dtypes():
        raise RuntimeError("std_mean only support floating point and complex dtypes")

    effective_correction = 1.0 if correction is None else float(correction)
    names = tuple(inp.names) if _has_names(inp) else None
    dims = (
        _normalize_named_dims(inp, dim) if named_dim else _normalize_dims(dim, inp.ndim)
    )
    shape = tuple(inp.shape)
    output_shape = _output_shape(shape, dims, keepdim)
    output_names = _output_names(names, dims, keepdim)
    dim_set = set(dims)
    N = 1 if inp.ndim == 0 else math.prod(shape[index] for index in dims)
    M = (
        1
        if inp.ndim == 0
        else math.prod(size for index, size in enumerate(shape) if index not in dim_set)
    )
    is_complex = inp.is_complex()
    std_dtype = _real_dtype(inp.dtype)

    user_out = out0 is not None or out1 is not None
    if user_out and (out0 is None or out1 is None):
        raise RuntimeError("std_mean.correction_out requires both out0 and out1")
    if user_out:
        out0 = _prepare_out(out0, output_shape, std_dtype, inp.device)
        out1 = _prepare_out(out1, output_shape, inp.dtype, inp.device)

    direct_out = user_out and out0.is_contiguous() and out1.is_contiguous()
    out_std = (
        out0
        if direct_out
        else torch.empty(output_shape, dtype=std_dtype, device=inp.device)
    )
    out_mean = (
        out1
        if direct_out
        else torch.empty(output_shape, dtype=inp.dtype, device=inp.device)
    )

    # Native uses a zero reduction factor when there are no output elements.
    if float(N if M else 0) - effective_correction <= 0:
        warnings.warn(_DOF_WARNING, UserWarning, stacklevel=3)
    denominator = float(N) - effective_correction if effective_correction < N else 0.0
    use_fp64_denominator = std_dtype == torch.float64
    denominator_low_bits, denominator_high_bits = (
        _float64_as_int32_bits(denominator) if use_fp64_denominator else (0, 0)
    )

    work = inp.rename(None) if names is not None else inp
    with torch_device_fn.device(inp.device):
        if M == 0:
            pass
        elif N == 0:
            mean_ptr = (
                torch.view_as_real(out_mean).reshape(-1) if is_complex else out_mean
            )
            _std_mean_nan_kernel[(triton.cdiv(M, _BLOCK_SIZE),)](
                out_std, M, BLOCK_SIZE=_BLOCK_SIZE
            )
            mean_elements = M * (2 if is_complex else 1)
            _std_mean_nan_kernel[(triton.cdiv(mean_elements, _BLOCK_SIZE),)](
                mean_ptr, mean_elements, BLOCK_SIZE=_BLOCK_SIZE
            )
        elif inp.ndim == 0 or len(dims) == inp.ndim:
            _run_global(
                work.contiguous(),
                out_std,
                out_mean,
                N,
                denominator,
                denominator_low_bits,
                denominator_high_bits,
                use_fp64_denominator,
                is_complex,
            )
        else:
            _run_reduction(
                work,
                out_std,
                out_mean,
                dims,
                N,
                denominator,
                denominator_low_bits,
                denominator_high_bits,
                use_fp64_denominator,
                is_complex,
            )

    if user_out and not direct_out:
        out0.copy_(out_std)
        out1.copy_(out_mean)
        out_std, out_mean = out0, out1

    out_std = _apply_names(out_std, output_names, user_out)
    out_mean = _apply_names(out_mean, output_names, user_out)
    return out_std, out_mean


def std_mean(inp, unbiased=True):
    logger.debug("GEMS STD_MEAN")
    return _std_mean_impl(inp, None, 1.0 if unbiased else 0.0, False)


def std_mean_dim(inp, dim, unbiased=True, keepdim=False):
    logger.debug("GEMS STD_MEAN.DIM")
    return _std_mean_impl(inp, dim, 1.0 if unbiased else 0.0, keepdim)


def std_mean_correction(inp, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS STD_MEAN.CORRECTION")
    return _std_mean_impl(inp, dim, correction, keepdim)


def std_mean_names_dim(inp, dim, unbiased=True, keepdim=False):
    logger.debug("GEMS STD_MEAN.NAMES_DIM")
    return _std_mean_impl(
        inp,
        dim,
        1.0 if unbiased else 0.0,
        keepdim,
        named_dim=True,
    )


def std_mean_correction_names(inp, dim, *, correction=None, keepdim=False):
    logger.debug("GEMS STD_MEAN.CORRECTION_NAMES")
    return _std_mean_impl(
        inp,
        dim,
        correction,
        keepdim,
        named_dim=True,
    )


def std_mean_correction_out(
    inp,
    dim=None,
    *,
    correction=None,
    keepdim=False,
    out0,
    out1,
):
    logger.debug("GEMS STD_MEAN.CORRECTION_OUT")
    return _std_mean_impl(
        inp,
        dim,
        correction,
        keepdim,
        out0=out0,
        out1=out1,
    )
