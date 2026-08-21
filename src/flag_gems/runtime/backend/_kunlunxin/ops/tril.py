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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _tril_tile_kernel(
    in_ptr,
    out_ptr,
    diag: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (offs_m < M) & (offs_n < N)
    keep = offs_n <= (offs_m + diag)

    in_ptr += pid_b * (M * N) + offs_m * N
    out_ptr += pid_b * (M * N) + offs_m * N

    x = tl.load(in_ptr + offs_n, mask=mask, other=0.0)
    result = tl.where(keep, x, 0.0)
    tl.store(out_ptr + offs_n, result, mask=mask)


@triton.jit
def _tril_rows_kernel(
    in_ptr,
    out_ptr,
    diag: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = offs_m < M
    in_ptr += pid_b * (M * N) + offs_m * N
    out_ptr += pid_b * (M * N) + offs_m * N

    for col_start in range(0, N, BLOCK_N):
        offs_n = col_start + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask & (offs_n < N)
        keep = offs_n <= (offs_m + diag)
        x = tl.load(in_ptr + offs_n, mask=mask, other=0.0)
        result = tl.where(keep, x, 0.0)
        tl.store(out_ptr + offs_n, result, mask=mask)


@triton.jit
def _tril_flat_kernel(
    in_ptr,
    out_ptr,
    total,
    diag,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    matrix_offsets = offsets % (M * N)
    rows = matrix_offsets // N
    cols = matrix_offsets - rows * N
    keep = cols <= rows + diag

    x = tl.load(in_ptr + offsets, mask=mask & keep, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def _tril_exact_row_kernel(
    in_ptr,
    out_ptr,
    diag,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_n = tl.arange(0, BLOCK_N)
    idxs = pid_b * (M * N) + pid_m * N + offs_n
    keep = offs_n <= pid_m + diag
    x = tl.load(in_ptr + idxs)
    result = tl.where(keep, x, 0.0)
    tl.store(out_ptr + idxs, result)


@triton.jit
def _tril_exact_diag0_tile_kernel(
    in_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (offs_m < M) & (offs_n < N)
    keep = offs_n <= offs_m
    offsets = pid_b * (M * N) + offs_m * N + offs_n
    x = tl.load(in_ptr + offsets, mask=mask & keep, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)


@libentry()
@triton.jit
def _tril_flat_inplace_kernel(
    ptr,
    active_total,
    MN,
    diag,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # In-place tril_ over a contiguous top-row prefix of one matrix.
    #
    # The old 2D-tile kernel (`offs_m * N + offs_n` addressing) is NOT proven
    # contiguous by XPU OffsetAnalysis and degrades to discrete access
    # (~1-3 GB/s, e.g. [4096,4096] took ~14ms, [10000,65536] ~543ms). The 1D-flat
    # form (scalar-base + stride-1 arange) is provably contiguous -> block DMA.
    # Same win as the triu.py rewrite (~10x on large shapes).
    #
    # pid_b pre-offsets the base pointer by pid_b * MN (a scalar), so each matrix
    # in a batch is handled by its own grid column while the inner offsets stay a
    # stride-1 arange. Only the first `active_total = active_rows * N` elements of
    # each matrix are visited: rows at/below the diagonal are fully kept and never
    # touched (true in-place). Offsets stay within [0, MN) so `off // N` is exact
    # even for the batched case (no `% MN` needed).
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    base = pid_b * MN

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < active_total
    rows = offsets // N
    cols = offsets - rows * N
    keep = cols <= rows + diag

    x = tl.load(ptr + base + offsets, mask=mask, other=0.0)
    y = tl.where(keep, x, 0.0)
    tl.store(ptr + base + offsets, y, mask=mask)


@triton.jit
def _tril_inplace_zero_strided_tile_kernel(
    ptr,
    diag: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B3: tl.constexpr,
    B4: tl.constexpr,
    B5: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    S4: tl.constexpr,
    S5: tl.constexpr,
    STRIDE_M: tl.constexpr,
    STRIDE_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    b = pid_b
    i5 = b % B5
    b = b // B5
    i4 = b % B4
    b = b // B4
    i3 = b % B3
    b = b // B3
    i2 = b % B2
    b = b // B2
    i1 = b % B1
    i0 = b // B1
    batch_offset = i0 * S0 + i1 * S1 + i2 * S2 + i3 * S3 + i4 * S4 + i5 * S5

    row = pid_m
    first_zero_col = tl.maximum(row + diag + 1, 0)
    offs_n = first_zero_col + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N
    ptr += batch_offset + row * STRIDE_M
    tl.store(ptr + offs_n * STRIDE_N, 0.0, mask=mask)


@libentry()
@triton.jit
def _tril_strided_out_tile_kernel(
    in_ptr,
    out_ptr,
    diag,
    M,
    N,
    B0,
    B1,
    B2,
    B3,
    B4,
    B5,
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    STRIDE_M,
    STRIDE_N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    b = pid_b
    i5 = b % B5
    b = b // B5
    i4 = b % B4
    b = b // B4
    i3 = b % B3
    b = b // B3
    i2 = b % B2
    b = b // B2
    i1 = b % B1
    i0 = b // B1
    out_batch_offset = i0 * S0 + i1 * S1 + i2 * S2 + i3 * S3 + i4 * S4 + i5 * S5

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (offs_m < M) & (offs_n < N)
    keep = offs_n <= (offs_m + diag)

    in_ptr += pid_b * (M * N) + offs_m * N
    out_ptr += out_batch_offset + offs_m * STRIDE_M

    x = tl.load(in_ptr + offs_n, mask=mask, other=0.0)
    result = tl.where(keep, x, 0.0)
    tl.store(out_ptr + offs_n * STRIDE_N, result, mask=mask)


# ---------------------------------------------------------------------------
# Flat/per-row out-of-place kernels (performance paths).
#
# On this XPU/triton, 2D-tiled kernels (`offs_m * N + offs_n` indexing) are not
# proven contiguous by OffsetAnalysis and degrade to discrete access (1-3 GB/s,
# e.g. [1024,1024] fp16 took ~3ms, [64,512,512] ~17ms, [100,65536,100] ~396ms).
# The winning primitive is the 1D-flat kernel (scalar base + stride-1 arange ->
# block DMA) with per-row recovery via integer divide, plus a per-row kernel
# for wide N that drops the per-element div/mod. NEED_MASK is a constexpr so
# always-true masks vanish (masked-memory path is slow on this XPU). Same
# pattern as triu.py, which PASSed on XPU 5.
#
# Native `_copy_from` (aten::_copy_from is NOT registered by flag_gems -> always
# dispatches to the vendor kernel) is used for the all-kept bottom band and the
# keep-everything edge case: under use_gems, `copy_(input)` redispatch through
# the gems copy_ kernel, which is ~1400x slower than the vendor copy
# ([10000,65536] fp16 1.4ms -> ~1.96s), and `zero_` (= gems memset) is only
# competitive when full > 1M elements (heavy fixed ~77us below that).
# ---------------------------------------------------------------------------


@triton.jit
def _tril_flat2d_kernel(
    in_ptr,
    out_ptr,
    total,
    diag,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Single matrix (or a contiguous top-row prefix of one): no `% MN`.
    # Offsets stay in [0, M*N) so row = offset // N is exact.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rows = offsets // N
    cols = offsets - rows * N
    keep = cols <= rows + diag
    if NEED_MASK:
        mask = offsets < total
        x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
        y = tl.where(keep, x, 0.0)
        tl.store(out_ptr + offsets, y, mask=mask)
    else:
        x = tl.load(in_ptr + offsets)
        y = tl.where(keep, x, 0.0)
        tl.store(out_ptr + offsets, y)


@triton.jit
def _tril_flat_batched_kernel(
    in_ptr,
    out_ptr,
    total,
    diag,
    MN,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Many small matrices: one pass with `% MN` folding the flat offset into
    # one matrix; preferred over the per-matrix 2D grid when MN is tiny
    # (else the 2D grid is launch-bound on this XPU).
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    matrix_offsets = offsets % MN
    rows = matrix_offsets // N
    cols = matrix_offsets - rows * N
    keep = cols <= rows + diag
    if NEED_MASK:
        mask = offsets < total
        x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
        y = tl.where(keep, x, 0.0)
        tl.store(out_ptr + offsets, y, mask=mask)
    else:
        x = tl.load(in_ptr + offsets)
        y = tl.where(keep, x, 0.0)
        tl.store(out_ptr + offsets, y)


@triton.jit
def _tril_flat_batchgrid_kernel(
    in_ptr,
    out_ptr,
    diag,
    N: tl.constexpr,
    MN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # grid = (tiles_per_matrix, batch). pid_b pre-offsets the base pointer by
    # pid_b * MN (scalar), inner offsets stay a stride-1 arange and `N` is
    # constexpr. For large matrices this beats the `% MN` variant (runtime
    # division per element).
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rows = offsets // N
    cols = offsets - rows * N
    keep = cols <= rows + diag
    base = pid_b * MN
    if NEED_MASK:
        mask = offsets < MN
        x = tl.load(in_ptr + base + offsets, mask=mask, other=0.0)
        y = tl.where(keep, x, 0.0)
        tl.store(out_ptr + base + offsets, y, mask=mask)
    else:
        x = tl.load(in_ptr + base + offsets)
        y = tl.where(keep, x, 0.0)
        tl.store(out_ptr + base + offsets, y)


@triton.jit
def _tril_row2d_kernel(
    in_ptr,
    out_ptr,
    M,
    N,
    diag,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # One program per row (grid = M*BATCH for the full matrix, or `num_rows`
    # band rows for the band prefix). `row = pid % M` is one mod PER PROGRAM;
    # each row streams its N columns as contiguous BLOCK_N chunks (block DMA).
    pid = tl.program_id(0)
    row = pid % M
    base = pid * N
    for c0 in range(0, N, BLOCK_N):
        cols = c0 + tl.arange(0, BLOCK_N)
        keep = cols <= row + diag
        if NEED_MASK:
            m = cols < N
            x = tl.load(in_ptr + base + cols, mask=m, other=0.0)
            tl.store(out_ptr + base + cols, tl.where(keep, x, 0.0), mask=m)
        else:
            x = tl.load(in_ptr + base + cols)
            tl.store(out_ptr + base + cols, tl.where(keep, x, 0.0))


@triton.jit
def _tril_zero_flat_kernel(
    ptr,
    total,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Store-only memset for small outputs: GEMS zero_() has a heavy fixed cost
    # (~77us) even for tiny tensors, while a single small flat store launch is
    # ~25us. For >1M elements zero_() wins again (bulk vendor memset).
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offsets < total
        tl.store(ptr + offsets, 0.0, mask=mask)
    else:
        tl.store(ptr + offsets, 0.0)


_BLOCK_SIZE = 16384
_ROW_N_THRESHOLD = 2048
# Above this width the pre-existing `_launch_flat` (BLOCK 4096 / 4 warps) beats
# both the flat kernel and the per-row kernel on this XPU (e.g. [10000,65536]).
_FLAT_WIDE_N = 8192
_SMALL_TOTAL_ZERO = 1 << 20
_BAND_MIN_TOTAL = 1 << 20


def _vendor_copy_from(src: torch.Tensor, dst: torch.Tensor):
    # aten::_copy_from is not registered by flag_gems -> dispatches straight to
    # the vendor native copy (fast), unlike copy_() which redispatch to the
    # gems kernel under use_gems (catastrophically slower on this XPU).
    torch.ops.aten._copy_from(src, dst)
    return dst


def _launch_v2_flat(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    total: int = None,
    block_size: int = _BLOCK_SIZE,
    num_warps: int = 8,
):
    if total is None:
        total = input.numel()
    grid = (triton.cdiv(total, block_size),)
    need_mask = total % block_size != 0
    with torch_device_fn.device(input.device):
        _tril_flat2d_kernel[grid](
            input,
            out,
            total,
            int(diagonal),
            input.shape[-1],
            block_size,
            need_mask,
            num_warps=num_warps,
        )
    return out


def _launch_v2_flat_batched(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    block_size: int = _BLOCK_SIZE,
    num_warps: int = 8,
):
    total = input.numel()
    M, N = input.shape[-2:]
    MN = M * N
    grid = (triton.cdiv(total, block_size),)
    need_mask = total % block_size != 0
    with torch_device_fn.device(input.device):
        _tril_flat_batched_kernel[grid](
            input,
            out,
            total,
            int(diagonal),
            MN,
            N,
            block_size,
            need_mask,
            num_warps=num_warps,
        )
    return out


def _launch_v2_flat_batchgrid(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    block_size: int = _BLOCK_SIZE,
    num_warps: int = 8,
):
    M, N = input.shape[-2:]
    MN = M * N
    batch = input.numel() // MN
    tiles = triton.cdiv(MN, block_size)
    need_mask = MN % block_size != 0
    with torch_device_fn.device(input.device):
        _tril_flat_batchgrid_kernel[(tiles, batch)](
            input,
            out,
            int(diagonal),
            N,
            MN,
            block_size,
            need_mask,
            num_warps=num_warps,
        )
    return out


def _launch_v2_rows(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    num_rows: int,
    num_warps: int = 4,
):
    # Per-row kernel; `num_rows` rows are covered (full matrix or band prefix).
    M, N = input.shape[-2:]
    block_n = min(triton.next_power_of_2(N), _BLOCK_SIZE)
    need_mask = N % block_n != 0
    with torch_device_fn.device(input.device):
        _tril_row2d_kernel[(num_rows,)](
            input,
            out,
            M,
            N,
            int(diagonal),
            block_n,
            need_mask,
            num_warps=num_warps,
        )


def _launch_v2_zero(
    out: torch.Tensor,
    block_size: int = _BLOCK_SIZE,
    num_warps: int = 8,
):
    total = out.numel()
    grid = (triton.cdiv(total, block_size),)
    need_mask = total % block_size != 0
    with torch_device_fn.device(out.device):
        _tril_zero_flat_kernel[grid](
            out,
            total,
            block_size,
            need_mask,
            num_warps=num_warps,
        )


def _launch_v2_band(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    band_lo: int,
):
    # Rows [band_lo, M) are entirely at/below the diagonal -> pure copy via
    # the native vendor path. Only the band prefix [0, band_lo*N) needs the
    # tril kernel.
    N = input.shape[-1]
    total = band_lo * N
    if total > 0:
        if N >= _ROW_N_THRESHOLD:
            _launch_v2_rows(input, out, diagonal, num_rows=band_lo)
        else:
            _launch_v2_flat(input, out, diagonal, total)
    if band_lo < input.shape[-2]:
        _vendor_copy_from(input[band_lo:], out[band_lo:])


def _check_input(input: torch.Tensor):
    if input.dim() < 2:
        raise RuntimeError("tril: input tensor must have at least 2 dimensions")


def _empty_contiguous_like(input: torch.Tensor):
    if input.is_contiguous():
        return torch.empty_like(input)
    return torch.empty_like(input, memory_format=torch.contiguous_format)


def _zero_out(out: torch.Tensor):
    if out.numel() == 0:
        return out
    if out.is_contiguous():
        return out.zero_()
    return out.fill_(0)


def _is_power_of_2(value: int):
    return value > 0 and (value & (value - 1)) == 0


def _has_internal_overlap_from_strides(tensor: torch.Tensor):
    span = 1
    strides_and_sizes = sorted(
        (stride, size)
        for size, stride in zip(tensor.shape, tensor.stride())
        if size > 1
    )
    for stride, size in strides_and_sizes:
        if stride < span:
            return True
        span += stride * (size - 1)
    return False


def _tensors_overlap(left: torch.Tensor, right: torch.Tensor):
    try:
        return torch._C._overlaps(left, right)
    except AttributeError:
        return True


def _can_use_strided_out_kernel(input: torch.Tensor, out: torch.Tensor):
    if out.is_contiguous() or out.numel() == 0:
        return False
    if out.dim() - 2 > 6:
        return False
    if _has_internal_overlap_from_strides(out):
        return False
    if input.is_contiguous() and _tensors_overlap(input, out):
        return False
    return True


_WIDE_EXACT_ROW_MIN_N = 2048
_WIDE_EXACT_ROW_MAX_N = 8192
_WIDE_EXACT_ROW_MIN_ROWS = 256
_WIDE_EXACT_ROW_ALWAYS_ROW_M = 512
_TINY_BATCHED_TILE_MIN_BATCH = 128


def _use_wide_exact_row(M: int, N: int, batch: int):
    # One exact-row program covers one matrix row with BLOCK_N == N.  Use it for
    # wide power-of-two rows where it avoids the flat kernel's div/mod indexing,
    # but require enough row programs to keep occupancy reasonable.
    if N < _WIDE_EXACT_ROW_MIN_N or N > _WIDE_EXACT_ROW_MAX_N or not _is_power_of_2(N):
        return False

    rows = M * batch
    if M >= _WIDE_EXACT_ROW_ALWAYS_ROW_M:
        return True
    return N <= 4096 and rows >= _WIDE_EXACT_ROW_MIN_ROWS


def _use_tiny_batched_tile(M: int, N: int, batch: int):
    return batch >= _TINY_BATCHED_TILE_MIN_BATCH and M <= 32 and N <= 32


def _wide_exact_row_warps(N: int):
    if N <= 4096:
        return 2
    return 4


def _launch_tile(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    block_m: int = 32,
    block_n: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
):
    M, N = input.shape[-2:]
    total = input.numel()
    if total == 0:
        return out

    batch = total // (M * N)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n), batch)
    with torch_device_fn.device(input.device):
        _tril_tile_kernel[grid](
            input,
            out,
            int(diagonal),
            M,
            N,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


def _launch_flat(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    block_size: int = 1024,
    num_warps: int = 4,
    num_stages: int = 2,
):
    M, N = input.shape[-2:]
    total = input.numel()
    if total == 0:
        return out

    grid = (triton.cdiv(total, block_size),)
    with torch_device_fn.device(input.device):
        _tril_flat_kernel[grid](
            input,
            out,
            total,
            int(diagonal),
            M,
            N,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


def _launch_rows(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    block_m: int = 32,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
):
    M, N = input.shape[-2:]
    total = input.numel()
    if total == 0:
        return out

    batch = total // (M * N)
    grid = (triton.cdiv(M, block_m), batch)
    with torch_device_fn.device(input.device):
        _tril_rows_kernel[grid](
            input,
            out,
            int(diagonal),
            M,
            N,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


def _launch_exact_row(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    num_warps: int = 4,
    num_stages: int = 2,
):
    M, N = input.shape[-2:]
    total = input.numel()
    if total == 0:
        return out

    batch = total // (M * N)
    grid = (M, batch)
    with torch_device_fn.device(input.device):
        _tril_exact_row_kernel[grid](
            input,
            out,
            int(diagonal),
            M,
            N,
            BLOCK_N=N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


def _launch_exact_diag0_tile(
    input: torch.Tensor,
    out: torch.Tensor,
    block_m: int,
    block_n: int,
    num_warps: int = 4,
    num_stages: int = 2,
):
    M, N = input.shape[-2:]
    total = input.numel()
    if total == 0:
        return out

    batch = total // (M * N)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n), batch)
    with torch_device_fn.device(input.device):
        _tril_exact_diag0_tile_kernel[grid](
            input,
            out,
            M,
            N,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


_INPLACE_FLAT_BLOCK = 8192


def _launch_tril_inplace_contiguous(
    input: torch.Tensor,
    diagonal: int,
    block_size: int = _INPLACE_FLAT_BLOCK,
    num_warps: int = 8,
    num_stages: int = 2,
):
    M, N = input.shape[-2:]
    if input.numel() == 0:
        return input

    # Rows [active_rows, M) sit entirely at/below the diagonal -> fully kept,
    # nothing to zero. Only the first `active_rows` rows of each matrix contain
    # strict-upper elements that must be zeroed.
    active_rows = min(M, max(0, N - 1 - diagonal))
    if active_rows == 0:
        return input

    MN = M * N
    active_total = active_rows * N
    batch = input.numel() // MN

    grid = (triton.cdiv(active_total, block_size), batch)
    with torch_device_fn.device(input.device):
        _tril_flat_inplace_kernel[grid](
            input,
            active_total,
            MN,
            int(diagonal),
            N,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return input


def _launch_tril_inplace_strided(
    input: torch.Tensor,
    diagonal: int,
    block_m: int = 1,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
):
    M, N = input.shape[-2:]
    if input.numel() == 0:
        return input

    active_rows = min(M, max(0, N - 1 - diagonal))
    if active_rows == 0:
        return input

    batch_shape = list(input.shape[:-2])
    batch_strides = list(input.stride()[:-2])
    batch = 1
    for size in batch_shape:
        batch *= size

    if len(batch_shape) > 6:
        tmp = _empty_contiguous_like(input)
        _launch_tril(input, tmp, diagonal)
        input.copy_(tmp)
        return input

    batch_shape.extend([1] * (6 - len(batch_shape)))
    batch_strides.extend([0] * (6 - len(batch_strides)))
    stride_m, stride_n = input.stride()[-2:]

    grid = (triton.cdiv(active_rows, block_m), triton.cdiv(N, block_n), batch)
    with torch_device_fn.device(input.device):
        _tril_inplace_zero_strided_tile_kernel[grid](
            input,
            int(diagonal),
            M,
            N,
            B0=batch_shape[0],
            B1=batch_shape[1],
            B2=batch_shape[2],
            B3=batch_shape[3],
            B4=batch_shape[4],
            B5=batch_shape[5],
            S0=batch_strides[0],
            S1=batch_strides[1],
            S2=batch_strides[2],
            S3=batch_strides[3],
            S4=batch_strides[4],
            S5=batch_strides[5],
            STRIDE_M=stride_m,
            STRIDE_N=stride_n,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return input


def _launch_tril_strided_out(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    block_m: int = 32,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
):
    M, N = input.shape[-2:]
    if input.numel() == 0:
        return out

    input_to_use = input if input.is_contiguous() else input.contiguous()
    batch_shape = list(out.shape[:-2])
    batch_strides = list(out.stride()[:-2])
    batch = 1
    for size in batch_shape:
        batch *= size

    batch_shape.extend([1] * (6 - len(batch_shape)))
    batch_strides.extend([0] * (6 - len(batch_strides)))
    stride_m, stride_n = out.stride()[-2:]

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n), batch)
    with torch_device_fn.device(input.device):
        _tril_strided_out_tile_kernel[grid](
            input_to_use,
            out,
            int(diagonal),
            M,
            N,
            B0=batch_shape[0],
            B1=batch_shape[1],
            B2=batch_shape[2],
            B3=batch_shape[3],
            B4=batch_shape[4],
            B5=batch_shape[5],
            S0=batch_strides[0],
            S1=batch_strides[1],
            S2=batch_strides[2],
            S3=batch_strides[3],
            S4=batch_strides[4],
            S5=batch_strides[5],
            STRIDE_M=stride_m,
            STRIDE_N=stride_n,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


def _launch_tril(input: torch.Tensor, out: torch.Tensor, diagonal: int):
    M, N = input.shape[-2:]
    total = input.numel()
    if total == 0:
        return out

    if diagonal <= -M:
        # Everything zeros out. GEMS zero_() has a heavy fixed cost (~77us);
        # a small flat store-only launch is cheaper below ~1M elements.
        if total <= _SMALL_TOTAL_ZERO:
            _launch_v2_zero(out)
        else:
            out.zero_()
        return out
    if diagonal >= N - 1:
        # Everything is kept: pure copy. Use the native vendor copy (fast);
        # gems copy_() under use_gems is ~1000x slower on this XPU. When out
        # aliases input there is nothing to write.
        if input.data_ptr() != out.data_ptr():
            _vendor_copy_from(input, out)
        return out

    input_to_use = input if input.is_contiguous() else input.contiguous()
    batch = input_to_use.numel() // (M * N)

    # Band split: rows [band_lo, M) are entirely at/below the diagonal -> pure
    # copy (vendor native). Only the band prefix [0, band_lo*N) needs the
    # masking kernel. Gated on batch == 1 (contiguous bottom slice), the band
    # being a small fraction of the matrix, and total being large enough for
    # the extra launch to pay off.
    if batch == 1:
        band_lo = min(M, max(0, N - 1 - diagonal))
        if band_lo < M and band_lo * N <= (M * N) // 4 and total >= _BAND_MIN_TOTAL:
            _launch_v2_band(input_to_use, out, diagonal, band_lo)
            return out

    if batch == 1:
        if _use_wide_exact_row(M, N, batch):
            # Pre-existing exact per-row kernel (2D grid, unmasked pow2 rows):
            # fastest measured on this XPU for wide pow2 single matrices
            # ([4096,4096] fp32 ~0.38ms vs ~0.49ms for the flat variants).
            return _launch_exact_row(
                input_to_use,
                out,
                diagonal,
                num_warps=_wide_exact_row_warps(N),
            )
        if N > _FLAT_WIDE_N:
            # Very wide single matrix: the pre-existing flat kernel (BLOCK
            # 4096 / 4 warps) is the fastest measured variant on this XPU for
            # N > 8192 (e.g. [10000,65536] fp16 ~12ms). Note the default block
            # size is 1024 -> always pass 4096/4 explicitly.
            return _launch_flat(input_to_use, out, diagonal, block_size=4096, num_warps=4)
        if N >= _ROW_N_THRESHOLD:
            _launch_v2_rows(input_to_use, out, diagonal, num_rows=M)
            return out
        return _launch_v2_flat(input_to_use, out, diagonal)
    # Batched
    if M * N <= 4096:
        # Many tiny matrices: the % MN single pass beats a 2D grid
        # (launch-bound otherwise on this XPU).
        return _launch_v2_flat_batched(input_to_use, out, diagonal)
    _launch_v2_flat_batchgrid(input_to_use, out, diagonal)
    return out


def tril(input: torch.Tensor, diagonal: int = 0):
    logger.debug("GEMS_KUNLUNXIN TRIL")
    _check_input(input)

    out = _empty_contiguous_like(input)
    return _launch_tril(input, out, int(diagonal))


def tril_(input: torch.Tensor, diagonal: int = 0):
    logger.debug("GEMS_KUNLUNXIN TRIL_")
    _check_input(input)

    diagonal = int(diagonal)
    if input.numel() == 0:
        return input

    M, N = input.shape[-2:]
    if diagonal >= N - 1:
        return input
    if diagonal <= -M:
        return _zero_out(input)

    if input.is_contiguous():
        return _launch_tril_inplace_contiguous(input, diagonal)

    return _launch_tril_inplace_strided(input, diagonal)


def tril_out(input: torch.Tensor, diagonal: int = 0, *, out: torch.Tensor = None):
    logger.debug("GEMS_KUNLUNXIN TRIL_OUT")

    if out is None:
        return tril(input, diagonal)

    _check_input(input)
    if out.dtype != input.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {input.dtype}, but got {out.dtype} instead"
        )
    if out.device != input.device:
        raise RuntimeError(
            f"Expected out tensor to be on device {input.device}, but got {out.device} instead"
        )
    if out.shape != input.shape:
        out.resize_(input.shape)

    if out.is_contiguous():
        return _launch_tril(input, out, int(diagonal))

    if input.numel() == 0:
        return out
    M, N = input.shape[-2:]
    if diagonal <= -M:
        return _zero_out(out)
    if diagonal >= N - 1:
        if input.data_ptr() != out.data_ptr():
            _vendor_copy_from(input, out)
        return out

    # NOTE: the strided 2D-tile out kernel (`_launch_tril_strided_out`) is
    # 10-50x slower than the fast contiguous path on this XPU (its
    # `offs_m * N + offs_n` indexing is not proven contiguous by the XPU
    # OffsetAnalysis and degrades to discrete access; measured on fp16:
    # [1024,1024] transposed 1.13ms, [10000,65536] sliced ~735ms). Rerouting
    # every non-contiguous out through a contiguous temp (the same flat/row
    # kernels tril() uses) + the vendor native strided copy (aten::_copy_from,
    # not registered by flag_gems -> dispatches to the vendor engine) is
    # ~25-50x faster: [1024,1024] T 45us, [4096,4096] T 0.50ms,
    # [10000,65536] T 14.9ms, [100,65536,100] T 18.2ms. Safe wrt aliasing:
    # input is fully read into tmp before any write to out.
    tmp = _empty_contiguous_like(input)
    _launch_tril(input, tmp, int(diagonal))
    _vendor_copy_from(tmp, out)
    return out
