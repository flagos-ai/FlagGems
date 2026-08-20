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

logger = logging.getLogger(__name__)


# NOTE: kunlunxin's original triu used the fixed-cluster `M//12` heuristic with
# an in-kernel N loop (slow on large shapes + per-shape first-compile spikes),
# and did NOT override triu_ at all (it fell to the generic slow path).
#
# On this XPU/triton, 2D tiled kernels (`offs_m * N + offs_n` indexing) are not
# proven contiguous by OffsetAnalysis and degrade to discrete access (~1-3 GB/s,
# e.g. a plain [1024,1024] tile copy took ~3ms). The winning primitive is the
# 1D-flat kernel: contiguous load/store with row/col recovered via div/mod. Both
# triu (out-of-place) and triu_ (in-place, in_ptr == out_ptr) are built on it.
# triu keeps col >= row + diag; the complement (col < row + diag) is zeroed.
#
# A pure masked scalar store (`tl.store(ptr, 0.0, mask=...)`) is MISCOMPILED here
# (it ignores the per-lane predicate and zeros the whole block), so we always use
# the load + `tl.where` + store form, which is verified correct.
#
# The flat kernel's cost is dominated by the per-element integer div/mod used to
# recover (row, col) from the flat offset (~112 GB/s, ~4x below pure bandwidth).
# For large N we instead launch one program PER ROW (grid = batch*M): `row` is
# recovered once per program via `pid % M` (one mod per program, not per element)
# and each row streams its N columns as contiguous blocks. Isolation on this XPU:
# [4096,4096] 0.72ms -> 0.52ms, [10000,65536] 27.8ms -> 15.2ms. But for small N
# the row grid is launch-bound (e.g. [100,65536,100] N=100 blows up to 391ms), so
# we gate on N and keep the flat kernel for small last dims.


@triton.jit
def _triu_flat_kernel(
    in_ptr,
    out_ptr,
    total,
    diag,
    MN,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Batched single-pass triu: keep where col >= row + diag, else write 0.
    # `% MN` folds the flat index into one matrix, so this works for any batch.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    matrix_offsets = offsets % MN
    rows = matrix_offsets // N
    cols = matrix_offsets - rows * N
    keep = cols >= rows + diag

    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    y = tl.where(keep, x, 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _triu_flat_2d_kernel(
    in_ptr,
    out_ptr,
    total,
    diag,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Single-matrix fast path: no `% MN`. Works out-of-place (in != out),
    # in-place (in == out), and over a contiguous top-row prefix of one matrix
    # (offsets stay within [0, M*N) so row = offset // N is exact).
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    rows = offsets // N
    cols = offsets - rows * N
    keep = cols >= rows + diag

    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    y = tl.where(keep, x, 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)


def _check_input(A):
    assert A.dim() >= 2, "Input tensor must have at least 2 dimensions"


@triton.jit
def _triu_row_kernel(
    in_ptr,
    out_ptr,
    M,
    N,
    diag,
    BLOCK_N: tl.constexpr,
):
    # One program per row (grid = batch*M). `row = pid % M` is a single mod PER
    # PROGRAM, replacing the flat kernel's per-element div/mod. Each row streams
    # its N columns as contiguous BLOCK_N chunks (block DMA), keeping the entries
    # with col >= row + diag and zeroing the rest.
    pid = tl.program_id(0)
    row = pid % M
    base = pid * N
    for c0 in range(0, N, BLOCK_N):
        cols = c0 + tl.arange(0, BLOCK_N)
        m = cols < N
        keep = cols >= row + diag
        x = tl.load(in_ptr + base + cols, mask=m, other=0.0)
        tl.store(out_ptr + base + cols, tl.where(keep, x, 0.0), mask=m)


# Large flat tiles hide the div/where compute better than small ones on this XPU
# (measured: BLOCK 1024 -> 8192 roughly halves latency on [4096,4096]).
_BLOCK_SIZE = 8192

# Route to the per-row kernel when the last dim is at least this wide. Below it
# the flat kernel wins (row grid becomes launch-bound); at/above it the per-row
# kernel wins by dropping the per-element div/mod. [1024,1024] flat wins,
# [4096,4096] row wins, so the crossover sits between them.
_ROW_N_THRESHOLD = 2048


def _launch_flat(input_c, out, diagonal, total=None):
    M, N = input_c.shape[-2:]
    MN = M * N
    if total is None:
        total = input_c.numel()
    if N >= _ROW_N_THRESHOLD:
        # Large N: one program per row (avoids per-element div/mod). `total` may
        # be a top-row prefix (band_hi * N) -> that many rows; row = pid % M.
        num_rows = total // N
        block_n = min(triton.next_power_of_2(N), _BLOCK_SIZE)
        with torch_device_fn.device(input_c.device):
            _triu_row_kernel[(num_rows,)](
                input_c, out, M, N, diagonal, block_n, num_warps=8
            )
        return
    grid = (triton.cdiv(total, _BLOCK_SIZE),)
    with torch_device_fn.device(input_c.device):
        if total <= MN:
            # single matrix (or a top-row prefix of one) -> skip the `% MN`.
            _triu_flat_2d_kernel[grid](
                input_c, out, total, diagonal, N, _BLOCK_SIZE, num_warps=8
            )
        else:
            _triu_flat_kernel[grid](
                input_c, out, total, diagonal, MN, N, _BLOCK_SIZE, num_warps=8
            )


def _launch_triu_inplace_contiguous(A, diagonal):
    M, N = A.shape[-2:]
    if diagonal >= N:  # zero everything
        A.zero_()
        return
    # rows [band_hi, M) are entirely below the diagonal. When that slice is
    # contiguous (2D / batch == 1) we memset it in bulk (vendor bandwidth) and
    # run the flat kernel only over the contiguous band prefix [0, band_hi*N).
    # For batched tensors the slice is strided (zero_()/fill_() would miszero a
    # contiguous block on this XPU), so we fall back to one flat pass over all.
    band_hi = min(M, max(0, N - diagonal))
    if band_hi < M:
        below = A[..., band_hi:, :]
        if below.is_contiguous():
            below.zero_()
            if band_hi > 0:
                _launch_flat(A, A, diagonal, total=band_hi * N)
            return
    _launch_flat(A, A, diagonal)


def triu(A, diagonal=0):
    logger.debug("GEMS_KUNLUNXIN TRIU")
    _check_input(A)
    A = A.contiguous()
    out = torch.empty_like(A)
    M, N = A.shape[-2:]
    if diagonal <= 1 - M:  # keep everything
        out.copy_(A)
        return out
    if diagonal >= N:  # zero everything
        out.zero_()
        return out
    _launch_flat(A, out, diagonal)
    return out


def triu_(A, diagonal=0):
    logger.debug("GEMS_KUNLUNXIN TRIU_")
    _check_input(A)
    M, N = A.shape[-2:]
    if diagonal <= 1 - M:  # keep everything, nothing to zero
        return A
    if A.is_contiguous():
        _launch_triu_inplace_contiguous(A, diagonal)
        return A
    # Non-contiguous: operate on a contiguous copy, then write back with copy_
    # to preserve the original data_ptr and stride (true in-place semantics).
    A_c = A.contiguous()
    _launch_triu_inplace_contiguous(A_c, diagonal)
    A.copy_(A_c)
    return A
