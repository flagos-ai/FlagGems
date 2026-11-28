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

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# L600 (GCU400): 48 SPs; a program whose DSM footprint stays <= ~448KB keeps
# a single program resident per SP, and ceil-balanced aligned chunks give
# <= 48 programs (single wave) for any input size. More programs only
# regressed (96/192-program spot check: +32-100%), so the chunk count is
# capped by construction.
_NUM_SPC = 48
# Wide flat copy op width in lanes (blessed 1D streaming form; the 2D
# [BLOCK_T, BLOCK_D] masked tile of the generic path is slow on this
# backend). Rescan on this repo's serving shapes picked 32768 lanes for
# >= 2^20-element copies and 8192 below (32768 on the 40K-element ragged
# shape collapsed to 2 programs and +130%); fp32 at 32768 lanes keeps the
# per-program DSM peak (load + store staging) at 256KB, within the 448KB
# single-program budget.
_COPY_BLOCK_LARGE = 32768
_COPY_BLOCK_SMALL = 8192
_LARGE_COPY_ELEMS = 2**20
_NUM_WARPS = 2
_NUM_STAGES = 2
# Host-side memo for triton.next_power_of_2(B): B is a host integer, so the
# memo is a pure function cache with no tensor-state involvement.
_NEXT_POW2: dict = {}


@triton.jit
def _unpack_seq_kernel(
    packed_ptr,  # [B, Lmax, D] contiguous
    out_ptr,  # [out_rows, D] contiguous, out_rows >= rows written
    lengths_ptr,  # *i32, [B]
    rows_per_chunk,
    out_rows,
    B: tl.constexpr,
    Lmax: tl.constexpr,
    D: tl.constexpr,
    BLOCK_B: tl.constexpr,  # rounded-up batch entries for prefix sum
    BLOCK: tl.constexpr,  # flat elements per maskless load/store
    WIDE: tl.constexpr,
):
    # One program per contiguous packed-row chunk of the flat [B*Lmax] row
    # space. The host ceil-divides the rows into <= NUM_SPC balanced chunks,
    # so program pid owns exactly chunk pid. Within a chunk, valid rows are
    # moved as flat runs:
    #   - a full batch (len_b == Lmax) is source- and destination-contiguous
    #     with every following full batch, so consecutive full batches merge
    #     into one long flat run (bounded by the chunk's row budget)
    #   - a partial batch is its own run: its source rows stop at len_b, the
    #     destination rows keep walking the packed layout, so merging past
    #     the batch boundary would read the padding region
    #   - a chunk that starts inside a batch's padding copies nothing and
    #     resumes at the next batch's first row
    chunk = tl.program_id(0)
    TP = B * Lmax
    off_b = tl.arange(0, BLOCK_B)
    r0 = chunk * rows_per_chunk
    r_end = tl.minimum(r0 + rows_per_chunk, TP)
    b = r0 // Lmax
    t = r0 - b * Lmax
    prev = tl.load(lengths_ptr + off_b, mask=off_b < b, other=0)
    out_row = tl.sum(prev, axis=0) + t
    r = r0
    while r < r_end:
        len_b = tl.load(lengths_ptr + b)
        valid_b = len_b - t
        if valid_b > 0:
            run = valid_b
            if len_b == Lmax:
                # Batch b is completely full, so its remaining rows are
                # source-contiguous with every following full batch: merge
                # them into one flat run. The peek is bounded by the
                # chunk's row budget — runs are clamped to the chunk
                # anyway, so scanning past it is wasted work.
                nb = b + 1
                go = 1
                budget = r_end - r
                while (nb < B) & (go == 1) & (valid_b + (nb - b) * Lmax <= budget):
                    len_nb = tl.load(lengths_ptr + nb)
                    go = tl.where(len_nb == Lmax, 1, 0)
                    nb += go
                run = valid_b + (nb - b - 1) * Lmax
            run = tl.minimum(run, r_end - r)
            # The walk's destination row can exceed the allocation only in
            # the degenerate len_b > Lmax case (prefix sums are unclamped);
            # clamp the copy to the buffer and keep advancing the walk.
            copy_run = tl.minimum(run, out_rows - out_row)
            if copy_run > 0:
                if WIDE:
                    n_elem = copy_run.to(tl.int64) * D
                    src = (b * Lmax + t).to(tl.int64) * D
                    dst = out_row.to(tl.int64) * D
                else:
                    n_elem = copy_run * D
                    src = (b * Lmax + t) * D
                    dst = out_row * D
                # Block count always fits i32: >= 2^31 blocks would need a
                # > 2^46-element tensor, impossible on this hardware.
                n_blocks = (n_elem // BLOCK).to(tl.int32)
                for j in range(0, n_blocks):
                    if WIDE:
                        offs = j.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
                    else:
                        offs = j * BLOCK + tl.arange(0, BLOCK)
                    tl.store(out_ptr + (dst + offs), tl.load(packed_ptr + (src + offs)))
                if WIDE:
                    base_full = n_blocks.to(tl.int64) * BLOCK
                else:
                    base_full = n_blocks * BLOCK
                rem = n_elem - base_full
                if rem > 0:
                    offs = base_full + tl.arange(0, BLOCK)
                    m = offs < n_elem
                    tl.store(
                        out_ptr + (dst + offs),
                        tl.load(packed_ptr + (src + offs), mask=m),
                        mask=m,
                    )
            r += run
            out_row += run
        else:
            # Rows [t, Lmax) of batch b are padding: nothing to copy, resume
            # at the next batch's first row (clamped by r_end above).
            out_row += len_b - t
            r = (b + 1) * Lmax
        b = r // Lmax
        t = r - b * Lmax


def unpack_seq_triton(
    packed_tensor: torch.Tensor,
    lengths: torch.Tensor,
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    # block_t/block_d are accepted for signature compatibility with the
    # generic implementation; the fixed-grid flat-run kernel derives its
    # own tiling and ignores them.
    logger.debug("GEMS_ENFLAME UNPACK_SEQ_TRITON")
    original_shape = packed_tensor.shape
    if len(original_shape) > 3:
        B, Lmax = original_shape[:2]
        packed_reshaped = packed_tensor.reshape(B, Lmax, -1)
        D = packed_reshaped.shape[2]
    else:
        B, Lmax, D = packed_tensor.shape
        packed_reshaped = packed_tensor

    # The kernel indexes lengths linearly and reads int32 entries; the hot
    # path (bench/tests already pass int32 contiguous) is a zero-cost pass
    # through, and only mismatched callers pay a metadata-normalizing copy.
    if lengths.dtype is torch.int32 and lengths.is_contiguous():
        lengths_i32 = lengths
    else:
        lengths_i32 = lengths.to(torch.int32)
        if not lengths_i32.is_contiguous():
            lengths_i32 = lengths_i32.contiguous()

    # Launch the N reduction before the remaining host prep so its device
    # execution overlaps the python below; only the .item() waits.
    total = lengths_i32.sum()

    total_packed_rows = B * Lmax
    pow2_b = _NEXT_POW2.get(B)
    if pow2_b is None:
        pow2_b = triton.next_power_of_2(B)
        _NEXT_POW2[B] = pow2_b

    # Upper-bound output allocation: the packed row count B*Lmax is a host
    # known >= sum(min(len_b, Lmax)) bound on the rows the walk writes, so
    # the kernel can launch before N is read back and the .item() roundtrip
    # overlaps the copy instead of serializing ahead of the allocation.
    out_full = torch.empty(
        (total_packed_rows, D), device=packed_tensor.device, dtype=packed_tensor.dtype
    )

    # Balanced chunking with block alignment: target ~48 chunks, then round
    # each chunk up to a whole number of BLOCKs so the maskless inner loop
    # covers the chunk exactly (no predicated-off tail lanes — the masked
    # tail form measured +0.5-2.5us per shape). One program per chunk, one
    # O(B) prefix per chunk, and no idle programs for small inputs.
    if total_packed_rows > 0 and D > 0:
        total_elems = total_packed_rows * D
        if total_elems >= _LARGE_COPY_ELEMS:
            block = _COPY_BLOCK_LARGE
        else:
            block = _COPY_BLOCK_SMALL
        target = -(-total_elems // _NUM_SPC)
        blocks_per_chunk = max(1, -(-target // block))
        rows_per_chunk = -(-blocks_per_chunk * block // D)
        n_chunks = -(-total_packed_rows // rows_per_chunk)
        # Element offsets can exceed i32 for very large tensors; widen the
        # whole address chain (single addptr per trap-note) instead of
        # wrapping. N is not needed for the wide decision: the walk reads
        # at most TP rows and writes at most min(N, TP) rows, both bounded
        # by TP*D on this side; N*D is checked after the readback below.
        wide = 1 if total_elems > 2**31 - 1 else 0

        _unpack_seq_kernel[(n_chunks,)](
            packed_reshaped,
            out_full,
            lengths_i32,
            rows_per_chunk,
            total_packed_rows,
            B=B,
            Lmax=Lmax,
            D=D,
            BLOCK_B=pow2_b,
            BLOCK=block,
            WIDE=wide,
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
        )

    N = int(total.item())
    if N > total_packed_rows:
        # Degenerate caller (some len_b > Lmax): the slice view would claim
        # rows past the allocation. Redo the launch into a right-sized
        # buffer on this cold path; rows past the packed horizon stay
        # uninitialized, matching the generic kernel's behavior there.
        out = torch.empty(
            (N, D), device=packed_tensor.device, dtype=packed_tensor.dtype
        )
        if total_packed_rows > 0 and D > 0:
            if total_packed_rows * D > 2**31 - 1 or N * D > 2**31 - 1:
                wide = 1
            else:
                wide = 0
            _unpack_seq_kernel[(n_chunks,)](
                packed_reshaped,
                out,
                lengths_i32,
                rows_per_chunk,
                N,
                B=B,
                Lmax=Lmax,
                D=D,
                BLOCK_B=pow2_b,
                BLOCK=block,
                WIDE=wide,
                num_warps=_NUM_WARPS,
                num_stages=_NUM_STAGES,
            )
        result = out
    else:
        # The walk fills destination rows [0, N) densely (batch b starts at
        # the unclamped prefix sum, so N <= TP keeps the slice in-bounds).
        result = out_full[:N]

    if len(original_shape) > 3:
        output_shape = (N,) + original_shape[2:]
        result = result.reshape(output_shape)

    return result
