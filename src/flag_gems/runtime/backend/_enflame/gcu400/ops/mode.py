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
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

from .sort import sort_stable

logger = logging.getLogger(__name__)


def _mode_sort(inp, dim=-1, descending=False):
    """Stable sort along ``dim``; reuses the backend's tested radix sort.

    The private radix sort this module used to carry duplicated the sweep
    kernel and was wrong on gcu400 for two independent reasons:

    * the associate (index) buffer was int64 while the value buffer is int32.
      64-bit tiles are lowered as 256-lane vectors against 512-lane 32-bit
      tiles, and the scattered int64 store then lands on the wrong addresses
      (adjacent output slots receive 2*index, 2*index+1), which also runs past
      the buffer and corrupts neighbouring allocations.
    * rows longer than one 2048-wide tile need the cross-tile running offset
      carried through a dynamic loop; that carry is miscompiled.

    ``sort_stable`` keeps the indices in int32 internally and builds the
    cross-tile prefix with ``block_base_kernel``, so both hazards are gone.
    """
    return sort_stable(inp, stable=True, dim=dim, descending=descending)


@triton.jit
def _mode_max_combine(left, right):
    return tl.maximum(left, right)


@triton.jit
def _mode_scan_body(
    sorted_inp,
    out_col,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Tiled run-length scan over already sorted rows.

    The row is split into BLOCK_N wide tiles; every tile resolves all runs that
    end inside it and only the run still open at the tile end is carried to the
    next tile through (prev_value, prev_count), so the loop runs N / BLOCK_N
    times instead of N times.

    The kernel stores the *column* of the winning run end instead of the value
    and the original index: values and indices are gathered from the sorted
    buffers afterwards.  Keeping int64 out of the kernel matters on gcu400 (see
    ``_mode_sort``); the column alone identifies both.
    """
    ELEM_TY: tl.constexpr = sorted_inp.dtype.element_ty
    IS_BYTE: tl.constexpr = ELEM_TY == tl.int8 or ELEM_TY == tl.uint8
    IS_I64: tl.constexpr = (
        ELEM_TY == tl.int64 or ELEM_TY == tl.uint64 or ELEM_TY == tl.float64
    )
    # Byte tiles are widened to int32 so the loop-carried state never changes
    # dtype (tl.sum promotes int8/int16 and fp16, so the carries are cast back).
    if IS_BYTE:
        VALUE_TY: tl.constexpr = tl.int32
    else:
        VALUE_TY: tl.constexpr = ELEM_TY

    pid_m = tle.program_id(0)
    row_ids = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = row_ids < M
    col_ids = tl.arange(0, BLOCK_N)

    row_base = row_ids[:, None] * N

    # Value and length of the run still open at the previous tile boundary.
    prev_value = tl.load(
        sorted_inp + row_base + tl.zeros([1, 1], tl.int32),
        mask=mask_m[:, None],
        other=0,
    ).reshape([BLOCK_M])
    if IS_BYTE:
        prev_value = prev_value.to(tl.int32)
    prev_count = tl.zeros([BLOCK_M], dtype=tl.int32)

    best_count = tl.zeros([BLOCK_M], dtype=tl.int32)
    best_col = tl.zeros([BLOCK_M], dtype=tl.int32)

    n_tiles = tl.cdiv(N, BLOCK_N)
    for tile in range(0, n_tiles):
        n_offset = tile * BLOCK_N + col_ids
        valid = (n_offset < N)[None, :] & mask_m[:, None]
        col2 = col_ids[None, :]

        val = tl.load(sorted_inp + row_base + n_offset[None, :], mask=valid, other=0)
        if IS_BYTE:
            val = val.to(tl.int32)
        # Previous element, taken from memory so it also covers the tile head;
        # column 0 is patched with the carried value.
        prev_val = tl.load(
            sorted_inp + row_base + tl.maximum(n_offset - 1, 0)[None, :],
            mask=valid,
            other=0,
        )
        if IS_BYTE:
            prev_val = prev_val.to(tl.int32)
        prev_val = tl.where(col2 == 0, prev_value[:, None], prev_val)
        # Next element (may live in the next tile, hence the memory load).
        next_val = tl.load(
            sorted_inp + row_base + tl.minimum(n_offset + 1, N - 1)[None, :],
            mask=valid & ((n_offset + 1) < N)[None, :],
            other=0,
        )
        if IS_BYTE:
            next_val = next_val.to(tl.int32)

        if IS_I64:
            # 64-bit compares stay in the two-signed-compare form to avoid the
            # v512i1 concat_vectors legalization failure.
            same_prev = ~((val - prev_val < 0) | (prev_val - val < 0))
            same_next = ~((next_val - val < 0) | (val - next_val < 0))
        else:
            same_prev = val == prev_val
            same_next = val == next_val

        run_start = (~same_prev) & valid
        # The last column of the row terminates its run in every case: the
        # next-element load is masked out there, so `same_next` alone would
        # keep a trailing run of zeros open forever.
        run_end = ((~same_next) & valid) | (valid & (n_offset == N - 1)[None, :])

        start_pos = tl.associative_scan(
            tl.where(run_start, col2, -1), 1, _mode_max_combine
        )
        cnt = tl.where(
            start_pos >= 0, col2 - start_pos + 1, prev_count[:, None] + col2 + 1
        )

        # Candidate length of every run that ends inside this tile.
        cand = tl.where(run_end, cnt, -1)
        tile_best = tl.max(cand, axis=1)
        tile_pos = tl.min(tl.where(cand == tile_best[:, None], col2, BLOCK_N), axis=1)

        # Tiles are visited in ascending order and the comparison is strict, so
        # ties keep the earliest (smallest) value, matching the torch result.
        better = (tile_pos < BLOCK_N) & (tile_best > best_count)
        best_count = tl.where(better, tile_best, best_count)
        best_col = tl.where(better, tile * BLOCK_N + tile_pos, best_col)

        # Carry the run that is still open at the last valid column.
        last_col = tl.minimum(N - 1 - tile * BLOCK_N, BLOCK_N - 1)
        is_last = col2 == last_col
        prev_count = tl.sum(tl.where(is_last, cnt, 0), axis=1)
        prev_value = tl.sum(tl.where(is_last, tl.where(valid, val, 0), 0), axis=1).to(
            VALUE_TY
        )

    tl.store(out_col + row_ids, best_col, mask=mask_m)


@libentry()
@triton.jit
def mode_kernel(
    sorted_inp,
    out_col,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    _mode_scan_body(sorted_inp, out_col, M, N, BLOCK_M, BLOCK_N)


def _scan_config(M):
    """Static scan tiling: wide row groups for tall inputs, single rows else."""
    if M >= 256:
        block_m, block_n = 8, 256
    else:
        block_m, block_n = 1, 1024
    return block_m, block_n


def mode(inp, dim=-1, keepdim=False):
    logger.debug("GEMS_ENFLAME MODE")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"

    shape = list(inp.shape)
    dim = dim % inp.ndim
    N = shape[dim]
    M = inp.numel() // N

    sorted_inp, sorted_indices = _mode_sort(inp, dim=dim)

    # Move dim to last for 2D processing
    sorted_inp = torch.movedim(sorted_inp, dim, -1).contiguous()
    sorted_indices = torch.movedim(sorted_indices, dim, -1).contiguous()

    sorted_flat = sorted_inp.reshape(M, N)
    indices_flat = sorted_indices.reshape(M, N)

    out_col = torch.empty(max(M, 1), dtype=torch.int32, device=inp.device)
    if M > 0:
        block_m, block_n = _scan_config(M)
        grid = (triton.cdiv(M, block_m),)
        with torch_device_fn.device(inp.device):
            mode_kernel[grid](
                sorted_flat,
                out_col,
                M,
                N,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
            )
        # Gather with tensor ops: the winning column identifies both the value
        # and the original index, so no int64 work is needed on the device side.
        sel = out_col[:M].to(torch.int64).unsqueeze(1)
        out_value = torch.gather(sorted_flat, 1, sel).squeeze(1)
        out_index = torch.gather(indices_flat, 1, sel).squeeze(1)
    else:
        out_value = torch.empty(0, dtype=inp.dtype, device=inp.device)
        out_index = torch.empty(0, dtype=torch.int64, device=inp.device)

    out_shape = list(shape)
    out_shape[dim] = 1
    out_value = out_value.reshape(out_shape)
    out_index = out_index.reshape(out_shape)

    if not keepdim:
        out_value = out_value.squeeze(dim)
        out_index = out_index.squeeze(dim)

    Mode_out = namedtuple("mode", ["values", "indices"])
    return Mode_out(values=out_value, indices=out_index)
