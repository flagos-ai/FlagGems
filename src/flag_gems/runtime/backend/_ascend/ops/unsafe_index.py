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

"""Ascend-tuned ``unsafe_index`` (aten._unsafe_index).

Three fixed kernel families (msprof-tuned on Ascend910B4):
  D  1-D gather: out[i] = inp[idx[i]], contiguous 1-D input only.
  G  point gather: no trailing slice dims; fully strided on the input side,
     so non-contiguous inputs qualify.
  R  row copy: trailing slice dims form a contiguous run copied in blocks;
     indexed/outer dims may be strided.

Contiguous index tensors of any rank are flattened to rank 1 on the host (a
view): the kernels walk the broadcast subspace linearly, matching both the
index storage and the contiguous output.  Anything else (>4 index tensors,
non-contiguous index views with rank > 2, non-contiguous trailing run,
>1 outer slice dim, int32 offsets exceeded, oversized grids) falls back to
the generic codegen implementation in ``flag_gems.ops``.

Measured Ascend pitfalls avoided:
  * int64 vector where/min/max scalarize catastrophically: index math is
    int32, gated on all offsets fitting int32.
  * per-lane ``%``/``//`` breaks load/store contiguity analysis: family R
    walks the contiguous run with a bare arange.
  * small problems need ~32 blocks to fill the device; tiles are capped at
    64 KiB to avoid UB (local buffer) overflow with multi-buffering.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.unsafe_index import (
    _broadcast_index_tensors,
    _check_indices,
    _eliminate_scalar_indices,
    _unsafe_index_func,
)
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_INT32_LIMIT = 2**31
_MAX_IDX = 4  # padded index-tensor slots in the fixed kernels
_MAX_GRID_AXIS = 65535


@triton.jit
def _gather_1d_kernel(
    input_ptr,
    idx_ptr,
    out_ptr,
    ish0,
    idx_stride0,
    M,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < M
    i = tl.load(idx_ptr + off * idx_stride0, mask=m, other=0).to(tl.int32)
    i = tl.where(i < 0, i + ish0, i)
    i = tl.minimum(tl.maximum(i, 0), ish0 - 1)
    v = tl.load(input_ptr + i, mask=m)
    tl.store(out_ptr + off, v, mask=m)


@triton.jit
def _gather_point_kernel(
    input_ptr,
    out_ptr,
    idx0_ptr,
    idx1_ptr,
    idx2_ptr,
    idx3_ptr,
    ish0,
    ish1,
    ish2,
    ish3,
    is0,
    is1,
    is2,
    is3,
    i0s0,
    i0s1,
    i1s0,
    i1s1,
    i2s0,
    i2s1,
    i3s0,
    i3s1,
    bs1,
    in_ss,
    out_bs,
    out_ss,
    M,
    N,
    NI: tl.constexpr,
    HAS_SLICE: tl.constexpr,
    BLOCK0: tl.constexpr,
    BLOCK1: tl.constexpr,
):
    pid0 = ext.program_id(axis=0)
    pid1 = ext.program_id(axis=1)
    off0 = pid0 * BLOCK0 + tl.arange(0, BLOCK0)[:, None]
    if HAS_SLICE:
        off1 = pid1 * BLOCK1 + tl.arange(0, BLOCK1)[None, :]
    else:
        off1 = tl.zeros([1, 1], tl.int32)
    m0 = off0 < M
    c1 = off0 % bs1
    c0 = off0 // bs1

    i0 = tl.load(idx0_ptr + c0 * i0s0 + c1 * i0s1, mask=m0, other=0).to(tl.int32)
    i0 = tl.where(i0 < 0, i0 + ish0, i0)
    i0 = tl.minimum(tl.maximum(i0, 0), ish0 - 1)
    addr = i0 * is0
    if NI > 1:
        i1 = tl.load(idx1_ptr + c0 * i1s0 + c1 * i1s1, mask=m0, other=0).to(tl.int32)
        i1 = tl.where(i1 < 0, i1 + ish1, i1)
        i1 = tl.minimum(tl.maximum(i1, 0), ish1 - 1)
        addr += i1 * is1
    if NI > 2:
        i2 = tl.load(idx2_ptr + c0 * i2s0 + c1 * i2s1, mask=m0, other=0).to(tl.int32)
        i2 = tl.where(i2 < 0, i2 + ish2, i2)
        i2 = tl.minimum(tl.maximum(i2, 0), ish2 - 1)
        addr += i2 * is2
    if NI > 3:
        i3 = tl.load(idx3_ptr + c0 * i3s0 + c1 * i3s1, mask=m0, other=0).to(tl.int32)
        i3 = tl.where(i3 < 0, i3 + ish3, i3)
        i3 = tl.minimum(tl.maximum(i3, 0), ish3 - 1)
        addr += i3 * is3

    mask = m0
    out_addr = off0 * out_bs
    if HAS_SLICE:
        addr = addr + off1 * in_ss
        out_addr = out_addr + off1 * out_ss
        mask = mask & (off1 < N)
    v = tl.load(input_ptr + addr, mask=mask)
    tl.store(out_ptr + out_addr, v, mask=mask)


@triton.jit
def _gather_rows_kernel(
    input_ptr,
    out_ptr,
    idx0_ptr,
    idx1_ptr,
    idx2_ptr,
    idx3_ptr,
    ish0,
    ish1,
    ish2,
    ish3,
    is0,
    is1,
    is2,
    is3,
    i0s0,
    i0s1,
    i1s0,
    i1s1,
    i2s0,
    i2s1,
    i3s0,
    i3s1,
    bs1,
    in_os,
    out_rs,
    out_os,
    M,
    R,
    NI: tl.constexpr,
    HAS_OUTER: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid0 = ext.program_id(axis=0)
    pid1 = ext.program_id(axis=1)
    pid2 = ext.program_id(axis=2)
    rows = pid0 * ROWS + tl.arange(0, ROWS)
    off = pid1 * BLOCK + tl.arange(0, BLOCK)[None, :]
    m0 = rows < M
    c1 = rows % bs1
    c0 = rows // bs1

    i0 = tl.load(idx0_ptr + c0 * i0s0 + c1 * i0s1, mask=m0, other=0).to(tl.int32)
    i0 = tl.where(i0 < 0, i0 + ish0, i0)
    i0 = tl.minimum(tl.maximum(i0, 0), ish0 - 1)
    base = i0 * is0
    if NI > 1:
        i1 = tl.load(idx1_ptr + c0 * i1s0 + c1 * i1s1, mask=m0, other=0).to(tl.int32)
        i1 = tl.where(i1 < 0, i1 + ish1, i1)
        i1 = tl.minimum(tl.maximum(i1, 0), ish1 - 1)
        base += i1 * is1
    if NI > 2:
        i2 = tl.load(idx2_ptr + c0 * i2s0 + c1 * i2s1, mask=m0, other=0).to(tl.int32)
        i2 = tl.where(i2 < 0, i2 + ish2, i2)
        i2 = tl.minimum(tl.maximum(i2, 0), ish2 - 1)
        base += i2 * is2
    if NI > 3:
        i3 = tl.load(idx3_ptr + c0 * i3s0 + c1 * i3s1, mask=m0, other=0).to(tl.int32)
        i3 = tl.where(i3 < 0, i3 + ish3, i3)
        i3 = tl.minimum(tl.maximum(i3, 0), ish3 - 1)
        base += i3 * is3

    in_addr = base[:, None] + off
    out_addr = rows[:, None] * out_rs + off
    if HAS_OUTER:
        in_addr = in_addr + pid2 * in_os
        out_addr = out_addr + pid2 * out_os
    mask = m0[:, None] & (off < R)
    v = tl.load(input_ptr + in_addr, mask=mask)
    tl.store(out_ptr + out_addr, v, mask=mask)


def _pow2_ceil(n):
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _launch_1d(inp, idx, out):
    M = out.numel()
    BLOCK = min(128, max(2, _pow2_ceil(M)))
    grid = (triton.cdiv(M, BLOCK), 1, 1)
    _gather_1d_kernel[grid](inp, idx, out, inp.shape[0], idx.stride(0), M, BLOCK=BLOCK)


def _pick_point_blocks(M, N):
    # ~32+ blocks win on Ascend; keep BLOCK0 >= 2, a 1-lane tile breaks the
    # Ascend BlockPtr analysis.
    if N == 1:
        return min(256, max(2, _pow2_ceil(max(1, M // 32)))), 1
    BLOCK0 = min(64, _pow2_ceil(M))
    BLOCK1 = min(32, _pow2_ceil(N))
    while ((M + BLOCK0 - 1) // BLOCK0) * ((N + BLOCK1 - 1) // BLOCK1) < 32:
        if BLOCK1 > 16:
            BLOCK1 //= 2
        elif BLOCK0 > 2:
            BLOCK0 //= 2
        else:
            break
    return BLOCK0, BLOCK1


def _pad_index_args(inp, kidx, idx_dims, index_rank):
    ptrs = [inp] * _MAX_IDX
    ish = [1] * _MAX_IDX
    iss = [0] * _MAX_IDX
    bstr = [0] * (2 * _MAX_IDX)
    for i, t in enumerate(kidx):
        ptrs[i] = t
        d = idx_dims[i]
        ish[i] = inp.shape[d]
        iss[i] = inp.stride(d)
        st = t.stride()
        if index_rank == 2:
            bstr[2 * i] = st[0]
            bstr[2 * i + 1] = st[1]
        else:
            bstr[2 * i] = 0
            bstr[2 * i + 1] = st[0]
    return ptrs, ish, iss, bstr


def _launch_point(inp, kidx, out, idx_dims, slice_dim, bcast_pos, index_rank):
    ptrs, ish, iss, bstr = _pad_index_args(inp, kidx, idx_dims, index_rank)
    M = kidx[0].numel()
    bs1 = kidx[0].shape[1] if index_rank == 2 else M
    N = out.numel() // M
    if slice_dim is None:
        in_ss, out_bs, out_ss = 0, 1, 0
    elif bcast_pos == 0:
        # split subspace: out = (bcast..., slice)
        in_ss, out_bs, out_ss = inp.stride(slice_dim), N, 1
    else:
        # slice dim precedes the bcast block: out = (slice, bcast...)
        in_ss, out_bs, out_ss = inp.stride(slice_dim), 1, M
    BLOCK0, BLOCK1 = _pick_point_blocks(M, N)
    if (N + BLOCK1 - 1) // BLOCK1 > _MAX_GRID_AXIS:
        return False
    grid = (triton.cdiv(M, BLOCK0), triton.cdiv(N, BLOCK1), 1)
    args = (
        inp,
        out,
        ptrs[0],
        ptrs[1],
        ptrs[2],
        ptrs[3],
        ish[0],
        ish[1],
        ish[2],
        ish[3],
        iss[0],
        iss[1],
        iss[2],
        iss[3],
        bstr[0],
        bstr[1],
        bstr[2],
        bstr[3],
        bstr[4],
        bstr[5],
        bstr[6],
        bstr[7],
        bs1,
        in_ss,
        out_bs,
        out_ss,
        M,
        N,
    )
    _gather_point_kernel[grid](
        *args,
        NI=len(kidx),
        HAS_SLICE=slice_dim is not None,
        BLOCK0=BLOCK0,
        BLOCK1=BLOCK1,
    )
    return True


def _pick_row_blocks(M, R, S, elem_size):
    tile_cap = 65536 // elem_size  # 64 KiB per tile (UB multibuffer limit)
    BLOCK = max(2, min(2048, _pow2_ceil(R), tile_cap))
    ROWS = min(_pow2_ceil(M), max(1, tile_cap // BLOCK))
    # Keep ~32 blocks when possible; ROWS stays >= 2 (a 1-row tile breaks the
    # Ascend BlockPtr analysis).
    while ROWS > 2 and ((M + ROWS - 1) // ROWS) * ((R + BLOCK - 1) // BLOCK) * S < 32:
        ROWS //= 2
    while BLOCK > 64 and ((M + ROWS - 1) // ROWS) * ((R + BLOCK - 1) // BLOCK) * S < 32:
        BLOCK //= 2
    return max(ROWS, 2), BLOCK


def _launch_rows(inp, kidx, out, idx_dims, outer_dim, bcast_pos, R, S, index_rank):
    ptrs, ish, iss, bstr = _pad_index_args(inp, kidx, idx_dims, index_rank)
    M = kidx[0].numel()
    bs1 = kidx[0].shape[1] if index_rank == 2 else M
    if outer_dim is None:
        in_os, out_rs, out_os = 0, R, 0
    else:
        in_os = inp.stride(outer_dim)
        if bcast_pos == 0:
            # split subspace: out = (bcast..., outer..., trailing run)
            out_rs, out_os = R * S, R
        else:
            # out = (outer..., bcast..., trailing run)
            out_rs, out_os = R, M * R
    ROWS, BLOCK = _pick_row_blocks(M, R, S, inp.element_size())
    if (R + BLOCK - 1) // BLOCK > _MAX_GRID_AXIS:
        return False
    grid = (triton.cdiv(M, ROWS), triton.cdiv(R, BLOCK), S)
    args = (
        inp,
        out,
        ptrs[0],
        ptrs[1],
        ptrs[2],
        ptrs[3],
        ish[0],
        ish[1],
        ish[2],
        ish[3],
        iss[0],
        iss[1],
        iss[2],
        iss[3],
        bstr[0],
        bstr[1],
        bstr[2],
        bstr[3],
        bstr[4],
        bstr[5],
        bstr[6],
        bstr[7],
        bs1,
        in_os,
        out_rs,
        out_os,
        M,
        R,
    )
    _gather_rows_kernel[grid](
        *args,
        NI=len(kidx),
        HAS_OUTER=outer_dim is not None,
        ROWS=ROWS,
        BLOCK=BLOCK,
    )
    return True


def unsafe_index(inp, indices):
    """Ascend fast path for ``aten._unsafe_index``; generic fallback otherwise."""
    logger.debug("GEMS_ASCEND UNSAFE_INDEX")
    if not indices:
        raise ValueError("at least one index must be provided")
    indices = _check_indices(inp, list(indices))
    if len(indices) > inp.ndim:
        raise IndexError(
            f"too many indices for tensor of dimension {inp.ndim} (got {len(indices)})"
        )

    # Subspace placement is decided over the *original* advanced indices:
    # aten counts scalar (0-d) tensors as advanced indices.
    advanced_dims = [i for i, idx in enumerate(indices) if idx is not None]
    subspace_split = bool(advanced_dims) and advanced_dims != list(
        range(advanced_dims[0], advanced_dims[0] + len(advanced_dims))
    )

    inp, indices = _eliminate_scalar_indices(inp, indices)
    if not indices:
        return inp  # every index was a scalar (0-d) tensor

    indices = indices + [None] * (inp.ndim - len(indices))
    kernel_indices = [idx for idx in indices if idx is not None]
    if not kernel_indices:
        return inp.contiguous()
    if len(kernel_indices) > 1:
        shapes = [idx.shape for idx in kernel_indices]
        if any(s != shapes[0] for s in shapes[1:]):
            kernel_indices = _broadcast_index_tensors(kernel_indices)

    idx_dims = tuple(i for i, idx in enumerate(indices) if idx is not None)
    slice_dims = [i for i, idx in enumerate(indices) if idx is None]
    bcast_pos = 0 if subspace_split else idx_dims[0]
    index_rank = kernel_indices[0].ndim
    out_rank = index_rank + len(slice_dims)
    bcast_out = list(range(bcast_pos, bcast_pos + index_rank))
    rest = [p for p in range(out_rank) if p not in bcast_out]
    out_shape = [0] * out_rank
    for r, p in enumerate(bcast_out):
        out_shape[p] = kernel_indices[0].shape[r]
    for d, p in zip(slice_dims, rest):
        out_shape[p] = inp.shape[d]

    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    if out.numel() == 0:
        return out

    # ---- fast-path eligibility ----
    # int32 gate: max linearized input offset, not numel (strided views can
    # have holes).
    max_in_offset = sum((s - 1) * st for s, st in zip(inp.shape, inp.stride()))
    if (
        len(kernel_indices) <= _MAX_IDX
        and max_in_offset < _INT32_LIMIT
        and out.numel() < _INT32_LIMIT
    ):
        # Flatten contiguous index tensors to rank 1 (a view); non-contiguous
        # broadcast views keep the rank-2 kernel path.
        kidx = kernel_indices
        k_rank = index_rank
        if k_rank > 1 and all(t.is_contiguous() for t in kidx):
            kidx = [t.reshape(-1) for t in kidx]
            k_rank = 1
        if k_rank <= 2:
            # n_trail: trailing suffix of all-slice dims; family R additionally
            # needs this run contiguous in the input.
            idx_set = set(idx_dims)
            n_trail = 0
            for d in range(inp.ndim - 1, -1, -1):
                if d in idx_set:
                    break
                n_trail += 1
            outer_slices = slice_dims[: len(slice_dims) - n_trail]
            trail_contig = True
            if n_trail:
                expect = 1
                for d in range(inp.ndim - 1, inp.ndim - n_trail - 1, -1):
                    if inp.stride(d) != expect:
                        trail_contig = False
                        break
                    expect *= inp.shape[d]
            if n_trail == 0 and len(slice_dims) <= 1:
                if (
                    inp.ndim == 1
                    and len(kidx) == 1
                    and k_rank == 1
                    and inp.is_contiguous()
                ):
                    _launch_1d(inp, kidx[0], out)
                    return out
                slice_dim = slice_dims[0] if slice_dims else None
                if _launch_point(
                    inp, kidx, out, idx_dims, slice_dim, bcast_pos, k_rank
                ):
                    return out
            elif n_trail > 0 and len(outer_slices) <= 1 and trail_contig:
                R = 1
                for d in range(inp.ndim - n_trail, inp.ndim):
                    R *= inp.shape[d]
                S = inp.shape[outer_slices[0]] if outer_slices else 1
                if S <= _MAX_GRID_AXIS:
                    outer_dim = outer_slices[0] if outer_slices else None
                    if _launch_rows(
                        inp,
                        kidx,
                        out,
                        idx_dims,
                        outer_dim,
                        bcast_pos,
                        R,
                        S,
                        k_rank,
                    ):
                        return out

    # Generic fallback (already-prepared arguments, full-stride codegen).
    _unsafe_index_func(inp, kernel_indices, out, idx_dims, bcast_pos)
    return out
