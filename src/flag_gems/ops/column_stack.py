# Copyright 2026, The FlagOS Contributors.
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
from typing import List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


_BLOCK = 1024
_BLOCK_MIN = 64
_BLOCK_MAX = 4096
_SPAN_ELEMS = 16384
_MIN_PROGRAMS = 64
_TILE_MAX_ELEMS = 1 << 16
_TILE_LANES = 4096
# Big-part path: one program copies _BIG_TILE_ROWS contiguous rows.
_BIG_TILE_ROWS = 16
_BIG_TILE_MIN = 1 << 16
_I32_MAX = 1 << 31


def _pick_block(row_len):
    """Per-part program granularity: cover a whole row when it is small."""
    block = _BLOCK_MIN
    while block < row_len and block < _BLOCK_MAX:
        block <<= 1
    return block


@triton.jit
def _copy_rows_kernel(
    in_ptr,
    out_ptr,
    in_row_stride,
    out_row_stride,
    col_base,
    row_len,
    n_col_blocks,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy one part, viewed as a logical (rows, row_len) tile, into ``out``.

    Each program owns one contiguous run of ``BLOCK`` columns of a single
    logical row, so no per-element index division is needed: the program
    identifier is split into (row, column block) once.
    """
    pid = tl.program_id(0).to(tl.int64)
    row = pid // n_col_blocks
    col = (pid - row * n_col_blocks) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    src = in_ptr + row * in_row_stride + col
    dst = out_ptr + row * out_row_stride + col_base + col
    if EVEN:
        val = tl.load(src)
        tl.store(dst, val.to(out_ptr.dtype.element_ty))
    else:
        mask = col < row_len
        val = tl.load(src, mask=mask)
        tl.store(dst, val.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _tile_copy(
    in_ptr,
    out_ptr,
    in_row_stride,
    out_row_stride,
    col_base,
    pid_r,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """One unmasked 2-D tile: ``BLOCK_R`` rows x ``BLOCK_C`` columns."""
    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)
    c = tl.arange(0, BLOCK_C).to(tl.int64)
    src = in_ptr + r[:, None] * in_row_stride + c[None, :]
    dst = out_ptr + r[:, None] * out_row_stride + col_base + c[None, :]
    val = tl.load(src)
    tl.store(dst, val.to(out_ptr.dtype.element_ty))


@triton.jit
def _copy_slots_kernel(
    out_ptr,
    in0,
    in1,
    in2,
    in3,
    out_row_stride,
    rs0,
    rs1,
    rs2,
    rs3,
    cb0,
    cb1,
    cb2,
    cb3,
    K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """One launch writing several same-shaped small parts into ``out``.

    Grid axis 1 selects the part and each program takes exactly one branch, so
    every access stays unmasked (per-lane masks measured ~25x slower here).
    """
    pid_r = tl.program_id(0).to(tl.int64)
    slot = tl.program_id(1)
    if slot == 0:
        _tile_copy(in0, out_ptr, rs0, out_row_stride, cb0, pid_r, BLOCK_R, BLOCK_C)
    if K > 1:
        if slot == 1:
            _tile_copy(in1, out_ptr, rs1, out_row_stride, cb1, pid_r, BLOCK_R, BLOCK_C)
    if K > 2:
        if slot == 2:
            _tile_copy(in2, out_ptr, rs2, out_row_stride, cb2, pid_r, BLOCK_R, BLOCK_C)
    if K > 3:
        if slot == 3:
            _tile_copy(in3, out_ptr, rs3, out_row_stride, cb3, pid_r, BLOCK_R, BLOCK_C)


@triton.jit
def _copy_big_tile_kernel(
    in_ptr,
    out_ptr,
    in_row_stride,
    out_row_stride,
    col_base,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Tall unmasked tile (BLOCK_R rows x BLOCK_C columns) with 32-bit offsets.

    Large parts are pure bandwidth; on this backend the realised bandwidth of a
    copy grows with the number of elements per program (measured on gcu:2,
    4096x4096 fp16 part: 1 row/program = 31 GB/s, 4 = 91, 16 = 187 GB/s) because
    the fixed per-program cost is amortised.  Addresses stay in int32, which is
    why this kernel is only used when the host proved every offset < 2**31.
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    val = tl.load(in_ptr + r[:, None] * in_row_stride + c[None, :])
    tl.store(
        out_ptr + r[:, None] * out_row_stride + col_base + c[None, :],
        val.to(out_ptr.dtype.element_ty),
    )


@triton.jit
def _copy_tile_kernel(
    in_ptr,
    out_ptr,
    in_row_stride,
    out_row_stride,
    col_base,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Unmasked 2-D tile copy: ``BLOCK_R`` whole rows of ``BLOCK_C`` columns.

    Used for small parts, where one program per row would spend all its time
    in per-program overhead.  The grid exactly covers the tile space, so no
    masks are needed.
    """
    pid_r = tl.program_id(0).to(tl.int64)
    pid_c = tl.program_id(1).to(tl.int64)
    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C).to(tl.int64)
    src = in_ptr + r[:, None] * in_row_stride + c[None, :]
    dst = out_ptr + r[:, None] * out_row_stride + col_base + c[None, :]
    val = tl.load(src)
    tl.store(dst, val.to(out_ptr.dtype.element_ty))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=["row_len", "n_elem"],
)
@triton.jit
def _copy_span_kernel(
    in_ptr,
    out_ptr,
    n_elem,
    in_row_delta,
    out_row_delta,
    col_base,
    row_len,
    n_chunks,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy a contiguous multi-chunk span of one part in a single program.

    ``BLOCK`` divides ``row_len``, so a chunk never crosses a logical row:
    ``row = chunk_start // row_len`` is a scalar per iteration and the in/out
    offsets only need the per-row stride correction ``*_row_delta``.  When the
    span covers whole chunks the mask is dropped so the backend can emit wide
    vector accesses.
    """
    pid = tl.program_id(0).to(tl.int64)
    chunk = pid * n_chunks
    for i in range(0, n_chunks):
        start = (chunk + i) * BLOCK
        row = start // row_len
        offs = start + tl.arange(0, BLOCK).to(tl.int64)
        src = in_ptr + offs + row * in_row_delta
        dst = out_ptr + offs + row * out_row_delta + col_base
        if EVEN:
            val = tl.load(src)
            tl.store(dst, val.to(out_ptr.dtype.element_ty))
        else:
            mask = offs < n_elem
            val = tl.load(src, mask=mask)
            tl.store(dst, val.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _copy_generic_kernel(
    in_ptr,
    out_ptr,
    n_elem,
    base_out,
    s0,
    s1,
    s2,
    s3,
    i0,
    i1,
    i2,
    i3,
    o0,
    o1,
    o2,
    o3,
    BLOCK: tl.constexpr,
):
    """Strided copy of one part (rank <= 4) into a possibly strided ``out``."""
    pid = tl.program_id(0).to(tl.int64)
    idx = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < n_elem
    rem = idx
    off_in = tl.zeros((BLOCK,), dtype=tl.int64)
    off_out = tl.zeros((BLOCK,), dtype=tl.int64) + base_out

    c3 = rem % s3
    rem = rem // s3
    off_in += c3 * i3
    off_out += c3 * o3

    c2 = rem % s2
    rem = rem // s2
    off_in += c2 * i2
    off_out += c2 * o2

    c1 = rem % s1
    rem = rem // s1
    off_in += c1 * i1
    off_out += c1 * o1

    off_in += rem * i0
    off_out += rem * o0

    val = tl.load(in_ptr + off_in, mask=mask)
    tl.store(out_ptr + off_out, val.to(out_ptr.dtype.element_ty), mask=mask)


def _rows_fast_path(t, ndim):
    """True when the trailing dims of ``t`` are exactly contiguous."""
    st = t.stride()
    acc = 1
    for d in range(ndim - 1, 0, -1):
        if st[d] != acc:
            return False
        acc *= t.shape[d]
    return True


def column_stack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("GEMS COLUMN_STACK")
    parts = []
    for t in tensors:
        if t.dim() < 2:
            t = t.view(t.numel(), 1)
        parts.append(t)
    if not parts:
        raise RuntimeError("column_stack expects a non-empty TensorList")

    first = parts[0]
    ndim = first.dim()
    shape0 = first.shape
    out_shape = list(shape0)
    result_dtype = first.dtype
    width = first.size(1)
    for idx in range(1, len(parts)):
        t = parts[idx]
        tshape = t.shape
        if t.dim() != ndim:
            raise RuntimeError(
                "Tensors must have same number of dimensions: got "
                f"{ndim} and {t.dim()}"
            )
        for dim in range(ndim):
            if dim != 1 and tshape[dim] != shape0[dim]:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension 1. "
                    f"Expected size {shape0[dim]} but got size {tshape[dim]} "
                    f"for tensor number {idx} in the list."
                )
        if t.dtype != result_dtype:
            result_dtype = torch.promote_types(result_dtype, t.dtype)
        width += t.size(1)
    out_shape[1] = width

    if out is None:
        out = torch.empty(out_shape, dtype=result_dtype, device=first.device)

    if not torch.can_cast(result_dtype, out.dtype):
        raise TypeError(
            "torch.cat(): input types can't be cast to the desired "
            f"output type {out.dtype}"
        )

    if list(out.shape) != out_shape:
        out.resize_(out_shape)
    if out.numel() == 0:
        return out

    rows = shape0[0]
    trailing = 1
    for dim in range(2, ndim):
        trailing *= shape0[dim]
    out_contig = out.is_contiguous()
    out_row_stride = out.numel() // rows if out_contig else 0

    n_parts = len(parts)
    if out_contig and 2 <= n_parts <= 4:
        dtype0 = parts[0].dtype
        row_len0 = parts[0].size(1) * trailing
        slot_ok = rows * row_len0 <= _TILE_MAX_ELEMS
        if slot_ok:
            for t in parts:
                if (
                    t.dtype != dtype0
                    or t.numel() != rows * row_len0
                    or not (t.is_contiguous() or _rows_fast_path(t, ndim))
                ):
                    slot_ok = False
                    break
        if slot_ok:
            block = _pick_block(row_len0)
            if row_len0 == block:
                max_br = _TILE_LANES // block
                br = 1
                while rows // br > 8 and br * 2 <= max_br:
                    br *= 2
                while rows % br:
                    br //= 2
                rs = [0, 0, 0, 0]
                cb = [0, 0, 0, 0]
                off = 0
                for i, t in enumerate(parts):
                    rs[i] = t.stride(0)
                    cb[i] = off * trailing
                    off += t.size(1)
                _copy_slots_kernel[(rows // br, n_parts)](
                    out,
                    parts[0],
                    parts[1],
                    parts[2] if n_parts > 2 else parts[0],
                    parts[3] if n_parts > 3 else parts[0],
                    out_row_stride,
                    rs[0],
                    rs[1],
                    rs[2],
                    rs[3],
                    cb[0],
                    cb[1],
                    cb[2],
                    cb[3],
                    K=n_parts,
                    BLOCK_R=br,
                    BLOCK_C=block,
                )
                return out

    col = 0
    for t in parts:
        n_elem = t.numel()
        width_t = t.size(1)
        if n_elem:
            row_len = width_t * trailing
            if out_contig and _rows_fast_path(t, ndim):
                block = _pick_block(row_len)
                col_start = col * trailing
                if (
                    row_len % block == 0
                    and n_elem >= _BIG_TILE_MIN
                    and rows >= _BIG_TILE_ROWS
                    and out_row_stride * rows + col_start + row_len <= _I32_MAX
                    and n_elem <= _I32_MAX
                ):
                    br = _BIG_TILE_ROWS
                    while rows % br:
                        br //= 2
                    if br > 1:
                        _copy_big_tile_kernel[(rows // br, row_len // block)](
                            t,
                            out,
                            t.stride(0),
                            out_row_stride,
                            col_start,
                            BLOCK_R=br,
                            BLOCK_C=block,
                        )
                        col += width_t
                        continue
                n_col_blocks = (row_len + block - 1) // block
                n_chunks = 1
                if row_len >= block and row_len % block == 0:
                    chunks_total = n_elem // block
                    if chunks_total >= _MIN_PROGRAMS:
                        span = max(1, _SPAN_ELEMS // block)
                        n_chunks = min(span, chunks_total // _MIN_PROGRAMS)
                if n_chunks > 1:
                    _copy_span_kernel[(triton.cdiv(n_elem // block, n_chunks),)](
                        t,
                        out,
                        n_elem,
                        t.stride(0) - row_len,
                        out_row_stride - row_len,
                        col_start,
                        row_len,
                        n_chunks,
                        EVEN=(n_elem % (block * n_chunks) == 0),
                        BLOCK=block,
                    )
                elif n_elem <= _TILE_MAX_ELEMS and row_len % block == 0:
                    max_br = _TILE_LANES // block
                    br = 1
                    while rows // br > 8 and br * 2 <= max_br:
                        br *= 2
                    while rows % br:
                        br //= 2
                    _copy_tile_kernel[(rows // br, row_len // block)](
                        t,
                        out,
                        t.stride(0),
                        out_row_stride,
                        col_start,
                        BLOCK_R=br,
                        BLOCK_C=block,
                    )
                else:
                    _copy_rows_kernel[(rows * n_col_blocks,)](
                        t,
                        out,
                        t.stride(0),
                        out_row_stride,
                        col_start,
                        row_len,
                        n_col_blocks,
                        EVEN=(row_len % block == 0),
                        BLOCK=block,
                    )
            else:
                if ndim > 4:
                    raise NotImplementedError(
                        "column_stack_out supports at most 4 dimensions"
                    )
                ss = [1, 1, 1, 1]
                ii = [0, 0, 0, 0]
                oo = [0, 0, 0, 0]
                for dim in range(ndim):
                    ss[dim] = t.shape[dim]
                    ii[dim] = t.stride(dim)
                    oo[dim] = out.stride(dim)
                _copy_generic_kernel[(triton.cdiv(n_elem, _BLOCK),)](
                    t,
                    out,
                    n_elem,
                    col * out.stride(1),
                    ss[0],
                    ss[1],
                    ss[2],
                    ss[3],
                    ii[0],
                    ii[1],
                    ii[2],
                    ii[3],
                    oo[0],
                    oo[1],
                    oo[2],
                    oo[3],
                    BLOCK=_BLOCK,
                )
        col += width_t

    return out


def column_stack_out(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS COLUMN_STACK_OUT")
    return column_stack(tensors, out=out)
