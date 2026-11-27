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
"""Kunlunxin XPU specialization of ``aten::dstack``.

``dstack`` promotes every input with ``atleast_3d`` and concatenates along the
promoted depth axis (dim 2), so it is pure data movement.  The generic Triton
implementation indexes every element with a per-lane ``div/mod`` (``o = g // s``,
``t = g - o * s``) and an int64 pointer path; on XPU that loses affine block-DMA
and lands on the expensive gather-store path (measured ~0.0008-0.06x on the
core benchmark).  This specialization keeps both sides of every copy affine:

* ``inner == 1`` and every promoted depth is 1 (single-token rows): the output
  row is ``[in_0[o], in_1[o], ..., in_{P-1}[o]]``, i.e. the transpose of a
  contiguous ``(P, outer)`` slab.  Copy each input into its slab row with a
  contiguous block-DMA kernel and return the ``(d0, d1, P, ...)`` permuted view
  (the same zero-copy view pattern the vendored ``stack``/``cat`` inner-dim
  paths use); no per-lane interleave select is needed.
* otherwise: each input occupies ``(outer, K_i * inner)`` copy units of the
  output ``(outer, total * inner)`` row-major layout.  That is exactly the
  shape the vendor SDNN row-copy DMA handles natively (independent source and
  destination row strides over unit-stride rows), so the copy rides the
  sdnn ``tle.dsa`` path (``utils.tle_copy``) -- the same path the vendored
  ``copy``/``cat``/``transpose`` family uses.  On KL3 this is tens of times
  faster than per-row Triton block DMAs (core case ``(64,512,512)x3``:
  ~0.25ms vs ~9.7ms fp16, both bit-exact).  Whatever ``tle_copy`` cannot
  express (an innermost extent of 1, exotic layouts, or a triton build without
  the TLE extension) falls back to the unrolled row-slab Triton kernel, which
  stays correct and matches the round-1 performance envelope.

Complex tensors are not accepted by Triton; they are copied through a zero-copy
``.view()`` onto the companion real dtype (copy units), which doubles the copy
length of the last extent.  On this platform ``complex128``/``float64`` cannot
be allocated on the device (the runtime silently downcasts to complex64 /
float32), so the copy-unit dtype is always one the backend supports.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

from _kunlunxin.utils.tle_copy import tle_copy, tle_dma_available

logger = logging.getLogger(__name__)

# Flat slab copy (interleave path): 65536-lane blocks with 16 warps stay near
# bandwidth for large slabs; small shapes pick a block close to the slab size so
# the giant masked block (a 65536-wide masked op costs ~5us) is avoided.
_FLAT_BLOCK = 65536
_FLAT_WARPS = 16

# Row-slab kernel (fallback of the block path): ROWS_PER_PROG rows x BLOCK
# columns per program keeps the program count bounded while every load/store
# stays ``scalar + arange``.
_ROWS_PER_PROG = 8
_ROW_SLAB_WARPS = 4

_REAL_OF_COMPLEX = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


@triton.jit
def _dstack_flat_kernel(
    in_a,
    in_b,
    in_c,
    in_d,
    out_ptr,
    S,
    BOFF,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Copy every input into its contiguous ``(P, S)`` slab row, one launch.

    Program ``(pid_x, n)`` moves ``BLOCK`` elements of input ``(BOFF + n)``
    (addresses ``pid_x*BLOCK + arange``) to ``out + (BOFF + n) * S + arange``.
    Both sides are contiguous block DMA; the only branch is the per-program
    input pointer (no per-lane div/mod).
    """
    pid_x = tl.program_id(0)
    n = tl.program_id(1)
    offs = pid_x * BLOCK + tl.arange(0, BLOCK)
    ptr = tl.where(n == 0, in_a, tl.where(n == 1, in_b, tl.where(n == 2, in_c, in_d)))
    if NEED_MASK:
        m = offs < S
        v = tl.load(ptr + offs, mask=m)
        tl.store(out_ptr + (BOFF + n) * S + offs, v, mask=m)
    else:
        v = tl.load(ptr + offs)
        tl.store(out_ptr + (BOFF + n) * S + offs, v)


@triton.jit
def _dstack_row_slab_kernel(
    in_ptr,
    out_ptr,
    S,
    row_stride,
    col_off,
    rows,
    ROWS_PER_PROG: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Copy ``rows`` contiguous runs of ``S`` copy units into a strided slab.

    Program ``(bx, rowh)`` owns rows ``[rowh*RP, rowh*RP + RP)`` and copies block
    ``[bx*BLOCK, bx*BLOCK + BLOCK)`` of each row: source ``r*S + offs`` (row-major
    contiguous run), destination ``r*row_stride + col_off + offs`` (same-width
    slot inside the output row).  Both addresses are ``scalar + tl.arange`` so
    the XPU backend keeps the contiguous block-DMA path.  This is the fallback
    when the SDNN row-copy DMA cannot express the layout (see ``_tle_row_copy``).
    """
    bx = tl.program_id(0)
    rh = tl.program_id(1).to(tl.int64)
    offs = bx * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    for i in tl.static_range(ROWS_PER_PROG):
        r = rh * ROWS_PER_PROG + i
        src = in_ptr + r * S + offs
        dst = out_ptr + r * row_stride + col_off + offs
        if NEED_MASK:
            m = (offs < S) & (r < rows)
            v = tl.load(src, mask=m)
            tl.store(dst, v, mask=m)
        else:
            v = tl.load(src)
            tl.store(dst, v)


def _atleast_3d(shape):
    """atleast_3d promotion: 0-D -> (1,1,1), 1-D -> (1,N,1), 2-D -> (M,N,1)."""
    dims = [int(x) for x in shape]
    if len(dims) == 0:
        return [1, 1, 1]
    if len(dims) == 1:
        return [1, dims[0], 1]
    if len(dims) == 2:
        return [dims[0], dims[1], 1]
    return dims


def _prod(values) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return int(out)


def _real_dtype(dtype):
    return _REAL_OF_COMPLEX.get(dtype)


def _as_copy_tensor(tensor, real_dtype):
    """Zero-copy reinterpretation of ``tensor`` in the copy element dtype."""
    if real_dtype is None:
        return tensor
    if tensor.dim() == 0:
        return tensor.reshape(1).view(real_dtype)
    return tensor.view(real_dtype)


def _pick_row_block(seg: int) -> int:
    """Largest power-of-two <= 1024 that divides ``seg`` (fallback 1024+mask)."""
    for block in (512, 256, 128):
        if seg % block == 0:
            return block
    return 1024


def _tle_row_copy(source, out_view, outer, seg, row, col_off):
    """Move ``(outer, seg)`` copy units into the output row window via the SDNN
    row-copy DMA (``utils.tle_copy``).

    Source rows are contiguous runs of ``seg`` units; the destination window is
    the matching ``(outer, seg)`` slice of the output rows of ``row`` units.
    Independent src/dst row strides are exactly what the SDNN ``dma_cfg_2d``
    engine takes, so the copy stays on the vendor DMA path.  ``tle_copy`` is
    best-effort: any layout it cannot express (or any triton build without the
    TLE extension) returns False and the caller keeps its Triton fallback.
    """
    if not tle_dma_available() or outer == 0 or seg == 0:
        return False
    try:
        src2d = source.view(outer, seg)
        dst2d = out_view.view(outer, row)[:, col_off : col_off + seg]
    except RuntimeError:
        return False
    if src2d.numel() != dst2d.numel():
        return False
    return tle_copy(src2d, dst2d)


def _dstack_forward(tensors, out=None, **kwargs):
    """``aten::dstack(Tensor[] tensors) -> Tensor``, plus the ``out=`` overload."""
    del kwargs
    items = [tensors] if isinstance(tensors, torch.Tensor) else list(tensors)
    if len(items) == 0:
        raise RuntimeError("dstack expects a non-empty TensorList")
    for item in items:
        if not isinstance(item, torch.Tensor):
            raise TypeError(
                "dstack(): argument 'tensors' must be a list of Tensors, "
                f"but found {type(item).__name__}"
            )

    # Both kernels address their input with a flat linear index, which only
    # matches logical element order for contiguous tensors.
    items = [item if item.is_contiguous() else item.contiguous() for item in items]

    promoted = [_atleast_3d(item.shape) for item in items]
    first = promoted[0]
    outer = first[0] * first[1]
    inner = _prod(first[3:])
    for shape in promoted[1:]:
        if shape[0] != first[0] or shape[1] != first[1] or _prod(shape[3:]) != inner:
            raise RuntimeError(
                "Sizes of tensors must match except in dimension 2. "
                f"Expected size {first[0]} and {first[1]} but got size {shape[0]} "
                f"and {shape[1]}"
            )

    depth = [shape[2] for shape in promoted]
    total = sum(depth)
    result_shape = [first[0], first[1], total, *first[3:]]
    dtype = items[0].dtype
    for item in items[1:]:
        dtype = torch.promote_types(dtype, item.dtype)
    result = (
        torch.empty(result_shape, dtype=dtype, device=items[0].device)
        if out is None
        else out
    )
    # Cast every input to the promoted dtype so the copy-element width used by
    # the kernels matches the one implied by the result.  On this platform the
    # cast is a vendor device kernel (no CPU fallback).
    items = [item.to(dtype=dtype).contiguous() for item in items]

    real_dtype = _real_dtype(dtype)
    unit = 1 if real_dtype is None else 2

    if result.numel() == 0:
        return result

    # Single-token interleave: every input contributes exactly one element
    # (complex: one copy-unit pair) per outer row.  Build the contiguous
    # ``(P, outer[, unit])`` slab and return the matching transposed view, so
    # every copy is a contiguous block DMA with no per-lane interleave selects.
    if inner == 1 and len(set(depth)) == 1 and depth[0] == 1:
        if unit == 1:
            sources = [items[i] if i < len(items) else items[0] for i in range(4)]
            slab = torch.empty((len(items), outer), dtype=dtype, device=items[0].device)
            block = _FLAT_BLOCK
            while block > 1024 and block // 2 >= outer:
                block //= 2
            grid = (triton.cdiv(outer, block), len(items))
            _dstack_flat_kernel[grid](
                sources[0],
                sources[1],
                sources[2],
                sources[3],
                slab,
                outer,
                0,
                BLOCK=block,
                NEED_MASK=outer % block != 0,
                num_warps=_FLAT_WARPS if block >= _FLAT_BLOCK else 4,
            )
            nd = len(first)
            view_shape = [len(items)] + list(first)
            perm = [1, 2, 0] + list(range(3, nd))
            res = slab.reshape(view_shape).permute(tuple(perm))
            if out is None:
                return res
            out.copy_(res)
            return result
        # Complex single-token rows: copy units are 2-wide groups, so the flat
        # slab cannot be transposed by metadata alone; fall through to the
        # general row-slab (per-input, S == 2) path, which is correct for all
        # the (small) complex test shapes.

    out_view = _as_copy_tensor(result, real_dtype)
    out_row = total * inner * unit
    offset = 0
    for item, depth_i in zip(items, depth):
        if item.numel() > 0:
            source = _as_copy_tensor(item, _real_dtype(item.dtype))
            seg = depth_i * inner * unit
            count = outer * seg
            if count > 0:
                col_off = offset * inner * unit
                if not _tle_row_copy(source, out_view, outer, seg, out_row, col_off):
                    block = _pick_row_block(seg)
                    need_mask = (seg % block != 0) or (outer % _ROWS_PER_PROG != 0)
                    grid = (triton.cdiv(seg, block), triton.cdiv(outer, _ROWS_PER_PROG))
                    _dstack_row_slab_kernel[grid](
                        source,
                        out_view,
                        seg,
                        out_row,
                        col_off,
                        outer,
                        ROWS_PER_PROG=_ROWS_PER_PROG,
                        BLOCK=block,
                        NEED_MASK=need_mask,
                        num_warps=_ROW_SLAB_WARPS,
                    )
        offset += depth_i
    return result


class _Dstack(torch.autograd.Function):
    """Autograd bridge for ``dstack``: backward is a slice of the output grad."""

    @staticmethod
    def forward(ctx, *items):
        ctx.shapes = [tuple(t.shape) for t in items]
        ctx.dtypes = [t.dtype for t in items]
        ctx.depths = [_atleast_3d(t.shape)[2] for t in items]
        return _dstack_forward(list(items))

    @staticmethod
    def backward(ctx, grad_out):
        grads = []
        offset = 0
        for shape, dtype, depth, needs in zip(
            ctx.shapes, ctx.dtypes, ctx.depths, ctx.needs_input_grad
        ):
            if needs:
                grads.append(grad_out.narrow(2, offset, depth).reshape(shape).to(dtype))
            else:
                grads.append(None)
            offset += depth
        return tuple(grads)


def run(tensors, out=None, **kwargs):
    """``aten::dstack(Tensor[] tensors) -> Tensor``, plus the ``out=`` overload."""
    items = [tensors] if isinstance(tensors, torch.Tensor) else list(tensors)
    if (
        out is None
        and torch.is_grad_enabled()
        and all(isinstance(t, torch.Tensor) for t in items)
        and any(t.requires_grad for t in items)
    ):
        return _Dstack.apply(*items)
    return _dstack_forward(tensors, out=out, **kwargs)


dstack = run

__all__ = ["run", "dstack"]
