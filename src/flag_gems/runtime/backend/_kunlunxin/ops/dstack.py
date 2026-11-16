# Copyright 2026 The FlagOS Contributors
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
"""Kunlunxin (XPU3) specialization of ``aten::dstack``.

The generic implementation interleaves the per-input blocks with two kernels
whose inner index is a *runtime* ``int64`` scalar:

* ``_dstack_copy_kernel`` computes ``o = g // seg`` per element (soft integer
  division) and forms every address in ``int64`` from a lane-varying vector.
* ``_dstack_token_kernel`` builds an ``[OB, KP]`` 2-D tile, performs four
  masked loads against four different pointers and selects with a 3-deep
  ``tl.where`` chain.

Both lower badly on XPU3.  Measured on the benchmark matrix they run at
~1.2 GB/s (copy path) and a fixed ~270 ms per launch (token path, independent
of the payload), i.e. 0.000-0.066x of ``torch.dstack``.

This override keeps the exact same operator contract (atleast_3d promotion,
dim-2 concatenation, dtype promotion, complex handling, empty / mismatched
shape / non-tensor errors) but replaces the two kernels with three backends
that match what the hardware does well:

1. **Copy path** (``inner > 1`` or unequal depths): the destination of an input
   is a run of ``seg`` contiguous elements repeated over ``outer`` rows with a
   larger row stride.  That is exactly a 2-D row copy, so it goes through the
   vendor SDNN DMA (``tle_copy``: one contiguous ``cols``-wide row transfer per
   program, independent source/destination row strides).  For small inputs the
   per-call TLE host setup dominates, so a masked 1-D row kernel with a scalar
   int64 row base is used instead (no per-element division).

2. **Token path** (``inner == 1`` and every depth is 1): the output is the
   ``P`` inputs interleaved element by element.  Rather than gather across four
   pointers, the interleave is done in registers by **packing P input elements
   into one 4-byte / 8-byte word** with shifts, so both the loads and the
   stores are contiguous 1-D streams.  Supported: P=2 (16/32-bit elements),
   P=3 (16/32-bit elements, even ``outer``), P=4 (16-bit -> one int64 word,
   32-bit -> two int64 words per row).  Anything else falls back to a flat
   masked kernel.

No CPU / ATen / native computation is used: the only torch calls are
``torch.empty`` (allocation), metadata ``view``/``reshape`` and ``.contiguous()``
for non-contiguous inputs (a device-side data move that the generic operator
also performs).
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)

_BLOCK_MAX = 65536
_WARPS = 8
_MAX_GROUP = 4

# Copies at or below this element count are dominated by the TLE host setup
# (12-40 us per call), so they take the plain masked row kernel instead.
_SMALL_COPY_ELEMS = 1 << 17

_REAL_OF_COMPLEX = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}

# The multi-pointer select cannot compile for fp8 element types (the masked load
# cannot produce an fp8 value from an integer default); those stay on the copy path.
_MULTI_SELECT_UNSUPPORTED = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
    torch.float8_e8m0fnu,
}


def _next_pow2(value: int) -> int:
    power = 1
    while power < value:
        power *= 2
    return power


# ---------------------------------------------------------------------------
# Copy path kernels
# ---------------------------------------------------------------------------
@triton.jit
def _dstack_copy_row_kernel(src, dst, seg, out_row, out_off, BLOCK: tl.constexpr):
    """Copy one contiguous chunk of one outer row of one input.

    ``program_id(0)`` is the chunk index inside the ``seg``-element run,
    ``program_id(1)`` is the outer row.  Both row bases are scalar ``int64``
    (``o * seg`` / ``o * out_row``); only the intra-row offset, bounded by
    ``BLOCK``, is a lane-varying vector.  The mask is mandatory: an unmasked
    vector access from a runtime scalar base is miscompiled on this backend
    (see ``technique-xpu3-per-run-affine-copy``).
    """
    c = tl.program_id(0)
    o = tl.program_id(1)
    start = c * BLOCK
    idx = tl.arange(0, BLOCK)
    lim = tl.minimum(seg - start, BLOCK)
    mask = idx < lim
    so = o.to(tl.int64) * seg + start
    do = o.to(tl.int64) * out_row + out_off + start
    tl.store(dst + do + idx, tl.load(src + so + idx, mask=mask), mask=mask)


# ---------------------------------------------------------------------------
# Token path kernels (packed interleave)
# ---------------------------------------------------------------------------
@triton.jit
def _dstack_pack2_w32(s0, s1, dst, n, BLOCK: tl.constexpr, MASK: tl.constexpr):
    """Two 16-bit inputs -> one 32-bit word per output row."""
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if MASK:
        m = r < n
        a = tl.load(s0 + r, mask=m, other=0)
        b = tl.load(s1 + r, mask=m, other=0)
        w = (a.to(tl.int32) & 0xFFFF) | (b.to(tl.int32) << 16)
        tl.store(dst + r, w, mask=m)
    else:
        a = tl.load(s0 + r)
        b = tl.load(s1 + r)
        w = (a.to(tl.int32) & 0xFFFF) | (b.to(tl.int32) << 16)
        tl.store(dst + r, w)


@triton.jit
def _dstack_pack2_w64(s0, s1, dst, n, BLOCK: tl.constexpr, MASK: tl.constexpr):
    """Two 32-bit inputs -> one 64-bit word per output row."""
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if MASK:
        m = r < n
        a = tl.load(s0 + r, mask=m, other=0)
        b = tl.load(s1 + r, mask=m, other=0)
        w = (a.to(tl.int64) & 0xFFFFFFFF) | (b.to(tl.int64) << 32)
        tl.store(dst + r, w, mask=m)
    else:
        a = tl.load(s0 + r)
        b = tl.load(s1 + r)
        w = (a.to(tl.int64) & 0xFFFFFFFF) | (b.to(tl.int64) << 32)
        tl.store(dst + r, w)


@triton.jit
def _dstack_pack3_w32(s0, s1, s2, dst, nw, BLOCK: tl.constexpr, MASK: tl.constexpr):
    """Three 16-bit inputs -> three 32-bit words for every two output rows.

    A source 32-bit word holds two adjacent elements.  Group ``g`` covers two
    output rows, and its three output words are
    ``[a0,b0] [c0,a1] [b1,c1]`` - a pure bit shuffle of the three source words.
    One lane owns one output word, so the store is contiguous; the three loads
    are contiguous as well (three consecutive lanes read the same source word).
    """
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = q // 3
    t = q - g * 3
    if MASK:
        m = q < nw
        aw = tl.load(s0 + g, mask=m, other=0).to(tl.uint32)
        bw = tl.load(s1 + g, mask=m, other=0).to(tl.uint32)
        cw = tl.load(s2 + g, mask=m, other=0).to(tl.uint32)
    else:
        aw = tl.load(s0 + g).to(tl.uint32)
        bw = tl.load(s1 + g).to(tl.uint32)
        cw = tl.load(s2 + g).to(tl.uint32)
    w0 = (aw & 0xFFFF) | ((bw & 0xFFFF) << 16)
    w1 = (cw & 0xFFFF) | (aw & 0xFFFF0000)
    w2 = (bw >> 16) | (cw & 0xFFFF0000)
    w = tl.where(t == 0, w0, tl.where(t == 1, w1, w2)).to(tl.int32)
    if MASK:
        tl.store(dst + q, w, mask=q < nw)
    else:
        tl.store(dst + q, w)


@triton.jit
def _dstack_pack3_w64(s0, s1, s2, dst, nw, BLOCK: tl.constexpr, MASK: tl.constexpr):
    """Three 32-bit inputs -> three 64-bit words for every two output rows."""
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = q // 3
    t = q - g * 3
    m = q < nw
    aw = tl.load(s0 + g, mask=m, other=0).to(tl.uint64)
    bw = tl.load(s1 + g, mask=m, other=0).to(tl.uint64)
    cw = tl.load(s2 + g, mask=m, other=0).to(tl.uint64)
    low = 0xFFFFFFFF
    w0 = (aw & low) | ((bw & low) << 32)
    w1 = (cw & low) | (aw & 0xFFFFFFFF00000000)
    w2 = (bw >> 32) | (cw & 0xFFFFFFFF00000000)
    w = tl.where(t == 0, w0, tl.where(t == 1, w1, w2)).to(tl.int64)
    tl.store(dst + q, w, mask=m)


@triton.jit
def _dstack_pack4_w64_16(s0, s1, s2, s3, dst, n, BLOCK: tl.constexpr, MASK: tl.constexpr):
    """Four 16-bit inputs -> one 64-bit word per output row."""
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = r < n
    a = tl.load(s0 + r, mask=m, other=0)
    b = tl.load(s1 + r, mask=m, other=0)
    c = tl.load(s2 + r, mask=m, other=0)
    e = tl.load(s3 + r, mask=m, other=0)
    w0 = (a.to(tl.int32) & 0xFFFF) | (b.to(tl.int32) << 16)
    w1 = (c.to(tl.int32) & 0xFFFF) | (e.to(tl.int32) << 16)
    w = (w0.to(tl.int64) & 0xFFFFFFFF) | (w1.to(tl.int64) << 32)
    if MASK:
        tl.store(dst + r, w, mask=m)
    else:
        tl.store(dst + r, w)


@triton.jit
def _dstack_pack4_w64_32(s0, s1, s2, s3, dst, nw, BLOCK: tl.constexpr, MASK: tl.constexpr):
    """Four 32-bit inputs -> two 64-bit words per output row (one per lane)."""
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r = q // 2
    t = q - r * 2
    m = q < nw
    a = tl.load(s0 + r, mask=m, other=0).to(tl.int64)
    b = tl.load(s1 + r, mask=m, other=0).to(tl.int64)
    c = tl.load(s2 + r, mask=m, other=0).to(tl.int64)
    e = tl.load(s3 + r, mask=m, other=0).to(tl.int64)
    low = 0xFFFFFFFF
    w0 = (a & low) | (b << 32)
    w1 = (c & low) | (e << 32)
    w = tl.where(t == 0, w0, w1)
    tl.store(dst + q, w, mask=m)


@triton.jit
def _dstack_interleave_kernel(s0, s1, s2, s3, dst, n, P: tl.constexpr, BLOCK: tl.constexpr):
    """Fallback interleave: one lane per output element, masked source loads."""
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = j < n
    row = j // P
    sid = j - row * P
    v0 = tl.load(s0 + row, mask=mask, other=0)
    v1 = tl.load(s1 + row, mask=mask, other=0)
    v2 = tl.load(s2 + row, mask=mask, other=0)
    v3 = tl.load(s3 + row, mask=mask, other=0)
    value = tl.where(sid == 0, v0, tl.where(sid == 1, v1, tl.where(sid == 2, v2, v3)))
    tl.store(dst + j, value, mask=mask)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _prod(values) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return int(out)


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


def _real_dtype(dtype):
    return _REAL_OF_COMPLEX.get(dtype)


def _as_copy_tensor(tensor, real_dtype):
    """Zero-copy reinterpretation of ``tensor`` in the copy element dtype."""
    if real_dtype is None:
        return tensor
    if tensor.dim() == 0:
        # A 0-D complex tensor cannot be reinterpreted in place (the element
        # size changes); materialise the single element as a 1-D view first.
        return tensor.reshape(1).view(real_dtype)
    return tensor.view(real_dtype)


def _copy_block(seg: int) -> int:
    return min(_next_pow2(max(int(seg), 1)), 4096)


def _token_block(kind: str, n: int) -> int:
    """Bounded tile size per packed kernel, from local measurements."""
    if kind in ("p2_16", "p2_32", "p3_32"):
        cap = 16384
    elif kind == "p3_16":
        cap = 65536
    elif kind == "p4_16":
        cap = 8192
    else:  # p4_32
        cap = 16384
    if n >= 2 ** 14:
        return cap
    if n >= 2 ** 12:
        return min(4096, cap)
    if n >= 2 ** 10:
        return 1024
    return 256


def _aligned(tensor) -> int:
    return tensor.data_ptr()


def _run_copy_path(
    items, depth, outer, inner, unit, out_view, result_dtype, real_dtype
):
    total = sum(depth)
    out_row = total * inner * unit
    offset = 0
    any_real_tle = False
    out_dtype = out_view.dtype
    for item, depth_i in zip(items, depth):
        if item.numel() == 0:
            offset += depth_i
            continue
        source = _as_copy_tensor(item, _real_dtype(item.dtype))
        seg = depth_i * inner * unit
        count = outer * seg
        if count == 0:
            offset += depth_i
            continue
        dst = out_view
        if (
            count <= _SMALL_COPY_ELEMS
            and source.element_size() in (2, 4)
            and source.dtype == out_dtype
        ):
            # Host-bound small copy: plain masked row kernel, no per-element division.
            block = _copy_block(seg)
            grid = (triton.cdiv(seg, block), outer)
            _dstack_copy_row_kernel[grid](
                source,
                dst,
                seg,
                out_row,
                offset * inner * unit,
                BLOCK=block,
                num_warps=_WARPS,
            )
        else:
            # 2-D vendor DMA: contiguous ``seg``-wide rows, independent row strides.
            src2d = source.reshape(-1).view(outer, seg)
            dst2d = (
                out_view.reshape(-1)
                .view(outer, out_row)[:, offset * inner * unit: offset * inner * unit + seg]
            )
            if not tle_copy(src2d, dst2d):
                # Layout / dtype tle cannot express: element-wise row kernel.
                block = _copy_block(seg)
                grid = (triton.cdiv(seg, block), outer)
                _dstack_copy_row_kernel[grid](
                    source,
                    dst,
                    seg,
                    out_row,
                    offset * inner * unit,
                    BLOCK=block,
                    num_warps=_WARPS,
                )
            else:
                any_real_tle = True
        offset += depth_i
    return any_real_tle


def _launch_pack(items, result, outer):
    """Try a packed-interleave kernel.  Returns True on success, False if the
    configuration is not packable (caller then uses the flat fallback)."""
    P = len(items)
    es = items[0].element_size()
    if any(it.dtype != result.dtype for it in items):
        return False
    if any(it.dtype in _MULTI_SELECT_UNSUPPORTED for it in items):
        return False
    if not result.is_contiguous():
        return False
    try:
        if P == 2 and es == 2:
            if result.data_ptr() % 4 != 0:
                return False
            n = outer
            blk = _token_block("p2_16", n)
            mask = (n % blk) != 0
            _dstack_pack2_w32[(triton.cdiv(n, blk),)](
                items[0].reshape(-1).view(torch.int16),
                items[1].reshape(-1).view(torch.int16),
                result.reshape(-1).view(torch.int32),
                n,
                BLOCK=blk,
                MASK=mask,
                num_warps=_WARPS,
            )
            return True
        if P == 2 and es == 4:
            if result.data_ptr() % 8 != 0:
                return False
            n = outer
            blk = _token_block("p2_32", n)
            mask = (n % blk) != 0
            _dstack_pack2_w64[(triton.cdiv(n, blk),)](
                items[0].reshape(-1).view(torch.int32),
                items[1].reshape(-1).view(torch.int32),
                result.reshape(-1).view(torch.int64),
                n,
                BLOCK=blk,
                MASK=mask,
                num_warps=_WARPS,
            )
            return True
        if P == 3 and es == 2:
            if outer % 2 != 0 or result.data_ptr() % 4 != 0:
                return False
            srcs = [it.reshape(-1).view(torch.int32) for it in items]
            if any(it.data_ptr() % 4 != 0 for it in items):
                return False
            nw = (outer * 3) // 2
            blk = _token_block("p3_16", nw)
            mask = (nw % blk) != 0
            _dstack_pack3_w32[(triton.cdiv(nw, blk),)](
                srcs[0],
                srcs[1],
                srcs[2],
                result.reshape(-1).view(torch.int32),
                nw,
                BLOCK=blk,
                MASK=mask,
                num_warps=_WARPS,
            )
            return True
        if P == 3 and es == 4:
            if outer % 2 != 0 or result.data_ptr() % 8 != 0:
                return False
            if any(it.data_ptr() % 8 != 0 for it in items):
                return False
            srcs = [it.reshape(-1).view(torch.int64) for it in items]
            nw = (outer * 3) // 2
            blk = _token_block("p3_32", nw)
            mask = (nw % blk) != 0
            _dstack_pack3_w64[(triton.cdiv(nw, blk),)](
                srcs[0],
                srcs[1],
                srcs[2],
                result.reshape(-1).view(torch.int64),
                nw,
                BLOCK=blk,
                MASK=mask,
                num_warps=_WARPS,
            )
            return True
        if P == 4 and es == 2:
            if result.data_ptr() % 8 != 0:
                return False
            n = outer
            blk = _token_block("p4_16", n)
            mask = (n % blk) != 0
            _dstack_pack4_w64_16[(triton.cdiv(n, blk),)](
                *(it.reshape(-1).view(torch.int16) for it in items),
                result.reshape(-1).view(torch.int64),
                n,
                BLOCK=blk,
                MASK=mask,
                num_warps=_WARPS,
            )
            return True
        if P == 4 and es == 4:
            if result.data_ptr() % 8 != 0 or any(it.data_ptr() % 4 != 0 for it in items):
                return False
            nw = outer * 2
            blk = _token_block("p4_32", nw)
            mask = (nw % blk) != 0
            _dstack_pack4_w64_32[(triton.cdiv(nw, blk),)](
                *(it.reshape(-1).view(torch.int32) for it in items),
                result.reshape(-1).view(torch.int64),
                nw,
                BLOCK=blk,
                MASK=mask,
                num_warps=_WARPS,
            )
            return True
    except (RuntimeError, ValueError):
        return False
    return False


def _launch_flat_interleave(items, result, outer):
    P = len(items)
    flat = [it.reshape(-1) for it in items]
    pad = [flat[i] if i < P else flat[0] for i in range(_MAX_GROUP)]
    n = outer * P
    blk = 4096 if n >= 2 ** 12 else (1024 if n >= 2 ** 10 else 256)
    _dstack_interleave_kernel[(triton.cdiv(n, blk),)](
        pad[0],
        pad[1],
        pad[2],
        pad[3],
        result.reshape(-1),
        n,
        P=P,
        BLOCK=blk,
        num_warps=_WARPS,
    )


def run(tensors, out=None, **kwargs):
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
    # matches logical element order when the tensor is contiguous.  Strided,
    # transposed and expanded views have to be materialised first, otherwise
    # they are read in storage order and silently give the wrong answer.
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

    real_dtype = _real_dtype(result.dtype)
    unit = 1 if real_dtype is None else 2
    if result.numel() == 0:
        return result

    out_view = _as_copy_tensor(result, real_dtype)

    # Single-token inputs interleave one element of every input into each output
    # row.  Packed register interleave covers the common P; anything else uses
    # the flat fallback.
    if (
        inner == 1
        and unit == 1
        and 2 <= len(items) <= _MAX_GROUP
        and len(set(depth)) == 1
        and depth[0] == 1
        and real_dtype is None
        and all(item.dtype not in _MULTI_SELECT_UNSUPPORTED for item in items)
    ):
        if _launch_pack(items, result, outer):
            return result
        _launch_flat_interleave(items, result, outer)
        return result

    _run_copy_path(items, depth, outer, inner, unit, out_view, result.dtype, real_dtype)
    return result


dstack = run

__all__ = ["run", "dstack"]
