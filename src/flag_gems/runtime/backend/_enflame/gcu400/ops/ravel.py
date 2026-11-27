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

"""Triton implementation of aten::ravel for the FlagGems (Gems) backend.

Contract (from the Gems pytest, which is authoritative):

* ``ravel(inp)`` returns a 1-D tensor with ``inp.numel()`` elements and stride
  ``(1,)``.
* VIEW path -- taken exactly when the flattened input is already contiguous
  after collapsing size-1 dimensions (``inp.is_contiguous()``).  The result
  aliases the input storage, preserves ``storage_offset`` and preserves the
  lazy neg/conj bits.  No device work is performed: ``inp.view(-1)``.
* COPY path -- every other strided layout.  A fresh contiguous buffer (offset 0,
  no storage sharing) is materialised with the lazy neg/conj bits folded into
  the stored values.
* A non-Tensor ``self`` raises TypeError, a non-strided (sparse) layout is
  rejected with RuntimeError.
* Backward only rearranges gradients back to the input shape and touches no
  backing storage.

The copy path is dispatched on the canonical layout shape (GCU tuning):

* rank 1 with a constant stride ``T0``      -> ``_ravel_strided1d_kernel``
* rank >= 2, i32-safe, >= 64Ki elements, whose outermost canonical dim has
  stride 1 and splits into power-of-two tiles of the stride-1 extent
  -> ``_ravel_tile_kernel`` with ``TRANSPOSE=True``: the (B_A, B_B) tile loads
  along the stride-1 axis and stores through swapped index roles, so both
  sides issue long contiguous runs without ``tl.trans``.  Tile geometry is
  regime-dependent (measured): a narrow stride-1 axis (s0 < 64) takes
  (1024, 32) tiles at 1 warp, the bandwidth-bound regime (numel >= 8Mi)
  takes (512, 256) tiles at 2 warps, the launch-bound regime keeps
  (256, 256) at 4 warps.  Row counts that do not divide into full tiles are
  covered by flat-gather tail programs appended to the same launch.
* rank >= 2, i32-safe, >= 64Ki elements, whose innermost canonical dim has
  stride 1 and the inner extent splits into a >= 32-wide power-of-two tile
  -> ``_ravel_tile_kernel`` with ``TRANSPOSE=False`` (straight block copy),
  with the same tail mechanism for non-divisible row counts.
* anything else, and every complex dtype    -> ``_ravel_copy_kernel`` /
  ``_ravel_copy_cplx_kernel`` (rank-agnostic flat gather).

GCU specifics baked into this copy (measured on GCU400 / L600):

* int64 vector address arithmetic is software-emulated and dominates the old
  kernels' runtime (4096x4096 fp32 transpose: 2650us -> 67us).  Every hot path
  therefore computes offsets in int32 inside a host-checked safe domain
  (``numel`` and the max source offset < 2**31); the int64 variants remain as
  the fallback for larger tensors.
* Masked 2D transpose tiles are a known silent-corruption class on this
  backend (compiler problem P15), so the tile kernel runs unmasked and the
  store side only ever issues contiguous masked 1D-style runs; every layout
  that cannot be covered by full tiles falls back to the flat gather.
* Tile rows whose count does not split into power-of-two tiles (e.g. a
  (4100, 2048) transpose) no longer demote the whole copy to the flat
  gather: the same single kernel launch appends "tail" programs after the
  full-tile programs on the same grid axis.  A tail program copies the
  remainder region with the flat-gather access pattern (contiguous masked
  1D loads/stores, the proven-correct class of ``_ravel_copy_kernel``),
  enumerated in output order so its stores stay contiguous.  This keeps the
  launch count at one and avoids masked transposed-tile stores entirely.
* Grid limits are enforced (axis 0 <= 65534, axis 1 <= 255).  The previous
  tile dispatch placed the flattened rest-extent on grid axis 1 and crashed
  with ``OutOfResources: grid.y`` for e.g. a (100000, 32) transposed input.
* The flat/strided-1D kernels cap their 1-D grid at 65534 blocks and spill
  onto a second grid axis for huge tensors.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

__all__ = ["ravel"]

_MAX_DIMS = 8
# int32 safe domain: every element offset computed by the i32 kernels stays
# below this bound (checked on the host before dispatch).
_I32_OFF_LIMIT = 2**31 - 1
# Flat-gather / strided-1D block: 4096 elements per program measured fastest
# (a 100000x32 transposed gather: 138us vs 200us at BLOCK=1024).
_FLAT_BLOCK = 4096
_FLAT_WARPS = 4
# P19 grid hard limits: axis 0 is unsafe at the 65535 edge, axis 1 caps at 255.
_GRID_X_MAX = 65534
_GRID_Y_MAX = 255
# Tiles below this element count stay on the flat gather (launch-bound sizes
# show no tile benefit); above it, 256-wide tiles are the measured optimum.
_TILE_MIN_NUMEL = 65536
_TILE_CAP = 256
# At this element count the copy leaves the launch-bound regime and wider
# 512-row tiles pay off (measured on GCU400/L600: 4096x4096 fp32 transpose
# 65.5us at (512, 256, nw=2) vs 69.7us at (256, 256, nw=4)).
_TILE_BIG_NUMEL = 8 * 1024 * 1024
# Tail programs (the non-power-of-two remainder of the row axis) reuse the
# flat-gather lane shape: 4096 lanes per program is the proven-correct block
# size of _ravel_copy_kernel on this backend.
_TAIL_BLOCK = 4096


def _prev_pow2(n):
    """Largest power of two <= ``n`` (the tile row width; the remainder rows
    are covered by tail programs, so divisibility is no longer required)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


@triton.jit
def _ravel_tile_kernel(
    in_ptr,
    out_ptr,
    EXT_A,
    EXT_B,
    d0,
    s0,
    t0,
    d1,
    s1,
    t1,
    d2,
    s2,
    t2,
    d3,
    s3,
    t3,
    d4,
    s4,
    t4,
    d5,
    s5,
    t5,
    d6,
    s6,
    t6,
    d7,
    s7,
    t7,
    NA_FULL,
    R_A,
    R2: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    A_FIRST: tl.constexpr,
    NEG: tl.constexpr,
    B_A: tl.constexpr,
    B_B: tl.constexpr,
    TAIL: tl.constexpr,
    TAIL_BLOCK: tl.constexpr,
):
    """Unmasked (B_A, B_B) tile copy over a stride-1 axis (int32 offsets).

    ``a`` walks the flattened remaining axes (via the div/mod chain ``g``),
    ``b`` walks the stride-1 axis, so loads run along ``b``.  With
    ``TRANSPOSE`` the output row index is ``b`` (the stride-1 axis is the
    outermost canonical dim) and the store swaps the index roles, keeping both
    sides coalesced without ``tl.trans``; otherwise the output row index is
    ``a`` and the store is a straight block copy.

    With ``TAIL`` the grid appends extra programs after the ``NA_FULL``
    full-tile programs on the same ``a`` axis.  Each tail program copies one
    ``TAIL_BLOCK`` slice of the ``R_A * EXT_B`` remainder elements with the
    flat-gather pattern (contiguous masked 1D access, like
    ``_ravel_copy_kernel``), enumerated in output order so the masked store
    stays a contiguous 1D run -- never a masked transposed-tile store (P15).
    """
    if A_FIRST:
        pid_a = tl.program_id(0)
        pid_b = tl.program_id(1)
    else:
        pid_a = tl.program_id(1)
        pid_b = tl.program_id(0)
    if TAIL and pid_a >= NA_FULL:
        tt = (pid_a - NA_FULL) * TAIL_BLOCK + tl.arange(0, TAIL_BLOCK)
        tm = tt < R_A * EXT_B
        if TRANSPOSE:
            tb = tt // R_A
            tao = tt % R_A
        else:
            tao = tt // EXT_B
            tb = tt % EXT_B
        ta = NA_FULL * B_A + tao
        if R2 == 1:
            tg = ta * t0
        else:
            tg = tl.zeros((TAIL_BLOCK,), dtype=tl.int32)
            if R2 > 0:
                tg += ((ta // d0) % s0) * t0
            if R2 > 1:
                tg += ((ta // d1) % s1) * t1
            if R2 > 2:
                tg += ((ta // d2) % s2) * t2
            if R2 > 3:
                tg += ((ta // d3) % s3) * t3
            if R2 > 4:
                tg += ((ta // d4) % s4) * t4
            if R2 > 5:
                tg += ((ta // d5) % s5) * t5
            if R2 > 6:
                tg += ((ta // d6) % s6) * t6
        tsrc = tg + tb
        tv = tl.load(in_ptr + tsrc, mask=tm, other=0)
        if NEG:
            tv = -tv
        if TRANSPOSE:
            tdst = tb * EXT_A + ta
        else:
            tdst = ta * EXT_B + tb
        tl.store(out_ptr + tdst, tv, mask=tm)
    else:
        a = pid_a * B_A + tl.arange(0, B_A)
        b = pid_b * B_B + tl.arange(0, B_B)
        if R2 == 1:
            g = a * t0
        else:
            g = tl.zeros((B_A,), dtype=tl.int32)
            if R2 > 0:
                g += ((a // d0) % s0) * t0
            if R2 > 1:
                g += ((a // d1) % s1) * t1
            if R2 > 2:
                g += ((a // d2) % s2) * t2
            if R2 > 3:
                g += ((a // d3) % s3) * t3
            if R2 > 4:
                g += ((a // d4) % s4) * t4
            if R2 > 5:
                g += ((a // d5) % s5) * t5
            if R2 > 6:
                g += ((a // d6) % s6) * t6
        src = g[:, None] + b[None, :]
        v = tl.load(in_ptr + src)
        if NEG:
            v = -v
        if TRANSPOSE:
            dst = b[None, :] * EXT_A + a[:, None]
        else:
            dst = a[:, None] * EXT_B + b[None, :]
        tl.store(out_ptr + dst, v)


@triton.jit
def _ravel_copy_kernel(
    in_ptr,
    out_ptr,
    N,
    NPROG,
    d0,
    s0,
    t0,
    d1,
    s1,
    t1,
    d2,
    s2,
    t2,
    d3,
    s3,
    t3,
    d4,
    s4,
    t4,
    d5,
    s5,
    t5,
    d6,
    s6,
    t6,
    d7,
    s7,
    t7,
    R: tl.constexpr,
    I32: tl.constexpr,
    TWO_D: tl.constexpr,
    NEG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Rank-agnostic 1-D gather: one output element per lane."""
    pid0 = tl.program_id(0)
    if TWO_D:
        pid = pid0 + tl.program_id(1) * NPROG
    else:
        pid = pid0
    if I32:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        src = tl.zeros((BLOCK,), dtype=tl.int32)
    else:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        src = tl.zeros((BLOCK,), dtype=tl.int64)
    mask = offs < N
    if R > 0:
        src += ((offs // d0) % s0) * t0
    if R > 1:
        src += ((offs // d1) % s1) * t1
    if R > 2:
        src += ((offs // d2) % s2) * t2
    if R > 3:
        src += ((offs // d3) % s3) * t3
    if R > 4:
        src += ((offs // d4) % s4) * t4
    if R > 5:
        src += ((offs // d5) % s5) * t5
    if R > 6:
        src += ((offs // d6) % s6) * t6
    if R > 7:
        src += ((offs // d7) % s7) * t7
    v = tl.load(in_ptr + src, mask=mask, other=0)
    if NEG:
        v = -v
    tl.store(out_ptr + offs, v, mask=mask)


@triton.jit
def _ravel_broadcast1d_kernel(
    in_ptr,
    out_ptr,
    N,
    NPROG,
    I32: tl.constexpr,
    TWO_D: tl.constexpr,
    NEG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Rank-1 stride-0 (expanded) input: every element copies the single
    source value.  The source is loaded once as a scalar -- a vector load of
    one address from every lane is a degenerate broadcast pattern that
    faults the device on this backend (see
    compiler_issues/flaggems_ravel_stride0_tile_device_fault.md)."""
    pid0 = tl.program_id(0)
    if TWO_D:
        pid = pid0 + tl.program_id(1) * NPROG
    else:
        pid = pid0
    if I32:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < N
    v = tl.load(in_ptr)
    if NEG:
        v = -v
    tl.store(out_ptr + offs, v, mask=mask)


@triton.jit
def _ravel_strided1d_kernel(
    in_ptr,
    out_ptr,
    N,
    T0,
    NPROG,
    I32: tl.constexpr,
    TWO_D: tl.constexpr,
    NEG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Rank-1 canonical form with a constant stride ``T0``: coalesced store,
    strided load."""
    pid0 = tl.program_id(0)
    if TWO_D:
        pid = pid0 + tl.program_id(1) * NPROG
    else:
        pid = pid0
    if I32:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < N
    v = tl.load(in_ptr + offs * T0, mask=mask, other=0)
    if NEG:
        v = -v
    tl.store(out_ptr + offs, v, mask=mask)


@triton.jit
def _ravel_copy_cplx_kernel(
    in_addr,
    out_addr,
    N,
    NPROG,
    d0,
    s0,
    t0,
    d1,
    s1,
    t1,
    d2,
    s2,
    t2,
    d3,
    s3,
    t3,
    d4,
    s4,
    t4,
    d5,
    s5,
    t5,
    d6,
    s6,
    t6,
    d7,
    s7,
    t7,
    R: tl.constexpr,
    WIDE: tl.constexpr,
    I32: tl.constexpr,
    TWO_D: tl.constexpr,
    NEG: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """complex64/128 copy path: Triton has no complex pointer type, so the raw
    storage address is reinterpreted as its real float width and the real /
    imaginary halves are gathered separately (imaginary negated for conj)."""
    if WIDE:
        in_ptr = in_addr.to(tl.pointer_type(tl.float64))
        out_ptr = out_addr.to(tl.pointer_type(tl.float64))
    else:
        in_ptr = in_addr.to(tl.pointer_type(tl.float32))
        out_ptr = out_addr.to(tl.pointer_type(tl.float32))

    pid0 = tl.program_id(0)
    if TWO_D:
        pid = pid0 + tl.program_id(1) * NPROG
    else:
        pid = pid0
    if I32:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        src = tl.zeros((BLOCK,), dtype=tl.int32)
    else:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        src = tl.zeros((BLOCK,), dtype=tl.int64)
    mask = offs < N
    if R > 0:
        src += ((offs // d0) % s0) * t0
    if R > 1:
        src += ((offs // d1) % s1) * t1
    if R > 2:
        src += ((offs // d2) % s2) * t2
    if R > 3:
        src += ((offs // d3) % s3) * t3
    if R > 4:
        src += ((offs // d4) % s4) * t4
    if R > 5:
        src += ((offs // d5) % s5) * t5
    if R > 6:
        src += ((offs // d6) % s6) * t6
    if R > 7:
        src += ((offs // d7) % s7) * t7

    re = tl.load(in_ptr + 2 * src, mask=mask, other=0.0)
    im = tl.load(in_ptr + 2 * src + 1, mask=mask, other=0.0)
    if CONJ:
        im = -im
    if NEG:
        re = -re
        im = -im
    tl.store(out_ptr + 2 * offs, re, mask=mask)
    tl.store(out_ptr + 2 * offs + 1, im, mask=mask)


def _canonical_dims(inp):
    """Minimal (size, stride) list: size-1 dims dropped, then adjacent dims are
    merged while the outer dim exactly continues the inner dim's reach."""
    dims = []
    for size, stride in zip(inp.shape, inp.stride()):
        size = int(size)
        if size == 1:
            continue
        dims.append((size, int(stride)))
    merged = []
    for size, stride in dims:
        if merged and merged[-1][1] == stride * size:
            # The outer dim spans exactly this dim's reach, so the pair is a
            # dense run whose unit step is the *inner* dim's stride.
            merged[-1] = (merged[-1][0] * size, stride)
        else:
            merged.append((size, stride))
    return merged


def _pack(dims):
    """(div, size, stride) triples for every canonical dim, padded to _MAX_DIMS."""
    rank = len(dims)
    divs = [1] * _MAX_DIMS
    sizes = [1] * _MAX_DIMS
    strides = [0] * _MAX_DIMS
    for k in range(rank):
        div = 1
        for j in range(k + 1, rank):
            div *= dims[j][0]
        divs[k] = div
        sizes[k] = dims[k][0]
        strides[k] = dims[k][1]
    packed = []
    for k in range(_MAX_DIMS):
        packed += [divs[k], sizes[k], strides[k]]
    return packed


def _pow2div(n):
    """Largest power-of-two divisor of ``n`` (tile widths must divide the
    extent exactly so the unmasked tile kernel needs no bounds masks)."""
    p = 1
    while n % (p * 2) == 0:
        p *= 2
    return p


def _max_src_offset(dims):
    """Conservative bound on the largest source element offset touched."""
    return sum((size - 1) * stride for size, stride in dims)


def _flat_grid(numel, block):
    """1-D grid for the flat kernels, spilled onto axis 1 past _GRID_X_MAX."""
    nblocks = triton.cdiv(numel, block)
    if nblocks <= _GRID_X_MAX:
        return (nblocks,), 1, False
    ny = triton.cdiv(nblocks, _GRID_X_MAX)
    if ny > _GRID_Y_MAX:
        raise RuntimeError("ravel: copy path grid overflow (%d blocks)" % nblocks)
    return (_GRID_X_MAX, ny), _GRID_X_MAX, True


def _launch_tile(inp, out, rest, ext_a, ext_b, transpose, neg, ba, bb, num_warps=4):
    """Launch the tile kernel (full tiles + flat tail programs); False when
    the grid limits reject it."""
    na_full = ext_a // ba
    if na_full == 0:
        return False
    rem = ext_a - na_full * ba
    n_tail = triton.cdiv(rem * ext_b, _TAIL_BLOCK) if rem else 0
    na = na_full + n_tail
    nb = ext_b // bb
    a_first = na >= nb
    grid = (na, nb) if a_first else (nb, na)
    if grid[0] > _GRID_X_MAX or grid[1] > _GRID_Y_MAX:
        return False
    if n_tail and (
        n_tail * _TAIL_BLOCK > _I32_OFF_LIMIT or rem * ext_b > _I32_OFF_LIMIT
    ):
        return False
    _ravel_tile_kernel[grid](
        inp,
        out,
        ext_a,
        ext_b,
        *_pack(rest),
        na_full,
        rem,
        R2=len(rest),
        TRANSPOSE=transpose,
        A_FIRST=a_first,
        NEG=neg,
        B_A=ba,
        B_B=bb,
        TAIL=bool(rem),
        TAIL_BLOCK=_TAIL_BLOCK,
        num_warps=num_warps,
    )
    return True


def _copy_impl(inp):
    numel = inp.numel()
    out = torch.empty(numel, dtype=inp.dtype, device=inp.device)
    if numel == 0:
        return out

    dims = _canonical_dims(inp)
    rank = len(dims)
    if rank == 0:  # a single element is always contiguous: unreachable
        return out
    if rank > _MAX_DIMS:
        raise RuntimeError(
            "ravel: unsupported strided layout rank %d (max %d)" % (rank, _MAX_DIMS)
        )

    neg = bool(inp.is_neg())
    max_src = _max_src_offset(dims)
    i32_ok = numel - 1 < _I32_OFF_LIMIT and max_src < _I32_OFF_LIMIT

    if inp.dtype.is_complex:
        # Offsets double (real/imag halves live two scalars apart).
        cplx_i32 = (
            i32_ok
            and 2 * max_src + 1 < _I32_OFF_LIMIT
            and 2 * numel - 1 < _I32_OFF_LIMIT
        )
        grid, nprog, two_d = _flat_grid(numel, _FLAT_BLOCK)
        _ravel_copy_cplx_kernel[grid](
            inp.data_ptr(),
            out.data_ptr(),
            numel,
            nprog,
            *_pack(dims),
            R=rank,
            WIDE=(inp.dtype == torch.complex128),
            I32=cplx_i32,
            TWO_D=two_d,
            NEG=neg,
            CONJ=bool(inp.is_conj()),
            BLOCK=_FLAT_BLOCK,
            num_warps=_FLAT_WARPS,
        )
        return out

    if rank == 1:
        t0 = dims[0][1]
        grid, nprog, two_d = _flat_grid(numel, _FLAT_BLOCK)
        if t0 == 0:
            # Expanded rank-1 input: scalar load + broadcast store (a
            # per-lane load of the single source address is the degenerate
            # broadcast pattern that faults the device).
            _ravel_broadcast1d_kernel[grid](
                inp,
                out,
                numel,
                nprog,
                I32=(numel - 1 < _I32_OFF_LIMIT),
                TWO_D=two_d,
                NEG=neg,
                BLOCK=_FLAT_BLOCK,
                num_warps=_FLAT_WARPS,
            )
            return out
        _ravel_strided1d_kernel[grid](
            inp,
            out,
            numel,
            t0,
            nprog,
            I32=((numel - 1) * t0 < _I32_OFF_LIMIT),
            TWO_D=two_d,
            NEG=neg,
            BLOCK=_FLAT_BLOCK,
            num_warps=_FLAT_WARPS,
        )
        return out

    if i32_ok and numel >= _TILE_MIN_NUMEL:
        # Transposing tile: the outermost canonical dim is the stride-1 axis.
        # Stride-0 (expanded) rest dims are excluded: the unmasked tile with a
        # zero-stride row multiply is a process-kill trigger on this backend
        # (measured twice on GCU400), and expands copy fine on the flat path.
        s0, t0 = dims[0]
        if t0 == 1 and all(st > 0 for _, st in dims[1:]):
            rest = dims[1:]
            ext_a = numel // s0
            bb = min(_TILE_CAP, _pow2div(s0))
            # Remainder rows on the a axis go to tail programs, so the tile
            # row width no longer needs to divide ext_a exactly; only the
            # load-side width must still divide the stride-1 extent.  The
            # store side is contiguous along a, so its geometry trades
            # against load coalescing, measured per regime below.
            if s0 % bb == 0 and bb >= 32:
                if bb < 64:
                    # Narrow stride-1 axis: long store rows dominate.  A
                    # (100000, 32) transpose runs 21.6us at (1024, 32, nw=1)
                    # vs 140us flat and 77.4us native.
                    ba = min(1024, _prev_pow2(ext_a))
                    if ba >= 64:
                        if _launch_tile(
                            inp, out, rest, ext_a, s0, True, neg, ba, bb, num_warps=1
                        ):
                            return out
                elif numel >= _TILE_BIG_NUMEL:
                    # Bandwidth-bound regime: 512-row tiles with 2 warps
                    # (measured optimum family on 4096x4096 transposes).
                    ba = min(512, _prev_pow2(ext_a))
                    if ba >= 64:
                        if _launch_tile(
                            inp, out, rest, ext_a, s0, True, neg, ba, bb, num_warps=2
                        ):
                            return out
                else:
                    # Launch-bound regime: the historical 256x256 tile at 4
                    # warps stays optimal (measured on a 1024x1024
                    # transpose: 8.7us vs 8.9-9.6us for 512-wide rows).
                    ba = min(_TILE_CAP, _prev_pow2(ext_a))
                    if ba >= 64:
                        if _launch_tile(
                            inp, out, rest, ext_a, s0, True, neg, ba, bb, num_warps=4
                        ):
                            return out
        # Straight block tile: the innermost canonical dim is the stride-1
        # axis and only the inner width gates coalescing.  Same stride-0
        # exclusion as the transposing arm; the a-axis remainder again runs
        # on tail programs inside the same launch.
        sN, tN = dims[-1]
        if tN == 1 and all(st > 0 for _, st in dims[:-1]):
            prefix = dims[:-1]
            ext_a = numel // sN
            ba = min(_TILE_CAP, _prev_pow2(ext_a))
            bb = min(_TILE_CAP, _pow2div(sN))
            if ba >= 16 and bb >= 32 and sN % bb == 0:
                if _launch_tile(inp, out, prefix, ext_a, sN, False, neg, ba, bb):
                    return out

    grid, nprog, two_d = _flat_grid(numel, _FLAT_BLOCK)
    _ravel_copy_kernel[grid](
        inp,
        out,
        numel,
        nprog,
        *_pack(dims),
        R=rank,
        I32=i32_ok,
        TWO_D=two_d,
        NEG=neg,
        BLOCK=_FLAT_BLOCK,
        num_warps=_FLAT_WARPS,
    )
    return out


class _RavelCopy(torch.autograd.Function):
    """Copy path with the native backward: the gradient is only rearranged
    back to the input shape and touches no backing storage."""

    @staticmethod
    def forward(ctx, inp):
        ctx.input_shape = inp.shape
        return _copy_impl(inp)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.reshape(ctx.input_shape)


def ravel(self):
    """``aten::ravel(Tensor(a) self) -> Tensor(a)``."""
    logger.debug("GEMS_ENFLAME RAVEL")
    # Hot path first: a dense strided tensor is contiguous in the standard
    # row-major sense exactly when it flattens to a pure view, and
    # ``flatten()`` is the cheapest C-bound primitive that produces it
    # (measured 0.95us vs 1.01us for ``view(-1)`` on GCU400).  Sparse and
    # non-strided layouts report a non-contiguous row-major layout and fall
    # through to the contract checks below.
    if isinstance(self, torch.Tensor) and self.is_contiguous():
        return self.flatten()
    if not isinstance(self, torch.Tensor):
        raise TypeError(
            "ravel(): argument 'self' must be a Tensor, not %s" % type(self).__name__
        )
    if self.layout != torch.strided:
        raise RuntimeError("unsupported memory format option Contiguous")

    if self.requires_grad and torch.is_grad_enabled():
        return _RavelCopy.apply(self)
    return _copy_impl(self)
