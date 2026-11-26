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
"""Bilinear 2-D upsampling for Ascend NPU, expressed with TLE/DSA staging.

Architecture (per output tile = one (plane, column-block, row-chunk) triple):

  * the column lattice (source column, left/right index and lerp weight) is
    loop invariant and therefore computed once per program;
  * the two source rows needed by an output row are brought into UB with
    ``tle.dsa.copy`` (contiguous MTE burst, one per row) instead of the four
    scattered global gathers the naive formulation needs;
  * the two staged rows are combined vertically FIRST, while they are still at
    source-column resolution (SEG lanes).  The fp16/bf16 -> fp32 widening and
    the vertical lerp therefore cost SEG lanes each instead of the BLOCK_W
    lanes a horizontal-first ordering pays for them, and only the two
    horizontal neighbours have to be picked out of the already-lerped row, so
    an output row costs 2 in-UB ``tl.gather`` instead of 4;
  * the row loop is a ``tle.dsa.pipeline`` so the next iteration's MTE transfer
    overlaps the current iteration's vector work, and it advances by a BAND of
    output rows per iteration: the row-dependent scalars (``rh``, ``h0``/``h1``,
    ``lh``) and the addresses of every row in the band are produced in one
    straight-line prologue, so the PIPE_S -> PIPE_V handshake and the pipeline's
    flag/parity round-trip that surround that prologue are paid once per BAND
    rows instead of once per row.  A chunk whose row count is not a multiple of
    BAND would spend the last iteration recomputing already-written rows (the
    out-of-range rows are clamped onto the chunk's last row, which is correct but
    wasted work), so ``_plan`` rounds the per-chunk row count up to a whole
    number of bands and shrinks the chunk count to match.

Arithmetic follows ATen's ``upsample_bilinear2d``: fp32 source-index scale,
truncated integer part, ``lambda = src - int(src)``, ATen's ``lh == 0`` / ``lw
== 0`` renormalisation when the upper tap is clamped, and fp32 accumulation for
every input dtype.  The four taps are associated vertically-first
(``(a*lh0 + c*lh)*lw0 + (b*lh0 + d*lh)*lw``) instead of ATen's
``lh0*(lw0*a + lw*b) + lh*(lw0*c + lw*d)``; both evaluate the same bilinear
polynomial and the two differ only by an fp32 reassociation of a convex
combination.
"""

import logging
import math
import struct

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl

logger = logging.getLogger(__name__)

try:  # fp32 arithmetic for the scale factor, exactly as ATen computes it
    import numpy as _np
except Exception:  # pragma: no cover - numpy is always present with torch
    _np = None

_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}

_MAX_BLOCK_W = 2048
_MIN_SEG = 32
# Output rows handled per pipeline iteration.  Every iteration pays the row
# prologue handshake once, so a wider band divides that fixed cost further; it
# also costs 2*BAND live SEG-element UB buffers, which caps it, and it needs
# _plan to make the per-chunk row count an exact multiple of the band.
_BAND = 4
_NCT = None


def _div32(num, den):
    """float32 division of two ints (ATen's accscalar_t scale computation)."""
    if _np is not None:
        return float(_np.float32(num) / _np.float32(den))
    return struct.unpack("f", struct.pack("f", float(num) / float(den)))[0]


def _vector_cores():
    """Number of vector cores on the active device (never hard-coded)."""
    global _NCT
    if _NCT is None:
        n = 40
        try:
            from triton.runtime import driver as _driver

            props = _driver.active.utils.get_device_properties(
                torch.npu.current_device()
            )
            n = int(props.get("num_vectorcore") or props.get("num_aicore") or n)
            if n <= 0:
                n = 40
        except Exception:
            n = 40
        _NCT = n
    return _NCT


def _recip_scale(src, dst, align_corners, scale):
    if align_corners:
        if dst > 1:
            return _div32(src - 1, dst - 1)
        return 0.0
    if scale is not None and scale > 0:
        return struct.unpack("f", struct.pack("f", 1.0 / scale))[0]
    return _div32(src, dst)


@triton.jit
def _ub_bilinear2d_kernel(
    ptr_o,
    ptr_i,
    IH,
    IW,
    OH,
    OW,
    rsh,
    rsw,
    NCB,
    NCHUNK,
    ROWS,
    BAND: tl.constexpr,
    BLOCK_W: tl.constexpr,
    SEG: tl.constexpr,
    ALIGN_CORNERS: tl.constexpr,
    MASK_W: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    rc = pid % NCHUNK
    q = pid // NCHUNK
    cb = q % NCB
    plane = q // NCB

    # ---- column lattice: loop invariant, hoisted out of the row loop --------
    # og is the GLOBAL output column; the per-block lane index must never be
    # used as a coordinate or as a store address when NCB > 1.  COL_START is the
    # source column of the block's FIRST output column, so it must follow the
    # same (og + 0.5) * rsw - 0.5 expression, not cb * BLOCK_W * rsw.
    og = cb * BLOCK_W + tl.arange(0, BLOCK_W)
    first = cb.to(tl.float32) * BLOCK_W
    if ALIGN_CORNERS:
        cs = first * rsw
    else:
        cs = (first + 0.5) * rsw - 0.5
    col_start = tl.maximum(cs.to(tl.int32), 0)
    col_start = tl.minimum(col_start, IW - 1)
    ext = tl.maximum(tl.minimum(SEG, IW - col_start), 1)

    if ALIGN_CORNERS:
        rw = og.to(tl.float32) * rsw
    else:
        rw = tl.maximum((og.to(tl.float32) + 0.5) * rsw - 0.5, 0.0)
    w0 = tl.minimum(tl.maximum(rw.to(tl.int32), 0), IW - 1)
    w1 = tl.minimum(w0 + 1, IW - 1)
    lw = tl.where(w0 < IW - 1, rw - w0.to(tl.float32), 0.0)
    lw0 = 1.0 - lw
    sel0 = tl.minimum(tl.maximum(w0 - col_start, 0), ext - 1)
    sel1 = tl.minimum(tl.maximum(w1 - col_start, 0), ext - 1)

    off = tl.arange(0, SEG)
    ib = plane.to(tl.int64) * (IH.to(tl.int64) * IW.to(tl.int64))
    ob = plane.to(tl.int64) * (OH.to(tl.int64) * OW.to(tl.int64))

    r0 = rc * ROWS
    nrows = tl.minimum(ROWS, OH - r0)
    # last output row of this chunk: the extra rows of a partial final band are
    # clamped onto it, which makes the out-of-range case an idempotent rewrite
    # of that row (same source rows, same weights) instead of a predicated store.
    rlast = r0 + nrows - 1

    # ---- row loop: one BAND of output rows per iteration -------------------
    # The per-row scalar prologue (rh/h0/h1/lh plus the int64 row address) and
    # the pipeline's per-iteration flag/parity bookkeeping are the fixed cost of
    # the loop, and they are serialised against the vector pipe by a
    # PIPE_S -> PIPE_V handshake.  Handling BAND rows per iteration divides the
    # number of those handshakes and flag rounds per output row by BAND, which
    # is what the narrow-row workloads (BLOCK_W = 256) pay most heavily.
    nit = (nrows + (BAND - 1)) // BAND
    for k in tle.dsa.pipeline(0, nit, 1, num_stages=2):
        ra_ = r0 + k * BAND
        rb_ = tl.minimum(ra_ + 1, rlast)
        rc_ = tl.minimum(ra_ + 2, rlast)
        rd_ = tl.minimum(ra_ + 3, rlast)

        # ---- band prologue: every row's scalars in one straight-line region --
        if ALIGN_CORNERS:
            rha = ra_.to(tl.float32) * rsh
            rhb = rb_.to(tl.float32) * rsh
            rhc = rc_.to(tl.float32) * rsh
            rhd = rd_.to(tl.float32) * rsh
        else:
            rha = tl.maximum((ra_.to(tl.float32) + 0.5) * rsh - 0.5, 0.0)
            rhb = tl.maximum((rb_.to(tl.float32) + 0.5) * rsh - 0.5, 0.0)
            rhc = tl.maximum((rc_.to(tl.float32) + 0.5) * rsh - 0.5, 0.0)
            rhd = tl.maximum((rd_.to(tl.float32) + 0.5) * rsh - 0.5, 0.0)
        h0a = tl.minimum(tl.maximum(rha.to(tl.int32), 0), IH - 1)
        h1a = tl.minimum(h0a + 1, IH - 1)
        lha = tl.where(h0a < IH - 1, rha - h0a.to(tl.float32), 0.0)
        h0b = tl.minimum(tl.maximum(rhb.to(tl.int32), 0), IH - 1)
        h1b = tl.minimum(h0b + 1, IH - 1)
        lhb = tl.where(h0b < IH - 1, rhb - h0b.to(tl.float32), 0.0)
        h0c = tl.minimum(tl.maximum(rhc.to(tl.int32), 0), IH - 1)
        h1c = tl.minimum(h0c + 1, IH - 1)
        lhc = tl.where(h0c < IH - 1, rhc - h0c.to(tl.float32), 0.0)
        h0d = tl.minimum(tl.maximum(rhd.to(tl.int32), 0), IH - 1)
        h1d = tl.minimum(h0d + 1, IH - 1)
        lhd = tl.where(h0d < IH - 1, rhd - h0d.to(tl.float32), 0.0)

        # ---- two staged rows per output row, issued back to back -----------
        ba0 = tle.dsa.alloc([SEG], dtype=DTYPE, mem_addr_space=tle.dsa.ascend.UB)
        ba1 = tle.dsa.alloc([SEG], dtype=DTYPE, mem_addr_space=tle.dsa.ascend.UB)
        bb0 = tle.dsa.alloc([SEG], dtype=DTYPE, mem_addr_space=tle.dsa.ascend.UB)
        bb1 = tle.dsa.alloc([SEG], dtype=DTYPE, mem_addr_space=tle.dsa.ascend.UB)
        bc0 = tle.dsa.alloc([SEG], dtype=DTYPE, mem_addr_space=tle.dsa.ascend.UB)
        bc1 = tle.dsa.alloc([SEG], dtype=DTYPE, mem_addr_space=tle.dsa.ascend.UB)
        bd0 = tle.dsa.alloc([SEG], dtype=DTYPE, mem_addr_space=tle.dsa.ascend.UB)
        bd1 = tle.dsa.alloc([SEG], dtype=DTYPE, mem_addr_space=tle.dsa.ascend.UB)
        tle.dsa.copy(
            ptr_i + (ib + (h0a * IW).to(tl.int64) + col_start) + off, ba0, [ext]
        )
        tle.dsa.copy(
            ptr_i + (ib + (h1a * IW).to(tl.int64) + col_start) + off, ba1, [ext]
        )
        tle.dsa.copy(
            ptr_i + (ib + (h0b * IW).to(tl.int64) + col_start) + off, bb0, [ext]
        )
        tle.dsa.copy(
            ptr_i + (ib + (h1b * IW).to(tl.int64) + col_start) + off, bb1, [ext]
        )
        tle.dsa.copy(
            ptr_i + (ib + (h0c * IW).to(tl.int64) + col_start) + off, bc0, [ext]
        )
        tle.dsa.copy(
            ptr_i + (ib + (h1c * IW).to(tl.int64) + col_start) + off, bc1, [ext]
        )
        tle.dsa.copy(
            ptr_i + (ib + (h0d * IW).to(tl.int64) + col_start) + off, bd0, [ext]
        )
        tle.dsa.copy(
            ptr_i + (ib + (h1d * IW).to(tl.int64) + col_start) + off, bd1, [ext]
        )

        # Vertical interpolation FIRST, at SOURCE-column resolution (SEG lanes),
        # then the two horizontal neighbours are picked out of the single
        # lerped row: same weights, same ATen grouping, half the gather count.
        va = (
            tle.dsa.to_tensor(ba0).to(tl.float32) * (1.0 - lha)
            + tle.dsa.to_tensor(ba1).to(tl.float32) * lha
        )
        vb = (
            tle.dsa.to_tensor(bb0).to(tl.float32) * (1.0 - lhb)
            + tle.dsa.to_tensor(bb1).to(tl.float32) * lhb
        )
        vc = (
            tle.dsa.to_tensor(bc0).to(tl.float32) * (1.0 - lhc)
            + tle.dsa.to_tensor(bc1).to(tl.float32) * lhc
        )
        vd = (
            tle.dsa.to_tensor(bd0).to(tl.float32) * (1.0 - lhd)
            + tle.dsa.to_tensor(bd1).to(tl.float32) * lhd
        )
        resa = tl.gather(va, sel0, 0) * lw0 + tl.gather(va, sel1, 0) * lw
        resb = tl.gather(vb, sel0, 0) * lw0 + tl.gather(vb, sel1, 0) * lw
        resc = tl.gather(vc, sel0, 0) * lw0 + tl.gather(vc, sel1, 0) * lw
        resd = tl.gather(vd, sel0, 0) * lw0 + tl.gather(vd, sel1, 0) * lw

        if MASK_W:
            om = og < OW
            tl.store(ptr_o + (ob + ra_ * OW) + og, resa.to(DTYPE), mask=om)
            tl.store(ptr_o + (ob + rb_ * OW) + og, resb.to(DTYPE), mask=om)
            tl.store(ptr_o + (ob + rc_ * OW) + og, resc.to(DTYPE), mask=om)
            tl.store(ptr_o + (ob + rd_ * OW) + og, resd.to(DTYPE), mask=om)
        else:
            tl.store(ptr_o + (ob + ra_ * OW) + og, resa.to(DTYPE))
            tl.store(ptr_o + (ob + rb_ * OW) + og, resb.to(DTYPE))
            tl.store(ptr_o + (ob + rc_ * OW) + og, resc.to(DTYPE))
            tl.store(ptr_o + (ob + rd_ * OW) + og, resd.to(DTYPE))


def _plan(O_elems, IH, IW, OH, OW, rsh, rsw, nct):
    """Pick block/segmentation parameters for one launch (host side only)."""
    ncb = 1 if OW <= _MAX_BLOCK_W else int(math.ceil(OW / float(_MAX_BLOCK_W)))
    block_w = triton.next_power_of_2(max(OW, _MIN_SEG)) if ncb == 1 else _MAX_BLOCK_W
    block_w = max(block_w, _MIN_SEG)

    window = int(math.ceil(block_w * rsw)) + 4
    seg = triton.next_power_of_2(max(min(window, max(IW, 1)), _MIN_SEG))
    seg = max(seg, _MIN_SEG)

    tiles0 = max(O_elems, 1) * ncb
    target = 8 * nct
    nchunk = 1
    if tiles0 < target and OH > 1:
        best = None
        for k in range(2, min(OH, 64) + 1):
            if tiles0 * k >= target and (tiles0 * k) % nct == 0:
                best = k
                break
        if best is None:
            best = max(1, int(math.ceil(target / float(tiles0))))
        nchunk = min(best, OH)
    # The row loop advances by _BAND rows per iteration and clamps the surplus rows
    # of a partial band onto the chunk's last row, so a row count that is not a
    # multiple of _BAND would spend whole iterations recomputing rows that were
    # already written.  Snap the per-chunk row count up to a whole number of bands
    # and shrink the chunk count to compensate; the snapped-up tail rows are simply
    # never reached because nrows is recomputed from the true chunk origin.
    rows = int(math.ceil(OH / float(nchunk)))
    rows = int(math.ceil(rows / float(_BAND))) * _BAND
    nchunk = max(1, int(math.ceil(OH / float(rows))))
    return ncb, block_w, seg, nchunk, rows


_SUPPORTED_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


def upsample_bilinear2d(
    input: torch.Tensor,
    output_size,
    align_corners: bool = False,
    scales_h=None,
    scales_w=None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE_BILINEAR2D")
    if input.dtype not in _SUPPORTED_DTYPES:
        from flag_gems.ops.upsample_bilinear2d import upsample_bilinear2d as _generic

        return _generic(input, output_size, align_corners, scales_h, scales_w)

    inp = input if input.is_contiguous() else input.contiguous()
    N, C, IH, IW = (int(s) for s in inp.shape)
    OH, OW = int(output_size[0]), int(output_size[1])

    out = torch.empty((N, C, OH, OW), device=inp.device, dtype=inp.dtype)
    planes = N * C
    if planes == 0 or OH == 0 or OW == 0:
        return out

    rsh = _recip_scale(IH, OH, align_corners, scales_h)
    rsw = _recip_scale(IW, OW, align_corners, scales_w)

    nct = _vector_cores()
    ncb, block_w, seg, nchunk, rows = _plan(planes, IH, IW, OH, OW, rsh, rsw, nct)
    grid = (planes * ncb * nchunk,)

    _ub_bilinear2d_kernel[grid](
        out,
        inp,
        IH,
        IW,
        OH,
        OW,
        rsh,
        rsw,
        ncb,
        nchunk,
        rows,
        BAND=_BAND,
        BLOCK_W=block_w,
        SEG=seg,
        ALIGN_CORNERS=bool(align_corners),
        MASK_W=(ncb * block_w != OW),
        DTYPE=_TL_DTYPE[inp.dtype],
    )
    return out
