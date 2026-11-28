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

logger = logging.getLogger(__name__)


@triton.jit
def _aap3d_bwd_uniform_bcast_kernel(
    gout_ptr,
    gin_ptr,
    oD,
    oH,
    oW,
    oHW,
    iW,
    iHW,
    iDHW,
    oPlane,
    rD: tl.constexpr,
    rH: tl.constexpr,
    rW: tl.constexpr,
    CNT: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    RW_PAD: tl.constexpr,
):
    """Exact integer-ratio downsampling with load reuse across the ratio block.

    One lane owns one output column group (rW contiguous input columns) for one
    (p, od, oh); the single grad_output load is then broadcast to the rD*rH
    input rows it feeds, and each store writes rW contiguous elements.
    Grid: (column blocks, oH/ROWS, planes*oD).
    """
    cb = tl.program_id(0)
    oh0 = tl.program_id(1) * ROWS
    pid = tl.program_id(2)
    p = pid // oD
    od = pid - p * oD

    oh = oh0 + tl.arange(0, ROWS)
    c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    m = (oh < oH)[:, None] & (c < oW)[None, :]

    g = tl.load(
        gout_ptr + p * oPlane + od * oHW + oh[:, None] * oW + c[None, :],
        mask=m,
        other=0.0,
    )
    v = (g.to(tl.float32) / CNT).to(gin_ptr.dtype.element_ty)[:, :, None]

    kw = tl.arange(0, RW_PAD)
    col = c[None, :, None] * rW + kw[None, None, :]
    store_mask = m[:, :, None] & (kw[None, None, :] < rW)
    in_plane = p * iDHW
    for kd in tl.static_range(rD):
        dbase = in_plane + (od * rD + kd) * iHW
        for kh in tl.static_range(rH):
            rbase = dbase + (oh[:, None, None] * rH + kh) * iW
            tl.store(gin_ptr + rbase + col, v, mask=store_mask)


@triton.jit
def _aap3d_bwd_uniform_down_kernel(
    gout_ptr,
    gin_ptr,
    iD,
    iH,
    iW,
    oHW,
    oW,
    iHW,
    iDHW,
    oPlane,
    rD: tl.constexpr,
    rH: tl.constexpr,
    rW: tl.constexpr,
    CNT: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Exact integer-ratio downsampling: every input has exactly one contributor.

    With iD = oD*rD (and likewise H, W) each input position i is covered by the
    single output position i // r per axis, and the pooled window size is the
    constant rD*rH*rW. Grid: (w blocks, iH/ROWS, planes*iD). The block spans
    ROWS consecutive input rows x BLOCK_W columns, so one CTA carries ROWS
    independent 64-lane memory chains instead of a single one.
    """
    wb = tl.program_id(0)
    ih0 = tl.program_id(1) * ROWS
    pid = tl.program_id(2)
    p = pid // iD
    id = pid - p * iD

    offs = tl.arange(0, ROWS * BLOCK_W)
    rr = offs // BLOCK_W
    iw = offs - rr * BLOCK_W + wb * BLOCK_W
    ih = ih0 + rr
    lane = (iw < iW) & (ih < iH)

    g = tl.load(
        gout_ptr + p * oPlane + (id // rD) * oHW + (ih // rH) * oW + iw // rW,
        mask=lane,
        other=0.0,
    )
    res = g.to(tl.float32) / CNT
    tl.store(
        gin_ptr + p * iDHW + id * iHW + ih * iW + iw,
        res.to(gin_ptr.dtype.element_ty),
        mask=lane,
    )


@triton.jit
def _aap3d_bwd_contig_kernel(
    gout_ptr,
    gin_ptr,
    total,
    iD,
    iH,
    iW,
    oD,
    oH,
    oW,
    BLOCK: tl.constexpr,
):
    """grad_input[i] = sum over output windows containing i of grad_output[o]/count(o).

    General contiguous path: exact per-input output ranges with dynamic loop
    bounds, fp32 accumulation, single dtype-preserving store.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    lane = offs < total

    iHW = iH * iW
    iDHW = iD * iHW
    p = offs // iDHW
    s = offs - p * iDHW
    id = s // iHW
    r = s - id * iHW
    ih = r // iW
    iw = r - ih * iW

    # Exact output range contributing to this input position.
    od0 = (id * oD) // iD
    od1 = ((id + 1) * oD + iD - 1) // iD
    oh0 = (ih * oH) // iH
    oh1 = ((ih + 1) * oH + iH - 1) // iH
    ow0 = (iw * oW) // iW
    ow1 = ((iw + 1) * oW + iW - 1) // iW

    MD = tl.max(od1 - od0)
    MH = tl.max(oh1 - oh0)
    MW = tl.max(ow1 - ow0)

    oHW = oH * oW
    base = p * (oD * oHW)

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for kd in range(0, MD):
        od = od0 + kd
        md = od < od1
        ds = (od * iD) // oD
        de = ((od + 1) * iD + oD - 1) // oD
        cd = (de - ds).to(tl.float32)
        for kh in range(0, MH):
            oh = oh0 + kh
            mh = md & (oh < oh1)
            hs = (oh * iH) // oH
            he = ((oh + 1) * iH + oH - 1) // oH
            cdh = cd * (he - hs).to(tl.float32)
            for kw in range(0, MW):
                ow = ow0 + kw
                m = mh & (ow < ow1)
                ws = (ow * iW) // oW
                we = ((ow + 1) * iW + oW - 1) // oW
                cnt = cdh * (we - ws).to(tl.float32)
                g = tl.load(
                    gout_ptr + base + od * oHW + oh * oW + ow,
                    mask=m & lane,
                    other=0.0,
                )
                acc += tl.where(m, g.to(tl.float32) / cnt, 0.0)

    tl.store(gin_ptr + offs, acc.to(gin_ptr.dtype.element_ty), mask=lane)


@triton.jit
def _aap3d_bwd_strided_kernel(
    gout_ptr,
    gin_ptr,
    total,
    iD,
    iH,
    iW,
    oD,
    oH,
    oW,
    C,
    sN,
    sC,
    sD,
    sH,
    sW,
    gN,
    gC,
    gD,
    gH,
    gW,
    BLOCK: tl.constexpr,
):
    """Generic layout path: explicit strides for both tensors."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    lane = offs < total

    iHW = iH * iW
    iDHW = iD * iHW
    p = offs // iDHW
    s = offs - p * iDHW
    id = s // iHW
    r = s - id * iHW
    ih = r // iW
    iw = r - ih * iW
    n = p // C
    c = p - n * C

    od0 = (id * oD) // iD
    od1 = ((id + 1) * oD + iD - 1) // iD
    oh0 = (ih * oH) // iH
    oh1 = ((ih + 1) * oH + iH - 1) // iH
    ow0 = (iw * oW) // iW
    ow1 = ((iw + 1) * oW + iW - 1) // iW

    MD = tl.max(od1 - od0)
    MH = tl.max(oh1 - oh0)
    MW = tl.max(ow1 - ow0)

    base = n * sN + c * sC
    obase = n * gN + c * gC + id * gD + ih * gH + iw * gW

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for kd in range(0, MD):
        od = od0 + kd
        md = od < od1
        ds = (od * iD) // oD
        de = ((od + 1) * iD + oD - 1) // oD
        cd = (de - ds).to(tl.float32)
        for kh in range(0, MH):
            oh = oh0 + kh
            mh = md & (oh < oh1)
            hs = (oh * iH) // oH
            he = ((oh + 1) * iH + oH - 1) // oH
            cdh = cd * (he - hs).to(tl.float32)
            for kw in range(0, MW):
                ow = ow0 + kw
                m = mh & (ow < ow1)
                ws = (ow * iW) // oW
                we = ((ow + 1) * iW + oW - 1) // oW
                cnt = cdh * (we - ws).to(tl.float32)
                g = tl.load(
                    gout_ptr + base + od * sD + oh * sH + ow * sW,
                    mask=m & lane,
                    other=0.0,
                )
                acc += tl.where(m, g.to(tl.float32) / cnt, 0.0)

    tl.store(gin_ptr + obase, acc.to(gin_ptr.dtype.element_ty), mask=lane)


_BLOCK = 256


def _adaptive_avg_pool3d_backward(grad_output, self, grad_input=None):
    logger.debug("GEMS_HYGON _ADAPTIVE_AVG_POOL3D_BACKWARD")
    shape = self.shape
    iD = shape[-3]
    iH = shape[-2]
    iW = shape[-1]
    oshape = grad_output.shape
    oD = oshape[-3]
    oH = oshape[-2]
    oW = oshape[-1]

    if grad_input is None:
        grad_input = torch.empty(shape, dtype=self.dtype, device=self.device)

    total = self.numel()
    if total == 0:
        return grad_input

    if grad_output.is_contiguous() and grad_input.is_contiguous():
        # Large exact integer-ratio downsampling: reuse one grad_output load
        # across the whole rD*rH*rW coverage block and store rW elements per
        # lane, which cuts memory instructions per output by ~3x.
        if (
            total >= 65536
            and oD
            and oH
            and oW
            and iD % oD == 0
            and iH % oH == 0
            and iW % oW == 0
        ):
            iHW = iH * iW
            iDHW = iD * iHW
            pd = (total // iDHW) * oD
            bc = min(256, triton.next_power_of_2(oW))
            lim = min(max(1, 256 // bc), oH)
            rows = 1 << (lim.bit_length() - 1)
            rW = iW // oW
            nb = (oH + rows - 1) // rows
            if pd <= 65535 and nb <= 65535:
                grid = ((oW + bc - 1) // bc, nb, pd)
                _aap3d_bwd_uniform_bcast_kernel[grid](
                    grad_output,
                    grad_input,
                    oD,
                    oH,
                    oW,
                    oH * oW,
                    iW,
                    iHW,
                    iDHW,
                    oD * oH * oW,
                    rD=iD // oD,
                    rH=iH // oH,
                    rW=rW,
                    CNT=float((iD // oD) * (iH // oH) * rW),
                    ROWS=rows,
                    BLOCK_C=bc,
                    RW_PAD=triton.next_power_of_2(rW),
                    num_warps=max(1, (rows * bc) // 64),
                )
                return grad_input

        # Exact integer-ratio downsampling: one contribution per input, so no
        # per-lane window search, division or block reduction is needed.
        if (
            oD
            and oH
            and oW
            and iW >= 32
            and iD % oD == 0
            and iH % oH == 0
            and iW % oW == 0
        ):
            iHW = iH * iW
            iDHW = iD * iHW
            pd = (total // iDHW) * iD
            if pd <= 65535:
                rD = iD // oD
                rH = iH // oH
                rW = iW // oW
                bw = min(256, triton.next_power_of_2(iW))
                rows = 8 if total >= 65536 else (4 if total >= 16384 else 1)
                if rows * bw > 1024:
                    rows = max(1, 1024 // bw)
                nh = (iH + rows - 1) // rows
                if nh <= 65535:
                    nw = max(1, (rows * bw) // 64)
                    grid = ((iW + bw - 1) // bw, nh, pd)
                    _aap3d_bwd_uniform_down_kernel[grid](
                        grad_output,
                        grad_input,
                        iD,
                        iH,
                        iW,
                        oH * oW,
                        oW,
                        iHW,
                        iDHW,
                        oD * oH * oW,
                        rD=rD,
                        rH=rH,
                        rW=rW,
                        CNT=float(rD * rH * rW),
                        ROWS=rows,
                        BLOCK_W=bw,
                        num_warps=nw,
                    )
                    return grad_input

        # Small tensors are launch-floor bound, so spread the window search over
        # more CTAs instead of filling only a handful of SMs with 256-lane blocks.
        blk = _BLOCK if total >= 8192 else 64
        grid = ((total + blk - 1) // blk,)
        _aap3d_bwd_contig_kernel[grid](
            grad_output,
            grad_input,
            total,
            iD,
            iH,
            iW,
            oD,
            oH,
            oW,
            BLOCK=blk,
            num_warps=4 if blk == _BLOCK else 1,
        )
    else:
        sN, sC, sD, sH, sW = grad_output.stride()
        gN, gC, gD, gH, gW = grad_input.stride()
        C = oshape[-4]
        grid = ((total + _BLOCK - 1) // _BLOCK,)
        _aap3d_bwd_strided_kernel[grid](
            grad_output,
            grad_input,
            total,
            iD,
            iH,
            iW,
            oD,
            oH,
            oW,
            C,
            sN,
            sC,
            sD,
            sH,
            sW,
            gN,
            gC,
            gD,
            gH,
            gW,
            BLOCK=_BLOCK,
            num_warps=4,
        )
    return grad_input
