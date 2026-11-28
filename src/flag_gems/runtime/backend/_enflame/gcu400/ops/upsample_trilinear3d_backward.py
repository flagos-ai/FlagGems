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
import math
from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def upsample_trilinear3d_backward_kernel(
    grad_out_ptr,
    grad_in_ptr,
    NC,
    ID,
    IH,
    IW,
    OD,
    OH,
    OW,
    scale_d,
    scale_h,
    scale_w,
    bias_d,
    bias_h,
    bias_w,
    SAME_D: tl.constexpr,
    SAME_H: tl.constexpr,
    SAME_W: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    nc0 = tl.program_id(1)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total_out = OD * OH * OW
    mask = offs < total_out

    ow = offs % OW
    oh = (offs // OW) % OH
    od = offs // (OW * OH)

    # Map output coordinates back to source (input) space.
    if SAME_D:
        src_d = od.to(tl.float32)
    else:
        src_d = od.to(tl.float32) * scale_d + bias_d
    if SAME_H:
        src_h = oh.to(tl.float32)
    else:
        src_h = oh.to(tl.float32) * scale_h + bias_h
    if SAME_W:
        src_w = ow.to(tl.float32)
    else:
        src_w = ow.to(tl.float32) * scale_w + bias_w

    src_d = tl.maximum(0.0, tl.minimum(src_d, ID - 1.0))
    src_h = tl.maximum(0.0, tl.minimum(src_h, IH - 1.0))
    src_w = tl.maximum(0.0, tl.minimum(src_w, IW - 1.0))

    id0 = tl.floor(src_d).to(tl.int32)
    ih0 = tl.floor(src_h).to(tl.int32)
    iw0 = tl.floor(src_w).to(tl.int32)

    id1 = tl.minimum(id0 + 1, ID - 1)
    ih1 = tl.minimum(ih0 + 1, IH - 1)
    iw1 = tl.minimum(iw0 + 1, IW - 1)

    td = src_d - id0.to(tl.float32)
    th = src_h - ih0.to(tl.float32)
    tw = src_w - iw0.to(tl.float32)

    wd0 = 1.0 - td
    wh0 = 1.0 - th
    ww0 = 1.0 - tw

    d_stride = IH * IW
    h_stride = IW

    rel_000 = id0 * d_stride + ih0 * h_stride + iw0
    rel_001 = id0 * d_stride + ih0 * h_stride + iw1
    rel_010 = id0 * d_stride + ih1 * h_stride + iw0
    rel_011 = id0 * d_stride + ih1 * h_stride + iw1
    rel_100 = id1 * d_stride + ih0 * h_stride + iw0
    rel_101 = id1 * d_stride + ih0 * h_stride + iw1
    rel_110 = id1 * d_stride + ih1 * h_stride + iw0
    rel_111 = id1 * d_stride + ih1 * h_stride + iw1

    # GCU grid.y hardware limit is 255; each program covers NUM_TILE
    # consecutive (n, c) slices so grid.y stays within the cap.
    for i in range(NUM_TILE):
        nc = nc0 * NUM_TILE + i
        m = mask & (nc < NC)

        g = tl.load(grad_out_ptr + nc * total_out + offs, mask=m, other=0.0).to(
            tl.float32
        )

        in_base = nc * (ID * IH * IW)

        tl.atomic_add(grad_in_ptr + in_base + rel_000, g * wd0 * wh0 * ww0, mask=m)
        tl.atomic_add(grad_in_ptr + in_base + rel_001, g * wd0 * wh0 * tw, mask=m)
        tl.atomic_add(grad_in_ptr + in_base + rel_010, g * wd0 * th * ww0, mask=m)
        tl.atomic_add(grad_in_ptr + in_base + rel_011, g * wd0 * th * tw, mask=m)
        tl.atomic_add(grad_in_ptr + in_base + rel_100, g * td * wh0 * ww0, mask=m)
        tl.atomic_add(grad_in_ptr + in_base + rel_101, g * td * wh0 * tw, mask=m)
        tl.atomic_add(grad_in_ptr + in_base + rel_110, g * td * th * ww0, mask=m)
        tl.atomic_add(grad_in_ptr + in_base + rel_111, g * td * th * tw, mask=m)


@triton.jit
def _backward_weight(o, i, scale, bias, in_sz):
    # Weight of output coordinate `o` contributing to input coordinate `i`.
    src = o * scale + bias
    src = tl.maximum(0.0, tl.minimum(src, in_sz - 1.0))
    i0 = tl.floor(src).to(tl.int32)
    t = src - i0.to(tl.float32)
    w = tl.where(i0 == i, 1.0 - t, 0.0)
    w = w + tl.where(i0 + 1 == i, t, 0.0)
    return w


@triton.jit
def upsample_trilinear3d_backward_gather_kernel(
    grad_out_ptr,
    grad_in_ptr,
    NC,
    ID,
    IH,
    IW,
    OD,
    OH,
    OW,
    scale_d,
    scale_h,
    scale_w,
    bias_d,
    bias_h,
    bias_w,
    NROWS,
    RPP,
    HT,
    BH: tl.constexpr,
    BW: tl.constexpr,
):
    # Input-anchored scatter-free backward: each program owns a [BH, BW] input
    # tile of one (nc, id) plane and accumulates every output voxel whose
    # trilinear stencil touches it, replacing 8 atomics per output voxel with
    # gathered loads and coalesced single-writer stores. The 2D lane tile keeps
    # the per-program loop trip count identical while covering BH times more
    # elements per load instruction. Programs cover up to RPP consecutive rows
    # so grid.x stays below the 65535 hardware limit.
    pid = tl.program_id(0)
    num_wtiles = tl.cdiv(IW, BW)
    wt = pid % num_wtiles
    row0 = (pid // num_wtiles) * RPP
    row_end = tl.minimum(row0 + RPP, NROWS)

    iw = wt * BW + tl.arange(0, BW)
    iwf = iw.to(tl.float32)
    lm_w = iw < IW
    o_lo_w = tl.maximum(0, (-tl.floor(-(iwf - 1.0 - bias_w) / scale_w)).to(tl.int32))
    o_hi_w = tl.minimum(
        OW - 1, (-tl.floor(-(iwf + 1.0 - bias_w) / scale_w) - 1.0).to(tl.int32)
    )
    win_w = tl.maximum(0, o_hi_w - o_lo_w + 1)
    win_w = tl.where(lm_w, win_w, 0)
    win_max = tl.max(win_w, axis=0)
    total_out = OD * OH * OW

    for row in range(row0, row_end):
        nc = row // (ID * HT)
        rid = row % (ID * HT)
        id_ = rid // HT
        it = rid % HT
        idf = id_.to(tl.float32)

        ih_vec = it * BH + tl.arange(0, BH)
        ihf = ih_vec.to(tl.float32)
        lm_h = ih_vec < IH
        o_lo_h = tl.maximum(
            0, (-tl.floor(-(ihf - 1.0 - bias_h) / scale_h)).to(tl.int32)
        )
        o_hi_h = tl.minimum(
            OH - 1,
            (-tl.floor(-(ihf + 1.0 - bias_h) / scale_h) - 1.0).to(tl.int32),
        )
        win_h = tl.maximum(0, o_hi_h - o_lo_h + 1)
        win_h = tl.where(lm_h, win_h, 0)
        win_h_max = tl.max(win_h, axis=0)

        od_lo = tl.maximum(0, (-tl.floor(-(idf - 1.0 - bias_d) / scale_d)).to(tl.int32))
        od_hi = tl.minimum(
            OD - 1,
            (-tl.floor(-(idf + 1.0 - bias_d) / scale_d) - 1.0).to(tl.int32),
        )

        acc = tl.zeros([BH, BW], dtype=tl.float32)
        in_base = nc * (ID * IH * IW)

        for dw in range(0, win_max):
            o_w = o_lo_w + dw
            m_w = (dw < win_w) & lm_w
            w_w = _backward_weight(o_w.to(tl.float32), iwf, scale_w, bias_w, IW)
            for od in range(od_lo, od_hi + 1):
                wd = _backward_weight(od.to(tl.float32), idf, scale_d, bias_d, ID)
                base_d = nc * total_out + od * (OH * OW)
                for dh in range(0, win_h_max):
                    o_h = o_lo_h + dh
                    m_h = (dh < win_h) & lm_h
                    w_h = _backward_weight(o_h.to(tl.float32), ihf, scale_h, bias_h, IH)
                    m2 = m_h[:, None] & m_w[None, :]
                    g = tl.load(
                        grad_out_ptr + base_d + o_h[:, None] * OW + o_w[None, :],
                        mask=m2,
                        other=0.0,
                    ).to(tl.float32)
                    acc += g * (wd * (w_h[:, None] * w_w[None, :]))

        in_off = in_base + id_ * (IH * IW) + ih_vec[:, None] * IW + iw[None, :]
        tl.store(
            grad_in_ptr + in_off,
            acc.to(grad_in_ptr.dtype.element_ty),
            mask=lm_h[:, None] & lm_w[None, :],
        )


def _calc_scale_and_bias(in_sz, out_sz, align_corners, scale):
    if align_corners:
        if out_sz > 1:
            scale_val = (in_sz - 1.0) / (out_sz - 1.0)
        else:
            scale_val = 0.0
        bias_val = 0.0
    else:
        if scale is not None and scale > 0:
            real_scale = 1.0 / scale
        else:
            real_scale = in_sz / out_sz
        scale_val = real_scale
        bias_val = 0.5 * real_scale - 0.5
    return scale_val, bias_val


def _max_window_len(in_sz, out_sz, scale):
    # Conservative upper bound of the per-input-voxel output window width;
    # ceil(2/s)+2 always covers the true window including edge ceil jitter.
    if out_sz <= 0 or scale <= 0:
        return out_sz
    return min(out_sz, math.ceil(2.0 / scale) + 2)


def upsample_trilinear3d_backward(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
    align_corners: bool,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME UPSAMPLE_TRILINEAR3D_BACKWARD")

    N, C, ID, IH, IW = input_size
    OD, OH, OW = output_size
    NC = N * C

    grad_out = grad_output.contiguous().view(NC, OD, OH, OW)
    # Accumulate in fp32 for numerically stable atomic adds, cast back at the end.
    grad_in = torch.zeros(
        (NC, ID, IH, IW),
        device=grad_output.device,
        dtype=torch.float32,
    )

    if grad_in.numel() == 0 or grad_out.numel() == 0:
        return grad_in.view(N, C, ID, IH, IW).to(grad_output.dtype)

    scale_d, bias_d = _calc_scale_and_bias(ID, OD, align_corners, scales_d)
    scale_h, bias_h = _calc_scale_and_bias(IH, OH, align_corners, scales_h)
    scale_w, bias_w = _calc_scale_and_bias(IW, OW, align_corners, scales_w)

    win_d = _max_window_len(ID, OD, scale_d)
    win_h = _max_window_len(IH, OH, scale_h)
    win_w = _max_window_len(IW, OW, scale_w)

    if scale_d > 0 and scale_h > 0 and scale_w > 0 and max(win_d, win_h, win_w) <= 32:
        # Scatter-free input-anchored path: each input voxel gathers from the
        # output window covering it; single-writer stores replace all atomics,
        # so grad_in needs no zero-init and no post-cast.
        grad_in = torch.empty(
            (NC, ID, IH, IW),
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        itemsize = grad_output.element_size()
        # W-lane width: 256 lanes for 2-byte dtypes; fp32 must stay at 128 —
        # fp32 x BW=256 tiles hit the ld.lld ICALL relocation defect (P03).
        BW_CAP = 256 if itemsize <= 2 else 128
        BW = min(BW_CAP, max(16, triton.next_power_of_2(IW)))
        # 2D lane tile, num_warps=2: larger tiles amortize the fixed
        # per-iteration loop cost of the gather window loops; nw=2 measured
        # 2x faster than 4/8 for this kernel on GCU400.
        BH = min(32, max(1, triton.next_power_of_2(IH)))
        HT = triton.cdiv(IH, BH)
        nrows = NC * ID * HT
        wtiles = triton.cdiv(IW, BW)
        # Keep grid.x below the 65535 hardware limit by covering several input
        # rows per program.
        RPP = max(1, -(-nrows * wtiles // 60000))
        grid = (triton.cdiv(nrows, RPP) * wtiles,)
        upsample_trilinear3d_backward_gather_kernel[grid](
            grad_out,
            grad_in,
            NC,
            ID,
            IH,
            IW,
            OD,
            OH,
            OW,
            scale_d,
            scale_h,
            scale_w,
            bias_d,
            bias_h,
            bias_w,
            nrows,
            RPP,
            HT,
            BH=BH,
            BW=BW,
            num_warps=2,
        )
        return grad_in.view(N, C, ID, IH, IW)

    total_out = OD * OH * OW
    # BLOCK=256 (num_warps=4) empirically best: each program scatters into 8
    # input corner voxels via atomic_add, so larger blocks increase atomic
    # contention and regress fp32; 256 balances occupancy against contention.
    BLOCK = 256
    # GCU grid.y limit is 255: chunk the NC axis so each program handles
    # NUM_TILE consecutive slices and grid.y never exceeds 128.
    NUM_TILE = max(1, triton.cdiv(NC, 128))
    grid = (triton.cdiv(total_out, BLOCK), triton.cdiv(NC, NUM_TILE))

    upsample_trilinear3d_backward_kernel[grid](
        grad_out,
        grad_in,
        NC,
        ID,
        IH,
        IW,
        OD,
        OH,
        OW,
        scale_d,
        scale_h,
        scale_w,
        bias_d,
        bias_h,
        bias_w,
        SAME_D=(ID == OD),
        SAME_H=(IH == OH),
        SAME_W=(IW == OW),
        BLOCK=BLOCK,
        NUM_TILE=NUM_TILE,
        num_warps=4,
    )

    return grad_in.view(N, C, ID, IH, IW).to(grad_output.dtype)
