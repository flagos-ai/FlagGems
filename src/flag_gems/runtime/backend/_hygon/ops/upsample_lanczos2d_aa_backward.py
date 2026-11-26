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

"""Triton implementation of aten::_upsample_lanczos2d_aa_backward.

Y[n,c,h,w] = sum_{oh,ow} G[n,c,oh,ow] * Wh[oh,h] * Ww[ow,w]

with the antialiased Lanczos-3 weights of the reference formulation:

    scale    = (in-1)/(out-1)            if align_corners and out > 1 else 0
             = 1/scale_arg               if (not align_corners) and scale_arg > 0
             = in/out                    otherwise
    support  = 3*scale if scale >= 1 else 3
    invscale = 1/scale if scale >= 1 else 1
    center   = scale*(o + 0.5)
    i0       = max(floor(center - support + 0.5), 0)
    i1       = min(floor(center + support + 0.5), in_size)
    w(i)     = lanczos3((i - center + 0.5) * invscale),  normalised over [i0, i1)

Every program owns a BH x BW tile of grad_input for one (n, c) plane and walks
the contributing output rows in blocks of BOH.  The separable contraction is
done with two small matrix products (G[BOH, MAX_OW] @ Wblk[MAX_OW, BW] then
Wy[BH, BOH] @ mid[BOH, BW]), so the inner loop contains no cross-lane reduction
and no 1-D -> 2-D broadcast.

The Lanczos weights come from one of two sources:

* tiny tensors: computed inline in the same launch (one launch total, which is
  the launch-overhead floor for small shapes);
* larger tensors: a tiny prep kernel builds the normalised 1-D weight tables
  once and every program gathers them, so the expensive libdevice sin sequence
  is evaluated once per output index instead of once per program.

float64 uses a table based variant because tl.dot has no fp64 path.
"""

import logging
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Keep the module surface of the upstream implementation so that any harness
# monkeypatching those knobs keeps working.
_FUSE_THRESHOLD = 0
_TABLE_THRESHOLD = 65536


def _should_use_gemm_path(*args, **kwargs):  # pragma: no cover - compatibility
    return False


@triton.jit
def _lanczos3(x):
    ax = tl.abs(x)
    pix = 3.141592653589793 * ax
    pix3 = pix / 3.0
    s1 = tl.where(ax == 0.0, 1.0, tl.sin(pix) / pix)
    s3 = tl.where(ax == 0.0, 1.0, tl.sin(pix3) / pix3)
    return tl.where(ax < 3.0, s1 * s3, 0.0)


# ---------------------------------------------------------------------------
# float16 / bfloat16 / float32
# ---------------------------------------------------------------------------


@triton.jit
def _bwd_fused_kernel(
    go_ptr,
    out_ptr,
    tab_ptr,
    start_ptr,
    IH,
    IW,
    OH,
    OW,
    C,
    sn,
    sc,
    sh,
    sw,
    scale_h,
    scale_w,
    sup_h,
    sup_w,
    inv_h,
    inv_w,
    OS,
    BH: tl.constexpr,
    BW: tl.constexpr,
    BOH: tl.constexpr,
    MAX_OH: tl.constexpr,
    MAX_OW: tl.constexpr,
    MAX_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PRECOMPUTED: tl.constexpr,
):
    pid_w = tl.program_id(0)
    th = tl.program_id(1)
    nc = tl.program_id(2)
    n = nc // C
    c = nc - n * C

    h_lo = th * BH
    w_lo = pid_w * BW
    h = h_lo + tl.arange(0, BH)
    w = w_lo + tl.arange(0, BW)
    h_hi = tl.minimum(h_lo + BH, IH)
    w_hi = tl.minimum(w_lo + BW, IW)
    hmask = h < IH
    wmask = w < IW

    # contributing output-index window of this input tile (safe superset)
    inv_sh = tl.where(scale_h > 0.0, 1.0 / tl.where(scale_h > 0.0, scale_h, 1.0), 0.0)
    inv_sw = tl.where(scale_w > 0.0, 1.0 / tl.where(scale_w > 0.0, scale_w, 1.0), 0.0)
    oh_lo = tl.maximum(tl.floor((h_lo + 0.5 - sup_h) * inv_sh - 0.5), 0.0).to(tl.int32)
    oh_hi = tl.minimum(
        tl.ceil((h_hi - 0.5 + sup_h) * inv_sh - 0.5) + 1.0, OH.to(tl.float32)
    ).to(tl.int32)
    ow_lo = tl.maximum(tl.floor((w_lo + 0.5 - sup_w) * inv_sw - 0.5), 0.0).to(tl.int32)
    ow_hi = tl.minimum(
        tl.ceil((w_hi - 0.5 + sup_w) * inv_sw - 0.5) + 1.0, OW.to(tl.float32)
    ).to(tl.int32)
    oh_lo = tl.minimum(oh_lo, OH)
    ow_lo = tl.minimum(ow_lo, OW)

    kk = tl.arange(0, BLOCK_K)
    hf = h.to(tl.float32)
    wf = w.to(tl.float32)

    ow = ow_lo + tl.arange(0, MAX_OW)
    owv = ow < ow_hi
    owc = tl.minimum(ow, OW - 1)

    # ---- horizontal weight block: Wblk[k, j] = Ww[ow_lo + k, w_lo + j]
    if PRECOMPUTED:
        stw = tl.load(start_ptr + OS + owc, mask=owv, other=0)
        tapw = w[None, :] - stw[:, None]
        wblk = tl.load(
            tab_ptr + (OS + owc).to(tl.int64)[:, None] * MAX_K + tapw.to(tl.int64),
            mask=owv[:, None] & wmask[None, :] & (tapw >= 0) & (tapw < MAX_K),
            other=0.0,
        )
    else:
        kkf = kk.to(tl.float32)
        cw = scale_w * (owc.to(tl.float32) + 0.5)
        stw = tl.maximum(tl.floor(cw - sup_w + 0.5), 0.0)
        enw = tl.minimum(tl.floor(cw + sup_w + 0.5), IW.to(tl.float32))
        szw = (enw - stw).to(tl.int32)
        vw = _lanczos3((stw[:, None] + kkf[None, :] - cw[:, None] + 0.5) * inv_w)
        normw = tl.sum(tl.where(kk[None, :] < szw[:, None], vw, 0.0), axis=1)
        normw = tl.where(normw != 0.0, normw, 1.0)
        tapw = w[None, :] - stw.to(tl.int32)[:, None]
        wblk = _lanczos3((wf[None, :] - cw[:, None] + 0.5) * inv_w)
        wblk = (
            tl.where(
                owv[:, None] & wmask[None, :] & (tapw >= 0) & (tapw < szw[:, None]),
                wblk,
                0.0,
            )
            / normw[:, None]
        )

    base = n.to(tl.int64) * sn + c.to(tl.int64) * sc
    acc = tl.zeros((BH, BW), dtype=tl.float32)
    for ci in tl.static_range(MAX_OH // BOH):
        rows = oh_lo + ci * BOH + tl.arange(0, BOH)
        rmask = (rows < oh_hi) & (rows >= 0)
        rowc = tl.minimum(tl.maximum(rows, 0), OH - 1)
        if PRECOMPUTED:
            sth = tl.load(start_ptr + rowc, mask=rmask, other=0)
            taph = h[:, None] - sth[None, :]
            wy = tl.load(
                tab_ptr + rowc.to(tl.int64)[None, :] * MAX_K + taph.to(tl.int64),
                mask=hmask[:, None] & rmask[None, :] & (taph >= 0) & (taph < MAX_K),
                other=0.0,
            )
        else:
            ch = scale_h * (rowc.to(tl.float32) + 0.5)
            sth = tl.maximum(tl.floor(ch - sup_h + 0.5), 0.0)
            enh = tl.minimum(tl.floor(ch + sup_h + 0.5), IH.to(tl.float32))
            szh = (enh - sth).to(tl.int32)
            vh = _lanczos3((sth[:, None] + kkf[None, :] - ch[:, None] + 0.5) * inv_h)
            normh = tl.sum(tl.where(kk[None, :] < szh[:, None], vh, 0.0), axis=1)
            normh = tl.where(normh != 0.0, normh, 1.0)
            taph = h[:, None] - sth.to(tl.int32)[None, :]
            wy = _lanczos3((hf[:, None] - ch[None, :] + 0.5) * inv_h)
            wy = (
                tl.where(
                    hmask[:, None]
                    & rmask[None, :]
                    & (taph >= 0)
                    & (taph < szh[None, :]),
                    wy,
                    0.0,
                )
                / normh[None, :]
            )

        g = tl.load(
            go_ptr + base + rowc[:, None] * sh + ow[None, :] * sw,
            mask=rmask[:, None] & owv[None, :],
            other=0.0,
        ).to(tl.float32)
        mid = tl.dot(g, wblk, input_precision="ieee")
        acc = tl.dot(wy, mid, acc, input_precision="ieee")

    off = (
        nc.to(tl.int64) * (IH * IW)
        + h[:, None].to(tl.int64) * IW
        + w[None, :].to(tl.int64)
    )
    tl.store(
        out_ptr + off,
        acc.to(out_ptr.dtype.element_ty),
        mask=hmask[:, None] & wmask[None, :],
    )


# ---------------------------------------------------------------------------
# weight table prep (shared by the table path and the float64 path)
# ---------------------------------------------------------------------------


@triton.jit
def _prep_weights_kernel(
    tab_ptr,
    start_ptr,
    aux_ptr,
    OH,
    OW,
    IH,
    IW,
    scale_h,
    scale_w,
    sup_h,
    sup_w,
    inv_h,
    inv_w,
    MAX_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK: tl.constexpr,
    F64: tl.constexpr,
):
    DT: tl.constexpr = tl.float64 if F64 else tl.float32
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    is_h = o < OH
    oi = tl.where(is_h, o, o - OH)
    n_out = tl.where(is_h, OH, OW)
    in_sz = tl.where(is_h, IH, IW)
    if F64:
        sc = tl.where(is_h, tl.load(aux_ptr + 0), tl.load(aux_ptr + 3))
        sp = tl.where(is_h, tl.load(aux_ptr + 1), tl.load(aux_ptr + 4))
        iv = tl.where(is_h, tl.load(aux_ptr + 2), tl.load(aux_ptr + 5))
    else:
        sc = tl.where(is_h, scale_h, scale_w)
        sp = tl.where(is_h, sup_h, sup_w)
        iv = tl.where(is_h, inv_h, inv_w)
    sc = sc.to(DT)
    sp = sp.to(DT)
    iv = iv.to(DT)

    m = (oi >= 0) & (oi < n_out)
    center = sc * (oi.to(DT) + 0.5)
    i0 = tl.maximum(tl.floor(center - sp + 0.5), 0.0)
    i1 = tl.minimum(tl.floor(center + sp + 0.5), in_sz.to(DT))

    kk = tl.arange(0, BLOCK_K)
    x = (i0[:, None] + kk.to(DT)[None, :] - center[:, None] + 0.5) * iv[:, None]
    v = _lanczos3(x)
    inside = kk[None, :] < (i1 - i0).to(tl.int32)[:, None]
    v = tl.where(m[:, None] & inside, v, 0.0)
    tot = tl.sum(v, axis=1)
    tot = tl.where(tot != 0.0, tot, 1.0)
    v = v / tot[:, None]

    ok = m[:, None] & (kk[None, :] < MAX_K)
    tl.store(
        tab_ptr + o[:, None].to(tl.int64) * MAX_K + kk[None, :].to(tl.int64),
        v.to(tab_ptr.dtype.element_ty),
        mask=ok,
    )
    tl.store(start_ptr + o, i0.to(tl.int32), mask=m)


# ---------------------------------------------------------------------------
# float64
# ---------------------------------------------------------------------------


@triton.jit
def _bwd_table_kernel(
    go_ptr,
    out_ptr,
    tab_ptr,
    start_ptr,
    aux_ptr,
    IH,
    IW,
    OH,
    OW,
    C,
    sn,
    sc,
    sh,
    sw,
    scale_h,
    scale_w,
    sup_h,
    sup_w,
    inv_h,
    inv_w,
    OS,
    BH: tl.constexpr,
    BW: tl.constexpr,
    MAX_OH: tl.constexpr,
    MAX_OW: tl.constexpr,
    MAX_K: tl.constexpr,
):
    pid_w = tl.program_id(0)
    th = tl.program_id(1)
    nc = tl.program_id(2)
    n = nc // C
    c = nc - n * C

    sch = tl.load(aux_ptr + 0)
    sph = tl.load(aux_ptr + 1)
    scw = tl.load(aux_ptr + 3)
    spw = tl.load(aux_ptr + 4)

    h_lo = th * BH
    w_lo = pid_w * BW
    h = h_lo + tl.arange(0, BH)
    w = w_lo + tl.arange(0, BW)
    h_hi = tl.minimum(h_lo + BH, IH)
    hmask = h < IH
    wmask = w < IW

    inv_sh = tl.where(sch > 0.0, 1.0 / tl.where(sch > 0.0, sch, 1.0), 0.0)
    inv_sw = tl.where(scw > 0.0, 1.0 / tl.where(scw > 0.0, scw, 1.0), 0.0)
    oh_lo = tl.maximum(tl.floor((h_lo + 0.5 - sph) * inv_sh - 0.5), 0.0).to(tl.int32)
    oh_hi = tl.minimum(
        tl.ceil((h_hi - 0.5 + sph) * inv_sh - 0.5) + 1.0, OH.to(tl.float64)
    ).to(tl.int32)
    w_hi = tl.minimum(w_lo + BW, IW)
    ow_lo = tl.maximum(tl.floor((w_lo + 0.5 - spw) * inv_sw - 0.5), 0.0).to(tl.int32)
    ow_hi = tl.minimum(
        tl.ceil((w_hi - 0.5 + spw) * inv_sw - 0.5) + 1.0, OW.to(tl.float64)
    ).to(tl.int32)
    oh_lo = tl.minimum(oh_lo, OH)
    ow_lo = tl.minimum(ow_lo, OW)
    oh_hi = tl.maximum(oh_hi, oh_lo)
    ow_hi = tl.maximum(ow_hi, ow_lo)

    ow = ow_lo + tl.arange(0, MAX_OW)
    owv = ow < ow_hi
    owc = tl.minimum(ow, OW - 1)
    stw = tl.load(start_ptr + OS + owc, mask=owv, other=0)
    tapw = w[None, :] - stw[:, None]
    wt = tl.load(
        tab_ptr + (OS + owc).to(tl.int64)[:, None] * MAX_K + tapw.to(tl.int64),
        mask=owv[:, None] & wmask[None, :] & (tapw >= 0) & (tapw < MAX_K),
        other=0.0,
    )

    base = n.to(tl.int64) * sn + c.to(tl.int64) * sc
    acc = tl.zeros((BH, BW), dtype=tl.float64)
    for d in tl.static_range(MAX_OH):
        oh = oh_lo + d
        ohv = oh < oh_hi
        ohc = tl.minimum(tl.maximum(oh, 0), OH - 1)
        g = tl.load(
            go_ptr + base + ohc * sh + ow * sw,
            mask=owv & ohv,
            other=0.0,
        )
        mid = tl.sum(g[:, None] * wt, axis=0)
        sth = tl.load(start_ptr + ohc, mask=ohv, other=0)
        taph = h - sth
        wy = tl.load(
            tab_ptr + ohc.to(tl.int64) * MAX_K + taph.to(tl.int64),
            mask=hmask & ohv & (taph >= 0) & (taph < MAX_K),
            other=0.0,
        )
        acc += wy[:, None] * mid[None, :]

    off = (
        nc.to(tl.int64) * (IH * IW)
        + h[:, None].to(tl.int64) * IW
        + w[None, :].to(tl.int64)
    )
    tl.store(
        out_ptr + off,
        acc.to(out_ptr.dtype.element_ty),
        mask=hmask[:, None] & wmask[None, :],
    )


def _compute_scale(in_size, out_size, align_corners, scale):
    if align_corners:
        return (in_size - 1) / (out_size - 1) if out_size > 1 else 0.0
    if scale is not None and scale > 0:
        return 1.0 / scale
    return in_size / out_size


def _max_window(tile, scale, support, out_size):
    if scale <= 0.0:
        return 1
    span = (
        int(math.ceil((tile - 1) / scale)) + int(math.ceil(2.0 * support / scale)) + 3
    )
    return max(1, min(out_size, span))


def upsample_lanczos2d_aa_backward(
    grad_output,
    output_size,
    input_size,
    align_corners,
    scales_h=None,
    scales_w=None,
):
    logger.debug("GEMS_HYGON UPSAMPLE_LANCZOS2D_AA_BACKWARD")
    if grad_output.ndim != 4:
        raise RuntimeError(
            "Expected grad_output to be a tensor of dimension 4 but got: "
            f"dimension {grad_output.ndim}"
        )
    if len(output_size) != 2 or len(input_size) != 4:
        raise RuntimeError("Expected output_size[2] and input_size[4]")
    if not grad_output.is_floating_point():
        raise RuntimeError(
            f'"upsample_lanczos2d_aa_backward" not implemented for '
            f"{grad_output.dtype}"
        )

    N, C, IH, IW = (int(v) for v in input_size)
    OH, OW = int(output_size[0]), int(output_size[1])
    if tuple(grad_output.shape) != (N, C, OH, OW):
        raise RuntimeError(
            f"Expected grad_output shape ({N}, {C}, {OH}, {OW}), "
            f"but got {tuple(grad_output.shape)}"
        )

    NC = N * C
    if NC == 0 or IH == 0 or IW == 0 or OH == 0 or OW == 0:
        return grad_output.new_zeros((N, C, IH, IW))

    ac = bool(align_corners)
    scale_h = _compute_scale(IH, OH, ac, scales_h)
    scale_w = _compute_scale(IW, OW, ac, scales_w)
    support_h = 3.0 * scale_h if scale_h >= 1.0 else 3.0
    support_w = 3.0 * scale_w if scale_w >= 1.0 else 3.0
    invscale_h = 1.0 / scale_h if scale_h >= 1.0 else 1.0
    invscale_w = 1.0 / scale_w if scale_w >= 1.0 else 1.0

    dt = grad_output.dtype
    dev = grad_output.device
    out = torch.empty((NC, IH, IW), dtype=dt, device=dev)

    BH = 32 if IH >= 32 else max(16, triton.next_power_of_2(IH))
    BW = 32 if IW >= 32 else max(16, triton.next_power_of_2(IW))
    tiles_h = triton.cdiv(IH, BH)
    grid_w = triton.cdiv(IW, BW)
    sn, sc, sh, sw = grad_output.stride()
    grid = (grid_w, tiles_h, NC)

    if dt != torch.float64:
        BOH = 16
        max_oh = _max_window(BH, scale_h, support_h, OH)
        max_oh = max(BOH, BOH * int(math.ceil(max_oh / BOH)))
        max_ow = max(
            16, triton.next_power_of_2(_max_window(BW, scale_w, support_w, OW))
        )
        max_ow = min(max_ow, max(16, triton.next_power_of_2(OW)))
        block_k = triton.next_power_of_2(
            max(int(math.ceil(2.0 * support_h)), int(math.ceil(2.0 * support_w))) + 2
        )
        use_tables = NC * IH * IW > _TABLE_THRESHOLD
        if use_tables:
            tab = torch.empty((OH + OW, block_k), dtype=torch.float32, device=dev)
            start = torch.empty((OH + OW,), dtype=torch.int32, device=dev)
            prep_block = min(64, triton.next_power_of_2(OH + OW))
            _prep_weights_kernel[(triton.cdiv(OH + OW, prep_block),)](
                tab,
                start,
                out,
                OH,
                OW,
                IH,
                IW,
                scale_h,
                scale_w,
                support_h,
                support_w,
                invscale_h,
                invscale_w,
                MAX_K=block_k,
                BLOCK_K=block_k,
                BLOCK=prep_block,
                F64=False,
                num_warps=2,
            )
        else:
            tab = out
            start = out
        _bwd_fused_kernel[grid](
            grad_output,
            out,
            tab,
            start,
            IH,
            IW,
            OH,
            OW,
            C,
            sn,
            sc,
            sh,
            sw,
            scale_h,
            scale_w,
            support_h,
            support_w,
            invscale_h,
            invscale_w,
            OH,
            BH=BH,
            BW=BW,
            BOH=BOH,
            MAX_OH=max_oh,
            MAX_OW=max_ow,
            MAX_K=block_k,
            BLOCK_K=block_k,
            PRECOMPUTED=use_tables,
            num_warps=4,
        )
        return out.view(N, C, IH, IW)

    # ---- float64: table based path
    max_k = (
        max(
            int(math.ceil(support_h)),
            int(math.ceil(support_w)),
        )
        * 2
        + 2
    )
    max_ow = triton.next_power_of_2(_max_window(BW, scale_w, support_w, OW))
    max_oh = _max_window(BH, scale_h, support_h, OH)

    tab = torch.empty((OH + OW, max_k), dtype=torch.float64, device=dev)
    start = torch.empty((OH + OW,), dtype=torch.int32, device=dev)
    aux = torch.tensor(
        [scale_h, support_h, invscale_h, scale_w, support_w, invscale_w],
        dtype=torch.float64,
        device=dev,
    )
    prep_block = min(64, triton.next_power_of_2(OH + OW))
    _prep_weights_kernel[(triton.cdiv(OH + OW, prep_block),)](
        tab,
        start,
        aux,
        OH,
        OW,
        IH,
        IW,
        scale_h,
        scale_w,
        support_h,
        support_w,
        invscale_h,
        invscale_w,
        MAX_K=max_k,
        BLOCK_K=triton.next_power_of_2(max_k),
        BLOCK=prep_block,
        F64=True,
        num_warps=2,
    )
    _bwd_table_kernel[grid](
        grad_output,
        out,
        tab,
        start,
        aux,
        IH,
        IW,
        OH,
        OW,
        C,
        sn,
        sc,
        sh,
        sw,
        scale_h,
        scale_w,
        support_h,
        support_w,
        invscale_h,
        invscale_w,
        OH,
        BH=BH,
        BW=BW,
        MAX_OH=max_oh,
        MAX_OW=max_ow,
        MAX_K=max_k,
        num_warps=4,
    )
    return out.view(N, C, IH, IW)
