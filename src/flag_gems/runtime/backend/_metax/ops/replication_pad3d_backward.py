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

import torch
import triton
import triton.language as tl

_TILE_CONFIGS = [
    triton.Config({"BH": 16, "BW": 32}, num_warps=2, num_stages=1),
    triton.Config({"BH": 16, "BW": 16}, num_warps=2, num_stages=1),
    triton.Config({"BH": 16, "BW": 8}, num_warps=2, num_stages=1),
    triton.Config({"BH": 8, "BW": 16}, num_warps=2, num_stages=1),
    triton.Config({"BH": 8, "BW": 8}, num_warps=2, num_stages=1),
    triton.Config({"BH": 8, "BW": 8}, num_warps=1, num_stages=1),
    triton.Config({"BH": 16, "BW": 8}, num_warps=4, num_stages=1),
    triton.Config({"BH": 8, "BW": 16}, num_warps=4, num_stages=1),
]


@triton.jit
def _rep_pad3d_bwd_body(
    go_ptr,
    out_ptr,
    D,
    H,
    W,
    Od,
    Oh,
    Ow,
    pf,
    pt,
    pl,
    sN,
    sC,
    sD,
    sH,
    sW,
    C,
    W1: tl.constexpr,
    FLAT: tl.constexpr,
    IDX64: tl.constexpr,
    F64: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
):
    NT_W = tl.cdiv(W, BW)
    NT_H = tl.cdiv(H, BH)
    pid = tl.program_id(0)
    wt = pid % NT_W
    r = pid // NT_W
    ht = r % NT_H
    r2 = r // NT_H
    d = r2 % D
    nc = r2 // D

    if IDX64:
        ncl = nc.to(tl.int64)
    else:
        ncl = nc
    if FLAT:
        base = ncl * sC
    else:
        base = (ncl // C) * sN + (ncl % C) * sC

    h0 = ht * BH
    w0 = wt * BW
    h = h0 + tl.arange(0, BH)
    w = w0 + tl.arange(0, BW)
    if W1:
        w = tl.max_contiguous(tl.multiple_of(w, BW), BW)
    hmask = h < H
    wmask = w < W

    # Offset ranges centred on the natural index (i + pad).  A tile that touches
    # the low edge also needs the t<0 side, a tile that touches the high edge the
    # t>0 side.  min/max with 0 keeps the t=0 interior lane reachable for
    # negative pads.
    lo_d = tl.minimum(-pf, 0) if d == 0 else 0
    hi_d = tl.maximum(Od - D - pf, 0) if d == D - 1 else 0
    lo_h = tl.minimum(-pt, 0) if h0 == 0 else 0
    hi_h = tl.maximum(Oh - H - pt, 0) if h0 + BH >= H else 0
    lo_w = tl.minimum(-pl, 0) if w0 == 0 else 0
    hi_w = tl.maximum(Ow - W - pl, 0) if w0 + BW >= W else 0

    # Per-axis decomposition.  For an offset t != 0 on the h (resp. w) axis the
    # clamp predicate clamp(i + t, 0, I - 1) == i admits exactly one lane, i = 0
    # for t < 0 and i = I - 1 for t > 0, so those contributions are a single row
    # (resp. column) and are accumulated in 1-D and folded in once at the end.
    # Only extra od offsets are genuinely full-plane work.
    if F64:
        acc = tl.zeros((BH, BW), dtype=tl.float64)
        rt = tl.zeros((BW,), dtype=tl.float64)
        rb = tl.zeros((BW,), dtype=tl.float64)
        cl = tl.zeros((BH,), dtype=tl.float64)
        cr = tl.zeros((BH,), dtype=tl.float64)
    else:
        acc = tl.zeros((BH, BW), dtype=tl.float32)
        rt = tl.zeros((BW,), dtype=tl.float32)
        rb = tl.zeros((BW,), dtype=tl.float32)
        cl = tl.zeros((BH,), dtype=tl.float32)
        cr = tl.zeros((BH,), dtype=tl.float32)

    for td in range(lo_d, hi_d + 1):
        od = d + pf + td
        odv = (od >= 0) & (od < Od) & (tl.minimum(tl.maximum(od - pf, 0), D - 1) == d)
        p_d = base + od * sD

        # th == 0: the whole h tile is active, so w offsets are either the full
        # tile (tw == 0) or a single column (tw != 0).
        oh = h + pt
        hv = (
            (oh >= 0)
            & (oh < Oh)
            & (tl.minimum(tl.maximum(oh - pt, 0), H - 1) == h)
            & hmask
        )
        p_h = p_d + oh * sH
        ow0 = w + pl
        wv0 = (
            (ow0 >= 0)
            & (ow0 < Ow)
            & (tl.minimum(tl.maximum(ow0 - pl, 0), W - 1) == w)
            & wmask
        )
        if W1:
            owv0 = tl.max_contiguous(ow0, BW)
        else:
            owv0 = ow0
        acc += tl.load(
            go_ptr + p_h[:, None] + owv0[None, :] * sW,
            mask=hv[:, None] & wv0[None, :] & odv,
            other=0.0,
        )
        for tw in range(lo_w, 0):
            ows = pl + tw
            wok = (
                (ows >= 0)
                & (ows < Ow)
                & (tl.minimum(tl.maximum(ows - pl, 0), W - 1) == 0)
                & odv
            )
            cl += tl.load(go_ptr + p_h + ows * sW, mask=hv & wok, other=0.0)
        for tw in range(1, hi_w + 1):
            ows = W - 1 + pl + tw
            wok = (
                (ows >= 0)
                & (ows < Ow)
                & (tl.minimum(tl.maximum(ows - pl, 0), W - 1) == W - 1)
                & odv
            )
            cr += tl.load(go_ptr + p_h + ows * sW, mask=hv & wok, other=0.0)

        # th < 0: every offset lands on the h == 0 row.
        for th in range(lo_h, 0):
            ohs = pt + th
            hok = (
                (ohs >= 0)
                & (ohs < Oh)
                & (tl.minimum(tl.maximum(ohs - pt, 0), H - 1) == 0)
                & odv
            )
            p_r = p_d + ohs * sH
            for tw in range(lo_w, hi_w + 1):
                ow = w + pl + tw
                wv = (
                    (ow >= 0)
                    & (ow < Ow)
                    & (tl.minimum(tl.maximum(ow - pl, 0), W - 1) == w)
                    & wmask
                )
                if W1:
                    owv = tl.max_contiguous(ow, BW)
                else:
                    owv = ow
                rt += tl.load(go_ptr + p_r + owv * sW, mask=wv & hok, other=0.0)

        # th > 0: every offset lands on the h == H-1 row.
        for th in range(1, hi_h + 1):
            ohs = H - 1 + pt + th
            hok = (
                (ohs >= 0)
                & (ohs < Oh)
                & (tl.minimum(tl.maximum(ohs - pt, 0), H - 1) == H - 1)
                & odv
            )
            p_r = p_d + ohs * sH
            for tw in range(lo_w, hi_w + 1):
                ow = w + pl + tw
                wv = (
                    (ow >= 0)
                    & (ow < Ow)
                    & (tl.minimum(tl.maximum(ow - pl, 0), W - 1) == w)
                    & wmask
                )
                if W1:
                    owv = tl.max_contiguous(ow, BW)
                else:
                    owv = ow
                rb += tl.load(go_ptr + p_r + owv * sW, mask=wv & hok, other=0.0)

    acc += tl.where((h == 0)[:, None], rt[None, :], 0.0)
    acc += tl.where((h == H - 1)[:, None], rb[None, :], 0.0)
    acc += tl.where((w == 0)[None, :], cl[:, None], 0.0)
    acc += tl.where((w == W - 1)[None, :], cr[:, None], 0.0)

    hw = h[:, None] * W + w[None, :]
    if IDX64:
        o_off = ((nc.to(tl.int64) * D + d) * H) * W + hw
    else:
        o_off = ((nc * D + d) * H) * W + hw
    tl.store(out_ptr + o_off, acc, mask=hmask[:, None] & wmask[None, :])


# dense-grad_output fast path: the strides follow from the shape, so the launch
# carries 9 scalar args instead of 15.  The channel stride is Od*Oh*Ow (the
# grad_output plane count), not D*H*W.
@triton.autotune(
    configs=_TILE_CONFIGS,
    key=["DT", "D", "H", "W", "Od", "Oh", "Ow", "pf", "pt", "pl"],
    warmup=10,
    rep=25,
)
@triton.jit
def _rep_pad3d_bwd_c(
    go_ptr,
    out_ptr,
    DT: tl.constexpr,
    D,
    H,
    W,
    Od,
    Oh,
    Ow,
    pf,
    pt,
    pl,
    F64: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
):
    _rep_pad3d_bwd_body(
        go_ptr,
        out_ptr,
        D,
        H,
        W,
        Od,
        Oh,
        Ow,
        pf,
        pt,
        pl,
        0,
        Od * Oh * Ow,
        Oh * Ow,
        Ow,
        1,
        1,
        True,
        True,
        False,
        F64,
        BH,
        BW,
    )


@triton.autotune(
    configs=_TILE_CONFIGS,
    key=["DT", "W1", "D", "H", "W", "Od", "Oh", "Ow", "pf", "pt", "pl"],
    warmup=10,
    rep=25,
)
@triton.jit
def _rep_pad3d_bwd_g(
    go_ptr,
    out_ptr,
    DT: tl.constexpr,
    D,
    H,
    W,
    Od,
    Oh,
    Ow,
    pf,
    pt,
    pl,
    sN,
    sC,
    sD,
    sH,
    sW,
    C,
    W1: tl.constexpr,
    IDX64: tl.constexpr,
    F64: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
):
    _rep_pad3d_bwd_body(
        go_ptr,
        out_ptr,
        D,
        H,
        W,
        Od,
        Oh,
        Ow,
        pf,
        pt,
        pl,
        sN,
        sC,
        sD,
        sH,
        sW,
        C,
        W1,
        False,
        IDX64,
        F64,
        BH,
        BW,
    )


def replication_pad3d_backward(grad_output, self, padding):
    og = grad_output.shape
    sg = self.shape
    nd = len(og)
    if nd != 4 and nd != 5:
        raise ValueError(
            "replication_pad3d_backward expects a 4D or 5D grad_output, got %dD" % nd
        )
    if len(sg) != nd or og[:-3] != sg[:-3]:
        raise ValueError(
            "grad_output and self must have matching leading dimensions, got %s and %s"
            % (tuple(og[:-3]), tuple(sg[:-3]))
        )
    try:
        pl, pr, pt, pb, pf, pbk = padding
        if not isinstance(pl, int):
            raise TypeError
    except Exception:
        pad = [int(p) for p in padding]
        if len(pad) != 6:
            raise ValueError(
                "replication_pad3d_backward expects 6 padding values, got %d" % len(pad)
            )
        pl, pr, pt, pb, pf, pbk = pad

    D = sg[-3]
    H = sg[-2]
    W = sg[-1]
    Od = og[-3]
    Oh = og[-2]
    Ow = og[-1]

    if nd == 5:
        C = sg[1]
        NC = sg[0] * C
    else:
        C = sg[0]
        NC = C

    dt = grad_output.dtype
    out = torch.empty(sg, dtype=dt, device=grad_output.device)

    if D <= 0 or H <= 0 or W <= 0 or NC <= 0:
        return out

    if dt == torch.float16:
        dt_code = 0
    elif dt == torch.float32:
        dt_code = 1
    elif dt == torch.bfloat16:
        dt_code = 2
    else:
        dt_code = 3
    f64 = dt == torch.float64

    if grad_output.is_contiguous() and grad_output.numel() <= 2147483647:
        grid = lambda META: (
            NC
            * D
            * ((H + META["BH"] - 1) // META["BH"])
            * ((W + META["BW"] - 1) // META["BW"]),
        )
        _rep_pad3d_bwd_c[grid](
            grad_output,
            out,
            dt_code,
            D,
            H,
            W,
            Od,
            Oh,
            Ow,
            pf,
            pt,
            pl,
            F64=f64,
        )
        return out

    sN = grad_output.stride(0) if nd == 5 else 0
    if nd == 5:
        sC, sD, sH, sW = (
            grad_output.stride(1),
            grad_output.stride(2),
            grad_output.stride(3),
            grad_output.stride(4),
        )
    else:
        sC, sD, sH, sW = (
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            grad_output.stride(3),
        )
    grid = lambda META: (
        NC
        * D
        * ((H + META["BH"] - 1) // META["BH"])
        * ((W + META["BW"] - 1) // META["BW"]),
    )
    _rep_pad3d_bwd_g[grid](
        grad_output,
        out,
        dt_code,
        D,
        H,
        W,
        Od,
        Oh,
        Ow,
        pf,
        pt,
        pl,
        sN,
        sC,
        sD,
        sH,
        sW,
        C,
        W1=(sW == 1),
        IDX64=True,
        F64=f64,
    )
    return out
