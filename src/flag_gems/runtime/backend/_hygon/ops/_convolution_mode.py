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
def _conv_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    C_out,
    XSN: tl.constexpr,
    XSC: tl.constexpr,
    XST: tl.constexpr,
    XSH: tl.constexpr,
    XSW: tl.constexpr,
    WOC: tl.constexpr,
    WC: tl.constexpr,
    WST: tl.constexpr,
    WSH: tl.constexpr,
    WSW: tl.constexpr,
    COUT_G: tl.constexpr,
    c_in: tl.constexpr,
    OTT: tl.constexpr,
    OTH: tl.constexpr,
    OTW: tl.constexpr,
    XT: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    KT: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    ST: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PT: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    DT: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    G: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_FP32: tl.constexpr,
    KFULL: tl.constexpr,
    NFULL: tl.constexpr,
    NTAPS: tl.constexpr,
    NUM_K: tl.constexpr,
    NUM_N: tl.constexpr,
    UNROLL: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_ng = tl.program_id(2)

    n = pid_ng // G
    g = pid_ng - n * G
    ot = pid_hw // OTH
    oh = pid_hw - ot * OTH

    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    sw = ow < OTW
    owski = ow * SW
    owptr = ow * (SW * XSW)

    kk = tl.arange(0, BLOCK_K)
    kmask = kk < c_in

    x_base = x_ptr + n * XSN + (g * c_in) * XSC
    out_p = OTT * OTH * OTW
    o_off = (ot * OTH + oh) * OTW

    for _j in tl.static_range(NUM_N):
        offs_n = _j * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = offs_n < COUT_G
        oc = g * COUT_G + offs_n
        w_base = w_ptr + oc[None, :] * WOC + kk[:, None] * WC
        acc = tl.zeros((BLOCK_W, BLOCK_N), dtype=tl.float32)
        for tap in tl.range(0, NTAPS, num_stages=2, loop_unroll_factor=UNROLL):
            kt = tap // (KH * KW)
            r = tap - kt * (KH * KW)
            kh = r // KW
            kw = r - kh * KW
            it = ot * ST - PT + kt * DT
            ih = oh * SH - PH + kh * DH
            lo = kw * DW - PW
            a_off = it * XST + ih * XSH + lo * XSW
            w_off = kt * WST + kh * WSH + kw * WSW
            ok = (
                sw
                & (owski >= -lo)
                & (owski < XW - lo)
                & (it >= 0)
                & (it < XT)
                & (ih >= 0)
                & (ih < XH)
            )
            for _k in tl.static_range(NUM_K):
                koff = _k * BLOCK_K
                if KFULL:
                    a = tl.load(
                        x_base + a_off + owptr[:, None] + (koff + kk)[None, :] * XSC,
                        mask=ok[:, None],
                        other=0.0,
                    )
                else:
                    a = tl.load(
                        x_base + a_off + owptr[:, None] + (koff + kk)[None, :] * XSC,
                        mask=ok[:, None] & kmask[None, :],
                        other=0.0,
                    )
                if KFULL & NFULL:
                    b = tl.load(w_base + koff * WC + w_off)
                elif KFULL:
                    b = tl.load(
                        w_base + koff * WC + w_off,
                        mask=nmask[None, :],
                        other=0.0,
                    )
                else:
                    b = tl.load(
                        w_base + koff * WC + w_off,
                        mask=kmask[:, None] & nmask[None, :],
                        other=0.0,
                    )
                if IS_FP32:
                    acc = tl.dot(a, b, acc, input_precision="ieee")
                else:
                    acc = tl.dot(a, b, acc)

        if HAS_BIAS:
            bias = tl.load(b_ptr + oc, mask=nmask, other=0.0)
            acc = acc + bias[None, :]

        res = acc.to(out_ptr.dtype.element_ty)
        out_ptrs = out_ptr + (n * C_out + oc[None, :]) * out_p + o_off + ow[:, None]
        if NFULL:
            tl.store(out_ptrs, res, mask=sw[:, None])
        else:
            tl.store(out_ptrs, res, mask=sw[:, None] & nmask[None, :])


@triton.jit
def _nchw_to_nhwc_kernel(
    x_ptr,
    o_ptr,
    XSN: tl.constexpr,
    XSC: tl.constexpr,
    XST: tl.constexpr,
    XSH: tl.constexpr,
    XSW: tl.constexpr,
    XT: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    C: tl.constexpr,
    NUM_C: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_n = tl.program_id(2)

    t = pid_hw // XH
    h = pid_hw - t * XH
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    offs_c = tl.arange(0, BLOCK_C)
    wmask = offs_w < XW

    src0 = x_ptr + pid_n * XSN + t * XST + h * XSH + offs_w[None, :] * XSW
    dst0 = o_ptr + ((pid_n * XT + t) * XH + h) * XW * C + offs_w[:, None] * C
    for j in tl.static_range(NUM_C):
        c0 = j * BLOCK_C
        cmask = (c0 + offs_c) < C
        v = tl.load(
            src0 + (c0 + offs_c)[:, None] * XSC,
            mask=cmask[:, None] & wmask[None, :],
            other=0.0,
        )
        tl.store(
            dst0 + (c0 + offs_c)[None, :],
            tl.trans(v),
            mask=wmask[:, None] & cmask[None, :],
        )


@triton.jit
def _repack_weight_kernel(
    w_ptr,
    o_ptr,
    C_out,
    c_in,
    WOC: tl.constexpr,
    WC: tl.constexpr,
    WST: tl.constexpr,
    WSH: tl.constexpr,
    WSW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    NUM_C: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    tap = tl.program_id(0)
    pid_oc = tl.program_id(1)

    kt = tap // (KH * KW)
    r = tap - kt * (KH * KW)
    kh = r // KW
    kw = r - kh * KW
    w_off = kt * WST + kh * WSH + kw * WSW

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_c = tl.arange(0, BLOCK_C)
    ocmask = offs_oc < C_out

    src0 = w_ptr + offs_oc[:, None] * WOC + w_off
    dst0 = o_ptr + tap * (C_out * c_in) + offs_oc[:, None] * c_in
    for j in tl.static_range(NUM_C):
        c0 = j * BLOCK_C
        cmask = (c0 + offs_c) < c_in
        v = tl.load(
            src0 + (c0 + offs_c)[None, :] * WC,
            mask=ocmask[:, None] & cmask[None, :],
            other=0.0,
        )
        tl.store(
            dst0 + (c0 + offs_c)[None, :],
            v,
            mask=ocmask[:, None] & cmask[None, :],
        )


def _next_pow2(v):
    p = 16
    while p < v:
        p <<= 1
    return p


def _as_tuple(v, nd):
    if hasattr(v, "__len__"):
        return tuple(int(e) for e in v)
    return (int(v),) * nd


NHWC_MIN_BYTES = 1 << 24


def _convolution_mode(
    input, weight, bias=None, stride=1, padding="valid", dilation=1, groups=1
):
    logger.debug("GEMS_HYGON _CONVOLUTION_MODE")
    x = input
    w = weight
    nd = w.dim() - 2

    if padding == "same":
        if hasattr(stride, "__len__"):
            if any(s != 1 for s in stride):
                raise ValueError(
                    "padding='same' is not supported for strided convolutions"
                )
        elif stride != 1:
            raise ValueError("padding='same' is not supported for strided convolutions")
    elif padding != "valid":
        raise ValueError(
            "Invalid padding string: %r, should be one of {'same', 'valid'}"
            % (padding,)
        )

    st = _as_tuple(stride, nd)
    dl = _as_tuple(dilation, nd)

    N = x.shape[0]
    C_in = x.shape[1]
    C_out = w.shape[0]
    G = int(groups)
    c_in = w.shape[1]
    COUT_G = C_out // G

    mk = tuple(w.shape[2 + i] for i in range(nd))
    xd = tuple(x.shape[2 + i] for i in range(nd))
    if padding == "valid":
        pl = (0,) * nd
        outs = tuple((xd[i] - dl[i] * (mk[i] - 1) - 1) // st[i] + 1 for i in range(nd))
    else:
        pads = tuple(dl[i] * (mk[i] - 1) for i in range(nd))
        pl = tuple(p // 2 for p in pads)
        outs = tuple(
            (xd[i] + pads[i] - dl[i] * (mk[i] - 1) - 1) // st[i] + 1 for i in range(nd)
        )

    for o in outs:
        if o <= 0:
            raise RuntimeError("Kernel size can't be greater than actual input size")

    out = torch.empty((N, C_out) + outs, dtype=x.dtype, device=x.device)

    M = N
    for o in outs:
        M *= o
    if M == 0:
        return out

    xsn = x.stride(0)
    xsc = x.stride(1)
    xst = xsh = xsw = 0
    T_ = H_ = W_ = 1
    KT = KH = KW = 1
    ST_ = SH_ = SW_ = 1
    PT = PH = PW = 0
    DT = DH = DW = 1
    OTT = OTH = OTW = 1
    wst = wsh = wsw = 0
    woc = w.stride(0)
    wc = w.stride(1)

    if nd == 1:
        W_ = xd[0]
        KW = mk[0]
        SW_ = st[0]
        PW = pl[0]
        DW = dl[0]
        OTW = outs[0]
        xsw = x.stride(2)
        wsw = w.stride(2)
    elif nd == 2:
        H_ = xd[0]
        W_ = xd[1]
        KH = mk[0]
        KW = mk[1]
        SH_ = st[0]
        SW_ = st[1]
        PH = pl[0]
        PW = pl[1]
        DH = dl[0]
        DW = dl[1]
        OTH = outs[0]
        OTW = outs[1]
        xsh = x.stride(2)
        xsw = x.stride(3)
        wsh = w.stride(2)
        wsw = w.stride(3)
    else:
        T_ = xd[0]
        H_ = xd[1]
        W_ = xd[2]
        KT = mk[0]
        KH = mk[1]
        KW = mk[2]
        ST_ = st[0]
        SH_ = st[1]
        SW_ = st[2]
        PT = pl[0]
        PH = pl[1]
        PW = pl[2]
        DT = dl[0]
        DH = dl[1]
        DW = dl[2]
        OTT = outs[0]
        OTH = outs[1]
        OTW = outs[2]
        xst = x.stride(2)
        xsh = x.stride(3)
        xsw = x.stride(4)
        wst = w.stride(2)
        wsh = w.stride(3)
        wsw = w.stride(4)

    BLOCK_N = _next_pow2(COUT_G if COUT_G < 64 else 64)
    BLOCK_K = _next_pow2(c_in if c_in < 64 else 64)
    NUM_K = (c_in + BLOCK_K - 1) // BLOCK_K
    NUM_N = (COUT_G + BLOCK_N - 1) // BLOCK_N
    BLOCK_W = _next_pow2(OTW if OTW < 256 else 256)

    esz = x.element_size()
    NTAPS = KT * KH * KW

    # channels-last input plus a tap-major weight when the input is large enough
    # to amortise the two extra device kernels; the conv kernel only sees strides
    if N * C_in * T_ * H_ * W_ * esz >= NHWC_MIN_BYTES:
        xt = torch.empty((N, T_, H_, W_, C_in), dtype=x.dtype, device=x.device)
        BLOCK_C = _next_pow2(C_in if C_in < 64 else 64)
        NC = (C_in + BLOCK_C - 1) // BLOCK_C
        BW_T = 64
        _nchw_to_nhwc_kernel[(triton.cdiv(W_, BW_T), T_ * H_, N)](
            x,
            xt,
            XSN=xsn,
            XSC=xsc,
            XST=xst,
            XSH=xsh,
            XSW=xsw,
            XT=T_,
            XH=H_,
            XW=W_,
            C=C_in,
            NUM_C=NC,
            BLOCK_W=BW_T,
            BLOCK_C=BLOCK_C,
            num_warps=4,
            num_stages=1,
        )
        wt = torch.empty((NTAPS, C_out, c_in), dtype=w.dtype, device=w.device)
        BOC = _next_pow2(C_out if C_out < 64 else 64)
        BKC = _next_pow2(c_in if c_in < 64 else 64)
        NCW = (c_in + BKC - 1) // BKC
        _repack_weight_kernel[(NTAPS, triton.cdiv(C_out, BOC))](
            w,
            wt,
            C_out,
            c_in,
            WOC=woc,
            WC=wc,
            WST=wst,
            WSH=wsh,
            WSW=wsw,
            KH=KH,
            KW=KW,
            NUM_C=NCW,
            BLOCK_OC=BOC,
            BLOCK_C=BKC,
            num_warps=4,
            num_stages=1,
        )
        src = xt
        wsrc = wt
        xsn = T_ * H_ * W_ * C_in
        xsc = 1
        xst = H_ * W_ * C_in
        xsh = W_ * C_in
        xsw = C_in
        woc = c_in
        wc = 1
        wst = KH * KW * C_out * c_in
        wsh = KW * C_out * c_in
        wsw = C_out * c_in
    else:
        src = x
        wsrc = w

    num_stages = 2
    while (
        BLOCK_W > 16
        and (BLOCK_W * BLOCK_K + BLOCK_K * BLOCK_N) * esz * num_stages > 60000
    ):
        BLOCK_W >>= 1
    if (BLOCK_W * BLOCK_K + BLOCK_K * BLOCK_N) * esz * num_stages > 60000:
        num_stages = 1

    if BLOCK_W >= 256:
        num_warps = 8
    elif BLOCK_W >= 32:
        num_warps = 4
    else:
        num_warps = 2

    grid = (triton.cdiv(OTW, BLOCK_W), OTT * OTH, N * G)
    # small grids cannot hide per-tap latency, so fully unroll the tap loop where
    # nonzero padding makes the shifted tap masks worth constant-folding: kh/kw
    # then become constants and the pad offsets and bounds fold. Large grids keep
    # the pipelined dynamic loop; 'valid' keeps it as well (measured faster there).
    grid_ctas = grid[0] * grid[1] * grid[2]
    unroll = NTAPS if (grid_ctas <= 2048 and padding == "same") else 1
    _conv_kernel[grid](
        src,
        wsrc,
        out if bias is None else bias,
        out,
        C_out,
        XSN=xsn,
        XSC=xsc,
        XST=xst,
        XSH=xsh,
        XSW=xsw,
        WOC=woc,
        WC=wc,
        WST=wst,
        WSH=wsh,
        WSW=wsw,
        COUT_G=COUT_G,
        c_in=c_in,
        OTT=OTT,
        OTH=OTH,
        OTW=OTW,
        XT=T_,
        XH=H_,
        XW=W_,
        KT=KT,
        KH=KH,
        KW=KW,
        ST=ST_,
        SH=SH_,
        SW=SW_,
        PT=PT,
        PH=PH,
        PW=PW,
        DT=DT,
        DH=DH,
        DW=DW,
        G=G,
        HAS_BIAS=bias is not None,
        IS_FP32=x.dtype == torch.float32,
        KFULL=(BLOCK_K * NUM_K == c_in),
        NFULL=(BLOCK_N * NUM_N == COUT_G),
        NTAPS=NTAPS,
        NUM_K=NUM_K,
        NUM_N=NUM_N,
        UNROLL=unroll,
        BLOCK_W=BLOCK_W,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
