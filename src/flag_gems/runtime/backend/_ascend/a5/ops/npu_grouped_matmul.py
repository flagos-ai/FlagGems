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

from flag_gems.runtime import torch_device_fn

_MAX_E = 128
_BM = 64
_BN = 128
_BK = 32
_P = 128


@triton.jit
def _fp4_to_e4m3(c):
    """e2m1 nibble -> e4m3 byte (exact: every e2m1 magnitude is representable)."""
    s = (c >> 3) & 1
    e = (c >> 1) & 3
    m = c & 1
    ef = tl.where(e == 0, 6, e + 6)
    mf = tl.where(e == 0, 0, 4 * m)
    v = (s << 7) | (ef << 3) | mf
    return tl.where((c & 7) == 0, s << 7, v).to(tl.uint8)


@triton.jit(
    do_not_specialize=[
        "M",
        "K",
        "N",
        "E",
        "xs0",
        "wse",
        "wsk",
        "wsn",
        "sws0",
        "sws1",
        "sws2",
        "sws3",
        "sxs0",
        "sxs1",
        "sxs2",
        "gltype",
        "NT",
    ]
)
def _gmm_kernel(
    x_ptr,
    w_ptr,
    sx_ptr,
    sw_ptr,
    gl_ptr,
    o_ptr,
    M,
    K,
    N,
    E,
    xs0,
    wse,
    wsk,
    wsn,
    sws0,
    sws1,
    sws2,
    sws3,
    sxs0,
    sxs1,
    sxs2,
    gltype,
    NT,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    ME: tl.constexpr,
    FP4X: tl.constexpr,
    FP4W: tl.constexpr,
    NZW: tl.constexpr,
    P: tl.constexpr,
):
    pid = tl.program_id(0)
    eo = tl.arange(0, ME)
    em = eo < E
    gl = tl.load(gl_ptr + eo, mask=em, other=0).to(tl.int32)
    lower = eo[None, :] < eo[:, None]
    if gltype == 1:
        cnt = gl
    else:
        pv = tl.sum(tl.where(eo[None, :] == (eo[:, None] - 1), gl[None, :], 0), axis=1)
        cnt = gl - pv
    cnt = tl.maximum(tl.where(em, cnt, 0), 0)
    st = tl.minimum(
        tl.maximum(tl.sum(tl.where(lower, cnt[None, :].to(tl.int64), 0), axis=1), 0), M
    ).to(tl.int32)
    en = tl.minimum(st.to(tl.int64) + cnt, M).to(tl.int32)
    cnt = en - st
    mt = (cnt + BM - 1) // BM
    itn = mt * NT
    ipv = tl.sum(tl.where(lower, itn[None, :], 0), axis=1)
    icu = ipv + itn
    total = tl.sum(itn)

    it = pid
    while it < total:
        sel = (it >= ipv) & (it < icu)
        e = tl.sum(tl.where(sel, eo, 0))
        r0 = tl.sum(tl.where(sel, st, 0))
        r1 = tl.sum(tl.where(sel, en, 0))
        mte = tl.sum(tl.where(sel, mt, 0))
        loc = it - tl.sum(tl.where(sel, ipv, 0))
        mti = loc % mte
        nti = loc // mte

        rs = r0 + mti * BM + tl.arange(0, BM)
        rmk = rs < r1
        ns = nti * BN + tl.arange(0, BN)
        nmk = ns < N

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            b = k0 // 32
            sxb = tl.load(
                sx_ptr + rs * sxs0 + (b // 2) * sxs1 + (b % 2) * sxs2,
                mask=rmk,
                other=127,
            )
            sxf = tl.exp2(sxb.to(tl.float32) - 127.0)
            swb = tl.load(
                sw_ptr + e * sws0 + (b // 2) * sws1 + ns * sws2 + (b % 2) * sws3,
                mask=nmk,
                other=127,
            )
            swf = tl.exp2(swb.to(tl.float32) - 127.0)
            if FP4W:
                c = tl.arange(0, BK)
                bc = (c // 32) * 16 + ((c % 32) % 16)
                hi = (c % 32) >= 16
                cb = k0 // 2 + bc
                if FP4X:
                    xb = tl.load(
                        x_ptr + rs[:, None] * xs0 + cb[None, :],
                        mask=rmk[:, None],
                        other=0,
                    )
                    a8 = _fp4_to_e4m3(tl.where(hi[None, :], xb >> 4, xb & 15))
                else:
                    kk = k0 + (c // 32) * 32 + 2 * ((c % 32) % 16) + hi.to(tl.int32)
                    a8 = tl.load(
                        x_ptr + rs[:, None] * xs0 + kk[None, :],
                        mask=rmk[:, None],
                        other=0,
                    )
                if NZW:
                    woff = (
                        e * (K // 2) * N
                        + (cb[:, None] // 16) * (N * 16)
                        + ns[None, :] * 16
                        + (cb[:, None] % 16)
                    )
                else:
                    woff = e * wse + cb[:, None] * wsk + ns[None, :] * wsn
                wb = tl.load(w_ptr + woff, mask=nmk[None, :], other=0)
                b8 = _fp4_to_e4m3(tl.where(hi[:, None], wb >> 4, wb & 15))
            else:
                kc = k0 + tl.arange(0, BK)
                a8 = tl.load(
                    x_ptr + rs[:, None] * xs0 + kc[None, :], mask=rmk[:, None], other=0
                )
                if NZW:
                    woff = (
                        e * K * N
                        + (ns[None, :] // 32) * (K * 32)
                        + kc[:, None] * 32
                        + (ns[None, :] % 32)
                    )
                else:
                    woff = e * wse + kc[:, None] * wsk + ns[None, :] * wsn
                b8 = tl.load(w_ptr + woff, mask=nmk[None, :], other=0)

            a8 = a8.to(tl.float8e4nv, bitcast=True)
            b8 = b8.to(tl.float8e4nv, bitcast=True)
            part = tl.dot(a8, b8)
            acc += part * sxf[:, None] * swf[None, :]

        tl.store(
            o_ptr + rs[:, None] * N + ns[None, :],
            acc.to(tl.bfloat16),
            mask=rmk[:, None] & nmk[None, :],
        )
        it += P


def _as_list(v):
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _first(v):
    lst = _as_list(v)
    return lst[0] if lst else None


def _is_fp4(dt):
    s = str(dt)
    return ("float4" in s) or ("e2m1" in s)


def _u8(t):
    if t is None:
        return None
    if t.dtype == torch.uint8:
        return t
    try:
        return t.view(torch.uint8)
    except Exception:
        return t


def _npu_format(t):
    try:
        import torch_npu

        return int(torch_npu.get_npu_format(t))
    except Exception:
        return 0


def _launch(x, weight, *args, **kwargs):
    kw = kwargs
    scale = kw.get("scale")
    if scale is None:
        scale = kw.get("antiquant_scale")
    pts = kw.get("per_token_scale")
    group_list = kw.get("group_list")
    gltype = kw.get("group_list_type")
    gltype = 0 if gltype is None else int(gltype)

    x0 = _first(x)
    w0 = _first(weight)
    s0 = _first(scale)
    p0 = _first(pts)

    E = int(w0.shape[0])
    ka = int(w0.shape[1])
    N = int(w0.shape[2])
    M = int(x0.shape[0])
    fp4x = _is_fp4(x0.dtype)
    K = 2 * int(x0.shape[1]) if fp4x else int(x0.shape[1])

    dev = x0.device
    out = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    if M == 0 or E == 0 or N == 0:
        return [out]

    fp4w = ka != K
    fmt = _npu_format(w0)
    nzw = fmt in (29, 50)

    xs0 = int(x0.stride(0))
    wse, wsk, wsn = (int(v) for v in w0.stride())
    sws0, sws1, sws2, sws3 = (int(v) for v in s0.stride())
    sxs0, sxs1, sxs2 = (int(v) for v in p0.stride())

    nt = (N + _BN - 1) // _BN
    gl = (
        group_list
        if group_list is not None
        else torch.zeros(E, dtype=torch.int64, device=dev)
    )
    if gl.dtype != torch.int64:
        gl = gl.to(torch.int64)

    _gmm_kernel[(_P,)](
        _u8(x0),
        _u8(w0),
        _u8(p0),
        _u8(s0),
        gl,
        out,
        M,
        K,
        N,
        E,
        xs0,
        wse,
        wsk,
        wsn,
        sws0,
        sws1,
        sws2,
        sws3,
        sxs0,
        sxs1,
        sxs2,
        gltype,
        nt,
        BM=_BM,
        BN=_BN,
        BK=_BK,
        ME=_MAX_E,
        FP4X=fp4x,
        FP4W=fp4w,
        NZW=nzw,
        P=_P,
    )
    return [out]


def npu_grouped_matmul(
    x,
    weight,
    *,
    bias=None,
    scale=None,
    offset=None,
    antiquant_scale=None,
    antiquant_offset=None,
    per_token_scale=None,
    group_list=None,
    activation_input=None,
    activation_quant_scale=None,
    activation_quant_offset=None,
    split_item=0,
    group_type=None,
    group_list_type=0,
    act_type=0,
    tuning_config=None,
    output_dtype=None,
    x_dtype=None,
    weight_dtype=None,
    scale_dtype=None,
    per_token_scale_dtype=None,
):
    """Grouped matrix multiplication with per-block MX scales and BF16 outputs."""
    if (
        not isinstance(x, (tuple, list))
        or len(x) != 1
        or not isinstance(weight, (tuple, list))
        or len(weight) != 1
    ):
        raise ValueError("expected one activation and one expert-weight tensor")
    if (
        bias is not None
        or offset is not None
        or antiquant_offset is not None
        or activation_input is not None
        or activation_quant_scale is not None
        or activation_quant_offset is not None
        or act_type != 0
        or tuning_config is not None
    ):
        raise ValueError("unsupported grouped-matmul epilogue")
    if (
        (scale is None) == (antiquant_scale is None)
        or not isinstance(per_token_scale, (tuple, list))
        or len(per_token_scale) != 1
    ):
        raise ValueError("expected one weight scale and one activation scale list")
    weight_scales = scale if scale is not None else antiquant_scale
    if not isinstance(weight_scales, (tuple, list)) or len(weight_scales) != 1:
        raise ValueError("expected one weight scale")
    a, w, sw, sx = x[0], weight[0], weight_scales[0], per_token_scale[0]
    if (
        a.ndim != 2
        or w.ndim != 3
        or not a.is_contiguous()
        or a.device.type != "npu"
        or not all(t.device == a.device for t in (w, sw, sx))
    ):
        raise ValueError("expected [M,K] activation and [E,K,N] weights on one NPU")
    fp4x = _is_fp4(a.dtype)
    fp4w = w.dtype == torch.uint8
    if (
        not (fp4x or a.dtype == torch.float8_e4m3fn)
        or not (fp4w or w.dtype == torch.float8_e4m3fn)
        or (fp4x and not fp4w)
    ):
        raise ValueError("unsupported activation or weight dtype")
    if (x_dtype not in ((296,) if fp4x else (None, 24))) or (
        weight_dtype not in ((296,) if fp4w else (None, 24))
    ):
        raise ValueError("operand dtype codes must agree with storage")
    m, k = a.shape
    k *= 2 if fp4x else 1
    e, kw, n = w.shape
    if (
        not (
            m > 0
            and n > 0
            and 0 < e <= 128
            and k > 0
            and k % 64 == 0
            and kw == (k // 2 if fp4w else k)
        )
        or max(w.numel(), a.numel(), m * n) >= 2**31
    ):
        raise ValueError(
            "expected block-aligned K, at most 128 experts, and int32-addressable matrices"
        )
    if (
        group_list is None
        or group_list.dtype != torch.int64
        or not group_list.is_contiguous()
        or group_list.device != a.device
        or group_list.shape != (e,)
        or group_list_type not in (0, 1)
    ):
        raise ValueError(
            "expected contiguous int64 group boundaries or counts, one per expert"
        )
    if (
        group_type != 0
        or split_item != 2
        or output_dtype not in (torch.bfloat16, 15)
        or scale_dtype not in (None, 293)
        or per_token_scale_dtype not in (None, 293)
    ):
        raise ValueError(
            "only row grouping, split_item=2, E8M0 scaling and BF16 output are supported"
        )
    if (
        sw.shape != (e, k // 64, n, 2)
        or sx.shape != (m, k // 64, 2)
        or sw.dtype not in (torch.float8_e8m0fnu, torch.uint8)
        or sx.dtype not in (torch.float8_e8m0fnu, torch.uint8)
    ):
        raise ValueError(
            "expected block-32 weight scales [E,K/64,N,2] and activation scales [M,K/64,2]"
        )
    fmt = _npu_format(w)
    if fmt not in (0, 2, 29, 50) or (fmt == 50 and not fp4w) or (fmt == 29 and fp4w):
        raise ValueError("unsupported expert weight format")

    if not fp4x and fp4w and antiquant_scale is None:
        raise ValueError("mixed E4M3/E2M1 weights require antiquant_scale")
    if x[0].device.type != "npu":
        raise ValueError("expected inputs on the same NPU")
    with torch_device_fn.device(x[0].device):
        return _launch(
            x,
            weight,
            bias=bias,
            scale=scale,
            offset=offset,
            antiquant_scale=antiquant_scale,
            antiquant_offset=antiquant_offset,
            per_token_scale=per_token_scale,
            group_list=group_list,
            activation_input=activation_input,
            activation_quant_scale=activation_quant_scale,
            activation_quant_offset=activation_quant_offset,
            split_item=split_item,
            group_type=group_type,
            group_list_type=group_list_type,
            act_type=act_type,
            tuning_config=tuning_config,
            output_dtype=output_dtype,
            x_dtype=x_dtype,
            weight_dtype=weight_dtype,
            scale_dtype=scale_dtype,
            per_token_scale_dtype=per_token_scale_dtype,
        )
