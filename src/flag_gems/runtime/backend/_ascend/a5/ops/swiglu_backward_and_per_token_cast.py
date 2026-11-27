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


@triton.jit
def _div_ulp(a, b):

    z = a / b
    zi = z.to(tl.int32, bitcast=True)
    zp = (zi - 1).to(tl.float32, bitcast=True)
    zn = (zi + 1).to(tl.float32, bitcast=True)
    r0 = tl.abs(tl.math.fma(z, -b, a))
    rp = tl.abs(tl.math.fma(zp, -b, a))
    rn = tl.abs(tl.math.fma(zn, -b, a))
    tp = rp < r0
    br = tl.where(tp, rp, r0)
    bz = tl.where(tp, zp, z)
    return tl.where(rn < br, zn, bz)


@triton.jit
def _quant_store(
    v,
    sgidx,
    dbase,
    n,
    q8_ptr,
    sf_ptr,
    sf_stride,
    ROUND_SF: tl.constexpr,
    PACKED: tl.constexpr,
    COL: tl.constexpr,
    MASKED: tl.constexpr,
    mrow,
    m2,
):

    amax = tl.max(v.to(tl.int32, bitcast=True) & 0x7FFFFFFF, axis=1)
    amax = tl.maximum(amax.to(tl.float32, bitcast=True), 1e-4)
    dsf = _div_ulp(amax, 448.0)
    if ROUND_SF:
        di = dsf.to(tl.int32, bitcast=True)
        dri = (di + 0x007FFFFF) & 0x7F800000
        dq = dri.to(tl.float32, bitcast=True)
        e = (dri >> 23) & 0xFF
        qsf = ((254 - e) << 23).to(tl.float32, bitcast=True)
        q = v * qsf[:, None]
    else:
        dq = dsf
        qsf = _div_ulp(448.0, amax)
        q = v * qsf[:, None]
        q = tl.minimum(tl.maximum(q, -448.0), 448.0)
    if MASKED:
        tl.store(q8_ptr + dbase, q.to(tl.float8e4nv), mask=m2)
    else:
        tl.store(q8_ptr + dbase, q.to(tl.float8e4nv))
    if PACKED:
        if COL:
            bo = (sgidx // 2) * sf_stride + 2 * n + (sgidx % 2)
        else:
            bo = n * sf_stride + sgidx
        if MASKED:
            tl.store(sf_ptr + bo, e.to(tl.uint8), mask=mrow)
        else:
            tl.store(sf_ptr + bo, e.to(tl.uint8))
    else:
        if COL:
            so = sgidx * sf_stride + n
        else:
            so = n * sf_stride + sgidx
        if MASKED:
            tl.store(sf_ptr + so, dq, mask=mrow)
        else:
            tl.store(sf_ptr + so, dq)


@triton.jit
def _chunk(
    coff,
    n,
    xq_ptr,
    xsf_ptr,
    g_ptr,
    xgrad_ptr,
    q8_ptr,
    sf_ptr,
    out_ptr,
    sf_stride,
    w,
    wacc,
    cv,
    NGL: tl.constexpr,
    CW: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    DO_REC: tl.constexpr,
    ROUND_SF: tl.constexpr,
    PACKED: tl.constexpr,
    COL: tl.constexpr,
    MASKED: tl.constexpr,
):
    H: tl.constexpr = 32 * NGL
    lane = tl.arange(0, 32)
    gidx = coff + tl.arange(0, CW)
    off = n * (2 * H) + (gidx[:, None] * 32 + lane[None, :])
    goff = n * H + (gidx[:, None] * 32 + lane[None, :])
    if MASKED:
        mrow = gidx < NGL
        m2 = mrow[:, None]
        sfg = tl.load(xsf_ptr + (n * (2 * NGL) + gidx), mask=mrow, other=1.0)
        sfu = tl.load(xsf_ptr + (n * (2 * NGL) + NGL + gidx), mask=mrow, other=1.0)
        xq = tl.load(xq_ptr + off, mask=m2, other=0.0)
        yq = tl.load(xq_ptr + (off + H), mask=m2, other=0.0)
        gr = tl.load(g_ptr + goff, mask=m2, other=0.0).to(tl.float32)
    else:
        mrow = gidx
        m2 = gidx
        sfg = tl.load(xsf_ptr + (n * (2 * NGL) + gidx))
        sfu = tl.load(xsf_ptr + (n * (2 * NGL) + NGL + gidx))
        xq = tl.load(xq_ptr + off)
        yq = tl.load(xq_ptr + (off + H))
        gr = tl.load(g_ptr + goff).to(tl.float32)

    xv = xq.to(tl.float32) * sfg[:, None]
    yv = yq.to(tl.float32) * sfu[:, None]
    if HAS_CLAMP:
        xcl = xv > cv
        ycl = tl.abs(yv) > cv
        xv = tl.minimum(xv, cv)
        yv = tl.minimum(tl.maximum(yv, -cv), cv)
    t = 1.0 + tl.exp(-xv)
    s = _div_ulp(1.0, t)
    if HAS_W:
        gw = (gr * w) * s
    else:
        gw = gr * s
    xg = (gw * yv) * ((xv * (1.0 - s)) + 1.0)
    if HAS_CLAMP:
        xg = tl.where(xcl, 0.0, xg)

    if MASKED:
        tl.store(xgrad_ptr + off, xg.to(tl.bfloat16), mask=m2)
    else:
        tl.store(xgrad_ptr + off, xg.to(tl.bfloat16))

    _quant_store(
        xg,
        gidx,
        off,
        n,
        q8_ptr,
        sf_ptr,
        sf_stride,
        ROUND_SF,
        PACKED,
        COL,
        MASKED,
        mrow,
        m2,
    )

    yg = gw * xv
    if HAS_CLAMP:
        yg = tl.where(ycl, 0.0, yg)
    if MASKED:
        tl.store(xgrad_ptr + (off + H), yg.to(tl.bfloat16), mask=m2)
    else:
        tl.store(xgrad_ptr + (off + H), yg.to(tl.bfloat16))

    _quant_store(
        yg,
        gidx + NGL,
        off + H,
        n,
        q8_ptr,
        sf_ptr,
        sf_stride,
        ROUND_SF,
        PACKED,
        COL,
        MASKED,
        mrow,
        m2,
    )

    if HAS_W or DO_REC:
        act = _div_ulp(xv, t) * yv
        if DO_REC:
            av = act * w if HAS_W else act
            if MASKED:
                tl.store(out_ptr + goff, av.to(tl.bfloat16), mask=m2)
            else:
                tl.store(out_ptr + goff, av.to(tl.bfloat16))
        if HAS_W:
            wacc += tl.sum(gr * act)
    return wacc


@triton.jit
def _tok(
    n,
    xq_ptr,
    xsf_ptr,
    g_ptr,
    w_ptr,
    xgrad_ptr,
    q8_ptr,
    sf_ptr,
    out_ptr,
    wg_ptr,
    sf_stride,
    rsf,
    cv,
    NGL: tl.constexpr,
    EXACT: tl.constexpr,
    W0: tl.constexpr,
    O0: tl.constexpr,
    W1: tl.constexpr,
    O1: tl.constexpr,
    W2: tl.constexpr,
    O2: tl.constexpr,
    W3: tl.constexpr,
    O3: tl.constexpr,
    CG: tl.constexpr,
    NCH: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    DO_REC: tl.constexpr,
    ROUND_SF: tl.constexpr,
    PACKED: tl.constexpr,
    COL: tl.constexpr,
):

    w = 1.0
    if HAS_W:
        w = tl.load(w_ptr + n) * rsf
    wacc = tl.zeros((1,), tl.float32)
    if EXACT:
        if W0 > 0:
            wacc = _chunk(
                O0,
                n,
                xq_ptr,
                xsf_ptr,
                g_ptr,
                xgrad_ptr,
                q8_ptr,
                sf_ptr,
                out_ptr,
                sf_stride,
                w,
                wacc,
                cv,
                NGL,
                W0,
                HAS_W,
                HAS_CLAMP,
                DO_REC,
                ROUND_SF,
                PACKED,
                COL,
                False,
            )
        if W1 > 0:
            wacc = _chunk(
                O1,
                n,
                xq_ptr,
                xsf_ptr,
                g_ptr,
                xgrad_ptr,
                q8_ptr,
                sf_ptr,
                out_ptr,
                sf_stride,
                w,
                wacc,
                cv,
                NGL,
                W1,
                HAS_W,
                HAS_CLAMP,
                DO_REC,
                ROUND_SF,
                PACKED,
                COL,
                False,
            )
        if W2 > 0:
            wacc = _chunk(
                O2,
                n,
                xq_ptr,
                xsf_ptr,
                g_ptr,
                xgrad_ptr,
                q8_ptr,
                sf_ptr,
                out_ptr,
                sf_stride,
                w,
                wacc,
                cv,
                NGL,
                W2,
                HAS_W,
                HAS_CLAMP,
                DO_REC,
                ROUND_SF,
                PACKED,
                COL,
                False,
            )
        if W3 > 0:
            wacc = _chunk(
                O3,
                n,
                xq_ptr,
                xsf_ptr,
                g_ptr,
                xgrad_ptr,
                q8_ptr,
                sf_ptr,
                out_ptr,
                sf_stride,
                w,
                wacc,
                cv,
                NGL,
                W3,
                HAS_W,
                HAS_CLAMP,
                DO_REC,
                ROUND_SF,
                PACKED,
                COL,
                False,
            )
    else:
        for c in tl.static_range(0, NCH):
            wacc = _chunk(
                c * CG,
                n,
                xq_ptr,
                xsf_ptr,
                g_ptr,
                xgrad_ptr,
                q8_ptr,
                sf_ptr,
                out_ptr,
                sf_stride,
                w,
                wacc,
                cv,
                NGL,
                CG,
                HAS_W,
                HAS_CLAMP,
                DO_REC,
                ROUND_SF,
                PACKED,
                COL,
                ((c + 1) * CG > NGL),
            )
    if HAS_W:
        tl.store(wg_ptr + n, tl.sum(wacc) * rsf)


@triton.jit
def _swiglu_bwd_cast(
    xq_ptr,
    xsf_ptr,
    g_ptr,
    w_ptr,
    xgrad_ptr,
    q8_ptr,
    sf_ptr,
    out_ptr,
    wg_ptr,
    sf_stride,
    rsf,
    cv,
    nblk,
    NGL: tl.constexpr,
    EXACT: tl.constexpr,
    W0: tl.constexpr,
    O0: tl.constexpr,
    W1: tl.constexpr,
    O1: tl.constexpr,
    W2: tl.constexpr,
    O2: tl.constexpr,
    W3: tl.constexpr,
    O3: tl.constexpr,
    CG: tl.constexpr,
    NCH: tl.constexpr,
    BT: tl.constexpr,
    NPROG: tl.constexpr,
    SERIAL: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    DO_REC: tl.constexpr,
    ROUND_SF: tl.constexpr,
    PACKED: tl.constexpr,
    COL: tl.constexpr,
):

    pid = tl.program_id(0)
    if SERIAL:
        for blk in range(pid, nblk, NPROG):
            for t in tl.static_range(0, BT):
                _tok(
                    blk * BT + t,
                    xq_ptr,
                    xsf_ptr,
                    g_ptr,
                    w_ptr,
                    xgrad_ptr,
                    q8_ptr,
                    sf_ptr,
                    out_ptr,
                    wg_ptr,
                    sf_stride,
                    rsf,
                    cv,
                    NGL,
                    EXACT,
                    W0,
                    O0,
                    W1,
                    O1,
                    W2,
                    O2,
                    W3,
                    O3,
                    CG,
                    NCH,
                    HAS_W,
                    HAS_CLAMP,
                    DO_REC,
                    ROUND_SF,
                    PACKED,
                    COL,
                )
    else:
        for t in tl.static_range(0, BT):
            _tok(
                pid * BT + t,
                xq_ptr,
                xsf_ptr,
                g_ptr,
                w_ptr,
                xgrad_ptr,
                q8_ptr,
                sf_ptr,
                out_ptr,
                wg_ptr,
                sf_stride,
                rsf,
                cv,
                NGL,
                EXACT,
                W0,
                O0,
                W1,
                O1,
                W2,
                O2,
                W3,
                O3,
                CG,
                NCH,
                HAS_W,
                HAS_CLAMP,
                DO_REC,
                ROUND_SF,
                PACKED,
                COL,
            )


def _chunk_plan(ngl):

    widths = []
    rem = ngl
    while rem > 0:
        w = 1 << (rem.bit_length() - 1)
        widths.append(w)
        rem -= w
    if len(widths) <= 4:
        offs = []
        o = 0
        for w in widths:
            offs.append(o)
            o += w
        widths += [0] * (4 - len(widths))
        offs += [0] * (4 - len(offs))
        return True, widths, offs, 0, 0
    cg = 128
    return False, [0, 0, 0, 0], [0, 0, 0, 0], cg, (ngl + cg - 1) // cg


def swiglu_backward_and_per_token_cast(
    x,
    act_x_grad,
    fmt,
    num_per_channels,
    psum_num_tokens_per_expert=None,
    topk_weights=None,
    routed_scaling_factor=None,
    use_tma_aligned_col_major_sf=False,
    round_sf=False,
    use_packed_ue8m0=False,
    clamp_value=None,
    do_recompute=False,
):
    """Compute SwiGLU gradients and group-32 FP8 output gradients."""
    if fmt != "e4m3" or num_per_channels != 32:
        raise ValueError("expected group-32 E4M3 gradients")
    if (
        fmt != "e4m3"
        or num_per_channels != 32
        or not isinstance(x, (tuple, list))
        or len(x) != 2
    ):
        raise ValueError("expected an E4M3 tensor and group-32 input scales")
    if (
        x[0].ndim != 2
        or x[0].shape[1] != 2 * act_x_grad.shape[-1]
        or x[0].shape[0] != act_x_grad.shape[0]
        or act_x_grad.shape[-1] % 32
        or x[1].shape != (x[0].shape[0], x[0].shape[1] // 32)
    ):
        raise ValueError("input, grad and scale shapes must agree")
    xq, xsf = x
    hidden = xq.shape[-1] // 2
    num_tokens = xq.numel() // (2 * hidden)
    device = xq.device
    ngl = hidden // 32
    col = bool(use_tma_aligned_col_major_sf)
    packed = bool(use_packed_ue8m0)

    xgrad = torch.empty((num_tokens, 2 * hidden), dtype=torch.bfloat16, device=device)
    q8 = torch.empty((num_tokens, 2 * hidden), dtype=torch.float8_e4m3fn, device=device)

    if packed:
        if col:
            buf = torch.empty((ngl, num_tokens), dtype=torch.int16, device=device)
            sf_ret = buf.T
            sf_stride = 2 * num_tokens
        else:
            buf = torch.empty((num_tokens, ngl), dtype=torch.int16, device=device)
            sf_ret = buf
            sf_stride = 2 * ngl
        sf_arg = buf.view(torch.uint8)
    else:
        if col:
            buf = torch.empty((2 * ngl, num_tokens), dtype=torch.float32, device=device)
            sf_ret = buf.T
            sf_stride = num_tokens
        else:
            buf = torch.empty((num_tokens, 2 * ngl), dtype=torch.float32, device=device)
            sf_ret = buf
            sf_stride = 2 * ngl
        sf_arg = buf

    has_w = topk_weights is not None
    weight_grad = (
        torch.empty((num_tokens,), dtype=torch.float32, device=device)
        if has_w
        else None
    )
    act_out = (
        torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=device)
        if do_recompute
        else None
    )

    dummy = xgrad

    bt = 2 if (num_tokens % 2 == 0 and ngl <= 64) else 1
    nblk = num_tokens // bt

    serial = True

    nprog = (128 if ngl <= 64 else 64) if nblk >= 128 else nblk
    grid = (nprog,)
    exact, widths, offs, cg, nch = _chunk_plan(ngl)
    _swiglu_bwd_cast[grid](
        xq,
        xsf.view(-1),
        act_x_grad.view(-1),
        topk_weights if has_w else dummy,
        xgrad,
        q8,
        sf_arg,
        act_out if do_recompute else dummy,
        weight_grad if has_w else dummy,
        sf_stride,
        float(routed_scaling_factor) if routed_scaling_factor is not None else 1.0,
        float(clamp_value) if clamp_value is not None else 0.0,
        nblk,
        NGL=ngl,
        EXACT=exact,
        W0=widths[0],
        O0=offs[0],
        W1=widths[1],
        O1=offs[1],
        W2=widths[2],
        O2=offs[2],
        W3=widths[3],
        O3=offs[3],
        CG=cg,
        NCH=nch,
        BT=bt,
        NPROG=nprog,
        SERIAL=serial,
        HAS_W=has_w,
        HAS_CLAMP=clamp_value is not None,
        DO_REC=bool(do_recompute),
        ROUND_SF=bool(round_sf),
        PACKED=packed,
        COL=col,
    )

    result = [xgrad, (q8, sf_ret)]
    if has_w:
        result.append(weight_grad)
    if do_recompute:
        result.append(act_out)
    return tuple(result)
