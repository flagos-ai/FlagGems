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


def _ru(x, m):
    return (x + m - 1) // m * m


def _pair(v):
    if isinstance(v, (list, tuple)):
        if len(v) == 1:
            return int(v[0]), int(v[0])
        return int(v[0]), int(v[1])
    return int(v), int(v)


# ------------------------------------------------------------------
# 1. Generic pure-Triton fallback kernels (handles ANY shape, stride,
#    dilation, padding, output_padding, groups, small/odd dims).
# ------------------------------------------------------------------
@triton.jit
def _gen_fwd2d_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    N,
    G,
    Cin_g,
    Cout_g,
    H,
    W,
    Hout,
    Wout,
    KH,
    KW,
    sh,
    sw,
    ph,
    pw,
    dh,
    dw,
    sx_n,
    sx_c,
    sx_h,
    sx_w,
    sw_co,
    sw_ci,
    sw_kh,
    sw_kw,
    BLOCK_W: tl.constexpr,
):
    row = tl.program_id(0)
    wb = tl.program_id(1)
    oh = row % Hout
    t = row // Hout
    co_g = t % Cout_g
    t = t // Cout_g
    g = t % G
    n = t // G

    co_abs = g * Cout_g + co_g
    ci_base = g * Cin_g
    ow = wb * BLOCK_W + tl.arange(0, BLOCK_W)
    w_ok_row = ow < Wout
    kk = KH * KW
    n_steps = Cin_g * kk
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    for step in range(0, n_steps):
        ci = step // kk
        rem = step % kk
        kh = rem // KW
        kw = rem % KW
        ih = oh * sh + kh * dh - ph
        if (ih >= 0) & (ih < H):
            iw = ow * sw + kw * dw - pw
            v_ok = w_ok_row & (iw >= 0) & (iw < W)
            iw_s = tl.maximum(tl.minimum(iw, W - 1), 0)
            x_off = n * sx_n + (ci_base + ci) * sx_c + ih * sx_h + iw_s * sx_w
            w_off = co_abs * sw_co + ci * sw_ci + kh * sw_kh + kw * sw_kw
            xv = tl.load(x_ptr + x_off).to(tl.float32)
            wv = tl.load(w_ptr + w_off).to(tl.float32)
            acc += xv * wv * v_ok.to(tl.float32)

    tl.store(y_ptr + row * Wout + ow, acc, mask=w_ok_row)


@triton.jit
def _gen_tr2d_kernel(
    z_ptr,
    w_ptr,
    y_ptr,
    N,
    G,
    Cz_g,
    Co_g,
    Hz,
    Wz,
    Ho,
    Wo,
    KH,
    KW,
    sh,
    sw,
    ph,
    pw,
    dh,
    dw,
    sz_n,
    sz_c,
    sz_h,
    sz_w,
    sw_cz,
    sw_co,
    sw_kh,
    sw_kw,
    BLOCK_W: tl.constexpr,
):
    row = tl.program_id(0)
    wb = tl.program_id(1)
    oh = row % Ho
    t = row // Ho
    co_g = t % Co_g
    t = t // Co_g
    g = t % G
    n = t // G

    cz_base = g * Cz_g
    ow = wb * BLOCK_W + tl.arange(0, BLOCK_W)
    w_ok_row = ow < Wo
    kk = KH * KW
    n_steps = Cz_g * kk
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    for step in range(0, n_steps):
        cz = step // kk
        rem = step % kk
        kh = rem // KW
        kw = rem % KW
        nh = oh + ph - kh * dh
        if (nh >= 0) & ((nh % sh) == 0) & (nh < Hz * sh):
            hz = nh // sh
            nw = ow + pw - kw * dw
            vw = w_ok_row & (nw >= 0) & ((nw % sw) == 0) & (nw < Wz * sw)
            wz = tl.maximum(tl.minimum(nw // sw, Wz - 1), 0)
            z_off = n * sz_n + (cz_base + cz) * sz_c + hz * sz_h + wz * sz_w
            w_off = (cz_base + cz) * sw_cz + co_g * sw_co + kh * sw_kh + kw * sw_kw
            zv = tl.load(z_ptr + z_off).to(tl.float32)
            wv = tl.load(w_ptr + w_off).to(tl.float32)
            acc += zv * wv * vw.to(tl.float32)

    tl.store(y_ptr + row * Wo + ow, acc, mask=w_ok_row)


@triton.jit
def _gen_gw2d_kernel(
    a_ptr,
    b_ptr,
    gw_ptr,
    N,
    G,
    Cin_g,
    Cout_g,
    H,
    W,
    Hout,
    Wout,
    KH,
    KW,
    sh,
    sw,
    ph,
    pw,
    dh,
    dw,
    sa_n,
    sa_c,
    sa_h,
    sa_w,
    sb_n,
    sb_c,
    sb_h,
    sb_w,
    BLOCK_W: tl.constexpr,
):
    idx = tl.program_id(0)
    kw = idx % KW
    t = idx // KW
    kh = t % KH
    t = t // KH
    ci = t % Cin_g
    t = t // Cin_g
    co_g = t % Cout_g
    g = t // Cout_g

    co_abs = g * Cout_g + co_g
    ci_abs = g * Cin_g + ci
    w_chunks = (Wout + BLOCK_W - 1) // BLOCK_W
    n_steps = N * Hout * w_chunks
    rk = tl.arange(0, BLOCK_W)
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    for step in range(0, n_steps):
        wb = step % w_chunks
        r = step // w_chunks
        n = r // Hout
        oh = r % Hout
        ih = oh * sh + kh * dh - ph
        if (ih >= 0) & (ih < H):
            ow = wb * BLOCK_W + rk
            ow_ok = ow < Wout
            ow_s = tl.minimum(ow, Wout - 1)
            iw = ow * sw + kw * dw - pw
            v_ok = ow_ok & (iw >= 0) & (iw < W)
            iw_s = tl.maximum(tl.minimum(iw, W - 1), 0)
            a_off = n * sa_n + ci_abs * sa_c + ih * sa_h + iw_s * sa_w
            b_off = n * sb_n + co_abs * sb_c + oh * sb_h + ow_s * sb_w
            av = tl.load(a_ptr + a_off).to(tl.float32)
            bv = tl.load(b_ptr + b_off).to(tl.float32)
            acc += av * bv * v_ok.to(tl.float32)

    tl.store(gw_ptr + idx, tl.sum(acc, axis=0))


@triton.jit
def _gen_add_bcast_bias_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    out_ptr,
    N,
    C,
    HW,
    HAS_A: tl.constexpr,
    HAS_B: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    tot = N * C * HW
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    ok = idx < tot
    safe = tl.minimum(idx, tot - 1)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    if HAS_A:
        acc += tl.load(a_ptr + safe).to(tl.float32)
    if HAS_B:
        acc += tl.load(b_ptr + safe).to(tl.float32)
    if HAS_BIAS:
        c = (safe // HW) % C
        acc += tl.load(bias_ptr + c).to(tl.float32)
    tl.store(out_ptr + idx, acc, mask=ok)


def _gen_fwd2d(x, w, sh, sw, ph, pw, dh, dw, G):
    N, Cin, H, W = x.shape
    Cout, Cin_g, KH, KW = w.shape
    Cout_g = Cout // G
    Hout = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    Wout = (W + 2 * pw - dw * (KW - 1) - 1) // sw + 1
    y = torch.empty((N, Cout, Hout, Wout), device=x.device, dtype=torch.float32)
    if y.numel() == 0:
        return y
    bw = 64 if Wout >= 32 else 16
    _gen_fwd2d_kernel[(N * Cout * Hout, triton.cdiv(Wout, bw))](
        x,
        w,
        y,
        N,
        G,
        Cin_g,
        Cout_g,
        H,
        W,
        Hout,
        Wout,
        KH,
        KW,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        *x.stride(),
        *w.stride(),
        bw,
    )
    return y


def _gen_tr2d(z, w, sh, sw, ph, pw, dh, dw, oph, opw, G):
    N, Cz, Hz, Wz = z.shape
    Cz_g = Cz // G
    Co_g, KH, KW = w.shape[1], w.shape[2], w.shape[3]
    Ho = (Hz - 1) * sh - 2 * ph + dh * (KH - 1) + 1 + oph
    Wo = (Wz - 1) * sw - 2 * pw + dw * (KW - 1) + 1 + opw
    y = torch.empty((N, G * Co_g, Ho, Wo), device=z.device, dtype=torch.float32)
    if y.numel() == 0:
        return y
    bw = 64 if Wo >= 32 else 16
    _gen_tr2d_kernel[(N * G * Co_g * Ho, triton.cdiv(Wo, bw))](
        z,
        w,
        y,
        N,
        G,
        Cz_g,
        Co_g,
        Hz,
        Wz,
        Ho,
        Wo,
        KH,
        KW,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        *z.stride(),
        *w.stride(),
        bw,
    )
    return y


def _gen_gw2d(a, b, wshape, sh, sw, ph, pw, dh, dw, G):
    Cout, Cin_g, KH, KW = wshape
    Cout_g = Cout // G
    N, _, H, W = a.shape
    Hout, Wout = b.shape[2], b.shape[3]
    gw = torch.empty(wshape, device=a.device, dtype=torch.float32)
    tot = gw.numel()
    if tot == 0:
        return gw
    bw = 64 if Wout >= 32 else 16
    _gen_gw2d_kernel[(tot,)](
        a,
        b,
        gw,
        N,
        G,
        Cin_g,
        Cout_g,
        H,
        W,
        Hout,
        Wout,
        KH,
        KW,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        *a.stride(),
        *b.stride(),
        bw,
    )
    return gw


def _gen_combine_ggo(a, b, ggb, N, C, H, W, dtype, device):
    out = torch.empty((N, C, H, W), device=device, dtype=torch.float32)
    tot = out.numel()
    if tot == 0:
        return out.to(dtype)
    dummy = a if a is not None else (b if b is not None else ggb)
    _gen_add_bcast_bias_kernel[(triton.cdiv(tot, 256),)](
        a if a is not None else dummy,
        b if b is not None else dummy,
        ggb if ggb is not None else dummy,
        out,
        N,
        C,
        H * W,
        a is not None,
        b is not None,
        ggb is not None,
        256,
    )
    return out.to(dtype)


# ------------------------------------------------------------------
# 2. Fast-path pure-Triton kernels (direct implicit tl.dot +
#    2D DMA pad/slice + bitcast parity deinterleave/interleave).
# ------------------------------------------------------------------
@triton.jit
def _pad1d_2d_kernel(
    x_ptr, xp_ptr, L, Lp, pl, NC_BLK: tl.constexpr, L_BLK: tl.constexpr
):
    nb = tl.program_id(0)
    lb = tl.program_id(1)
    r = nb * NC_BLK + tl.arange(0, NC_BLK)
    c = tl.minimum(lb * L_BLK, L - L_BLK) + tl.arange(0, L_BLK)
    v = tl.load(x_ptr + r[:, None] * L + c[None, :])
    tl.store(xp_ptr + r[:, None] * Lp + (c + pl)[None, :], v)


@triton.jit
def _slice1d_2d_kernel(
    x_ptr, y_ptr, L_in, Lo, NC_BLK: tl.constexpr, L_BLK: tl.constexpr
):
    nb = tl.program_id(0)
    lb = tl.program_id(1)
    r = nb * NC_BLK + tl.arange(0, NC_BLK)
    c = tl.minimum(lb * L_BLK, Lo - L_BLK) + tl.arange(0, L_BLK)
    v = tl.load(x_ptr + r[:, None] * L_in + c[None, :])
    tl.store(y_ptr + r[:, None] * Lo + c[None, :], v)


@triton.jit
def _pad2d_tile_kernel(
    x_ptr,
    xp_ptr,
    H,
    W,
    Hp,
    Wp,
    ph,
    pw,
    R_BLK: tl.constexpr,
    W_BLK: tl.constexpr,
):
    nc = tl.program_id(0)
    hb = tl.program_id(1)
    wb = tl.program_id(2)
    r = tl.minimum(hb * R_BLK, H - R_BLK) + tl.arange(0, R_BLK)
    c = tl.minimum(wb * W_BLK, W - W_BLK) + tl.arange(0, W_BLK)
    v = tl.load(x_ptr + (nc * H + r[:, None]) * W + c[None, :])
    tl.store(xp_ptr + (nc * Hp + (r + ph)[:, None]) * Wp + (c + pw)[None, :], v)


@triton.jit
def _slice2d_tile_kernel(
    x_ptr,
    y_ptr,
    Hin,
    Win,
    Ho,
    Wo,
    R_BLK: tl.constexpr,
    W_BLK: tl.constexpr,
):
    nc = tl.program_id(0)
    hb = tl.program_id(1)
    wb = tl.program_id(2)
    r = tl.minimum(hb * R_BLK, Ho - R_BLK) + tl.arange(0, R_BLK)
    c = tl.minimum(wb * W_BLK, Wo - W_BLK) + tl.arange(0, W_BLK)
    v = tl.load(x_ptr + (nc * Hin + r[:, None]) * Win + c[None, :])
    tl.store(y_ptr + (nc * Ho + r[:, None]) * Wo + c[None, :], v)


@triton.jit
def _stack_2rows_kernel(
    er_ptr, or_ptr, out_ptr, Wfull, R_BLK: tl.constexpr, W_BLK: tl.constexpr
):
    rb = tl.program_id(0)
    wb = tl.program_id(1)
    r = rb * R_BLK + tl.arange(0, R_BLK)
    c = wb * W_BLK + tl.arange(0, W_BLK)
    ve = tl.load(er_ptr + r[:, None] * Wfull + c[None, :])
    vo = tl.load(or_ptr + r[:, None] * Wfull + c[None, :])
    tl.store(out_ptr + r[:, None] * (2 * Wfull) + c[None, :], ve)
    tl.store(out_ptr + r[:, None] * (2 * Wfull) + (c + Wfull)[None, :], vo)


@triton.jit
def _deint_i64_flat_kernel(x_i64_ptr, dst_i32_ptr, HALF, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    f = pid * BLOCK + tl.arange(0, BLOCK)
    w = tl.load(x_i64_ptr + f)
    tl.store(dst_i32_ptr + f, (w & 0xFFFFFFFF).to(tl.int32))
    tl.store(dst_i32_ptr + HALF + f, ((w >> 32) & 0xFFFFFFFF).to(tl.int32))


@triton.jit
def _deint_u32_flat_kernel(x_i32_ptr, dst_i16_ptr, HALF, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    f = pid * BLOCK + tl.arange(0, BLOCK)
    w = tl.load(x_i32_ptr + f)
    tl.store(dst_i16_ptr + f, (w & 0xFFFF).to(tl.int16))
    tl.store(dst_i16_ptr + HALF + f, ((w >> 16) & 0xFFFF).to(tl.int16))


@triton.jit
def _interleave_i64_mless_kernel(e_ptr, o_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    f = pid * BLOCK + tl.arange(0, BLOCK)
    lo = tl.load(e_ptr + f).to(tl.uint32).to(tl.uint64)
    hi = tl.load(o_ptr + f).to(tl.uint32).to(tl.uint64)
    tl.store(out_ptr + f, (lo | (hi << 32)).to(tl.int64))


@triton.jit
def _interleave_u32_mless_kernel(e_ptr, o_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    f = pid * BLOCK + tl.arange(0, BLOCK)
    lo = tl.load(e_ptr + f).to(tl.uint16).to(tl.uint32)
    hi = tl.load(o_ptr + f).to(tl.uint16).to(tl.uint32)
    tl.store(out_ptr + f, (lo | (hi << 16)).to(tl.int32))


@triton.jit
def _pack_w_fwd_kernel(w_ptr, out_ptr, G, Co_g, Ci_g, KK, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tot = G * KK * Co_g * Ci_g
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    ok = idx < tot
    safe = tl.minimum(idx, tot - 1)
    ci = safe % Ci_g
    t = safe // Ci_g
    co = t % Co_g
    t = t // Co_g
    ks = t % KK
    g = t // KK
    src = ((g * Co_g + co) * Ci_g + ci) * KK + ks
    tl.store(out_ptr + idx, tl.load(w_ptr + src), mask=ok)


@triton.jit
def _pack_w_flip_t_kernel(w_ptr, out_ptr, G, Co_g, Ci_g, KK, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tot = G * KK * Ci_g * Co_g
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    ok = idx < tot
    safe = tl.minimum(idx, tot - 1)
    co = safe % Co_g
    t = safe // Co_g
    ci = t % Ci_g
    t = t // Ci_g
    ks = t % KK
    g = t // KK
    src = ((g * Co_g + co) * Ci_g + ci) * KK + (KK - 1 - ks)
    tl.store(out_ptr + idx, tl.load(w_ptr + src), mask=ok)


@triton.jit
def _pack_w_sub_tr2d_kernel(
    w_ptr,
    out_ptr,
    G,
    Cz_g,
    Co_g,
    KH,
    KW,
    kh_start,
    kh_step,
    n_kh,
    kw_start,
    kw_step,
    n_kw,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    kk_sub = n_kh * n_kw
    tot = G * kk_sub * Co_g * Cz_g
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    ok = idx < tot
    safe = tl.minimum(idx, tot - 1)
    cz = safe % Cz_g
    t = safe // Cz_g
    co = t % Co_g
    t = t // Co_g
    ks = t % kk_sub
    g = t // kk_sub
    okh = ks // n_kw
    okw = ks % n_kw
    kh = kh_start + (n_kh - 1 - okh) * kh_step
    kw = kw_start + (n_kw - 1 - okw) * kw_step
    src = (((g * Cz_g + cz) * Co_g + co) * KH + kh) * KW + kw
    tl.store(out_ptr + idx, tl.load(w_ptr + src), mask=ok)


@triton.jit
def _unpack_gw_fp32_kernel(sm_ptr, gw_ptr, G, KK, Co_g, Ci_g, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tot = G * Co_g * Ci_g * KK
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    ok = idx < tot
    safe = tl.minimum(idx, tot - 1)
    ks = safe % KK
    t = safe // KK
    ci = t % Ci_g
    t = t // Ci_g
    co = t % Co_g
    g = t // Co_g
    src = ((g * KK + ks) * Co_g + co) * Ci_g + ci
    tl.store(gw_ptr + idx, tl.load(sm_ptr + src), mask=ok)


@triton.jit
def _splitk_sum_flat_kernel(
    part_ptr, out_ptr, stride_sp, SPLIT: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for sp in range(SPLIT):
        acc += tl.load(part_ptr + sp * stride_sp + idx)
    tl.store(out_ptr + idx, acc)


@triton.jit
def _fwd1d_direct_parity_kernel(
    xp1_ptr,
    w1_ptr,
    xp2_ptr,
    w2_ptr,
    bias_ptr,
    y_ptr,
    N,
    G,
    Cin_g,
    Cout_g,
    Lp,
    Lout_pad,
    KL,
    sl,
    dl,
    HAS_1: tl.constexpr,
    HAS_2: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid_mg = tl.program_id(0)
    pid_p = tl.program_id(1)
    m_per_g = Cout_g // BM
    g = pid_mg // m_per_g
    pm = pid_mg % m_per_g
    l_blks = Lout_pad // BN
    n = pid_p // l_blks
    ol0 = (pid_p % l_blks) * BN
    rm = pm * BM + tl.arange(0, BM)
    rn = ol0 + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    plane_stride = N * G * Cin_g * Lp
    nc_base = (n * G + g) * Cin_g
    w_g_base = g * KL * Cout_g * Cin_g
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kl in range(0, KL):
        kd = kl * dl
        par = kd % sl
        off = kd // sl
        par_base = par * plane_stride
        for ci0 in range(0, Cin_g, BK):
            ci = ci0 + rk
            if HAS_1:
                w_t = tl.load(
                    w1_ptr
                    + w_g_base
                    + (kl * Cout_g + rm[:, None]) * Cin_g
                    + ci[None, :]
                )
                x_t = tl.load(
                    xp1_ptr
                    + par_base
                    + (nc_base + ci[:, None]) * Lp
                    + (rn + off)[None, :]
                )
                if IEEE:
                    acc += tl.dot(
                        w_t, x_t, input_precision="ieee", out_dtype=tl.float32
                    )
                else:
                    acc += tl.dot(w_t, x_t, out_dtype=tl.float32)
            if HAS_2:
                w_t = tl.load(
                    w2_ptr
                    + w_g_base
                    + (kl * Cout_g + rm[:, None]) * Cin_g
                    + ci[None, :]
                )
                x_t = tl.load(
                    xp2_ptr
                    + par_base
                    + (nc_base + ci[:, None]) * Lp
                    + (rn + off)[None, :]
                )
                if IEEE:
                    acc += tl.dot(
                        w_t, x_t, input_precision="ieee", out_dtype=tl.float32
                    )
                else:
                    acc += tl.dot(w_t, x_t, out_dtype=tl.float32)
    if HAS_BIAS:
        bv = tl.load(bias_ptr + g * Cout_g + rm).to(tl.float32)
        acc += bv[:, None]
    tl.store(
        y_ptr + ((n * G + g) * Cout_g + rm[:, None]) * Lout_pad + rn[None, :],
        acc.to(y_ptr.dtype.element_ty),
    )


@triton.jit
def _gw1d_direct_parity_kernel(
    dy_ptr,
    xp_ptr,
    part_ptr,
    stride_sp,
    N,
    G,
    Cin_g,
    Cout_g,
    Lp,
    Lout_pad,
    KL,
    sl,
    dl,
    SPLIT_N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid_mg = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_z = tl.program_id(2)
    m_per_g = Cout_g // BM
    g = pid_mg // m_per_g
    pm = pid_mg % m_per_g
    kl = pid_z // SPLIT_N
    sp = pid_z % SPLIT_N
    rm = pm * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    kd = kl * dl
    par = kd % sl
    off = kd // sl
    par_base = par * N * G * Cin_g * Lp
    n_chunks = Lout_pad // BK
    total = N * n_chunks
    chunk = (total + SPLIT_N - 1) // SPLIT_N
    s_start = sp * chunk
    s_end = tl.minimum(total, s_start + chunk)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for s in range(s_start, s_end):
        n = s // n_chunks
        c = s % n_chunks
        ol = c * BK + rk
        a = tl.load(
            dy_ptr + ((n * G + g) * Cout_g + rm[:, None]) * Lout_pad + ol[None, :]
        )
        b = tl.load(
            xp_ptr
            + par_base
            + ((n * G + g) * Cin_g + rn[:, None]) * Lp
            + (ol + off)[None, :]
        )
        if IEEE:
            acc += tl.dot(a, tl.trans(b), input_precision="ieee", out_dtype=tl.float32)
        else:
            acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32)
    tl.store(
        part_ptr
        + sp * stride_sp
        + ((g * KL + kl) * Cout_g + rm[:, None]) * Cin_g
        + rn[None, :],
        acc,
    )


@triton.jit
def _fwd2d_direct_parity_kernel(
    xp1_ptr,
    w1_ptr,
    xp2_ptr,
    w2_ptr,
    bias_ptr,
    y_ptr,
    N,
    G,
    Cin_g,
    Cout_g,
    Hout,
    Wout_pad,
    Hp,
    Wp,
    KH,
    KW,
    sh,
    sw,
    dh,
    dw,
    HAS_1: tl.constexpr,
    HAS_2: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid_mg = tl.program_id(0)
    pid_p = tl.program_id(1)
    m_per_g = Cout_g // BM
    g = pid_mg // m_per_g
    pm = pid_mg % m_per_g
    w_blks = Wout_pad // BN
    wb = pid_p % w_blks
    r = pid_p // w_blks
    n = r // Hout
    oh = r % Hout
    rm = pm * BM + tl.arange(0, BM)
    rn = wb * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    HWp = Hp * Wp
    plane_stride = N * G * Cin_g * HWp
    nc_base = (n * G + g) * Cin_g
    kk = KH * KW
    w_g_base = g * kk * Cout_g * Cin_g
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for ks in range(0, kk):
        kh = ks // KW
        kw = ks % KW
        ih = oh * sh + kh * dh
        kdw = kw * dw
        par = kdw % sw
        w_off = kdw // sw
        par_base = par * plane_stride
        for ci0 in range(0, Cin_g, BK):
            ci = ci0 + rk
            if HAS_1:
                w_t = tl.load(
                    w1_ptr
                    + w_g_base
                    + (ks * Cout_g + rm[:, None]) * Cin_g
                    + ci[None, :]
                )
                x_t = tl.load(
                    xp1_ptr
                    + par_base
                    + (nc_base + ci[:, None]) * HWp
                    + ih * Wp
                    + (rn + w_off)[None, :]
                )
                if IEEE:
                    acc += tl.dot(
                        w_t, x_t, input_precision="ieee", out_dtype=tl.float32
                    )
                else:
                    acc += tl.dot(w_t, x_t, out_dtype=tl.float32)
            if HAS_2:
                w_t = tl.load(
                    w2_ptr
                    + w_g_base
                    + (ks * Cout_g + rm[:, None]) * Cin_g
                    + ci[None, :]
                )
                x_t = tl.load(
                    xp2_ptr
                    + par_base
                    + (nc_base + ci[:, None]) * HWp
                    + ih * Wp
                    + (rn + w_off)[None, :]
                )
                if IEEE:
                    acc += tl.dot(
                        w_t, x_t, input_precision="ieee", out_dtype=tl.float32
                    )
                else:
                    acc += tl.dot(w_t, x_t, out_dtype=tl.float32)
    if HAS_BIAS:
        bv = tl.load(bias_ptr + g * Cout_g + rm).to(tl.float32)
        acc += bv[:, None]
    tl.store(
        y_ptr
        + ((n * G + g) * Cout_g + rm[:, None]) * (Hout * Wout_pad)
        + oh * Wout_pad
        + rn[None, :],
        acc.to(y_ptr.dtype.element_ty),
    )


@triton.jit
def _gw2d_direct_parity_kernel(
    dy_ptr,
    xp_ptr,
    part_ptr,
    stride_sp,
    N,
    G,
    Cin_g,
    Cout_g,
    Hout,
    Wout_pad,
    Hp,
    Wp,
    KH,
    KW,
    sh,
    sw,
    dh,
    dw,
    SPLIT_R: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid_mg = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_z = tl.program_id(2)
    m_per_g = Cout_g // BM
    g = pid_mg // m_per_g
    pm = pid_mg % m_per_g
    kk = KH * KW
    ks = pid_z // SPLIT_R
    sp = pid_z % SPLIT_R
    kh = ks // KW
    kw = ks % KW
    rm = pm * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    kdw = kw * dw
    par = kdw % sw
    w_off = kdw // sw
    HWp = Hp * Wp
    par_base = par * N * G * Cin_g * HWp
    HWout = Hout * Wout_pad
    w_chunks = Wout_pad // BK
    total = N * Hout * w_chunks
    chunk = (total + SPLIT_R - 1) // SPLIT_R
    s_start = sp * chunk
    s_end = tl.minimum(total, s_start + chunk)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for s in range(s_start, s_end):
        wb = s % w_chunks
        r = s // w_chunks
        n = r // Hout
        oh = r % Hout
        ih = oh * sh + kh * dh
        ow = wb * BK + rk
        a = tl.load(
            dy_ptr
            + ((n * G + g) * Cout_g + rm[:, None]) * HWout
            + oh * Wout_pad
            + ow[None, :]
        )
        b = tl.load(
            xp_ptr
            + par_base
            + ((n * G + g) * Cin_g + rn[:, None]) * HWp
            + ih * Wp
            + (ow + w_off)[None, :]
        )
        if IEEE:
            acc += tl.dot(a, tl.trans(b), input_precision="ieee", out_dtype=tl.float32)
        else:
            acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32)
    tl.store(
        part_ptr
        + sp * stride_sp
        + ((g * kk + ks) * Cout_g + rm[:, None]) * Cin_g
        + rn[None, :],
        acc,
    )


@triton.jit
def _gw2d_s1_k3_all9_kernel(
    dy_ptr,
    xp_ptr,
    part_ptr,
    stride_sp,
    N,
    G,
    Cin_g,
    Cout_g,
    Hout,
    Wout_pad,
    Hp,
    Wp,
    sh,
    dh,
    dw,
    SPLIT_R: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid_mg = tl.program_id(0)
    pid_n = tl.program_id(1)
    sp = tl.program_id(2)
    m_per_g = Cout_g // BM
    g = pid_mg // m_per_g
    pm = pid_mg % m_per_g
    rm = pm * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    w_chunks = Wout_pad // BK
    total = N * Hout * w_chunks
    chunk = (total + SPLIT_R - 1) // SPLIT_R
    s_start = sp * chunk
    s_end = tl.minimum(total, s_start + chunk)
    HWout = Hout * Wout_pad
    HWp = Hp * Wp
    d01 = dw
    d02 = 2 * dw
    d10 = dh * Wp
    d11 = dh * Wp + dw
    d12 = dh * Wp + 2 * dw
    d20 = 2 * dh * Wp
    d21 = 2 * dh * Wp + dw
    d22 = 2 * dh * Wp + 2 * dw
    acc00 = tl.zeros((BM, BN), dtype=tl.float32)
    acc01 = tl.zeros((BM, BN), dtype=tl.float32)
    acc02 = tl.zeros((BM, BN), dtype=tl.float32)
    acc10 = tl.zeros((BM, BN), dtype=tl.float32)
    acc11 = tl.zeros((BM, BN), dtype=tl.float32)
    acc12 = tl.zeros((BM, BN), dtype=tl.float32)
    acc20 = tl.zeros((BM, BN), dtype=tl.float32)
    acc21 = tl.zeros((BM, BN), dtype=tl.float32)
    acc22 = tl.zeros((BM, BN), dtype=tl.float32)
    for s in range(s_start, s_end):
        wb = s % w_chunks
        r = s // w_chunks
        n = r // Hout
        oh = r % Hout
        ow = wb * BK + rk
        ih = oh * sh
        dy_off = (
            ((n * G + g) * Cout_g + rm[:, None]) * HWout + oh * Wout_pad + ow[None, :]
        )
        xb = ((n * G + g) * Cin_g + rn[:, None]) * HWp + ih * Wp + ow[None, :]
        a = tl.load(dy_ptr + dy_off)
        if IEEE:
            acc00 += tl.dot(
                a,
                tl.trans(tl.load(xp_ptr + xb)),
                input_precision="ieee",
                out_dtype=tl.float32,
            )
            acc01 += tl.dot(
                a,
                tl.trans(tl.load(xp_ptr + xb + d01)),
                input_precision="ieee",
                out_dtype=tl.float32,
            )
            acc02 += tl.dot(
                a,
                tl.trans(tl.load(xp_ptr + xb + d02)),
                input_precision="ieee",
                out_dtype=tl.float32,
            )
            acc10 += tl.dot(
                a,
                tl.trans(tl.load(xp_ptr + xb + d10)),
                input_precision="ieee",
                out_dtype=tl.float32,
            )
            acc11 += tl.dot(
                a,
                tl.trans(tl.load(xp_ptr + xb + d11)),
                input_precision="ieee",
                out_dtype=tl.float32,
            )
            acc12 += tl.dot(
                a,
                tl.trans(tl.load(xp_ptr + xb + d12)),
                input_precision="ieee",
                out_dtype=tl.float32,
            )
            acc20 += tl.dot(
                a,
                tl.trans(tl.load(xp_ptr + xb + d20)),
                input_precision="ieee",
                out_dtype=tl.float32,
            )
            acc21 += tl.dot(
                a,
                tl.trans(tl.load(xp_ptr + xb + d21)),
                input_precision="ieee",
                out_dtype=tl.float32,
            )
            acc22 += tl.dot(
                a,
                tl.trans(tl.load(xp_ptr + xb + d22)),
                input_precision="ieee",
                out_dtype=tl.float32,
            )
        else:
            acc00 += tl.dot(a, tl.trans(tl.load(xp_ptr + xb)), out_dtype=tl.float32)
            acc01 += tl.dot(
                a, tl.trans(tl.load(xp_ptr + xb + d01)), out_dtype=tl.float32
            )
            acc02 += tl.dot(
                a, tl.trans(tl.load(xp_ptr + xb + d02)), out_dtype=tl.float32
            )
            acc10 += tl.dot(
                a, tl.trans(tl.load(xp_ptr + xb + d10)), out_dtype=tl.float32
            )
            acc11 += tl.dot(
                a, tl.trans(tl.load(xp_ptr + xb + d11)), out_dtype=tl.float32
            )
            acc12 += tl.dot(
                a, tl.trans(tl.load(xp_ptr + xb + d12)), out_dtype=tl.float32
            )
            acc20 += tl.dot(
                a, tl.trans(tl.load(xp_ptr + xb + d20)), out_dtype=tl.float32
            )
            acc21 += tl.dot(
                a, tl.trans(tl.load(xp_ptr + xb + d21)), out_dtype=tl.float32
            )
            acc22 += tl.dot(
                a, tl.trans(tl.load(xp_ptr + xb + d22)), out_dtype=tl.float32
            )
    base = sp * stride_sp + g * 9 * Cout_g * Cin_g
    CC = Cout_g * Cin_g
    out_off = rm[:, None] * Cin_g + rn[None, :]
    tl.store(part_ptr + base + 0 * CC + out_off, acc00)
    tl.store(part_ptr + base + 1 * CC + out_off, acc01)
    tl.store(part_ptr + base + 2 * CC + out_off, acc02)
    tl.store(part_ptr + base + 3 * CC + out_off, acc10)
    tl.store(part_ptr + base + 4 * CC + out_off, acc11)
    tl.store(part_ptr + base + 5 * CC + out_off, acc12)
    tl.store(part_ptr + base + 6 * CC + out_off, acc20)
    tl.store(part_ptr + base + 7 * CC + out_off, acc21)
    tl.store(part_ptr + base + 8 * CC + out_off, acc22)


# ------------------------------------------------------------------
# 3. Host helpers for fast path
# ------------------------------------------------------------------
def _pad1d_fast(x, pl, Lp):
    L = x.shape[-1]
    NC = x.numel() // L
    xp = torch.zeros((*x.shape[:-1], Lp), device=x.device, dtype=x.dtype)
    ncb = 32 if NC % 32 == 0 else (16 if NC % 16 == 0 else 8)
    lb = 1024 if L >= 1024 else (512 if L >= 512 else (256 if L >= 256 else 128))
    _pad1d_2d_kernel[(NC // ncb, triton.cdiv(L, lb))](
        x.contiguous(), xp, L, Lp, pl, ncb, lb, is_sdnn=True
    )
    return xp


def _slice1d_fast(x, Lo):
    L_in = x.shape[-1]
    if L_in == Lo:
        return x
    NC = x.numel() // L_in
    y = torch.empty((*x.shape[:-1], Lo), device=x.device, dtype=x.dtype)
    ncb = 32 if NC % 32 == 0 else (16 if NC % 16 == 0 else 8)
    lb = 512 if Lo >= 512 else (256 if Lo >= 256 else 128)
    _slice1d_2d_kernel[(NC // ncb, triton.cdiv(Lo, lb))](
        x, y, L_in, Lo, ncb, lb, is_sdnn=True
    )
    return y


def _pad2d_fast(x, ph, pw, Hp, Wp):
    H, W = x.shape[-2], x.shape[-1]
    NC = x.numel() // (H * W)
    xp = torch.zeros((*x.shape[:-2], Hp, Wp), device=x.device, dtype=x.dtype)
    rb = 64 if H >= 64 else (32 if H >= 32 else 16)
    wb = 64 if W >= 64 else (32 if W >= 32 else 16)
    _pad2d_tile_kernel[(NC, triton.cdiv(H, rb), triton.cdiv(W, wb))](
        x.contiguous(), xp, H, W, Hp, Wp, ph, pw, rb, wb, is_sdnn=True
    )
    return xp


def _slice2d_fast(x, Ho, Wo):
    Hin, Win = x.shape[-2], x.shape[-1]
    if Hin == Ho and Win == Wo:
        return x
    NC = x.numel() // (Hin * Win)
    y = torch.empty((*x.shape[:-2], Ho, Wo), device=x.device, dtype=x.dtype)
    rb = 64 if Ho >= 64 else (32 if Ho >= 32 else 16)
    wb = 64 if Wo >= 64 else (32 if Wo >= 32 else 16)
    _slice2d_tile_kernel[(NC, triton.cdiv(Ho, rb), triton.cdiv(Wo, wb))](
        x, y, Hin, Win, Ho, Wo, rb, wb, is_sdnn=True
    )
    return y


def _pack_w_fwd(w, G):
    Cout, Cin_g = w.shape[0], w.shape[1]
    Cout_g = Cout // G
    KK = w.numel() // (Cout * Cin_g)
    out = torch.empty((G, KK, Cout_g, Cin_g), device=w.device, dtype=w.dtype)
    _pack_w_fwd_kernel[(triton.cdiv(w.numel(), 256),)](
        w.contiguous(), out, G, Cout_g, Cin_g, KK, 256
    )
    return out


def _pack_w_flip_t(w, G):
    Cout, Cin_g = w.shape[0], w.shape[1]
    Cout_g = Cout // G
    KK = w.numel() // (Cout * Cin_g)
    out = torch.empty((G, KK, Cin_g, Cout_g), device=w.device, dtype=w.dtype)
    _pack_w_flip_t_kernel[(triton.cdiv(w.numel(), 256),)](
        w.contiguous(), out, G, Cout_g, Cin_g, KK, 256
    )
    return out


def _pack_w_sub_tr(w, G, kh_start, kh_step, n_kh, kw_start, kw_step, n_kw):
    Cz, Co_g = w.shape[0], w.shape[1]
    Cz_g = Cz // G
    if w.ndim == 3:
        KH, KW = 1, w.shape[2]
    else:
        KH, KW = w.shape[2], w.shape[3]
    out = torch.empty((G, n_kh * n_kw, Co_g, Cz_g), device=w.device, dtype=w.dtype)
    _pack_w_sub_tr2d_kernel[(triton.cdiv(out.numel(), 256),)](
        w.contiguous(),
        out,
        G,
        Cz_g,
        Co_g,
        KH,
        KW,
        kh_start,
        kh_step,
        n_kh,
        kw_start,
        kw_step,
        n_kw,
        256,
    )
    return out


def _unpack_gw(sm, wshape, G, dtype):
    Cout, Cin_g = wshape[0], wshape[1]
    Cout_g = Cout // G
    KK = 1
    for d in wshape[2:]:
        KK *= d
    gw_f = torch.empty(wshape, device=sm.device, dtype=torch.float32)
    _unpack_gw_fp32_kernel[(triton.cdiv(gw_f.numel(), 256),)](
        sm, gw_f, G, KK, Cout_g, Cin_g, 256
    )
    return gw_f.to(dtype)


def _merge_4par_fast(p00, p01, p10, p11):
    N, C, Hh, Wh = p00.shape
    Wfull = 2 * Wh
    er = torch.empty((N, C, Hh, Wfull), device=p00.device, dtype=p00.dtype)
    or_ = torch.empty((N, C, Hh, Wfull), device=p00.device, dtype=p00.dtype)
    half = p00.numel()
    if p00.dtype == torch.float32:
        BL = 2048 if half % 2048 == 0 else (512 if half % 512 == 0 else 256)
        _interleave_i64_mless_kernel[(half // BL,)](
            p00.view(torch.int32), p01.view(torch.int32), er.view(torch.int64), BL
        )
        _interleave_i64_mless_kernel[(half // BL,)](
            p10.view(torch.int32),
            p11.view(torch.int32),
            or_.view(torch.int64),
            BL,
        )
    else:
        BL = 4096 if half % 4096 == 0 else (512 if half % 512 == 0 else 256)
        _interleave_u32_mless_kernel[(half // BL,)](
            p00.view(torch.int16), p01.view(torch.int16), er.view(torch.int32), BL
        )
        _interleave_u32_mless_kernel[(half // BL,)](
            p10.view(torch.int16),
            p11.view(torch.int16),
            or_.view(torch.int32),
            BL,
        )
    out = torch.empty((N, C, 2 * Hh, Wfull), device=p00.device, dtype=p00.dtype)
    NC_HH = N * C * Hh
    rb = 64 if NC_HH % 64 == 0 else (32 if NC_HH % 32 == 0 else 16)
    wb = 64 if Wfull >= 64 else 32
    _stack_2rows_kernel[(NC_HH // rb, Wfull // wb)](
        er, or_, out, Wfull, rb, wb, is_sdnn=True
    )
    return out


# ------------------------------------------------------------------
# 4. Fast 1D path
# ------------------------------------------------------------------
def _can_fast_1d(x, weight, gO, sl, pl, dl, transposed, G):
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if sl not in (1, 2) or dl != 1:
        return False
    KL = weight.shape[2]
    if KL < 1:
        return False
    N, Cin, Lin = x.shape
    Cout = gO.shape[1]
    Lout = gO.shape[2]
    Cin_g = Cin // G
    Cout_g = Cout // G
    if (
        (Cin_g % 16 != 0)
        or (Cout_g % 16 != 0)
        or (N * Cin % 8 != 0)
        or (N * Cout % 8 != 0)
    ):
        return False
    if Lin < 128 or Lout < 128:
        return False
    if sl == 2 and transposed:
        # Check sub-grid taps exist and min_sh >= 0 for both parities
        for par in (0, 1):
            taps = [
                (par + pl - kl * dl) // 2
                for kl in range(KL)
                if (par + pl - kl * dl) % 2 == 0
            ]
            if not taps or min(taps) < 0:
                return False
    if sl == 2 and (not transposed):
        for par in (0, 1):
            taps = [
                (par + pl - kl * dl) // 2
                for kl in range(KL)
                if (par + pl - kl * dl) % 2 == 0
            ]
            if not taps or min(taps) < 0:
                return False
    return True


def _prep_xp_1d(x, sl, pl, KL, dl, Lout_pad, G):
    N, Cin, Lin = x.shape
    Cin_g = Cin // G
    NC = N * Cin
    if sl == 1:
        Lp = max(Lin + pl, Lout_pad + (KL - 1) * dl)
        return _pad1d_fast(x, pl, Lp).reshape(1, N, G, Cin_g, Lp), Lp
    if Lin % 2 == 1:
        x = _pad1d_fast(x, 0, Lin + 1)
        Lin += 1
    L2 = Lin // 2
    half = NC * L2
    xeo = torch.empty((2, NC, L2), device=x.device, dtype=x.dtype)
    if x.dtype == torch.float32:
        BL = 2048 if half % 2048 == 0 else 256
        _deint_i64_flat_kernel[(half // BL,)](
            x.contiguous().view(torch.int64), xeo.view(torch.int32), half, BL
        )
    else:
        BL = 4096 if half % 4096 == 0 else 256
        _deint_u32_flat_kernel[(half // BL,)](
            x.contiguous().view(torch.int32), xeo.view(torch.int16), half, BL
        )
    Lp = max(L2 + (pl + 1) // 2 + 1, Lout_pad + ((KL - 1) * dl) // 2 + 2)
    xp = torch.zeros((2, NC, Lp), device=x.device, dtype=x.dtype)
    p0, o0 = pl % 2, pl // 2
    p1, o1 = (pl + 1) % 2, (pl + 1) // 2
    ncb = 32 if NC % 32 == 0 else (16 if NC % 16 == 0 else 8)
    lb = 512 if L2 >= 512 else (256 if L2 >= 256 else 128)
    _pad1d_2d_kernel[(NC // ncb, triton.cdiv(L2, lb))](
        xeo[0], xp[p0], L2, Lp, o0, ncb, lb, is_sdnn=True
    )
    _pad1d_2d_kernel[(NC // ncb, triton.cdiv(L2, lb))](
        xeo[1], xp[p1], L2, Lp, o1, ncb, lb, is_sdnn=True
    )
    return xp.reshape(2, N, G, Cin_g, Lp), Lp


def _run_tr1d_s2(z1, w1, z2, w2, bias, Lo, pl, dl, G, ieee):
    z = z1 if z1 is not None else z2
    dt_orig = z.dtype
    N, Cz, Lz = z.shape
    Cz_g = Cz // G
    w_ref = w1 if w1 is not None else w2
    Co_g = w_ref.shape[1]
    KL = w_ref.shape[2]
    use_i = ieee
    if (not ieee) and (Co_g < 32 or Cz_g < 32):
        z1 = z1.float() if z1 is not None else None
        z2 = z2.float() if z2 is not None else None
        w1 = w1.float() if w1 is not None else None
        w2 = w2.float() if w2 is not None else None
        bias = bias.float() if bias is not None else None
        z = z1 if z1 is not None else z2
        w_ref = w1 if w1 is not None else w2
        use_i = True
    Lhalf = (Lo + 1) // 2
    BN = 256 if Lhalf >= 256 else (128 if Lhalf >= 128 else 64)
    Lhalf_pad = _ru(Lhalf, BN)
    BM = min(64, Co_g)
    BK = min(64, Cz_g)
    Lzp = max(Lz, Lhalf_pad + (KL // 2) + 2)
    zp1 = (
        _pad1d_fast(z1, 0, Lzp)
        if (z1 is not None and Lzp != Lz)
        else (z1.contiguous() if z1 is not None else z)
    )
    zp2 = (
        _pad1d_fast(z2, 0, Lzp)
        if (z2 is not None and Lzp != Lz)
        else (z2.contiguous() if z2 is not None else z)
    )
    outs = []
    for par in (0, 1):
        kl_valid = [kl for kl in range(KL) if (par + pl - kl * dl) % 2 == 0]
        kw_start = kl_valid[0]
        n_kw = len(kl_valid)
        kw_step = kl_valid[1] - kl_valid[0] if n_kw > 1 else 1
        wk1 = (
            _pack_w_sub_tr(w1, G, 0, 1, 1, kw_start, kw_step, n_kw)
            if w1 is not None
            else w_ref
        )
        wk2 = (
            _pack_w_sub_tr(w2, G, 0, 1, 1, kw_start, kw_step, n_kw)
            if w2 is not None
            else w_ref
        )
        y_p = torch.empty((N, G * Co_g, Lhalf_pad), device=z.device, dtype=z.dtype)
        _fwd1d_direct_parity_kernel[(G * (Co_g // BM), N * (Lhalf_pad // BN))](
            zp1,
            wk1,
            zp2,
            wk2,
            bias if bias is not None else z,
            y_p,
            N,
            G,
            Cz_g,
            Co_g,
            Lzp,
            Lhalf_pad,
            n_kw,
            1,
            1,
            z1 is not None,
            z2 is not None,
            bias is not None,
            BM,
            BN,
            BK,
            use_i,
        )
        outs.append(y_p)
    out_e, out_o = outs[0], outs[1]
    out_full = torch.empty((N, G * Co_g, 2 * Lhalf_pad), device=z.device, dtype=z.dtype)
    half_tot = out_e.numel()
    if z.dtype == torch.float32:
        BL = 2048 if half_tot % 2048 == 0 else 256
        _interleave_i64_mless_kernel[(half_tot // BL,)](
            out_e.view(torch.int32),
            out_o.view(torch.int32),
            out_full.view(torch.int64),
            BL,
        )
    else:
        BL = 4096 if half_tot % 4096 == 0 else 256
        _interleave_u32_mless_kernel[(half_tot // BL,)](
            out_e.view(torch.int16),
            out_o.view(torch.int16),
            out_full.view(torch.int32),
            BL,
        )
    res = _slice1d_fast(out_full, Lo)
    return res.to(dt_orig) if res.dtype != dt_orig else res


def _run_gw1d(xp, Lp, b, wshape, sl, dl, G, ieee):
    Cout, Cin_g, KL = wshape
    Cout_g = Cout // G
    N = b.shape[0]
    Lout = b.shape[2]
    BK = 128 if Lout >= 256 else 64
    Lout_pad = _ru(Lout, BK)
    dy = _pad1d_fast(b, 0, Lout_pad) if Lout_pad != Lout else b.contiguous()
    BM = min(64, Cout_g)
    BN = min(64, Cin_g)
    dt_orig = b.dtype
    use_i = ieee
    if (not ieee) and (Cout_g < 32 or Cin_g < 32):
        dy = dy.float()
        xp = xp.float()
        use_i = True
    SP = 16
    tot = G * KL * Cout_g * Cin_g
    tot_pad = _ru(tot, 1024)
    part = torch.empty((SP, tot_pad), device=b.device, dtype=torch.float32)
    _gw1d_direct_parity_kernel[(G * (Cout_g // BM), Cin_g // BN, KL * SP)](
        dy,
        xp,
        part,
        tot_pad,
        N,
        G,
        Cin_g,
        Cout_g,
        Lp,
        Lout_pad,
        KL,
        sl,
        dl,
        SP,
        BM,
        BN,
        BK,
        use_i,
    )
    sm = torch.empty((tot_pad,), device=b.device, dtype=torch.float32)
    _splitk_sum_flat_kernel[(tot_pad // 1024,)](part, sm, tot_pad, SP, 1024)
    return _unpack_gw(sm, wshape, G, dt_orig)


def _cdb_1d_fast(ggI, ggW, ggb, gO, weight, x, sl, pl, dl, transposed, opad, G):
    ieee = x.dtype == torch.float32
    dt_orig = x.dtype
    KL = weight.shape[2]
    N, Cin, Lin = x.shape
    Cout = gO.shape[1]
    Lout = gO.shape[2]
    Cin_g = Cin // G
    Cout_g = Cout // G

    grad_ggO = None
    grad_self = None
    grad_weight = None

    if not transposed:
        BN = 256 if Lout >= 256 else 128
        Lout_pad = _ru(Lout, BN)
        xp1 = None
        Lp = 0
        if ggI is not None:
            xp1, Lp = _prep_xp_1d(ggI, sl, pl, KL, dl, Lout_pad, G)
        if ggI is not None or ggW is not None:
            xp2 = None
            if ggW is not None:
                xp2, Lp2 = _prep_xp_1d(x, sl, pl, KL, dl, Lout_pad, G)
                if xp1 is None:
                    Lp = Lp2
            w1 = _pack_w_fwd(weight, G) if ggI is not None else None
            w2 = _pack_w_fwd(ggW, G) if ggW is not None else None
            use_i = ieee
            x1_u, x2_u, w1_u, w2_u, b_u = xp1, xp2, w1, w2, ggb
            if (not ieee) and (Cout_g < 32 or Cin_g < 32):
                x1_u = xp1.float() if xp1 is not None else None
                x2_u = xp2.float() if xp2 is not None else None
                w1_u = w1.float() if w1 is not None else None
                w2_u = w2.float() if w2 is not None else None
                b_u = ggb.float() if ggb is not None else None
                use_i = True
            ref_t = x1_u if x1_u is not None else x2_u
            w_ref = w1_u if w1_u is not None else w2_u
            y = torch.empty((N, Cout, Lout_pad), device=x.device, dtype=ref_t.dtype)
            BM = min(64, Cout_g)
            BK = min(64, Cin_g)
            _fwd1d_direct_parity_kernel[(G * (Cout_g // BM), N * (Lout_pad // BN))](
                x1_u if x1_u is not None else ref_t,
                w1_u if w1_u is not None else w_ref,
                x2_u if x2_u is not None else ref_t,
                w2_u if w2_u is not None else w_ref,
                b_u if b_u is not None else ref_t,
                y,
                N,
                G,
                Cin_g,
                Cout_g,
                Lp,
                Lout_pad,
                KL,
                sl,
                dl,
                ggI is not None,
                ggW is not None,
                ggb is not None,
                BM,
                BN,
                BK,
                use_i,
            )
            grad_ggO = _slice1d_fast(y, Lout).to(dt_orig)
        elif ggb is not None:
            grad_ggO = _gen_combine_ggo(
                None, None, ggb, N, Cout, 1, Lout, dt_orig, x.device
            ).squeeze(2)

        if ggW is not None:
            if sl == 1:
                p_t = (KL - 1) * dl - pl
                BNs = 256 if Lin >= 256 else 128
                Lin_pad = _ru(Lin, BNs)
                z_xp, Lzp = _prep_xp_1d(gO, 1, p_t, KL, dl, Lin_pad, G)
                wt_k = _pack_w_flip_t(ggW, G)
                zu, wu, ui = z_xp, wt_k, ieee
                if (not ieee) and (Cin_g < 32 or Cout_g < 32):
                    zu, wu, ui = z_xp.float(), wt_k.float(), True
                gs = torch.empty((N, Cin, Lin_pad), device=x.device, dtype=zu.dtype)
                BMs = min(64, Cin_g)
                BKs = min(64, Cout_g)
                _fwd1d_direct_parity_kernel[(G * (Cin_g // BMs), N * (Lin_pad // BNs))](
                    zu,
                    wu,
                    zu,
                    wu,
                    zu,
                    gs,
                    N,
                    G,
                    Cout_g,
                    Cin_g,
                    Lzp,
                    Lin_pad,
                    KL,
                    1,
                    dl,
                    True,
                    False,
                    False,
                    BMs,
                    BNs,
                    BKs,
                    ui,
                )
                grad_self = _slice1d_fast(gs, Lin).to(dt_orig)
            else:
                grad_self = _run_tr1d_s2(
                    gO, ggW, None, None, None, Lin, pl, dl, G, ieee
                )

        if ggI is not None:
            grad_weight = _run_gw1d(xp1, Lp, gO, weight.shape, sl, dl, G, ieee)
        return grad_ggO, grad_self, grad_weight
    else:
        if ggI is not None or ggW is not None:
            if sl == 1:
                p_t = (KL - 1) * dl - pl
                BN = 256 if Lout >= 256 else 128
                Lout_pad = _ru(Lout, BN)
                xp1 = (
                    _prep_xp_1d(ggI, 1, p_t, KL, dl, Lout_pad, G)[0]
                    if ggI is not None
                    else None
                )
                xp2 = (
                    _prep_xp_1d(x, 1, p_t, KL, dl, Lout_pad, G)[0]
                    if ggW is not None
                    else None
                )
                Lp = (xp1 if xp1 is not None else xp2).shape[-1]
                w1 = _pack_w_flip_t(weight, G) if ggI is not None else None
                w2 = _pack_w_flip_t(ggW, G) if ggW is not None else None
                use_i = ieee
                x1_u, x2_u, w1_u, w2_u, b_u = xp1, xp2, w1, w2, ggb
                if (not ieee) and (Cout_g < 32 or Cin_g < 32):
                    x1_u = xp1.float() if xp1 is not None else None
                    x2_u = xp2.float() if xp2 is not None else None
                    w1_u = w1.float() if w1 is not None else None
                    w2_u = w2.float() if w2 is not None else None
                    b_u = ggb.float() if ggb is not None else None
                    use_i = True
                ref_t = x1_u if x1_u is not None else x2_u
                w_ref = w1_u if w1_u is not None else w2_u
                y = torch.empty((N, Cout, Lout_pad), device=x.device, dtype=ref_t.dtype)
                BM = min(64, Cout_g)
                BK = min(64, Cin_g)
                _fwd1d_direct_parity_kernel[(G * (Cout_g // BM), N * (Lout_pad // BN))](
                    x1_u if x1_u is not None else ref_t,
                    w1_u if w1_u is not None else w_ref,
                    x2_u if x2_u is not None else ref_t,
                    w2_u if w2_u is not None else w_ref,
                    b_u if b_u is not None else ref_t,
                    y,
                    N,
                    G,
                    Cin_g,
                    Cout_g,
                    Lp,
                    Lout_pad,
                    KL,
                    1,
                    dl,
                    ggI is not None,
                    ggW is not None,
                    ggb is not None,
                    BM,
                    BN,
                    BK,
                    use_i,
                )
                grad_ggO = _slice1d_fast(y, Lout).to(dt_orig)
            else:
                grad_ggO = _run_tr1d_s2(ggI, weight, x, ggW, ggb, Lout, pl, dl, G, ieee)
        elif ggb is not None:
            grad_ggO = _gen_combine_ggo(
                None, None, ggb, N, Cout, 1, Lout, dt_orig, x.device
            ).squeeze(2)

        gO_xp = None
        Lp = 0
        if ggW is not None or ggI is not None:
            BNs = 256 if Lin >= 256 else 128
            Lin_pad = _ru(Lin, BNs)
            gO_xp, Lp = _prep_xp_1d(gO, sl, pl, KL, dl, Lin_pad, G)
        if ggW is not None:
            wk = _pack_w_fwd(ggW, G)
            zu, wu, ui = gO_xp, wk, ieee
            if (not ieee) and (Cin_g < 32 or Cout_g < 32):
                zu, wu, ui = gO_xp.float(), wk.float(), True
            gs = torch.empty((N, Cin, Lin_pad), device=x.device, dtype=zu.dtype)
            BMs = min(64, Cin_g)
            BKs = min(64, Cout_g)
            _fwd1d_direct_parity_kernel[(G * (Cin_g // BMs), N * (Lin_pad // BNs))](
                zu,
                wu,
                zu,
                wu,
                zu,
                gs,
                N,
                G,
                Cout_g,
                Cin_g,
                Lp,
                Lin_pad,
                KL,
                sl,
                dl,
                True,
                False,
                False,
                BMs,
                BNs,
                BKs,
                ui,
            )
            grad_self = _slice1d_fast(gs, Lin).to(dt_orig)
        if ggI is not None:
            grad_weight = _run_gw1d(gO_xp, Lp, ggI, weight.shape, sl, dl, G, ieee)
        return grad_ggO, grad_self, grad_weight


# ------------------------------------------------------------------
# 5. Fast 2D path
# ------------------------------------------------------------------
def _can_fast_2d(x, weight, gO, sh, sw, ph, pw, dh, dw, transposed, G):
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if sh not in (1, 2) or sw not in (1, 2) or sh != sw or dh != 1 or dw != 1:
        return False
    KH, KW = weight.shape[2], weight.shape[3]
    if KH < 1 or KW < 1:
        return False
    N, Cin, H, W = x.shape
    Cout, Hout, Wout = gO.shape[1], gO.shape[2], gO.shape[3]
    Cin_g = Cin // G
    Cout_g = Cout // G
    if (Cin_g % 16 != 0) or (Cout_g % 16 != 0):
        return False
    if H < 16 or W < 16 or Hout < 16 or Wout < 16:
        return False
    if sw == 2:
        for par in (0, 1):
            th = [
                (par + ph - kh * dh) // 2
                for kh in range(KH)
                if (par + ph - kh * dh) % 2 == 0
            ]
            tw = [
                (par + pw - kw * dw) // 2
                for kw in range(KW)
                if (par + pw - kw * dw) % 2 == 0
            ]
            if not th or not tw or min(th) < 0 or min(tw) < 0:
                return False
    return True


def _prep_xp_2d(x, sh, sw, ph, pw, KH, KW, dh, dw, Hout, Wout_pad, G):
    N, Cin, H, W = x.shape
    Cin_g = Cin // G
    Hp = max(H + ph, (Hout - 1) * sh + (KH - 1) * dh + 1)
    if sw == 1:
        Wp = max(W + pw, Wout_pad + (KW - 1) * dw)
        return _pad2d_fast(x, ph, pw, Hp, Wp).reshape(1, N, G, Cin_g, Hp, Wp), Hp, Wp
    if W % 2 == 1:
        x = _pad2d_fast(x, 0, 0, H, W + 1)
        W += 1
    W2 = W // 2
    half = x.numel() // 2
    xeo = torch.empty((2, N, Cin, H, W2), device=x.device, dtype=x.dtype)
    if x.dtype == torch.float32:
        BL = 2048 if half % 2048 == 0 else 256
        _deint_i64_flat_kernel[(half // BL,)](
            x.contiguous().view(torch.int64), xeo.view(torch.int32), half, BL
        )
    else:
        BL = 4096 if half % 4096 == 0 else 256
        _deint_u32_flat_kernel[(half // BL,)](
            x.contiguous().view(torch.int32), xeo.view(torch.int16), half, BL
        )
    Wp = max(W2 + (pw + 1) // 2 + 1, Wout_pad + ((KW - 1) * dw) // 2 + 2)
    xp = torch.zeros((2, N, Cin, Hp, Wp), device=x.device, dtype=x.dtype)
    p0, o0 = pw % 2, pw // 2
    p1, o1 = (pw + 1) % 2, (pw + 1) // 2
    NC = N * Cin
    rb = 64 if H >= 64 else (32 if H >= 32 else 16)
    wb = 64 if W2 >= 64 else (32 if W2 >= 32 else 16)
    _pad2d_tile_kernel[(NC, triton.cdiv(H, rb), triton.cdiv(W2, wb))](
        xeo[0], xp[p0], H, W2, Hp, Wp, ph, o0, rb, wb, is_sdnn=True
    )
    _pad2d_tile_kernel[(NC, triton.cdiv(H, rb), triton.cdiv(W2, wb))](
        xeo[1], xp[p1], H, W2, Hp, Wp, ph, o1, rb, wb, is_sdnn=True
    )
    return xp.reshape(2, N, G, Cin_g, Hp, Wp), Hp, Wp


def _run_tr2d_s2(z1, w1, z2, w2, bias, Ho, Wo, ph, pw, dh, dw, G, ieee):
    z = z1 if z1 is not None else z2
    dt_orig = z.dtype
    N, Cz, Hz, Wz = z.shape
    Cz_g = Cz // G
    w_ref = w1 if w1 is not None else w2
    Co_g = w_ref.shape[1]
    KH, KW = w_ref.shape[2], w_ref.shape[3]
    Hh = (Ho + 1) // 2
    Wh = (Wo + 1) // 2
    BN = 64 if Wh >= 64 else (32 if Wh >= 32 else 16)
    Wh_pad = _ru(Wh, BN)
    BM = min(64, Co_g)
    BK = min(64, Cz_g)
    use_i = ieee
    if (not ieee) and (Co_g < 32 or Cz_g < 32 or BN < 32):
        z1 = z1.float() if z1 is not None else None
        z2 = z2.float() if z2 is not None else None
        w1 = w1.float() if w1 is not None else None
        w2 = w2.float() if w2 is not None else None
        bias = bias.float() if bias is not None else None
        z = z1 if z1 is not None else z2
        w_ref = w1 if w1 is not None else w2
        use_i = True
    Hzp = max(Hz, Hh + (KH // 2) + 2)
    Wzp = max(Wz, Wh_pad + (KW // 2) + 2)
    zp1 = _pad2d_fast(z1, 0, 0, Hzp, Wzp) if z1 is not None else z
    zp2 = _pad2d_fast(z2, 0, 0, Hzp, Wzp) if z2 is not None else z
    pars = {}
    for par_h in (0, 1):
        kh_v = [kh for kh in range(KH) if (par_h + ph - kh * dh) % 2 == 0]
        kh_s, n_kh = kh_v[0], len(kh_v)
        kh_st = kh_v[1] - kh_v[0] if n_kh > 1 else 1
        for par_w in (0, 1):
            kw_v = [kw for kw in range(KW) if (par_w + pw - kw * dw) % 2 == 0]
            kw_s, n_kw = kw_v[0], len(kw_v)
            kw_st = kw_v[1] - kw_v[0] if n_kw > 1 else 1
            wk1 = (
                _pack_w_sub_tr(w1, G, kh_s, kh_st, n_kh, kw_s, kw_st, n_kw)
                if w1 is not None
                else w_ref
            )
            wk2 = (
                _pack_w_sub_tr(w2, G, kh_s, kh_st, n_kh, kw_s, kw_st, n_kw)
                if w2 is not None
                else w_ref
            )
            y_p = torch.empty((N, G * Co_g, Hh, Wh_pad), device=z.device, dtype=z.dtype)
            _fwd2d_direct_parity_kernel[(G * (Co_g // BM), N * Hh * (Wh_pad // BN))](
                zp1,
                wk1,
                zp2,
                wk2,
                bias if bias is not None else z,
                y_p,
                N,
                G,
                Cz_g,
                Co_g,
                Hh,
                Wh_pad,
                Hzp,
                Wzp,
                n_kh,
                n_kw,
                1,
                1,
                1,
                1,
                z1 is not None,
                z2 is not None,
                bias is not None,
                BM,
                BN,
                BK,
                use_i,
            )
            pars[(par_h, par_w)] = y_p
    out_full = _merge_4par_fast(pars[(0, 0)], pars[(0, 1)], pars[(1, 0)], pars[(1, 1)])
    res = _slice2d_fast(out_full, Ho, Wo)
    return res.to(dt_orig) if res.dtype != dt_orig else res


def _run_gw2d(xp, Hp, Wp, b, wshape, sh, sw, dh, dw, G, ieee):
    Cout, Cin_g, KH, KW = wshape
    Cout_g = Cout // G
    N, _, Hout, Wout = b.shape
    BK = 64 if Wout >= 64 else (32 if Wout >= 32 else 16)
    Wout_pad = _ru(Wout, BK)
    dy = _pad2d_fast(b, 0, 0, Hout, Wout_pad) if Wout_pad != Wout else b.contiguous()
    BM = min(64, Cout_g)
    BN = min(64, Cin_g)
    dt_orig = b.dtype
    use_i = ieee
    if (not ieee) and (Cout_g < 32 or Cin_g < 32 or BK < 32):
        dy = dy.float()
        xp = xp.float()
        use_i = True
    tot = G * KH * KW * Cout_g * Cin_g
    tot_pad = _ru(tot, 1024)
    if sw == 1 and KH == 3 and KW == 3:
        SP = 32 if Hout >= 64 else 16
        part = torch.empty((SP, tot_pad), device=b.device, dtype=torch.float32)
        _gw2d_s1_k3_all9_kernel[(G * (Cout_g // BM), Cin_g // BN, SP)](
            dy,
            xp,
            part,
            tot_pad,
            N,
            G,
            Cin_g,
            Cout_g,
            Hout,
            Wout_pad,
            Hp,
            Wp,
            sh,
            dh,
            dw,
            SP,
            BM,
            BN,
            BK,
            use_i,
        )
    else:
        SP = 8
        part = torch.empty((SP, tot_pad), device=b.device, dtype=torch.float32)
        _gw2d_direct_parity_kernel[(G * (Cout_g // BM), Cin_g // BN, KH * KW * SP)](
            dy,
            xp,
            part,
            tot_pad,
            N,
            G,
            Cin_g,
            Cout_g,
            Hout,
            Wout_pad,
            Hp,
            Wp,
            KH,
            KW,
            sh,
            sw,
            dh,
            dw,
            SP,
            BM,
            BN,
            BK,
            use_i,
        )
    sm = torch.empty((tot_pad,), device=b.device, dtype=torch.float32)
    _splitk_sum_flat_kernel[(tot_pad // 1024,)](part, sm, tot_pad, SP, 1024)
    return _unpack_gw(sm, wshape, G, dt_orig)


def _cdb_2d_fast(
    ggI, ggW, ggb, gO, weight, x, sh, sw, ph, pw, dh, dw, transposed, oph, opw, G
):
    ieee = x.dtype == torch.float32
    dt_orig = x.dtype
    KH, KW = weight.shape[2], weight.shape[3]
    N, Cin, H, W = x.shape
    Cout, Hout, Wout = gO.shape[1], gO.shape[2], gO.shape[3]
    Cin_g = Cin // G
    Cout_g = Cout // G

    grad_ggO = None
    grad_self = None
    grad_weight = None

    if not transposed:
        BN = 64 if Wout >= 64 else (32 if Wout >= 32 else 16)
        Wout_pad = _ru(Wout, BN)
        xp1 = None
        Hp = Wp = 0
        if ggI is not None:
            xp1, Hp, Wp = _prep_xp_2d(
                ggI, sh, sw, ph, pw, KH, KW, dh, dw, Hout, Wout_pad, G
            )
        if ggI is not None or ggW is not None:
            xp2 = None
            if ggW is not None:
                xp2, Hp2, Wp2 = _prep_xp_2d(
                    x, sh, sw, ph, pw, KH, KW, dh, dw, Hout, Wout_pad, G
                )
                if xp1 is None:
                    Hp, Wp = Hp2, Wp2
            w1 = _pack_w_fwd(weight, G) if ggI is not None else None
            w2 = _pack_w_fwd(ggW, G) if ggW is not None else None
            BM = min(64, Cout_g)
            BK = min(64, Cin_g)
            use_i = ieee
            x1_u, x2_u, w1_u, w2_u, b_u = xp1, xp2, w1, w2, ggb
            if (not ieee) and (Cout_g < 32 or Cin_g < 32 or BN < 32):
                x1_u = xp1.float() if xp1 is not None else None
                x2_u = xp2.float() if xp2 is not None else None
                w1_u = w1.float() if w1 is not None else None
                w2_u = w2.float() if w2 is not None else None
                b_u = ggb.float() if ggb is not None else None
                use_i = True
            ref_t = x1_u if x1_u is not None else x2_u
            w_ref = w1_u if w1_u is not None else w2_u
            y = torch.empty(
                (N, Cout, Hout, Wout_pad), device=x.device, dtype=ref_t.dtype
            )
            _fwd2d_direct_parity_kernel[
                (G * (Cout_g // BM), N * Hout * (Wout_pad // BN))
            ](
                x1_u if x1_u is not None else ref_t,
                w1_u if w1_u is not None else w_ref,
                x2_u if x2_u is not None else ref_t,
                w2_u if w2_u is not None else w_ref,
                b_u if b_u is not None else ref_t,
                y,
                N,
                G,
                Cin_g,
                Cout_g,
                Hout,
                Wout_pad,
                Hp,
                Wp,
                KH,
                KW,
                sh,
                sw,
                dh,
                dw,
                ggI is not None,
                ggW is not None,
                ggb is not None,
                BM,
                BN,
                BK,
                use_i,
            )
            grad_ggO = _slice2d_fast(y, Hout, Wout).to(dt_orig)
        elif ggb is not None:
            grad_ggO = _gen_combine_ggo(
                None, None, ggb, N, Cout, Hout, Wout, dt_orig, x.device
            )

        if ggW is not None:
            if sh == 1 and sw == 1:
                pt_h = (KH - 1) * dh - ph
                pt_w = (KW - 1) * dw - pw
                BNs = 64 if W >= 64 else (32 if W >= 32 else 16)
                W_pad = _ru(W, BNs)
                z_xp, Hzp, Wzp = _prep_xp_2d(
                    gO, 1, 1, pt_h, pt_w, KH, KW, dh, dw, H, W_pad, G
                )
                wt_k = _pack_w_flip_t(ggW, G)
                BMs = min(64, Cin_g)
                BKs = min(64, Cout_g)
                zu, wu, ui = z_xp, wt_k, ieee
                if (not ieee) and (Cin_g < 32 or Cout_g < 32 or BNs < 32):
                    zu, wu, ui = z_xp.float(), wt_k.float(), True
                gs = torch.empty((N, Cin, H, W_pad), device=x.device, dtype=zu.dtype)
                _fwd2d_direct_parity_kernel[
                    (G * (Cin_g // BMs), N * H * (W_pad // BNs))
                ](
                    zu,
                    wu,
                    zu,
                    wu,
                    zu,
                    gs,
                    N,
                    G,
                    Cout_g,
                    Cin_g,
                    H,
                    W_pad,
                    Hzp,
                    Wzp,
                    KH,
                    KW,
                    1,
                    1,
                    dh,
                    dw,
                    True,
                    False,
                    False,
                    BMs,
                    BNs,
                    BKs,
                    ui,
                )
                grad_self = _slice2d_fast(gs, H, W).to(dt_orig)
            else:
                grad_self = _run_tr2d_s2(
                    gO, ggW, None, None, None, H, W, ph, pw, dh, dw, G, ieee
                )

        if ggI is not None:
            grad_weight = _run_gw2d(
                xp1, Hp, Wp, gO, weight.shape, sh, sw, dh, dw, G, ieee
            )
        return grad_ggO, grad_self, grad_weight
    else:
        if ggI is not None or ggW is not None:
            if sh == 1 and sw == 1:
                pt_h = (KH - 1) * dh - ph
                pt_w = (KW - 1) * dw - pw
                BN = 64 if Wout >= 64 else (32 if Wout >= 32 else 16)
                Wout_pad = _ru(Wout, BN)
                xp1 = (
                    _prep_xp_2d(
                        ggI, 1, 1, pt_h, pt_w, KH, KW, dh, dw, Hout, Wout_pad, G
                    )[0]
                    if ggI is not None
                    else None
                )
                xp2 = (
                    _prep_xp_2d(x, 1, 1, pt_h, pt_w, KH, KW, dh, dw, Hout, Wout_pad, G)[
                        0
                    ]
                    if ggW is not None
                    else None
                )
                ref_x = xp1 if xp1 is not None else xp2
                Hp, Wp = ref_x.shape[-2], ref_x.shape[-1]
                w1 = _pack_w_flip_t(weight, G) if ggI is not None else None
                w2 = _pack_w_flip_t(ggW, G) if ggW is not None else None
                BM = min(64, Cout_g)
                BK = min(64, Cin_g)
                use_i = ieee
                x1_u, x2_u, w1_u, w2_u, b_u = xp1, xp2, w1, w2, ggb
                if (not ieee) and (Cout_g < 32 or Cin_g < 32 or BN < 32):
                    x1_u = xp1.float() if xp1 is not None else None
                    x2_u = xp2.float() if xp2 is not None else None
                    w1_u = w1.float() if w1 is not None else None
                    w2_u = w2.float() if w2 is not None else None
                    b_u = ggb.float() if ggb is not None else None
                    use_i = True
                ref_t = x1_u if x1_u is not None else x2_u
                w_ref = w1_u if w1_u is not None else w2_u
                y = torch.empty(
                    (N, Cout, Hout, Wout_pad), device=x.device, dtype=ref_t.dtype
                )
                _fwd2d_direct_parity_kernel[
                    (G * (Cout_g // BM), N * Hout * (Wout_pad // BN))
                ](
                    x1_u if x1_u is not None else ref_t,
                    w1_u if w1_u is not None else w_ref,
                    x2_u if x2_u is not None else ref_t,
                    w2_u if w2_u is not None else w_ref,
                    b_u if b_u is not None else ref_t,
                    y,
                    N,
                    G,
                    Cin_g,
                    Cout_g,
                    Hout,
                    Wout_pad,
                    Hp,
                    Wp,
                    KH,
                    KW,
                    1,
                    1,
                    dh,
                    dw,
                    ggI is not None,
                    ggW is not None,
                    ggb is not None,
                    BM,
                    BN,
                    BK,
                    use_i,
                )
                grad_ggO = _slice2d_fast(y, Hout, Wout).to(dt_orig)
            else:
                grad_ggO = _run_tr2d_s2(
                    ggI, weight, x, ggW, ggb, Hout, Wout, ph, pw, dh, dw, G, ieee
                )
        elif ggb is not None:
            grad_ggO = _gen_combine_ggo(
                None, None, ggb, N, Cout, Hout, Wout, dt_orig, x.device
            )

        gO_xp = None
        Hp = Wp = 0
        if ggW is not None or ggI is not None:
            BNs = 64 if W >= 64 else (32 if W >= 32 else 16)
            W_pad = _ru(W, BNs)
            gO_xp, Hp, Wp = _prep_xp_2d(gO, sh, sw, ph, pw, KH, KW, dh, dw, H, W_pad, G)
        if ggW is not None:
            wk = _pack_w_fwd(ggW, G)
            BMs = min(64, Cin_g)
            BKs = min(64, Cout_g)
            zu, wu, ui = gO_xp, wk, ieee
            if (not ieee) and (Cin_g < 32 or Cout_g < 32 or BNs < 32):
                zu, wu, ui = gO_xp.float(), wk.float(), True
            gs = torch.empty((N, Cin, H, W_pad), device=x.device, dtype=zu.dtype)
            _fwd2d_direct_parity_kernel[(G * (Cin_g // BMs), N * H * (W_pad // BNs))](
                zu,
                wu,
                zu,
                wu,
                zu,
                gs,
                N,
                G,
                Cout_g,
                Cin_g,
                H,
                W_pad,
                Hp,
                Wp,
                KH,
                KW,
                sh,
                sw,
                dh,
                dw,
                True,
                False,
                False,
                BMs,
                BNs,
                BKs,
                ui,
            )
            grad_self = _slice2d_fast(gs, H, W).to(dt_orig)
        if ggI is not None:
            grad_weight = _run_gw2d(
                gO_xp, Hp, Wp, ggI, weight.shape, sh, sw, dh, dw, G, ieee
            )
        return grad_ggO, grad_self, grad_weight


# ------------------------------------------------------------------
# 6. Generic 2D fallback (also serves widened 1D fallback)
# ------------------------------------------------------------------
def _cdb_2d_generic(
    ggI, ggW, ggb, gO, weight, x, sh, sw, ph, pw, dh, dw, transposed, oph, opw, G
):
    dt = x.dtype
    N, Cin, H, W = x.shape
    Cout, Hout, Wout = gO.shape[1], gO.shape[2], gO.shape[3]
    kH, kW = weight.shape[2], weight.shape[3]

    p1 = None
    p2 = None
    if ggI is not None:
        if not transposed:
            p1 = _gen_fwd2d(ggI, weight, sh, sw, ph, pw, dh, dw, G)
        else:
            p1 = _gen_tr2d(ggI, weight, sh, sw, ph, pw, dh, dw, oph, opw, G)
    if ggW is not None:
        if not transposed:
            p2 = _gen_fwd2d(x, ggW, sh, sw, ph, pw, dh, dw, G)
        else:
            p2 = _gen_tr2d(x, ggW, sh, sw, ph, pw, dh, dw, oph, opw, G)

    grad_ggO = None
    if p1 is not None or p2 is not None or ggb is not None:
        grad_ggO = _gen_combine_ggo(p1, p2, ggb, N, Cout, Hout, Wout, dt, x.device)

    grad_self = None
    if ggW is not None:
        if not transposed:
            op_h = H - ((Hout - 1) * sh - 2 * ph + dh * (kH - 1) + 1)
            op_w = W - ((Wout - 1) * sw - 2 * pw + dw * (kW - 1) + 1)
            grad_self = _gen_tr2d(gO, ggW, sh, sw, ph, pw, dh, dw, op_h, op_w, G).to(dt)
        else:
            grad_self = _gen_fwd2d(gO, ggW, sh, sw, ph, pw, dh, dw, G).to(dt)

    grad_weight = None
    if ggI is not None:
        if not transposed:
            grad_weight = _gen_gw2d(
                ggI, gO, weight.shape, sh, sw, ph, pw, dh, dw, G
            ).to(dt)
        else:
            grad_weight = _gen_gw2d(
                gO, ggI, weight.shape, sh, sw, ph, pw, dh, dw, G
            ).to(dt)

    return grad_ggO, grad_self, grad_weight


def _convolution_double_backward(
    ggI,
    ggW,
    ggb,
    gO,
    weight,
    self,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    logger.debug("GEMS_KUNLUNXIN _CONVOLUTION_DOUBLE_BACKWARD")
    spatial_ndim = weight.ndim - 2
    if spatial_ndim == 1:
        sl = int(stride[0]) if isinstance(stride, (list, tuple)) else int(stride)
        pl = int(padding[0]) if isinstance(padding, (list, tuple)) else int(padding)
        dl = int(dilation[0]) if isinstance(dilation, (list, tuple)) else int(dilation)
        opl = (
            int(output_padding[0])
            if isinstance(output_padding, (list, tuple))
            else int(output_padding)
        )
        if _can_fast_1d(self, weight, gO, sl, pl, dl, transposed, groups):
            return _cdb_1d_fast(
                ggI, ggW, ggb, gO, weight, self, sl, pl, dl, transposed, opl, groups
            )
        ggI_2 = ggI.unsqueeze(2) if ggI is not None else None
        ggW_2 = ggW.unsqueeze(2) if ggW is not None else None
        gO_2 = gO.unsqueeze(2)
        w_2 = weight.unsqueeze(2)
        x_2 = self.unsqueeze(2)
        r0, r1, r2 = _cdb_2d_generic(
            ggI_2,
            ggW_2,
            ggb,
            gO_2,
            w_2,
            x_2,
            1,
            sl,
            0,
            pl,
            1,
            dl,
            transposed,
            0,
            opl,
            groups,
        )
        return (
            r0.squeeze(2) if r0 is not None else None,
            r1.squeeze(2) if r1 is not None else None,
            r2.squeeze(2) if r2 is not None else None,
        )
    if spatial_ndim == 2:
        sh, sw = _pair(stride)
        ph, pw = _pair(padding)
        dh, dw = _pair(dilation)
        oph, opw = _pair(output_padding)
        if _can_fast_2d(self, weight, gO, sh, sw, ph, pw, dh, dw, transposed, groups):
            return _cdb_2d_fast(
                ggI,
                ggW,
                ggb,
                gO,
                weight,
                self,
                sh,
                sw,
                ph,
                pw,
                dh,
                dw,
                transposed,
                oph,
                opw,
                groups,
            )
        return _cdb_2d_generic(
            ggI,
            ggW,
            ggb,
            gO,
            weight,
            self,
            sh,
            sw,
            ph,
            pw,
            dh,
            dw,
            transposed,
            oph,
            opw,
            groups,
        )
    raise NotImplementedError(
        f"_convolution_double_backward: FlagGems supports 1D and 2D convolutions, "
        f"got {spatial_ndim}D"
    )
