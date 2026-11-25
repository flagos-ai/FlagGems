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

"""Triton implementation of ``aten::_weight_norm``.

Reference semantics (verified against ``torch._weight_norm``):

    norm = ||v||_2 over every axis except the literal ``dim``
    out  = v * (g / norm)          # g broadcast along its single real axis

So ``dim == 0`` reduces the trailing axes (g indexed by axis 0), ``dim == r-1``
reduces the leading axes (g indexed by the last axis) and a negative ``dim``
removes nothing from the reduction set, i.e. it normalizes by the Frobenius
norm while ``g`` is still broadcast along its own axis.  No epsilon is added:
an exactly zero norm yields the same NaN/Inf as the reference.

Every path is expressed with the same pair of flattened indices
``(NG kept indices) x (NK reduced elements)`` plus explicit strides, so a
single program tile can cover the complete reduction and read ``v`` once.
The launched output is always freshly allocated and contiguous, so its offsets
are derived inside the kernels from ``NG``/``NK``.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_TILE_CAP = 4096
_INNER_K = 64
_INNER_G = 32
_APPLY_K = 128


# ------------------------------------------------------------------ kernels --


@triton.jit
def _wn_tile(
    out_ptr,
    v_ptr,
    g_ptr,
    NG: tl.constexpr,
    NK: tl.constexpr,
    v_gs: tl.constexpr,
    v_rs: tl.constexpr,
    g_s: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_K: tl.constexpr,
    RED_AXIS: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    gi = pid * BLOCK_G + tl.arange(0, BLOCK_G)
    ki = tl.arange(0, BLOCK_K)
    gm = gi < NG
    if RED_AXIS == 1:
        off_v = gi[:, None] * v_gs + ki[None, :] * v_rs
        off_o = gi[:, None] * NK + ki[None, :]
        msk = gm[:, None] & (ki < NK)[None, :]
    else:
        off_v = ki[:, None] * v_rs + gi[None, :] * v_gs
        off_o = ki[:, None] * NG + gi[None, :]
        msk = (ki < NK)[:, None] & gm[None, :]
    if EVEN:
        x = tl.load(v_ptr + off_v).to(tl.float32)
    else:
        x = tl.load(v_ptr + off_v, mask=msk, other=0.0).to(tl.float32)
    nrm = tl.sqrt(tl.sum(x * x, axis=RED_AXIS))
    gv = tl.load(g_ptr + gi * g_s, mask=gm, other=0.0).to(tl.float32)
    sc = gv / nrm
    if RED_AXIS == 1:
        o = x * sc[:, None]
    else:
        o = x * sc[None, :]
    if EVEN:
        tl.store(out_ptr + off_o, o.to(out_ptr.dtype.element_ty))
    else:
        tl.store(out_ptr + off_o, o.to(out_ptr.dtype.element_ty), mask=msk)


@triton.jit
def _wn_loop(
    out_ptr,
    v_ptr,
    g_ptr,
    NG: tl.constexpr,
    NK: tl.constexpr,
    v_gs: tl.constexpr,
    v_rs: tl.constexpr,
    g_s: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    gi = pid * BLOCK_G + tl.arange(0, BLOCK_G)
    gm = gi < NG
    ss = tl.zeros([BLOCK_G], tl.float32)
    for k0 in range(0, NK, BLOCK_K):
        ki = k0 + tl.arange(0, BLOCK_K)
        m = gm[:, None] & (ki < NK)[None, :]
        x = tl.load(
            v_ptr + gi[:, None] * v_gs + ki[None, :] * v_rs, mask=m, other=0.0
        ).to(tl.float32)
        ss += tl.sum(x * x, axis=1)
    sc = tl.load(g_ptr + gi * g_s, mask=gm, other=0.0).to(tl.float32) / tl.sqrt(ss)
    for k0 in range(0, NK, BLOCK_K):
        ki = k0 + tl.arange(0, BLOCK_K)
        m = gm[:, None] & (ki < NK)[None, :]
        x = tl.load(
            v_ptr + gi[:, None] * v_gs + ki[None, :] * v_rs, mask=m, other=0.0
        ).to(tl.float32)
        tl.store(
            out_ptr + gi[:, None] * NK + ki[None, :],
            (x * sc[:, None]).to(out_ptr.dtype.element_ty),
            mask=m,
        )


@triton.jit
def _wn_partial(
    part_ptr,
    v_ptr,
    NG: tl.constexpr,
    NK: tl.constexpr,
    v_gs: tl.constexpr,
    v_rs: tl.constexpr,
    rows_per: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_g = tl.program_id(1)
    gi = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    gm = gi < NG
    acc = tl.zeros([BLOCK_G], tl.float32)
    start = pid_s * rows_per
    for k0 in range(start, start + rows_per, BLOCK_K):
        ki = k0 + tl.arange(0, BLOCK_K)
        m = (ki < NK)[:, None] & gm[None, :]
        x = tl.load(
            v_ptr + ki[:, None] * v_rs + gi[None, :] * v_gs, mask=m, other=0.0
        ).to(tl.float32)
        acc += tl.sum(x * x, axis=0)
    tl.store(part_ptr + pid_s * NG + gi, acc, mask=gm)


@triton.jit
def _wn_global(
    ss_ptr,
    v_ptr,
    NG: tl.constexpr,
    NK: tl.constexpr,
    v_gs: tl.constexpr,
    v_rs: tl.constexpr,
    rows_per: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_g = tl.program_id(1)
    gi = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    gm = gi < NG
    acc = tl.zeros([BLOCK_G], tl.float32)
    start = pid_s * rows_per
    for k0 in range(start, start + rows_per, BLOCK_K):
        ki = k0 + tl.arange(0, BLOCK_K)
        m = (ki < NK)[:, None] & gm[None, :]
        x = tl.load(
            v_ptr + ki[:, None] * v_rs + gi[None, :] * v_gs, mask=m, other=0.0
        ).to(tl.float32)
        acc += tl.sum(x * x, axis=0)
    tl.atomic_add(ss_ptr, tl.sum(acc))


@triton.jit
def _wn_apply(
    out_ptr,
    v_ptr,
    g_ptr,
    part_ptr,
    NG: tl.constexpr,
    NK: tl.constexpr,
    npart: tl.constexpr,
    v_gs: tl.constexpr,
    v_rs: tl.constexpr,
    g_s: tl.constexpr,
    p_gs: tl.constexpr,
    p_gn: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SLOTS: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_g = tl.program_id(1)
    gi = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    gm = gi < NG
    s = tl.arange(0, SLOTS)
    pm = tl.load(
        part_ptr + s[:, None] * p_gs + (gi * p_gn)[None, :],
        mask=(s < npart)[:, None] & gm[None, :],
        other=0.0,
    )
    ss = tl.sum(pm, axis=0)
    ki = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    m = (ki < NK)[:, None] & gm[None, :]
    x = tl.load(v_ptr + ki[:, None] * v_rs + gi[None, :] * v_gs, mask=m, other=0.0).to(
        tl.float32
    )
    gv = tl.load(g_ptr + gi * g_s, mask=gm, other=0.0).to(tl.float32)
    sc = gv / tl.sqrt(ss)
    o = x * sc[None, :]
    tl.store(
        out_ptr + ki[:, None] * NG + gi[None, :],
        o.to(out_ptr.dtype.element_ty),
        mask=m,
    )


@triton.jit
def _wn_bwd_stats(
    ss_ptr,
    s_ptr,
    dg_ptr,
    v_ptr,
    grad_ptr,
    NG,
    NK,
    v_gs,
    v_rs,
    gr_gs,
    gr_rs,
    dg_s,
    BLOCK_G: tl.constexpr,
    BLOCK_K: tl.constexpr,
    RED_AXIS: tl.constexpr,
    WRITE_DG: tl.constexpr,
):
    pid = tl.program_id(0)
    gi = pid * BLOCK_G + tl.arange(0, BLOCK_G)
    gm = gi < NG
    acc_ss = tl.zeros([BLOCK_G], tl.float32)
    acc_s = tl.zeros([BLOCK_G], tl.float32)
    for k0 in range(0, NK, BLOCK_K):
        ki = k0 + tl.arange(0, BLOCK_K)
        if RED_AXIS == 1:
            m = gm[:, None] & (ki < NK)[None, :]
            x = tl.load(
                v_ptr + gi[:, None] * v_gs + ki[None, :] * v_rs, mask=m, other=0.0
            ).to(tl.float32)
            gr = tl.load(
                grad_ptr + gi[:, None] * gr_gs + ki[None, :] * gr_rs, mask=m, other=0.0
            ).to(tl.float32)
        else:
            m = (ki < NK)[:, None] & gm[None, :]
            x = tl.load(
                v_ptr + ki[:, None] * v_rs + gi[None, :] * v_gs, mask=m, other=0.0
            ).to(tl.float32)
            gr = tl.load(
                grad_ptr + ki[:, None] * gr_rs + gi[None, :] * gr_gs, mask=m, other=0.0
            ).to(tl.float32)
        acc_ss += tl.sum(x * x, axis=RED_AXIS)
        acc_s += tl.sum(x * gr, axis=RED_AXIS)
    tl.store(ss_ptr + gi, acc_ss, mask=gm)
    tl.store(s_ptr + gi, acc_s, mask=gm)
    if WRITE_DG:
        tl.store(
            dg_ptr + gi * dg_s,
            (acc_s / tl.sqrt(acc_ss)).to(dg_ptr.dtype.element_ty),
            mask=gm,
        )


@triton.jit
def _wn_bwd_dv(
    dv_ptr,
    v_ptr,
    grad_ptr,
    g_ptr,
    ss_ptr,
    s_ptr,
    NG,
    NK,
    v_gs,
    v_rs,
    gr_gs,
    gr_rs,
    dv_gs,
    dv_rs,
    g_s,
    stat_s,
    ss_off,
    s_off,
    BLOCK_G: tl.constexpr,
    BLOCK_K: tl.constexpr,
    RED_AXIS: tl.constexpr,
    GLOBAL: tl.constexpr,
):
    pid = tl.program_id(0)
    gi = pid * BLOCK_G + tl.arange(0, BLOCK_G)
    gm = gi < NG
    inv = 1.0 / tl.sqrt(tl.load(ss_ptr + ss_off + gi * stat_s, mask=gm, other=0.0))
    sv = tl.load(s_ptr + s_off + gi * stat_s, mask=gm, other=0.0)
    vsi = sv * inv * inv * inv
    gv = tl.load(g_ptr + gi * g_s, mask=gm, other=0.0).to(tl.float32)
    if not GLOBAL:
        vsi = vsi * gv
    for k0 in range(0, NK, BLOCK_K):
        ki = k0 + tl.arange(0, BLOCK_K)
        if RED_AXIS == 1:
            m = gm[:, None] & (ki < NK)[None, :]
            x = tl.load(
                v_ptr + gi[:, None] * v_gs + ki[None, :] * v_rs, mask=m, other=0.0
            ).to(tl.float32)
            gr = tl.load(
                grad_ptr + gi[:, None] * gr_gs + ki[None, :] * gr_rs, mask=m, other=0.0
            ).to(tl.float32)
            dv = gv[:, None] * gr * inv[:, None] - x * vsi[:, None]
            tl.store(
                dv_ptr + gi[:, None] * dv_gs + ki[None, :] * dv_rs,
                dv.to(dv_ptr.dtype.element_ty),
                mask=m,
            )
        else:
            m = (ki < NK)[:, None] & gm[None, :]
            x = tl.load(
                v_ptr + ki[:, None] * v_rs + gi[None, :] * v_gs, mask=m, other=0.0
            ).to(tl.float32)
            gr = tl.load(
                grad_ptr + ki[:, None] * gr_rs + gi[None, :] * gr_gs, mask=m, other=0.0
            ).to(tl.float32)
            dv = gv[None, :] * gr * inv[None, :] - x * vsi[None, :]
            tl.store(
                dv_ptr + ki[:, None] * dv_rs + gi[None, :] * dv_gs,
                dv.to(dv_ptr.dtype.element_ty),
                mask=m,
            )


@triton.jit
def _wn_bwd_final_c(
    scal_ptr,
    dg_ptr,
    ss_ptr,
    s_ptr,
    g_ptr,
    NG,
    g_s,
    dg_s,
    BLOCK: tl.constexpr,
):
    off = tl.arange(0, BLOCK)
    mask = off < NG
    tot_ss = tl.sum(tl.load(ss_ptr + off, mask=mask, other=0.0))
    sv = tl.load(s_ptr + off, mask=mask, other=0.0)
    inv = 1.0 / tl.sqrt(tot_ss)
    tl.store(dg_ptr + off * dg_s, (sv * inv).to(dg_ptr.dtype.element_ty), mask=mask)
    gv = tl.load(g_ptr + off * g_s, mask=mask, other=0.0).to(tl.float32)
    tl.store(scal_ptr + 0, tot_ss)
    tl.store(scal_ptr + 1, tl.sum(sv * gv))


# --------------------------------------------------------------------- host --


def _np2(x):
    return 1 << (x - 1).bit_length() if x > 1 else 1


def _chan_stride(t):
    """Flat stride of the (keepdim) per-channel tensor ``t``."""
    if t.is_contiguous():
        return 1
    idx = 0
    for i in range(t.dim()):
        if t.shape[i] != 1:
            idx = i
    return t.stride(idx)


def _plan_first(t):
    """Reduce every axis but axis 0: (NG, NK, kept_stride, reduce_stride)."""
    ng = t.shape[0]
    nk = t.numel() // ng
    if t.is_contiguous():
        return ng, nk, nk, (1 if t.dim() > 1 else 0)
    r = t.dim()
    gs = t.stride(0)
    rs = t.stride(r - 1) if r > 1 else 0
    step = rs
    for i in range(r - 2, 0, -1):
        if t.stride(i) != step * t.shape[i + 1]:
            raise NotImplementedError("unsupported layout for reduction over dim 0")
        step *= t.shape[i + 1]
    return ng, nk, gs, rs


def _plan_last(t):
    """Reduce every axis but the last: (NG, NK, kept_stride, reduce_stride)."""
    r = t.dim()
    ng = t.shape[r - 1]
    nk = t.numel() // ng
    if t.is_contiguous():
        return ng, nk, 1, ng
    gs = t.stride(r - 1)
    rs = t.stride(r - 2)
    step = rs
    for i in range(r - 3, -1, -1):
        if t.stride(i) != step * t.shape[i + 1]:
            raise NotImplementedError(
                "unsupported layout for reduction over the last dim"
            )
        step *= t.shape[i + 1]
    return ng, nk, gs, rs


def _pick_bg(ng, bk):
    cap = _TILE_CAP // bk
    if cap < 1:
        cap = 1
    bg = _np2((ng + 511) // 512)
    if bg > cap:
        bg = cap
    floor = 256 // bk
    if floor < 1:
        floor = 1
    if floor > cap:
        floor = cap
    if bg < floor:
        bg = floor
    return bg


def _bwd_bg(ng, bk):
    bg = _np2(ng)
    cap = _TILE_CAP // bk
    if cap < 1:
        cap = 1
    return bg if bg < cap else cap


def _warps(tile):
    if tile <= 512:
        return 1
    if tile <= 1024:
        return 2
    if tile <= 2048:
        return 4
    return 8


def _split(nk):
    if nk >= 2048:
        s = 32
    elif nk >= 1024:
        s = 16
    else:
        s = nk // 64
        if s < 1:
            s = 1
        elif s > 16:
            s = 16
    rows_per = -(-nk // s)
    return -(-nk // rows_per), rows_per


def _mk_first(ng, nk, v_gs, v_rs, g_s):
    bk = _np2(nk)
    if bk <= _TILE_CAP:
        bg = _pick_bg(ng, bk)
        even = (ng % bg == 0) and (nk == bk)
        grid = (-(-ng // bg),)
        nw = _warps(bg * bk)

        def launch(out, v, g):
            _wn_tile[grid](
                out, v, g, ng, nk, v_gs, v_rs, g_s, bg, bk, 1, even, num_warps=nw
            )

        return launch
    bk = 1024
    bg = _pick_bg(ng, bk)
    grid = (-(-ng // bg),)
    nw = _warps(bg * bk)

    def launch(out, v, g):
        _wn_loop[grid](out, v, g, ng, nk, v_gs, v_rs, g_s, bg, bk, num_warps=nw)

    return launch


def _mk_last(ng, nk, v_gs, v_rs, g_s, dev):
    bkp = _np2(nk)
    bgp = _np2(ng)
    if bkp * bgp <= _TILE_CAP:
        even = (ng == bgp) and (nk == bkp)
        nw = _warps(bgp * bkp)

        def launch(out, v, g):
            _wn_tile[(1,)](
                out, v, g, ng, nk, v_gs, v_rs, g_s, bgp, bkp, 0, even, num_warps=nw
            )

        return launch
    ns, rows_per = _split(nk)
    bn = _INNER_G if ng > _INNER_G else _np2(ng)
    bm = _APPLY_K if nk >= _APPLY_K else _np2(nk)
    nslots = _np2(ns)
    g1 = (ns, -(-ng // bn))
    g2 = (-(-nk // bm), -(-ng // bn))
    nw1 = _warps(bn * _INNER_K)
    nw2 = _warps(bn * bm)

    def launch(out, v, g):
        part = torch.empty(ns * ng, dtype=torch.float32, device=dev)
        _wn_partial[g1](
            part, v, ng, nk, v_gs, v_rs, rows_per, bn, _INNER_K, num_warps=nw1
        )
        _wn_apply[g2](
            out,
            v,
            g,
            part,
            ng,
            nk,
            ns,
            v_gs,
            v_rs,
            g_s,
            ng,
            1,
            bn,
            bm,
            nslots,
            num_warps=nw2,
        )

    return launch


def _mk_global(ng, nk, v_gs, v_rs, g_s, dev):
    ns, rows_per = _split(nk)
    bn = _INNER_G if ng > _INNER_G else _np2(ng)
    bm = _APPLY_K if nk >= _APPLY_K else _np2(nk)
    g1 = (ns, -(-ng // bn))
    g2 = (-(-nk // bm), -(-ng // bn))
    nw1 = _warps(bn * _INNER_K)
    nw2 = _warps(bn * bm)

    def launch(out, v, g):
        ss = torch.zeros(1, dtype=torch.float32, device=dev)
        _wn_global[g1](ss, v, ng, nk, v_gs, v_rs, rows_per, bn, _INNER_K, num_warps=nw1)
        _wn_apply[g2](
            out,
            v,
            g,
            ss,
            ng,
            nk,
            1,
            v_gs,
            v_rs,
            g_s,
            1,
            0,
            bn,
            bm,
            1,
            num_warps=nw2,
        )

    return launch


def _make_forward(v, g, dim):
    """Build the launch closure for this metadata signature (metadata only)."""
    r = v.dim()
    g_s = _chan_stride(g)
    dev = v.device
    if dim == 0:
        ng, nk, v_gs, v_rs = _plan_first(v)
        return _mk_first(ng, nk, v_gs, v_rs, g_s)
    if dim == r - 1:
        ng, nk, v_gs, v_rs = _plan_last(v)
        return _mk_last(ng, nk, v_gs, v_rs, g_s, dev)
    kept = 0
    for i in range(g.dim()):
        if g.shape[i] != 1:
            kept = i
    if kept == 0:
        ng, nk, v_gs, v_rs = _plan_first(v)
    elif kept == r - 1:
        ng, nk, v_gs, v_rs = _plan_last(v)
    else:
        raise NotImplementedError("g must be broadcast along the first or last axis")
    return _mk_global(ng, nk, v_gs, v_rs, g_s, dev)


_LAUNCH_CACHE = {}


def _run_forward(v, g, dim):
    key = (v.shape, v.stride(), v.dtype, dim, g.shape, g.stride())
    launch = _LAUNCH_CACHE.get(key)
    if launch is None:
        launch = _make_forward(v, g, dim)
        _LAUNCH_CACHE[key] = launch
    out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    launch(out, v, g)
    return out


def _backward(v, g, dim, grad):
    r = v.dim()
    dv = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    dg = torch.empty(g.shape, dtype=g.dtype, device=g.device)
    dev = v.device
    g_s = _chan_stride(g)
    dg_s = _chan_stride(dg)
    global_norm = not (dim == 0 or dim == r - 1)
    if global_norm:
        kept = 0
        for i in range(g.dim()):
            if g.shape[i] != 1:
                kept = i
        kept = 0 if kept == 0 else (r - 1 if r > 1 else 0)
    else:
        kept = dim
    if kept == 0:
        ng, nk, v_gs, v_rs = _plan_first(v)
        gr_gs, gr_rs = _plan_first(grad)[2:]
        dv_gs, dv_rs = _plan_first(dv)[2:]
        red_axis = 1
    else:
        ng, nk, v_gs, v_rs = _plan_last(v)
        gr_gs, gr_rs = _plan_last(grad)[2:]
        dv_gs, dv_rs = _plan_last(dv)[2:]
        red_axis = 0
    bk = 256
    bg = _bwd_bg(ng, bk)
    ssb = torch.empty(ng, dtype=torch.float32, device=dev)
    sb = torch.empty(ng, dtype=torch.float32, device=dev)
    grid = (-(-ng // bg),)
    _wn_bwd_stats[grid](
        ssb,
        sb,
        dg,
        v,
        grad,
        ng,
        nk,
        v_gs,
        v_rs,
        gr_gs,
        gr_rs,
        dg_s,
        bg,
        bk,
        red_axis,
        not global_norm,
        num_warps=_warps(bg * bk),
    )
    if global_norm:
        blk = min(65536, _np2(ng))
        scal = torch.empty(2, dtype=torch.float32, device=dev)
        _wn_bwd_final_c[(1,)](
            scal, dg, ssb, sb, g, ng, g_s, dg_s, blk, num_warps=_warps(blk)
        )
        _wn_bwd_dv[grid](
            dv,
            v,
            grad,
            g,
            scal,
            scal,
            ng,
            nk,
            v_gs,
            v_rs,
            gr_gs,
            gr_rs,
            dv_gs,
            dv_rs,
            g_s,
            0,
            0,
            1,
            bg,
            bk,
            red_axis,
            True,
            num_warps=_warps(bg * bk),
        )
    else:
        _wn_bwd_dv[grid](
            dv,
            v,
            grad,
            g,
            ssb,
            sb,
            ng,
            nk,
            v_gs,
            v_rs,
            gr_gs,
            gr_rs,
            dv_gs,
            dv_rs,
            g_s,
            1,
            0,
            0,
            bg,
            bk,
            red_axis,
            False,
            num_warps=_warps(bg * bk),
        )
    return dv, dg


class _WeightNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, g, dim):
        out = _run_forward(v, g, dim)
        ctx.save_for_backward(v, g)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, grad):
        v, g = ctx.saved_tensors
        dv, dg = _backward(v, g, ctx.dim, grad)
        return dv, dg, None


def _weight_norm(v, g, dim=0):
    logger.debug("GEMS_HYGON _WEIGHT_NORM")
    if not isinstance(v, torch.Tensor):
        raise TypeError("v must be a torch.Tensor")
    if v.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"_weight_norm does not support {v.dtype}")
    if v.dim() == 0:
        raise RuntimeError("_weight_norm requires a tensor with at least 1 dim")
    if g.numel() != v.shape[dim]:
        raise RuntimeError("g must have one element per index of the reduced dim")
    if (v.requires_grad or g.requires_grad) and torch.is_grad_enabled():
        return _WeightNorm.apply(v, g, dim)
    return _run_forward(v, g, dim)
