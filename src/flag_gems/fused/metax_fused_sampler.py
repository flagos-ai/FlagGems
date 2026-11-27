# SPDX-License-Identifier: Apache-2.0
"""Zero-sync BF16 histogram + inverse-CDF top-p categorical sampler for MetaX Wave64 GPUs."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

KEY_COUNT = tl.constexpr(16384)
MIN_MAG = tl.constexpr(96 * 128)  # BF16 exponent/mantissa for 2**-31
MAG_SPAN = tl.constexpr(48 * 128)  # Covers [2**-31, 2**17)
NINF_ORD16 = tl.constexpr(0x007F)
_STEP_SEED = (
    int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF
)  # ord16(-inf) where u16(-inf) == 0xFF80


@triton.jit
def _u16_to_ord16(u16_raw):
    """Order all non-NaN BF16 values monotonically (-0.0 normalized to +0.0)."""
    u16 = u16_raw.to(tl.int32) & 0xFFFF
    u16 = tl.where(u16 == 0x8000, 0, u16)
    neg = (u16 >> 15) != 0
    return tl.where(neg, (~u16) & 0xFFFF, u16 ^ 0x8000)


@triton.jit
def _ord16_to_f32(ord16):
    u16 = tl.where(ord16 < 0x8000, (~ord16) & 0xFFFF, ord16 ^ 0x8000)
    return (u16.to(tl.uint32) << 16).to(tl.float32, bitcast=True)


@triton.jit
def _u16_to_f32(u16_raw):
    u32 = (u16_raw.to(tl.int32) & 0xFFFF).to(tl.uint32) << 16
    return u32.to(tl.float32, bitcast=True)


@triton.jit
def _u16_to_hist_key(u16_raw):
    u16 = u16_raw.to(tl.int32) & 0xFFFF
    min_m = tl.full((), MIN_MAG, tl.int32)
    span = tl.full((), MAG_SPAN, tl.int32)
    mag = u16 & 0x7FFF
    neg = (u16 >> 15) != 0
    is_ninf = u16 == 0xFF80
    is_nan_or_pinf = mag >= 0x7F80
    is_zero = mag == 0
    in_span = (mag >= min_m) & (mag < min_m + span)
    valid = (~is_nan_or_pinf) & (is_ninf | is_zero | in_span)
    delta = mag - min_m
    key = tl.where(
        is_ninf,
        0,
        tl.where(
            is_zero,
            span + 1,
            tl.where(neg, span - delta, delta + span + 2),
        ),
    )
    return tl.where(valid, key, 0), valid


@triton.jit
def _hist_key_to_ord16(key):
    min_m = tl.full((), MIN_MAG, tl.int32)
    span = tl.full((), MAG_SPAN, tl.int32)
    neg = key <= span
    delta = tl.where(neg, span - key, key - (span + 2))
    u16 = tl.where(neg, 0x8000, 0) | (delta + min_m)
    u16 = tl.where(key == 0, 0xFF80, tl.where(key == span + 1, 0, u16))
    return _u16_to_ord16(u16)


@triton.jit
def _k1_hist_i16(LogitsI16, Counts, Rare, V: tl.constexpr, RS, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    m = col < V
    u16 = tl.load(LogitsI16 + row * RS + col, mask=m, other=-128)
    key, valid = _u16_to_hist_key(u16)
    if tl.sum((m & (~valid)).to(tl.int32), 0) != 0:
        tl.atomic_or(Rare + row, 1, sem="relaxed")
    tl.atomic_add(Counts + row * KEY_COUNT + key, 1, mask=m & valid, sem="relaxed")


@triton.jit
def _k2_fast_pivot_i16(Counts, Rare, Temp, TopP, OutPivOrd16, OutRemEq, OutMx, OutWPiv):
    row = tl.program_id(0)
    if tl.load(Rare + row) != 0:
        return
    inv_t = 1.0 / tl.load(Temp + row)
    p_val = tl.load(TopP + row)
    ninf_o = tl.full((), NINF_ORD16, tl.int32)
    c_base = Counts + row * KEY_COUNT

    max_ck = -1
    for blk in tl.static_range(3, -1, -1):
        offs = blk * 4096 + tl.arange(0, 4096)
        cnt = tl.load(c_base + offs)
        if tl.max(cnt, 0) > 0:
            cand = tl.max(tl.where((cnt > 0) & (offs > 0), offs, -1), 0)
            max_ck = tl.maximum(max_ck, cand)

    if max_ck < 0:
        tl.store(OutPivOrd16 + row, -1)
        tl.store(OutRemEq + row, 0)
        tl.store(OutMx + row, 0.0)
        tl.store(OutWPiv + row, 0.0)
        return

    max_ord = _hist_key_to_ord16(max_ck)
    mx = _ord16_to_f32(max_ord) * inv_t
    z_sum = 0.0
    for blk in tl.static_range(0, 4):
        offs = blk * 4096 + tl.arange(0, 4096)
        cnt = tl.load(c_base + offs)
        if tl.max(cnt, 0) > 0:
            valid = (cnt > 0) & (offs > 0)
            s = _ord16_to_f32(_hist_key_to_ord16(offs)) * inv_t
            w = tl.where(valid, cnt.to(tl.float32) * tl.exp(s - mx), 0.0)
            z_sum += tl.sum(w, 0)

    cutoff = (1.0 - p_val) * z_sum
    if (p_val >= 1.0) | (cutoff <= 0.0):
        tl.store(OutPivOrd16 + row, ninf_o + 1)
        tl.store(OutRemEq + row, 0)
        tl.store(OutMx + row, mx)
        tl.store(OutWPiv + row, 0.0)
        return

    running = 0.0
    piv_ord = ninf_o + 1
    rem_eq = 0
    w_piv = 0.0
    found = 0
    for blk in tl.static_range(0, 4):
        if found == 0:
            offs = blk * 4096 + tl.arange(0, 4096)
            cnt = tl.load(c_base + offs)
            if tl.max(cnt, 0) > 0:
                valid = (cnt > 0) & (offs > 0)
                ord_v = _hist_key_to_ord16(offs)
                s = _ord16_to_f32(ord_v) * inv_t
                wp_vec = tl.where(valid, tl.exp(s - mx), 0.0)
                w_bin = cnt.to(tl.float32) * wp_vec
                t_mass = tl.sum(w_bin, 0)
                if (running + t_mass > cutoff) | (blk == (max_ck // 4096)):
                    cw = running + tl.cumsum(w_bin, 0)
                    hit = valid & ((cw > cutoff) | (offs == max_ck))
                    if tl.sum(hit.to(tl.int32), 0) > 0:
                        ck = tl.min(tl.where(hit, offs, KEY_COUNT), 0)
                        piv_ord = _hist_key_to_ord16(ck)
                        mass_below = running + tl.sum(
                            tl.where(offs < ck, w_bin, 0.0), 0
                        )
                        piv_cnt = tl.sum(tl.where(offs == ck, cnt, 0), 0)
                        wp = tl.exp(_ord16_to_f32(piv_ord) * inv_t - mx)
                        w_piv = wp
                        rem = ((cutoff - mass_below) / wp).to(tl.int32)
                        rem = tl.maximum(0, tl.minimum(rem, piv_cnt))
                        rem = tl.where(
                            mass_below + rem.to(tl.float32) * wp <= cutoff,
                            rem,
                            tl.maximum(0, rem - 1),
                        )
                        rem = tl.where(
                            mass_below + (rem + 1).to(tl.float32) * wp <= cutoff,
                            tl.minimum(piv_cnt, rem + 1),
                            rem,
                        )
                        if ck == max_ck:
                            rem = tl.minimum(rem, piv_cnt - 1)
                        rem_eq = rem
                        found = 1
                running += t_mass

    tl.store(OutPivOrd16 + row, piv_ord)
    tl.store(OutRemEq + row, rem_eq)
    tl.store(OutMx + row, mx)
    tl.store(OutWPiv + row, w_piv)


@triton.jit
def _k2_rare_pivot_i16(
    LogitsI16,
    Rare,
    Temp,
    TopP,
    OutPivOrd16,
    OutRemEq,
    OutMx,
    OutWPiv,
    V: tl.constexpr,
    RS,
):
    """Exact 4-nibble 16-bit quantile resolver for rows outside the compact normal span."""
    row = tl.program_id(0)
    if tl.load(Rare + row) == 0:
        return
    inv_t = 1.0 / tl.load(Temp + row)
    p_val = tl.load(TopP + row)
    ninf_o = tl.full((), NINF_ORD16, tl.int32)
    row_ptr = LogitsI16 + row * RS

    v_maxo = tl.full((2048,), -1, tl.int32)
    v_nan = tl.zeros((2048,), tl.int32)
    v_pinf = tl.zeros((2048,), tl.int32)
    for base in range(0, V, 2048):
        cols = base + tl.arange(0, 2048)
        m = cols < V
        u16 = tl.load(row_ptr + cols, mask=m, other=-128)
        mag = u16.to(tl.int32) & 0x7FFF
        u16_clean = u16.to(tl.int32) & 0xFFFF
        v_nan += (m & (mag > 0x7F80)).to(tl.int32)
        v_pinf += (m & (u16_clean == 0x7F80)).to(tl.int32)
        o16 = _u16_to_ord16(u16)
        valid_f = m & (mag < 0x7F80) & (o16 > ninf_o)
        v_maxo = tl.maximum(v_maxo, tl.where(valid_f, o16, -1))

    if (tl.sum(v_nan, 0) > 0) | (tl.sum(v_pinf, 0) > 0):
        # Match PyTorch argmax fallback on NaN/+inf rows: mark -2 for special scan.
        tl.store(OutPivOrd16 + row, -2)
        tl.store(OutRemEq + row, 0)
        tl.store(OutMx + row, 0.0)
        tl.store(OutWPiv + row, 0.0)
        return

    max_o16 = tl.max(v_maxo, 0)
    if max_o16 < 0:
        tl.store(OutPivOrd16 + row, -1)
        tl.store(OutRemEq + row, 0)
        tl.store(OutMx + row, 0.0)
        tl.store(OutWPiv + row, 0.0)
        return

    mx = _ord16_to_f32(max_o16) * inv_t
    v_z = tl.zeros((2048,), tl.float32)
    for base in range(0, V, 2048):
        cols = base + tl.arange(0, 2048)
        m = cols < V
        u16 = tl.load(row_ptr + cols, mask=m, other=-128)
        mag = u16.to(tl.int32) & 0x7FFF
        o16 = _u16_to_ord16(u16)
        valid_f = m & (mag < 0x7F80) & (o16 > ninf_o)
        v_z += tl.where(valid_f, tl.exp(_u16_to_f32(u16) * inv_t - mx), 0.0)
    z_sum = tl.sum(v_z, 0)
    cutoff = (1.0 - p_val) * z_sum
    if (p_val >= 1.0) | (cutoff <= 0.0):
        tl.store(OutPivOrd16 + row, ninf_o + 1)
        tl.store(OutRemEq + row, 0)
        tl.store(OutMx + row, mx)
        tl.store(OutWPiv + row, 0.0)
        return

    lo_o = 0
    mass_below = 0.0
    for shift in tl.static_range(12, -1, -4):
        hi_o = lo_o + (16 << shift)
        vb0 = tl.zeros((2048,), tl.float32)
        vb1 = tl.zeros((2048,), tl.float32)
        vb2 = tl.zeros((2048,), tl.float32)
        vb3 = tl.zeros((2048,), tl.float32)
        vb4 = tl.zeros((2048,), tl.float32)
        vb5 = tl.zeros((2048,), tl.float32)
        vb6 = tl.zeros((2048,), tl.float32)
        vb7 = tl.zeros((2048,), tl.float32)
        vb8 = tl.zeros((2048,), tl.float32)
        vb9 = tl.zeros((2048,), tl.float32)
        vb10 = tl.zeros((2048,), tl.float32)
        vb11 = tl.zeros((2048,), tl.float32)
        vb12 = tl.zeros((2048,), tl.float32)
        vb13 = tl.zeros((2048,), tl.float32)
        vb14 = tl.zeros((2048,), tl.float32)
        for base in range(0, V, 2048):
            cols = base + tl.arange(0, 2048)
            m = cols < V
            u16 = tl.load(row_ptr + cols, mask=m, other=-128)
            mag = u16.to(tl.int32) & 0x7FFF
            o16 = _u16_to_ord16(u16)
            in_w = m & (mag < 0x7F80) & (o16 > ninf_o) & (o16 >= lo_o) & (o16 < hi_o)
            w = tl.where(in_w, tl.exp(_u16_to_f32(u16) * inv_t - mx), 0.0)
            nib = (o16 - lo_o) >> shift
            vb0 += tl.where(nib == 0, w, 0.0)
            vb1 += tl.where(nib == 1, w, 0.0)
            vb2 += tl.where(nib == 2, w, 0.0)
            vb3 += tl.where(nib == 3, w, 0.0)
            vb4 += tl.where(nib == 4, w, 0.0)
            vb5 += tl.where(nib == 5, w, 0.0)
            vb6 += tl.where(nib == 6, w, 0.0)
            vb7 += tl.where(nib == 7, w, 0.0)
            vb8 += tl.where(nib == 8, w, 0.0)
            vb9 += tl.where(nib == 9, w, 0.0)
            vb10 += tl.where(nib == 10, w, 0.0)
            vb11 += tl.where(nib == 11, w, 0.0)
            vb12 += tl.where(nib == 12, w, 0.0)
            vb13 += tl.where(nib == 13, w, 0.0)
            vb14 += tl.where(nib == 14, w, 0.0)
        b0 = tl.sum(vb0, 0)
        b1 = tl.sum(vb1, 0)
        b2 = tl.sum(vb2, 0)
        b3 = tl.sum(vb3, 0)
        b4 = tl.sum(vb4, 0)
        b5 = tl.sum(vb5, 0)
        b6 = tl.sum(vb6, 0)
        b7 = tl.sum(vb7, 0)
        b8 = tl.sum(vb8, 0)
        b9 = tl.sum(vb9, 0)
        b10 = tl.sum(vb10, 0)
        b11 = tl.sum(vb11, 0)
        b12 = tl.sum(vb12, 0)
        b13 = tl.sum(vb13, 0)
        b14 = tl.sum(vb14, 0)
        chosen = 0
        acc = mass_below
        if acc + b0 <= cutoff:
            acc += b0
            chosen = 1
            if acc + b1 <= cutoff:
                acc += b1
                chosen = 2
                if acc + b2 <= cutoff:
                    acc += b2
                    chosen = 3
                    if acc + b3 <= cutoff:
                        acc += b3
                        chosen = 4
                        if acc + b4 <= cutoff:
                            acc += b4
                            chosen = 5
                            if acc + b5 <= cutoff:
                                acc += b5
                                chosen = 6
                                if acc + b6 <= cutoff:
                                    acc += b6
                                    chosen = 7
                                    if acc + b7 <= cutoff:
                                        acc += b7
                                        chosen = 8
                                        if acc + b8 <= cutoff:
                                            acc += b8
                                            chosen = 9
                                            if acc + b9 <= cutoff:
                                                acc += b9
                                                chosen = 10
                                                if acc + b10 <= cutoff:
                                                    acc += b10
                                                    chosen = 11
                                                    if acc + b11 <= cutoff:
                                                        acc += b11
                                                        chosen = 12
                                                        if acc + b12 <= cutoff:
                                                            acc += b12
                                                            chosen = 13
                                                            if acc + b13 <= cutoff:
                                                                acc += b13
                                                                chosen = 14
                                                                if acc + b14 <= cutoff:
                                                                    acc += b14
                                                                    chosen = 15
        mass_below = acc
        lo_o = lo_o + (chosen << shift)

    piv_ord = tl.minimum(lo_o, max_o16)
    wp = tl.exp(_ord16_to_f32(piv_ord) * inv_t - mx)
    v_cnt = tl.zeros((2048,), tl.int32)
    for base in range(0, V, 2048):
        cols = base + tl.arange(0, 2048)
        m = cols < V
        u16 = tl.load(row_ptr + cols, mask=m, other=-128)
        mag = u16.to(tl.int32) & 0x7FFF
        v_cnt += (m & (mag < 0x7F80) & (_u16_to_ord16(u16) == piv_ord)).to(tl.int32)
    piv_cnt = tl.sum(v_cnt, 0)
    rem_eq = 0
    if (wp > 0.0) & (piv_cnt > 0):
        rem = ((cutoff - mass_below) / wp).to(tl.int32)
        rem = tl.maximum(0, tl.minimum(rem, piv_cnt))
        rem = tl.where(
            mass_below + rem.to(tl.float32) * wp <= cutoff, rem, tl.maximum(0, rem - 1)
        )
        rem = tl.where(
            mass_below + (rem + 1).to(tl.float32) * wp <= cutoff,
            tl.minimum(piv_cnt, rem + 1),
            rem,
        )
        if piv_ord == max_o16:
            rem = tl.minimum(rem, piv_cnt - 1)
        rem_eq = rem

    tl.store(OutPivOrd16 + row, piv_ord)
    tl.store(OutRemEq + row, rem_eq)
    tl.store(OutMx + row, mx)
    tl.store(OutWPiv + row, wp)


@triton.jit
def _k3a_tile_stats_ord16(
    LogitsI16,
    Temp,
    PivOrd16,
    Mx,
    TileGtMass,
    TileEqCnt,
    V: tl.constexpr,
    RS,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    pk = tl.load(PivOrd16 + row)
    if pk < 0:
        tl.store(TileGtMass + row * NT + tile, 0.0)
        tl.store(TileEqCnt + row * NT + tile, 0)
        return
    ninf_o = tl.full((), NINF_ORD16, tl.int32)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    m = col < V
    u16 = tl.load(LogitsI16 + row * RS + col, mask=m, other=-128)
    mag = u16.to(tl.int32) & 0x7FFF
    o16 = _u16_to_ord16(u16)
    valid = m & (mag < 0x7F80) & (o16 > ninf_o)
    eq = valid & (o16 == pk)
    tl.store(TileEqCnt + row * NT + tile, tl.sum(eq.to(tl.int32), 0))
    gt = valid & (o16 > pk)
    if tl.max(gt.to(tl.int32), 0) > 0:
        inv_t = 1.0 / tl.load(Temp + row)
        mx = tl.load(Mx + row)
        w_gt = tl.where(gt, tl.exp(_u16_to_f32(u16) * inv_t - mx), 0.0)
        tl.store(TileGtMass + row * NT + tile, tl.sum(w_gt, 0))
    else:
        tl.store(TileGtMass + row * NT + tile, 0.0)


@triton.jit
def _k3b_sample_ord16(
    LogitsI16,
    Temp,
    PivOrd16,
    RemEq,
    Mx,
    WPiv,
    StepSeed,
    TileGtMass,
    TileEqCnt,
    OutTok,
    V: tl.constexpr,
    RS,
    NT: tl.constexpr,
    PAD_NT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    pk = tl.load(PivOrd16 + row)
    if pk == -1:
        tl.store(OutTok + row, 0)
        return
    if pk == -2:
        # Degenerate NaN/+inf row: return first NaN/+inf column (consistent with softmax->NaN->argmax).
        first_bad = V
        for base in range(0, V, BLOCK):
            col = base + tl.arange(0, BLOCK)
            m = col < V
            u16 = tl.load(LogitsI16 + row * RS + col, mask=m, other=0)
            mag = u16.to(tl.int32) & 0x7FFF
            u16_clean = u16.to(tl.int32) & 0xFFFF
            is_deg = m & ((mag > 0x7F80) | (u16_clean == 0x7F80))
            first_bad = tl.minimum(first_bad, tl.min(tl.where(is_deg, col, V), 0))
        tl.store(OutTok + row, tl.where(first_bad < V, first_bad, 0).to(tl.int32))
        return

    ninf_o = tl.full((), NINF_ORD16, tl.int32)
    rem_ties = tl.load(RemEq + row)
    mx = tl.load(Mx + row)
    w_piv = tl.load(WPiv + row)
    t_idx = tl.arange(0, PAD_NT)
    t_mask = t_idx < NT
    eq_cnt = tl.load(TileEqCnt + row * NT + t_idx, mask=t_mask, other=0)
    gt_mass = tl.load(TileGtMass + row * NT + t_idx, mask=t_mask, other=0.0)

    cum_eq_inc = tl.cumsum(eq_cnt, 0)
    cum_eq_exc = cum_eq_inc - eq_cnt
    kept_eq = tl.maximum(0, cum_eq_inc - rem_ties) - tl.maximum(
        0, cum_eq_exc - rem_ties
    )
    tile_mass = gt_mass + kept_eq.to(tl.float32) * w_piv
    cum_mass_inc = tl.cumsum(tile_mass, 0)
    total_kept = tl.sum(tile_mass, 0)

    u = tl.rand(StepSeed, row)
    u = tl.minimum(tl.maximum(u, 0.0), 0.99999994)
    target = u * total_kept

    last_tile = tl.max(tl.where(t_mask & (tile_mass > 0.0), t_idx, 0), 0)
    win_hit = t_mask & (tile_mass > 0.0) & (cum_mass_inc > target)
    win_tile = tl.min(tl.where(win_hit, t_idx, NT), 0)
    win_tile = tl.where(win_tile < NT, win_tile, last_tile)

    mass_before = tl.sum(tl.where(t_idx < win_tile, tile_mass, 0.0), 0)
    ties_before = tl.sum(tl.where(t_idx < win_tile, eq_cnt, 0), 0)
    inv_t = 1.0 / tl.load(Temp + row)

    col = win_tile * BLOCK + tl.arange(0, BLOCK)
    m = col < V
    u16 = tl.load(LogitsI16 + row * RS + col, mask=m, other=-128)
    mag = u16.to(tl.int32) & 0x7FFF
    o16 = _u16_to_ord16(u16)
    valid = m & (mag < 0x7F80) & (o16 > ninf_o)
    eq = valid & (o16 == pk)
    tie_rank = ties_before + tl.cumsum(eq.to(tl.int32), 0)
    keep = valid & ((o16 > pk) | (eq & (tie_rank > rem_ties)))
    w = tl.where(keep, tl.exp(_u16_to_f32(u16) * inv_t - mx), 0.0)
    cw = mass_before + tl.cumsum(w, 0)
    last_col = tl.max(tl.where(keep, col, 0), 0)
    col_hit = keep & (cw > target)
    picked = tl.min(tl.where(col_hit, col, V), 0)
    picked = tl.where(picked < V, picked, last_col)
    tl.store(OutTok + row, picked.to(tl.int32))


def metax_fused_bf16_top_p_sample(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
) -> torch.Tensor:
    """Sample [B, 1] int32 token IDs directly from BF16 logits without FP32 full-vocab buffers or host sync."""
    batch, vocab = logits.shape
    device = logits.device
    if batch == 0:
        return torch.empty((0, 1), device=device, dtype=torch.int32)

    logits_i16 = logits.view(torch.int16)
    rs = logits_i16.stride(0)
    blk3 = 2048
    nt = triton.cdiv(vocab, blk3)
    pad_nt = triton.next_power_of_2(nt)

    global _STEP_SEED
    _STEP_SEED = (_STEP_SEED + 1000003) & 0x7FFFFFFF
    step_seed = _STEP_SEED
    with torch_device_fn.device(device):
        counts = torch.zeros((batch, KEY_COUNT), device=device, dtype=torch.int32)
        rare = torch.zeros(batch, device=device, dtype=torch.int32)
        piv_ord = torch.empty(batch, device=device, dtype=torch.int32)
        rem_eq = torch.empty(batch, device=device, dtype=torch.int32)
        mx = torch.empty(batch, device=device, dtype=torch.float32)
        w_piv = torch.empty(batch, device=device, dtype=torch.float32)
        gt_mass = torch.empty((batch, nt), device=device, dtype=torch.float32)
        eq_cnt = torch.empty((batch, nt), device=device, dtype=torch.int32)
        out_tokens = torch.empty((batch, 1), device=device, dtype=torch.int32)

        _k1_hist_i16[(batch, triton.cdiv(vocab, 1024))](
            logits_i16,
            counts,
            rare,
            vocab,
            rs,
            BLOCK=1024,
            num_warps=4,
        )
        _k2_fast_pivot_i16[(batch,)](
            counts,
            rare,
            temperature,
            top_p,
            piv_ord,
            rem_eq,
            mx,
            w_piv,
            num_warps=8,
        )
        _k2_rare_pivot_i16[(batch,)](
            logits_i16,
            rare,
            temperature,
            top_p,
            piv_ord,
            rem_eq,
            mx,
            w_piv,
            vocab,
            rs,
            num_warps=8,
        )
        _k3a_tile_stats_ord16[(batch, nt)](
            logits_i16,
            temperature,
            piv_ord,
            mx,
            gt_mass,
            eq_cnt,
            vocab,
            rs,
            NT=nt,
            BLOCK=blk3,
            num_warps=4,
        )
        _k3b_sample_ord16[(batch,)](
            logits_i16,
            temperature,
            piv_ord,
            rem_eq,
            mx,
            w_piv,
            step_seed,
            gt_mass,
            eq_cnt,
            out_tokens,
            vocab,
            rs,
            NT=nt,
            PAD_NT=pad_nt,
            BLOCK=blk3,
            num_warps=4,
        )
    return out_tokens
