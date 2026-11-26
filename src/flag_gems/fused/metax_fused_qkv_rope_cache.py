# SPDX-License-Identifier: Apache-2.0
"""Fused NeoX rotary embedding + paged FlashAttention KV-cache write (+ Split-K=2 INT32 epilogue,
direct contig KV, and V7 Dual-Buffer in-place INT8 KV) for MetaX Wave64.

V7 Dual-Buffer In-Place INT8 KV Layout (8192 bytes/block, 0 extra VRAM):
  key_cache   block `b` (8192 bytes):
    bytes [0..4095]   : K_i8     [Hkv=2, T=16, D=128] int8    (offset: kvh*2048 + t*128 + d)
    bytes [4096..4223]: K_sc/V_sc per-head adjacent float16 (same 64B cache line per head!):
                        K_sc: f16 offset 2048 + kvh*32 + t
                        V_sc: f16 offset 2048 + kvh*32 + 16 + t
  value_cache block `b` (8192 bytes):
    bytes [0..4095]   : V_i8_t   [Hkv=2, D=128, T=16] int8    (transposed, finalized when off==15)
    bytes [4096..8191]: V_i8_row [Hkv=2, T=16, D=128] int8    (row-major, 100% coalesced 128B write!)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _fused_qkv_rope_and_cache_4w_kernel(
    Q,
    K,
    V,
    Pos,
    CosSin,
    QOut,
    KCache,
    VCache,
    SlotMap,
    KOut,
    VOut,
    n_slots,
    max_pos,
    num_blocks,
    stride_q_m,
    stride_k_m,
    stride_v_m,
    stride_pos,
    stride_cs_m,
    stride_slot,
    stride_kc_b,
    stride_kc_t,
    stride_kc_h,
    stride_vc_b,
    stride_vc_t,
    stride_vc_h,
    SAVE_CONTIG_KV: tl.constexpr,
):
    tok = tl.program_id(0)
    raw_pos = tl.load(Pos + tok * stride_pos)
    pos = tl.minimum(tl.maximum(raw_pos, 0), max_pos - 1)

    idx = tl.arange(0, 256)
    h_sub = idx // 64
    d = idx % 64

    cs_base = CosSin + pos * stride_cs_m
    cos = tl.load(cs_base + d)
    sin = tl.load(cs_base + 64 + d)

    q_row = Q + tok * stride_q_m
    qo_row = QOut + tok * 2048
    cf = cos.to(cos.dtype).to(tl.float32)
    sf = sin.to(sin.dtype).to(tl.float32)
    for g in tl.static_range(0, 4):
        h = g * 4 + h_sub
        qh = q_row + h * 128
        x1 = tl.load(qh + d)
        x2 = tl.load(qh + 64 + d)
        dt = x1.dtype
        x1f = x1.to(tl.float32)
        x2f = x2.to(tl.float32)
        o1 = ((x1f * cf).to(dt).to(tl.float32) - (x2f * sf).to(dt).to(tl.float32)).to(
            dt
        )
        o2 = ((x2f * cf).to(dt).to(tl.float32) + (x1f * sf).to(dt).to(tl.float32)).to(
            dt
        )
        qo = qo_row + h * 128
        tl.store(qo + d, o1)
        tl.store(qo + 64 + d, o2)

    is_k = h_sub < 2
    kvh = h_sub % 2
    k_ptr = K + tok * stride_k_m + kvh * 128
    v_ptr = V + tok * stride_v_m + kvh * 128
    src_ptr = tl.where(is_k, k_ptr, v_ptr)

    v1 = tl.load(src_ptr + d)
    v2 = tl.load(src_ptr + 64 + d)
    kdt = v1.dtype
    v1f = v1.to(tl.float32)
    v2f = v2.to(tl.float32)
    k1 = ((v1f * cf).to(kdt).to(tl.float32) - (v2f * sf).to(kdt).to(tl.float32)).to(kdt)
    k2 = ((v2f * cf).to(kdt).to(tl.float32) + (v1f * sf).to(kdt).to(tl.float32)).to(kdt)
    out1 = tl.where(is_k, k1, v1)
    out2 = tl.where(is_k, k2, v2)

    if SAVE_CONTIG_KV:
        ko_ptr = KOut + tok * 256 + kvh * 128
        vo_ptr = VOut + tok * 256 + kvh * 128
        cdst = tl.where(is_k, ko_ptr, vo_ptr)
        tl.store(cdst + d, out1)
        tl.store(cdst + 64 + d, out2)

    if tok >= n_slots:
        return
    slot = tl.load(SlotMap + tok * stride_slot)
    if slot < 0:
        return
    blk = slot // 16
    if blk >= num_blocks:
        return
    off = slot % 16

    kc_ptr = KCache + blk * stride_kc_b + off * stride_kc_t + kvh * stride_kc_h
    vc_ptr = VCache + blk * stride_vc_b + off * stride_vc_t + kvh * stride_vc_h
    dst_ptr = tl.where(is_k, kc_ptr, vc_ptr)
    tl.store(dst_ptr + d, out1)
    tl.store(dst_ptr + 64 + d, out2)


@triton.jit
def _fused_qkv_splitk2_rope_and_cache_4w_kernel(
    PartQKV,
    XScale,
    WScale,
    Pos,
    CosSin,
    QOut,
    KCache,
    VCache,
    SlotMap,
    M,
    n_slots,
    max_pos,
    num_blocks,
    stride_pos,
    stride_cs_m,
    stride_slot,
    stride_kc_b,
    stride_kc_t,
    stride_kc_h,
    stride_vc_b,
    stride_vc_t,
    stride_vc_h,
):
    """Fused QKV Split-K=2 INT32 reduce + dequant + NeoX RoPE + paged KV cache write."""
    tok = tl.program_id(0)
    raw_pos = tl.load(Pos + tok * stride_pos)
    pos = tl.minimum(tl.maximum(raw_pos, 0), max_pos - 1)
    sx = tl.load(XScale + tok)

    idx = tl.arange(0, 256)
    h_sub = idx // 64
    d = idx % 64

    cs_base = CosSin + pos * stride_cs_m
    cos = tl.load(cs_base + d)
    sin = tl.load(cs_base + 64 + d)
    cf = cos.to(cos.dtype).to(tl.float32)
    sf = sin.to(sin.dtype).to(tl.float32)

    mn = M * 2560
    p0_row = PartQKV + tok * 2560
    p1_row = p0_row + mn
    qo_row = QOut + tok * 2048

    for g in tl.static_range(0, 4):
        h = g * 4 + h_sub
        col1 = h * 128 + d
        col2 = col1 + 64
        x1_i32 = tl.load(p0_row + col1) + tl.load(p1_row + col1)
        x2_i32 = tl.load(p0_row + col2) + tl.load(p1_row + col2)
        sw1 = tl.load(WScale + col1)
        sw2 = tl.load(WScale + col2)
        x1 = (x1_i32.to(tl.float32) * (sx * sw1)).to(tl.bfloat16)
        x2 = (x2_i32.to(tl.float32) * (sx * sw2)).to(tl.bfloat16)
        x1f = x1.to(tl.float32)
        x2f = x2.to(tl.float32)
        o1 = (
            (x1f * cf).to(tl.bfloat16).to(tl.float32)
            - (x2f * sf).to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16)
        o2 = (
            (x2f * cf).to(tl.bfloat16).to(tl.float32)
            + (x1f * sf).to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16)
        qo = qo_row + h * 128
        tl.store(qo + d, o1)
        tl.store(qo + 64 + d, o2)

    is_k = h_sub < 2
    kvh = h_sub % 2
    kv_off = tl.where(is_k, 2048 + kvh * 128, 2304 + kvh * 128)
    kc1 = kv_off + d
    kc2 = kc1 + 64
    v1_i32 = tl.load(p0_row + kc1) + tl.load(p1_row + kc1)
    v2_i32 = tl.load(p0_row + kc2) + tl.load(p1_row + kc2)
    v1 = (v1_i32.to(tl.float32) * (sx * tl.load(WScale + kc1))).to(tl.bfloat16)
    v2 = (v2_i32.to(tl.float32) * (sx * tl.load(WScale + kc2))).to(tl.bfloat16)
    v1f = v1.to(tl.float32)
    v2f = v2.to(tl.float32)
    k1 = (
        (v1f * cf).to(tl.bfloat16).to(tl.float32)
        - (v2f * sf).to(tl.bfloat16).to(tl.float32)
    ).to(tl.bfloat16)
    k2 = (
        (v2f * cf).to(tl.bfloat16).to(tl.float32)
        + (v1f * sf).to(tl.bfloat16).to(tl.float32)
    ).to(tl.bfloat16)
    out1 = tl.where(is_k, k1, v1)
    out2 = tl.where(is_k, k2, v2)

    if tok >= n_slots:
        return
    slot = tl.load(SlotMap + tok * stride_slot)
    if slot < 0:
        return
    blk = slot // 16
    if blk >= num_blocks:
        return
    off = slot % 16

    kc_ptr = KCache + blk * stride_kc_b + off * stride_kc_t + kvh * stride_kc_h
    vc_ptr = VCache + blk * stride_vc_b + off * stride_vc_t + kvh * stride_vc_h
    dst_ptr = tl.where(is_k, kc_ptr, vc_ptr)
    tl.store(dst_ptr + d, out1)
    tl.store(dst_ptr + 64 + d, out2)


# ==============================================================================
# V7 Dual-Buffer In-Place 8192B/block INT8 KV Cache Fused RoPE + Cache Kernels
# ==============================================================================
@triton.jit
def _finalize_v_t_from_slots_kernel(
    V_cache_i8,
    SlotMap,
    n_slots,
    num_blocks,
    stride_slot,
    STRIDE_VB_I8: tl.constexpr,
    USE_I64: tl.constexpr,
):
    """Race-free block transpose V_i8_row -> V_i8_t for any block completed (slot % 16 == 15)."""
    tok = tl.program_id(0)
    kvh = tl.program_id(1)
    if tok >= n_slots:
        return
    slot = tl.load(SlotMap + tok * stride_slot)
    if slot < 0 or (slot % 16) != 15:
        return
    blk = slot // 16
    if blk >= num_blocks:
        return
    blk_idx = blk.to(tl.int64) if USE_I64 else blk
    v_row_base = V_cache_i8 + blk_idx * STRIDE_VB_I8 + 4096 + kvh * 2048
    v_t_base = V_cache_i8 + blk_idx * STRIDE_VB_I8 + kvh * 2048
    offs_t = tl.arange(0, 16)
    d = tl.arange(0, 64)
    lo = tl.load(v_row_base + offs_t[:, None] * 128 + d[None, :])
    tl.store(v_t_base + d[:, None] * 16 + offs_t[None, :], tl.trans(lo))
    hi = tl.load(v_row_base + offs_t[:, None] * 128 + 64 + d[None, :])
    tl.store(v_t_base + (64 + d[:, None]) * 16 + offs_t[None, :], tl.trans(hi))


@triton.jit
def _fused_qkv_rope_and_cache_i8_4w_kernel(
    Q,
    K,
    V,
    Pos,
    CosSin,
    QOut,
    QI8Out,
    QScOut,
    K_cache_i8,
    K_cache_f16,
    V_cache_i8,
    SlotMap,
    KOut,
    VOut,
    n_slots,
    max_pos,
    num_blocks,
    stride_q_m,
    stride_k_m,
    stride_v_m,
    stride_pos,
    stride_cs_m,
    stride_slot,
    STRIDE_KB_I8: tl.constexpr,
    STRIDE_KB_F16: tl.constexpr,
    STRIDE_VB_I8: tl.constexpr,
    SAVE_CONTIG_KV: tl.constexpr,
    SAVE_Q_I8: tl.constexpr,
    USE_I64: tl.constexpr,
):
    """NeoX RoPE + V7 coalesced INT8 K_i8 / V_i8_row / co-located K_sc+V_sc write (paired with
    _finalize_v_t_from_slots_kernel)."""
    tok = tl.program_id(0)
    raw_pos = tl.load(Pos + tok * stride_pos)
    pos = tl.minimum(tl.maximum(raw_pos, 0), max_pos - 1)

    h_sub = tl.arange(0, 4)
    d = tl.arange(0, 64)

    cs_base = CosSin + pos * stride_cs_m
    cf = tl.load(cs_base + d).to(tl.float32)  # [64]
    sf = tl.load(cs_base + 64 + d).to(tl.float32)  # [64]

    q_row = Q + tok * stride_q_m
    qo_row = QOut + tok * 2048
    qi_row = QI8Out + tok * 2048
    qs_row = QScOut + tok * 16

    for g in tl.static_range(0, 4):
        h = g * 4 + h_sub
        qh = q_row + h[:, None] * 128
        x1 = tl.load(qh + d[None, :])
        x2 = tl.load(qh + 64 + d[None, :])
        dt = x1.dtype
        x1f = x1.to(tl.float32)
        x2f = x2.to(tl.float32)
        o1 = (
            (x1f * cf[None, :]).to(dt).to(tl.float32)
            - (x2f * sf[None, :]).to(dt).to(tl.float32)
        ).to(dt)
        o2 = (
            (x2f * cf[None, :]).to(dt).to(tl.float32)
            + (x1f * sf[None, :]).to(dt).to(tl.float32)
        ).to(dt)
        qo = qo_row + h[:, None] * 128
        tl.store(qo + d[None, :], o1)
        tl.store(qo + 64 + d[None, :], o2)
        if SAVE_Q_I8:
            o1f = o1.to(tl.float32)
            o2f = o2.to(tl.float32)
            qpk = tl.maximum(tl.max(tl.abs(o1f), axis=1), tl.max(tl.abs(o2f), axis=1))
            qpk = tl.maximum(qpk, 1e-8)
            qsc = qpk * (1.0 / 127.0)
            qinv = 127.0 / qpk
            tl.store(
                qi_row + h[:, None] * 128 + d[None, :],
                tl.extra.cuda.libdevice.rint(o1f * qinv[:, None])
                .to(tl.int32)
                .to(tl.int8),
            )
            tl.store(
                qi_row + h[:, None] * 128 + 64 + d[None, :],
                tl.extra.cuda.libdevice.rint(o2f * qinv[:, None])
                .to(tl.int32)
                .to(tl.int8),
            )
            tl.store(qs_row + h, qsc)

    is_k = h_sub < 2
    kvh = h_sub % 2
    k_ptr = K + tok * stride_k_m + kvh[:, None] * 128
    v_ptr = V + tok * stride_v_m + kvh[:, None] * 128
    src_ptr = tl.where(is_k[:, None], k_ptr, v_ptr)

    v1 = tl.load(src_ptr + d[None, :])
    v2 = tl.load(src_ptr + 64 + d[None, :])
    kdt = v1.dtype
    v1f = v1.to(tl.float32)
    v2f = v2.to(tl.float32)
    k1 = (
        (v1f * cf[None, :]).to(kdt).to(tl.float32)
        - (v2f * sf[None, :]).to(kdt).to(tl.float32)
    ).to(kdt)
    k2 = (
        (v2f * cf[None, :]).to(kdt).to(tl.float32)
        + (v1f * sf[None, :]).to(kdt).to(tl.float32)
    ).to(kdt)
    out1 = tl.where(is_k[:, None], k1, v1)
    out2 = tl.where(is_k[:, None], k2, v2)

    if SAVE_CONTIG_KV:
        ko_ptr = KOut + tok * 256 + kvh[:, None] * 128
        vo_ptr = VOut + tok * 256 + kvh[:, None] * 128
        cdst = tl.where(is_k[:, None], ko_ptr, vo_ptr)
        tl.store(cdst + d[None, :], out1)
        tl.store(cdst + 64 + d[None, :], out2)

    if tok >= n_slots:
        return
    slot = tl.load(SlotMap + tok * stride_slot)
    if slot < 0:
        return
    blk = slot // 16
    if blk >= num_blocks:
        return
    off = slot % 16

    f1 = out1.to(tl.float32)
    f2 = out2.to(tl.float32)
    peak = tl.maximum(tl.max(tl.abs(f1), axis=1), tl.max(tl.abs(f2), axis=1))
    peak = tl.maximum(peak, 1e-8)
    sc = peak * (1.0 / 127.0)
    inv_sc = 127.0 / peak
    q1 = tl.extra.cuda.libdevice.rint(f1 * inv_sc[:, None]).to(tl.int32).to(tl.int8)
    q2 = tl.extra.cuda.libdevice.rint(f2 * inv_sc[:, None]).to(tl.int32).to(tl.int8)

    blk_idx = blk.to(tl.int64) if USE_I64 else blk
    # 100% coalesced 128B writes for BOTH K_i8 (+0) and V_i8_row (+4096)!
    k_dst1 = (
        K_cache_i8
        + blk_idx * STRIDE_KB_I8
        + kvh[:, None] * 2048
        + off * 128
        + d[None, :]
    )
    v_dst1 = (
        V_cache_i8
        + blk_idx * STRIDE_VB_I8
        + 4096
        + kvh[:, None] * 2048
        + off * 128
        + d[None, :]
    )
    dst1 = tl.where(is_k[:, None], k_dst1, v_dst1)
    tl.store(dst1, q1)
    tl.store(dst1 + 64, q2)

    # Co-located K_sc (2048 + kvh*32 + off) and V_sc (2048 + kvh*32 + 16 + off) in key_cache!
    sc_off = 2048 + kvh * 32 + tl.where(is_k, 0, 16) + off
    tl.store(K_cache_f16 + blk_idx * STRIDE_KB_F16 + sc_off, sc.to(tl.float16))


@triton.jit
def _fused_qkv_splitk2_rope_and_cache_i8_6cta_kernel(
    PartQKV,
    XScale,
    WScale,
    Pos,
    CosSin,
    QI8Out,
    QScOut,
    K_cache_i8,
    K_cache_f16,
    V_cache_i8,
    SlotMap,
    M,
    n_slots,
    max_pos,
    num_blocks,
    stride_pos,
    stride_cs_m,
    stride_slot,
    STRIDE_KB_I8: tl.constexpr,
    STRIDE_KB_F16: tl.constexpr,
    STRIDE_VB_I8: tl.constexpr,
    USE_I64: tl.constexpr,
):
    """V7 ultra-fast 6-CTA Wave64 Decode kernel (grid=(M, 6), num_warps=1):
    gid=0..3: 4 Q heads each (Split-K=2 reduce + dequant + RoPE + INT8 Q quant; skips unused BF16 Q)
    gid=4,5 : K/V head (gid-4) -> K_i8 + V_i8_row (100% coalesced) + co-located K_sc/V_sc;
              finalizes V_i8_t inline when off==15 (6.25% of steps, pure-decode distinct blocks).
    """
    tok = tl.program_id(0)
    gid = tl.program_id(1)
    d = tl.arange(0, 64)

    raw_pos = tl.load(Pos + tok * stride_pos)
    pos = tl.minimum(tl.maximum(raw_pos, 0), max_pos - 1)
    sx = tl.load(XScale + tok)

    cs_base = CosSin + pos * stride_cs_m
    cf = tl.load(cs_base + d).to(tl.float32)
    sf = tl.load(cs_base + 64 + d).to(tl.float32)

    mn = M * 2560
    p0_row = PartQKV + tok * 2560
    p1_row = p0_row + mn

    if gid < 4:
        qi_row = QI8Out + tok * 2048
        qs_row = QScOut + tok * 16
        base_h = gid * 4
        for j in tl.static_range(0, 4):
            h = base_h + j
            col1 = h * 128 + d
            col2 = col1 + 64
            x1_i32 = tl.load(p0_row + col1) + tl.load(p1_row + col1)
            x2_i32 = tl.load(p0_row + col2) + tl.load(p1_row + col2)
            x1f = x1_i32.to(tl.float32) * (sx * tl.load(WScale + col1))
            x2f = x2_i32.to(tl.float32) * (sx * tl.load(WScale + col2))
            o1f = x1f * cf - x2f * sf
            o2f = x2f * cf + x1f * sf
            qpk = tl.maximum(tl.max(tl.abs(o1f), axis=0), tl.max(tl.abs(o2f), axis=0))
            qpk = tl.maximum(qpk, 1e-8)
            qsc = qpk * (1.0 / 127.0)
            qinv = 127.0 / qpk
            tl.store(
                qi_row + col1,
                tl.extra.cuda.libdevice.rint(o1f * qinv).to(tl.int32).to(tl.int8),
            )
            tl.store(
                qi_row + col2,
                tl.extra.cuda.libdevice.rint(o2f * qinv).to(tl.int32).to(tl.int8),
            )
            tl.store(qs_row + h, qsc)
    else:
        if tok >= n_slots:
            return
        slot = tl.load(SlotMap + tok * stride_slot)
        if slot < 0:
            return
        blk = slot // 16
        if blk >= num_blocks:
            return
        off = slot % 16
        blk_idx = blk.to(tl.int64) if USE_I64 else blk
        kvh = gid - 4

        kc1 = 2048 + kvh * 128 + d
        kc2 = kc1 + 64
        k1_i32 = tl.load(p0_row + kc1) + tl.load(p1_row + kc1)
        k2_i32 = tl.load(p0_row + kc2) + tl.load(p1_row + kc2)
        k1f = k1_i32.to(tl.float32) * (sx * tl.load(WScale + kc1))
        k2f = k2_i32.to(tl.float32) * (sx * tl.load(WScale + kc2))
        rk1 = k1f * cf - k2f * sf
        rk2 = k2f * cf + k1f * sf
        kpk = tl.maximum(tl.max(tl.abs(rk1), axis=0), tl.max(tl.abs(rk2), axis=0))
        kpk = tl.maximum(kpk, 1e-8)
        ksc = kpk * (1.0 / 127.0)
        kinv = 127.0 / kpk
        k_dst = K_cache_i8 + blk_idx * STRIDE_KB_I8 + kvh * 2048 + off * 128 + d
        tl.store(
            k_dst, tl.extra.cuda.libdevice.rint(rk1 * kinv).to(tl.int32).to(tl.int8)
        )
        tl.store(
            k_dst + 64,
            tl.extra.cuda.libdevice.rint(rk2 * kinv).to(tl.int32).to(tl.int8),
        )
        tl.store(
            K_cache_f16 + blk_idx * STRIDE_KB_F16 + 2048 + kvh * 32 + off,
            ksc.to(tl.float16),
        )

        vc1 = 2304 + kvh * 128 + d
        vc2 = vc1 + 64
        v1_i32 = tl.load(p0_row + vc1) + tl.load(p1_row + vc1)
        v2_i32 = tl.load(p0_row + vc2) + tl.load(p1_row + vc2)
        v1f = v1_i32.to(tl.float32) * (sx * tl.load(WScale + vc1))
        v2f = v2_i32.to(tl.float32) * (sx * tl.load(WScale + vc2))
        vpk = tl.maximum(tl.max(tl.abs(v1f), axis=0), tl.max(tl.abs(v2f), axis=0))
        vpk = tl.maximum(vpk, 1e-8)
        vsc = vpk * (1.0 / 127.0)
        vinv = 127.0 / vpk
        vq1 = tl.extra.cuda.libdevice.rint(v1f * vinv).to(tl.int32).to(tl.int8)
        vq2 = tl.extra.cuda.libdevice.rint(v2f * vinv).to(tl.int32).to(tl.int8)
        v_row_base = V_cache_i8 + blk_idx * STRIDE_VB_I8 + 4096 + kvh * 2048
        tl.store(v_row_base + off * 128 + d, vq1)
        tl.store(v_row_base + off * 128 + 64 + d, vq2)
        tl.store(
            K_cache_f16 + blk_idx * STRIDE_KB_F16 + 2048 + kvh * 32 + 16 + off,
            vsc.to(tl.float16),
        )

        if off == 15:
            v_t_base = V_cache_i8 + blk_idx * STRIDE_VB_I8 + kvh * 2048
            offs_t = tl.arange(0, 16)
            is_last = offs_t[:, None] == 15
            lo_old = tl.load(
                v_row_base + offs_t[:, None] * 128 + d[None, :],
                mask=offs_t[:, None] < 15,
                other=0,
            )
            lo_blk = tl.where(is_last, vq1[None, :], lo_old)
            tl.store(v_t_base + d[:, None] * 16 + offs_t[None, :], tl.trans(lo_blk))
            hi_old = tl.load(
                v_row_base + offs_t[:, None] * 128 + 64 + d[None, :],
                mask=offs_t[:, None] < 15,
                other=0,
            )
            hi_blk = tl.where(is_last, vq2[None, :], hi_old)
            tl.store(
                v_t_base + (64 + d[:, None]) * 16 + offs_t[None, :], tl.trans(hi_blk)
            )


def fused_qkv_rope_and_cache(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    return_contig_kv: bool = False,
):
    """Apply NeoX rotary embedding to [M, 16, 128] Q and [M, 2, 128] K while writing K/V directly to paged cache."""
    num_tokens = query.shape[0]
    q_out = torch.empty((num_tokens, 16, 128), dtype=query.dtype, device=query.device)
    if return_contig_kv:
        k_out = torch.empty(
            (num_tokens, 2, 128), dtype=query.dtype, device=query.device
        )
        v_out = torch.empty(
            (num_tokens, 2, 128), dtype=query.dtype, device=query.device
        )
    else:
        k_out = q_out
        v_out = q_out
    if num_tokens == 0:
        return (q_out, k_out, v_out) if return_contig_kv else q_out
    with torch_device_fn.device(query.device):
        _fused_qkv_rope_and_cache_4w_kernel[(num_tokens,)](
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            q_out,
            key_cache,
            value_cache,
            slot_mapping,
            k_out,
            v_out,
            slot_mapping.shape[0],
            cos_sin_cache.shape[0],
            key_cache.shape[0],
            query.stride(0),
            key.stride(0),
            value.stride(0),
            positions.stride(0),
            cos_sin_cache.stride(0),
            slot_mapping.stride(0),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            SAVE_CONTIG_KV=return_contig_kv,
            num_warps=4,
            num_stages=1,
        )
    if return_contig_kv:
        return q_out, k_out, v_out
    return q_out


def fused_qkv_splitk2_rope_and_cache(
    part_qkv: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor:
    """Fuse Split-K=2 INT32 QKV reduce + dequant + NeoX RoPE + paged KV cache write."""
    num_tokens = part_qkv.shape[1]
    q_out = torch.empty(
        (num_tokens, 16, 128), dtype=torch.bfloat16, device=part_qkv.device
    )
    if num_tokens == 0:
        return q_out
    with torch_device_fn.device(part_qkv.device):
        _fused_qkv_splitk2_rope_and_cache_4w_kernel[(num_tokens,)](
            part_qkv,
            x_scales,
            w_scales,
            positions,
            cos_sin_cache,
            q_out,
            key_cache,
            value_cache,
            slot_mapping,
            num_tokens,
            slot_mapping.shape[0],
            cos_sin_cache.shape[0],
            key_cache.shape[0],
            positions.stride(0),
            cos_sin_cache.stride(0),
            slot_mapping.stride(0),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            num_warps=4,
            num_stages=1,
        )
    return q_out


def fused_qkv_rope_and_cache_i8(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    return_contig_kv: bool = False,
    return_q_i8: bool = False,
):
    """Apply NeoX RoPE + race-free V7 dual-buffer INT8 KV write (+ optional contig BF16 KV or INT8 Q)."""
    num_tokens = query.shape[0]
    q_out = torch.empty((num_tokens, 16, 128), dtype=query.dtype, device=query.device)
    if return_contig_kv:
        k_out = torch.empty(
            (num_tokens, 2, 128), dtype=query.dtype, device=query.device
        )
        v_out = torch.empty(
            (num_tokens, 2, 128), dtype=query.dtype, device=query.device
        )
    else:
        k_out = q_out
        v_out = q_out
    if return_q_i8:
        q_i8 = torch.empty((num_tokens, 16, 128), dtype=torch.int8, device=query.device)
        q_sc = torch.empty((num_tokens, 16), dtype=torch.float32, device=query.device)
    else:
        q_i8 = q_out.view(torch.int8)
        q_sc = positions
    if num_tokens == 0:
        if return_contig_kv:
            return q_out, k_out, v_out
        if return_q_i8:
            return q_out, q_i8, q_sc
        return q_out

    sb_k_f16 = key_cache.stride(0)
    sb_k_i8 = sb_k_f16 * 2
    sb_v_i8 = value_cache.stride(0) * 2
    use_i64 = (key_cache.shape[0] * sb_k_i8 >= (1 << 31)) or (
        value_cache.shape[0] * sb_v_i8 >= (1 << 31)
    )
    v_i8 = value_cache.view(torch.int8)
    n_slots = slot_mapping.shape[0]

    with torch_device_fn.device(query.device):
        _fused_qkv_rope_and_cache_i8_4w_kernel[(num_tokens,)](
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            q_out,
            q_i8,
            q_sc,
            key_cache.view(torch.int8),
            key_cache.view(torch.float16),
            v_i8,
            slot_mapping,
            k_out,
            v_out,
            n_slots,
            cos_sin_cache.shape[0],
            key_cache.shape[0],
            query.stride(0),
            key.stride(0),
            value.stride(0),
            positions.stride(0),
            cos_sin_cache.stride(0),
            slot_mapping.stride(0),
            STRIDE_KB_I8=sb_k_i8,
            STRIDE_KB_F16=sb_k_f16,
            STRIDE_VB_I8=sb_v_i8,
            SAVE_CONTIG_KV=return_contig_kv,
            SAVE_Q_I8=return_q_i8,
            USE_I64=use_i64,
            num_warps=4,
            num_stages=1,
        )
        if n_slots > 0:
            _finalize_v_t_from_slots_kernel[(n_slots, 2)](
                v_i8,
                slot_mapping,
                n_slots,
                value_cache.shape[0],
                slot_mapping.stride(0),
                STRIDE_VB_I8=sb_v_i8,
                USE_I64=use_i64,
                num_warps=1,
                num_stages=1,
            )
    if return_contig_kv:
        return q_out, k_out, v_out
    if return_q_i8:
        return q_out, q_i8, q_sc
    return q_out


def fused_qkv_splitk2_rope_and_cache_i8(
    part_qkv: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-decode V7 6-CTA Split-K=2 QKV reduce + dequant + RoPE + INT8 Q + Dual-Buffer INT8 KV write.

    NOTE: Pure-decode only (1 token/request -> distinct blocks, BF16 q_out unused by INT8 decode attn).
    """
    num_tokens = part_qkv.shape[1]
    q_out = torch.empty(
        (num_tokens, 16, 128), dtype=torch.bfloat16, device=part_qkv.device
    )
    q_i8 = torch.empty((num_tokens, 16, 128), dtype=torch.int8, device=part_qkv.device)
    q_sc = torch.empty((num_tokens, 16), dtype=torch.float32, device=part_qkv.device)
    if num_tokens == 0:
        return q_out, q_i8, q_sc

    sb_k_f16 = key_cache.stride(0)
    sb_k_i8 = sb_k_f16 * 2
    sb_v_i8 = value_cache.stride(0) * 2
    use_i64 = (key_cache.shape[0] * sb_k_i8 >= (1 << 31)) or (
        value_cache.shape[0] * sb_v_i8 >= (1 << 31)
    )

    with torch_device_fn.device(part_qkv.device):
        _fused_qkv_splitk2_rope_and_cache_i8_6cta_kernel[(num_tokens, 6)](
            part_qkv,
            x_scales,
            w_scales,
            positions,
            cos_sin_cache,
            q_i8,
            q_sc,
            key_cache.view(torch.int8),
            key_cache.view(torch.float16),
            value_cache.view(torch.int8),
            slot_mapping,
            num_tokens,
            slot_mapping.shape[0],
            cos_sin_cache.shape[0],
            key_cache.shape[0],
            positions.stride(0),
            cos_sin_cache.stride(0),
            slot_mapping.stride(0),
            STRIDE_KB_I8=sb_k_i8,
            STRIDE_KB_F16=sb_k_f16,
            STRIDE_VB_I8=sb_v_i8,
            USE_I64=use_i64,
            num_warps=1,
            num_stages=1,
        )
    return q_out, q_i8, q_sc
