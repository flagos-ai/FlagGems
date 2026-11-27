# SPDX-License-Identifier: Apache-2.0
"""Pure FlagGems Triton V7 Dual-Buffer INT8 KV-Cache Decode Attention + Prefill Dequant-Compact for MetaX C500 (Wave64).

V7 Dual-Buffer In-Place 8192B/block INT8 KV Layout reusing vLLM's [num_blocks, 16, 2, 128] BF16
KV tensors (0 extra VRAM):
  key_cache   block `b` (8192 bytes):
    bytes [0..4095]   : K_i8     [Hkv=2, T=16, D=128] int8    (offset: kvh*2048 + t*128 + d)
    bytes [4096..4223]: K_sc/V_sc per-head adjacent float16 (same 64B cache line per head!):
                        K_sc: f16 offset 2048 + kvh*32 + t
                        V_sc: f16 offset 2048 + kvh*32 + 16 + t
  value_cache block `b` (8192 bytes):
    bytes [0..4095]   : V_i8_t   [Hkv=2, D=128, T=16] int8    (transposed, finalized when off==15; read by K2)
    bytes [4096..8191]: V_i8_row [Hkv=2, T=16, D=128] int8    (row-major, 100% coalesced write; read by Tail & Prefill)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_gems.fused.metax_fused_qkv_rope_cache import _finalize_v_t_from_slots_kernel
from flag_gems.runtime import torch_device_fn

_LOG2E = 1.4426950408889634
# C500 has 104 CUs: at B=64 (B*2=128), 128 * 26 = 3328 = 32.000 * 104 CUs (exact zero-tail wave alignment!)
_NUM_SPLITS = 26
_TOTAL_SLOTS = 27  # slot 0 = Tail (launched FIRST -> 100% hidden!), slots 1..26 = 26 balanced quad64 splits
_PAD_SLOTS = 32  # next power of 2 for Triton reduction
_MAX_QUADS_STATIC = (
    640  # supports up to 40,960 tokens/req at B<=64 inside CUDAGraph (~61 MB total)
)


@triton.jit
def _write_i8_kv_cache_4w_kernel(
    Key,
    Value,
    K_cache_i8,
    K_cache_f16,
    V_cache_i8,
    SlotMap,
    n_slots,
    num_blocks,
    stride_k_m,
    stride_v_m,
    stride_slot,
    STRIDE_KB_I8: tl.constexpr,
    STRIDE_KB_F16: tl.constexpr,
    STRIDE_VB_I8: tl.constexpr,
    USE_I64: tl.constexpr,
):
    """Fallback per-token V7 dual-buffer INT8 KV cache writer (replaces reshape_and_cache_flash)."""
    tok = tl.program_id(0)
    if tok >= n_slots:
        return
    slot = tl.load(SlotMap + tok * stride_slot)
    if slot < 0:
        return
    blk = slot // 16
    if blk >= num_blocks:
        return
    off = slot % 16

    h_sub = tl.arange(0, 4)
    d = tl.arange(0, 64)
    is_k = h_sub < 2
    kvh = h_sub % 2

    k_base = Key + tok * stride_k_m + kvh[:, None] * 128
    v_base = Value + tok * stride_v_m + kvh[:, None] * 128
    src = tl.where(is_k[:, None], k_base, v_base)

    v1 = tl.load(src + d[None, :]).to(tl.float32)  # [4, 64]
    v2 = tl.load(src + 64 + d[None, :]).to(tl.float32)  # [4, 64]

    peak = tl.maximum(tl.max(tl.abs(v1), axis=1), tl.max(tl.abs(v2), axis=1))
    peak = tl.maximum(peak, 1e-8)
    sc = peak * (1.0 / 127.0)
    inv_sc = 127.0 / peak
    q1 = tl.extra.cuda.libdevice.rint(v1 * inv_sc[:, None]).to(tl.int32).to(tl.int8)
    q2 = tl.extra.cuda.libdevice.rint(v2 * inv_sc[:, None]).to(tl.int32).to(tl.int8)

    blk_idx = blk.to(tl.int64) if USE_I64 else blk

    # 100% coalesced 128B writes for K_i8 (+0) and V_i8_row (+4096)
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

    sc_off = 2048 + kvh * 32 + tl.where(is_k, 0, 16) + off
    tl.store(K_cache_f16 + blk_idx * STRIDE_KB_F16 + sc_off, sc.to(tl.float16))


@triton.jit
def _compact_prefill_kv_i8_kernel(
    K_cache_i8,
    K_cache_f16,
    V_cache_i8,
    BlockTable,
    CuSeqLens,
    PK,
    PV,
    stride_bt_b,
    STRIDE_KB_I8: tl.constexpr,
    STRIDE_KB_F16: tl.constexpr,
    STRIDE_VB_I8: tl.constexpr,
    USE_I64: tl.constexpr,
):
    """Dequantize & pack paged V7 INT8 KV blocks into contiguous BF16 [total_kv, 2, 128] (reads
    V_i8_row directly, zero trans!)."""
    blk_idx = tl.program_id(0)
    req_idx = tl.program_id(1)

    first = tl.load(CuSeqLens + req_idx)
    length = tl.load(CuSeqLens + req_idx + 1) - first
    t0 = blk_idx * 16
    if t0 >= length:
        return

    raw_bid = tl.load(BlockTable + req_idx * stride_bt_b + blk_idx)
    bid = raw_bid.to(tl.int64) if USE_I64 else raw_bid

    offs_t = tl.max_contiguous(tl.multiple_of(tl.arange(0, 16), 16), 16)
    offs_d = tl.max_contiguous(tl.multiple_of(tl.arange(0, 128), 16), 128)
    t_mask = (t0 + offs_t) < length
    out_tok = first + t0 + offs_t

    for kvh in tl.static_range(0, 2):
        k_i8 = tl.load(
            K_cache_i8
            + bid * STRIDE_KB_I8
            + kvh * 2048
            + offs_t[:, None] * 128
            + offs_d[None, :]
        )
        k_sc = tl.load(K_cache_f16 + bid * STRIDE_KB_F16 + 2048 + kvh * 32 + offs_t).to(
            tl.float32
        )
        k_bf16 = (k_i8.to(tl.float32) * k_sc[:, None]).to(tl.bfloat16)
        tl.store(
            PK + out_tok[:, None] * 256 + kvh * 128 + offs_d[None, :],
            k_bf16,
            mask=t_mask[:, None],
        )

        v_i8_row = tl.load(
            V_cache_i8
            + bid * STRIDE_VB_I8
            + 4096
            + kvh * 2048
            + offs_t[:, None] * 128
            + offs_d[None, :]
        )
        v_sc = tl.load(
            K_cache_f16 + bid * STRIDE_KB_F16 + 2048 + kvh * 32 + 16 + offs_t
        ).to(tl.float32)
        v_bf16 = (v_i8_row.to(tl.float32) * v_sc[:, None]).to(tl.bfloat16)
        tl.store(
            PV + out_tok[:, None] * 256 + kvh * 128 + offs_d[None, :],
            v_bf16,
            mask=t_mask[:, None],
        )


@triton.jit
def _quant_q_i8_4w_kernel(
    Q_bf16,
    Q_i8,
    Q_sc,
    stride_q_b,
):
    """Per-head INT8 quantization fallback for decode Q [B, 16, 128] -> Q_i8 [B, 16, 128], Q_sc [B, 16]."""
    b = tl.program_id(0)
    h_sub = tl.arange(0, 4)
    d = tl.arange(0, 64)
    q_row = Q_bf16 + b * stride_q_b
    qi_row = Q_i8 + b * 2048
    qs_row = Q_sc + b * 16

    for g in tl.static_range(0, 4):
        h = g * 4 + h_sub
        x1 = tl.load(q_row + h[:, None] * 128 + d[None, :]).to(tl.float32)
        x2 = tl.load(q_row + h[:, None] * 128 + 64 + d[None, :]).to(tl.float32)
        peak = tl.maximum(tl.max(tl.abs(x1), axis=1), tl.max(tl.abs(x2), axis=1))
        peak = tl.maximum(peak, 1e-8)
        sc = peak * (1.0 / 127.0)
        inv_sc = 127.0 / peak
        q1 = tl.extra.cuda.libdevice.rint(x1 * inv_sc[:, None]).to(tl.int32).to(tl.int8)
        q2 = tl.extra.cuda.libdevice.rint(x2 * inv_sc[:, None]).to(tl.int32).to(tl.int8)
        tl.store(qi_row + h[:, None] * 128 + d[None, :], q1)
        tl.store(qi_row + h[:, None] * 128 + 64 + d[None, :], q2)
        tl.store(qs_row + h, sc)


@triton.jit
def _c500_i8kv_k1_quad64_kernel(
    Q_i8,
    Q_sc,
    K_cache_i8,
    K_cache_f16,
    BlockTable,
    SeqLens,
    P_i8,
    TileMax,
    TileSum,
    TileVMax,
    SplitMax,
    sm_scale_log2,
    stride_bt_b,
    stride_pb,
    stride_ph,
    stride_tsb,
    stride_tsh,
    stride_vmb,
    stride_vmh,
    STRIDE_KB_I8: tl.constexpr,
    STRIDE_KB_F16: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    USE_I64: tl.constexpr,
):
    """V7 K1: Single-Wave64 Quad64 INT8 QK^T + Softmax + co-located K_sc/V_sc (0 V_cache access, 0
    spills, balanced split)."""
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    b = pid_bh // 2
    kvh = pid_bh % 2

    seq_len = tl.load(SeqLens + b)
    n_quads = seq_len // 64
    base = n_quads // NUM_SPLITS
    rem = n_quads % NUM_SPLITS
    q0 = pid_s * base + tl.minimum(pid_s, rem)
    q1 = q0 + base + tl.where(pid_s < rem, 1, 0)

    offs_h = tl.arange(0, 16)
    h_mask = offs_h < 8
    offs_t = tl.max_contiguous(tl.multiple_of(tl.arange(0, 16), 16), 16)
    offs_64 = tl.max_contiguous(tl.multiple_of(tl.arange(0, 64), 16), 64)
    offs_d = tl.max_contiguous(tl.multiple_of(tl.arange(0, 128), 16), 128)
    q_h = kvh * 8 + offs_h

    if q0 >= q1:
        tl.store(SplitMax + (pid_bh * NUM_SPLITS + pid_s) * 16 + offs_h, -1e9)
        return

    q_i8 = tl.load(
        Q_i8 + b * 2048 + q_h[:, None] * 128 + offs_d[None, :],
        mask=h_mask[:, None],
        other=0,
    )
    q_sc = tl.load(Q_sc + b * 16 + q_h, mask=h_mask, other=0.0) * sm_scale_log2

    bt = BlockTable + b * stride_bt_b
    p_base = P_i8 + b * stride_pb + kvh * stride_ph
    tm_base = TileMax + b * stride_tsb + kvh * stride_tsh
    tl_base = TileSum + b * stride_tsb + kvh * stride_tsh
    vm_base = TileVMax + b * stride_vmb + kvh * stride_vmh
    m_split = tl.full((16,), -1e9, dtype=tl.float32)

    k_off_h = kvh * 2048
    ksc_off_h = 2048 + kvh * 32
    vsc_off_h = 2048 + kvh * 32 + 16

    for qi in range(q0, q1):
        bid0 = tl.load(bt + qi * 4 + 0)
        bid1 = tl.load(bt + qi * 4 + 1)
        bid2 = tl.load(bt + qi * 4 + 2)
        bid3 = tl.load(bt + qi * 4 + 3)
        if USE_I64:
            bid0 = bid0.to(tl.int64)
            bid1 = bid1.to(tl.int64)
            bid2 = bid2.to(tl.int64)
            bid3 = bid3.to(tl.int64)

        k0 = tl.load(
            K_cache_i8
            + bid0 * STRIDE_KB_I8
            + k_off_h
            + offs_t[:, None] * 128
            + offs_d[None, :]
        )
        ksc0 = tl.load(K_cache_f16 + bid0 * STRIDE_KB_F16 + ksc_off_h + offs_t).to(
            tl.float32
        )
        s0 = tl.dot(q_i8, tl.trans(k0), out_dtype=tl.int32).to(tl.float32) * (
            q_sc[:, None] * ksc0[None, :]
        )

        k1 = tl.load(
            K_cache_i8
            + bid1 * STRIDE_KB_I8
            + k_off_h
            + offs_t[:, None] * 128
            + offs_d[None, :]
        )
        ksc1 = tl.load(K_cache_f16 + bid1 * STRIDE_KB_F16 + ksc_off_h + offs_t).to(
            tl.float32
        )
        s1 = tl.dot(q_i8, tl.trans(k1), out_dtype=tl.int32).to(tl.float32) * (
            q_sc[:, None] * ksc1[None, :]
        )
        s01 = tl.reshape(tl.join(s0, s1), (16, 32))

        k2 = tl.load(
            K_cache_i8
            + bid2 * STRIDE_KB_I8
            + k_off_h
            + offs_t[:, None] * 128
            + offs_d[None, :]
        )
        ksc2 = tl.load(K_cache_f16 + bid2 * STRIDE_KB_F16 + ksc_off_h + offs_t).to(
            tl.float32
        )
        s2 = tl.dot(q_i8, tl.trans(k2), out_dtype=tl.int32).to(tl.float32) * (
            q_sc[:, None] * ksc2[None, :]
        )

        k3 = tl.load(
            K_cache_i8
            + bid3 * STRIDE_KB_I8
            + k_off_h
            + offs_t[:, None] * 128
            + offs_d[None, :]
        )
        ksc3 = tl.load(K_cache_f16 + bid3 * STRIDE_KB_F16 + ksc_off_h + offs_t).to(
            tl.float32
        )
        s3 = tl.dot(q_i8, tl.trans(k3), out_dtype=tl.int32).to(tl.float32) * (
            q_sc[:, None] * ksc3[None, :]
        )
        s23 = tl.reshape(tl.join(s2, s3), (16, 32))
        s_quad = tl.reshape(tl.join(s01, s23), (16, 64))

        m_j = tl.max(s_quad, axis=1)
        m_split = tl.maximum(m_split, m_j)
        p_j = tl.math.exp2(s_quad - m_j[:, None])
        l_j = tl.sum(p_j, axis=1)

        vsc0 = tl.load(K_cache_f16 + bid0 * STRIDE_KB_F16 + vsc_off_h + offs_t).to(
            tl.float32
        )
        vsc1 = tl.load(K_cache_f16 + bid1 * STRIDE_KB_F16 + vsc_off_h + offs_t).to(
            tl.float32
        )
        vsc2 = tl.load(K_cache_f16 + bid2 * STRIDE_KB_F16 + vsc_off_h + offs_t).to(
            tl.float32
        )
        vsc3 = tl.load(K_cache_f16 + bid3 * STRIDE_KB_F16 + vsc_off_h + offs_t).to(
            tl.float32
        )
        vsc_quad = tl.reshape(
            tl.join(
                tl.reshape(tl.join(vsc0, vsc1), (32,)),
                tl.reshape(tl.join(vsc2, vsc3), (32,)),
            ),
            (64,),
        )
        v_max = tl.maximum(tl.max(vsc_quad, axis=0), 1e-12)
        p_q = (
            (p_j * (vsc_quad[None, :] * (127.0 / v_max)) + 0.5).to(tl.int32).to(tl.int8)
        )

        tl.store(
            p_base + qi * 512 + offs_h[:, None] * 64 + offs_64[None, :],
            p_q,
            mask=h_mask[:, None],
        )
        tl.store(tm_base + qi * 16 + offs_h, m_j)
        tl.store(tl_base + qi * 16 + offs_h, l_j)
        tl.store(vm_base + qi, v_max)

    tl.store(SplitMax + (pid_bh * NUM_SPLITS + pid_s) * 16 + offs_h, m_split)


@triton.jit
def _c500_i8kv_k2_tailfirst_kernel(
    Q_i8,
    Q_sc,
    K_cache_i8,
    K_cache_f16,
    P_i8,
    TileMax,
    TileSum,
    TileVMax,
    SplitMax,
    V_cache_i8,
    BlockTable,
    SeqLens,
    OutPart,
    MaxPart,
    SumPart,
    sm_scale_log2,
    stride_bt_b,
    stride_pb,
    stride_ph,
    stride_tsb,
    stride_tsh,
    stride_vmb,
    stride_vmh,
    STRIDE_KB_I8: tl.constexpr,
    STRIDE_KB_F16: tl.constexpr,
    STRIDE_VB_I8: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    TOTAL_SLOTS: tl.constexpr,
    USE_I64: tl.constexpr,
):
    """V7 K2 + Fused Tail-First (pid_slot==0 launches Tail FIRST reading V_i8_row;
    pid_slot==1..NUM_SPLITS read V_i8_t)."""
    pid_bh = tl.program_id(0)
    pid_slot = tl.program_id(1)
    b = pid_bh // 2
    kvh = pid_bh % 2

    seq_len = tl.load(SeqLens + b)
    offs_h = tl.arange(0, 16)
    h_mask = offs_h < 8
    offs_t = tl.max_contiguous(tl.multiple_of(tl.arange(0, 16), 16), 16)
    offs_d = tl.max_contiguous(tl.multiple_of(tl.arange(0, 128), 16), 128)
    q_h = kvh * 8 + offs_h

    ob_base = (b * 16 + q_h[:, None]) * (TOTAL_SLOTS * 128) + pid_slot * 128
    mb_base = (b * 16 + q_h) * TOTAL_SLOTS + pid_slot
    bt = BlockTable + b * stride_bt_b
    kv_off_h = kvh * 2048

    if pid_slot == 0:
        tail_start = (seq_len // 64) * 64
        if tail_start >= seq_len:
            tl.store(
                OutPart + ob_base + offs_d[None, :],
                tl.zeros((16, 128), dtype=tl.bfloat16),
                mask=h_mask[:, None],
            )
            tl.store(MaxPart + mb_base, -1e9, mask=h_mask)
            tl.store(SumPart + mb_base, 0.0, mask=h_mask)
            return

        q_i8 = tl.load(
            Q_i8 + b * 2048 + q_h[:, None] * 128 + offs_d[None, :],
            mask=h_mask[:, None],
            other=0,
        )
        q_sc = tl.load(Q_sc + b * 16 + q_h, mask=h_mask, other=0.0) * sm_scale_log2
        ksc_off_h = 2048 + kvh * 32
        vsc_off_h = 2048 + kvh * 32 + 16

        m_acc = tl.full((16,), -1e9, dtype=tl.float32)
        l_acc = tl.zeros((16,), dtype=tl.float32)
        o_acc = tl.zeros((16, 128), dtype=tl.float32)

        start_blk = tail_start // 16
        end_blk = tl.cdiv(seq_len, 16)
        for bi in range(start_blk, end_blk):
            bid = tl.load(bt + bi)
            if USE_I64:
                bid = bid.to(tl.int64)
            t_mask = (bi * 16 + offs_t) < seq_len
            k_b = tl.load(
                K_cache_i8
                + bid * STRIDE_KB_I8
                + kv_off_h
                + offs_t[:, None] * 128
                + offs_d[None, :]
            )
            ksc = tl.load(K_cache_f16 + bid * STRIDE_KB_F16 + ksc_off_h + offs_t).to(
                tl.float32
            )
            s = tl.dot(q_i8, tl.trans(k_b), out_dtype=tl.int32).to(tl.float32) * (
                q_sc[:, None] * ksc[None, :]
            )
            s = tl.where(t_mask[None, :], s, -1e9)
            m_b = tl.max(s, axis=1)
            m_new = tl.maximum(m_acc, m_b)
            alpha = tl.math.exp2(m_acc - m_new)
            p_b = tl.where(t_mask[None, :], tl.math.exp2(s - m_new[:, None]), 0.0)

            v_row = tl.load(
                V_cache_i8
                + bid * STRIDE_VB_I8
                + 4096
                + kv_off_h
                + offs_t[:, None] * 128
                + offs_d[None, :]
            )
            vsc = tl.load(K_cache_f16 + bid * STRIDE_KB_F16 + vsc_off_h + offs_t).to(
                tl.float32
            )
            p_bf16 = (p_b * vsc[None, :]).to(tl.bfloat16)
            o_b = tl.dot(p_bf16, v_row.to(tl.bfloat16), out_dtype=tl.float32)

            o_acc = o_acc * alpha[:, None] + o_b
            l_acc = l_acc * alpha + tl.sum(p_b, axis=1)
            m_acc = m_new

        tl.store(
            OutPart + ob_base + offs_d[None, :],
            o_acc.to(tl.bfloat16),
            mask=h_mask[:, None],
        )
        tl.store(MaxPart + mb_base, m_acc, mask=h_mask)
        tl.store(SumPart + mb_base, l_acc, mask=h_mask)
        return

    pid_s = pid_slot - 1
    n_quads = seq_len // 64
    base = n_quads // NUM_SPLITS
    rem = n_quads % NUM_SPLITS
    q0 = pid_s * base + tl.minimum(pid_s, rem)
    q1 = q0 + base + tl.where(pid_s < rem, 1, 0)

    if q0 >= q1:
        tl.store(
            OutPart + ob_base + offs_d[None, :],
            tl.zeros((16, 128), dtype=tl.bfloat16),
            mask=h_mask[:, None],
        )
        tl.store(MaxPart + mb_base, -1e9, mask=h_mask)
        tl.store(SumPart + mb_base, 0.0, mask=h_mask)
        return

    offs_64 = tl.max_contiguous(tl.multiple_of(tl.arange(0, 64), 16), 64)
    p_base = P_i8 + b * stride_pb + kvh * stride_ph
    tm_base = TileMax + b * stride_tsb + kvh * stride_tsh
    tl_base = TileSum + b * stride_tsb + kvh * stride_tsh

    m_split = tl.load(SplitMax + (pid_bh * NUM_SPLITS + pid_s) * 16 + offs_h)
    inv_127 = 1.0 / 127.0
    l_acc = tl.zeros((16,), dtype=tl.float32)
    o_acc = tl.zeros((16, 128), dtype=tl.float32)

    for qi in range(q0, q1):
        bid0 = tl.load(bt + qi * 4 + 0)
        bid1 = tl.load(bt + qi * 4 + 1)
        bid2 = tl.load(bt + qi * 4 + 2)
        bid3 = tl.load(bt + qi * 4 + 3)
        if USE_I64:
            bid0 = bid0.to(tl.int64)
            bid1 = bid1.to(tl.int64)
            bid2 = bid2.to(tl.int64)
            bid3 = bid3.to(tl.int64)

        v0_t = tl.load(
            V_cache_i8
            + bid0 * STRIDE_VB_I8
            + kv_off_h
            + offs_d[:, None] * 16
            + offs_t[None, :]
        )
        v1_t = tl.load(
            V_cache_i8
            + bid1 * STRIDE_VB_I8
            + kv_off_h
            + offs_d[:, None] * 16
            + offs_t[None, :]
        )
        v01_t = tl.reshape(tl.join(v0_t, v1_t), (128, 32))
        v2_t = tl.load(
            V_cache_i8
            + bid2 * STRIDE_VB_I8
            + kv_off_h
            + offs_d[:, None] * 16
            + offs_t[None, :]
        )
        v3_t = tl.load(
            V_cache_i8
            + bid3 * STRIDE_VB_I8
            + kv_off_h
            + offs_d[:, None] * 16
            + offs_t[None, :]
        )
        v23_t = tl.reshape(tl.join(v2_t, v3_t), (128, 32))
        v_quad_t = tl.reshape(tl.join(v01_t, v23_t), (128, 64))

        p_q = tl.load(
            p_base + qi * 512 + offs_h[:, None] * 64 + offs_64[None, :],
            mask=h_mask[:, None],
            other=0,
        )
        o_i32 = tl.dot(p_q, tl.trans(v_quad_t), out_dtype=tl.int32)

        m_j = tl.load(tm_base + qi * 16 + offs_h)
        l_j = tl.load(tl_base + qi * 16 + offs_h)
        v_max = tl.load(TileVMax + b * stride_vmb + kvh * stride_vmh + qi)
        beta = tl.math.exp2(m_j - m_split)
        o_acc += o_i32.to(tl.float32) * (beta[:, None] * (v_max * inv_127))
        l_acc += l_j * beta

    tl.store(
        OutPart + ob_base + offs_d[None, :], o_acc.to(tl.bfloat16), mask=h_mask[:, None]
    )
    tl.store(MaxPart + mb_base, m_split, mask=h_mask)
    tl.store(SumPart + mb_base, l_acc, mask=h_mask)


@triton.jit
def _c500_i8kv_reduce_kernel(
    OutPart,
    MaxPart,
    SumPart,
    Out,
    stride_out_b,
    stride_out_h,
    TOTAL_SLOTS: tl.constexpr,
    PAD_SLOTS: tl.constexpr,
):
    """Single-Wave64 split-K reduction across TOTAL_SLOTS=27 (padded to PAD_SLOTS=32)."""
    pid = tl.program_id(0)
    b = pid // 16
    h = pid % 16
    offs_s = tl.arange(0, PAD_SLOTS)
    s_mask = offs_s < TOTAL_SLOTS
    offs_d = tl.max_contiguous(tl.multiple_of(tl.arange(0, 128), 16), 128)

    ms = tl.load(MaxPart + pid * TOTAL_SLOTS + offs_s, mask=s_mask, other=-1e9)
    ls = tl.load(SumPart + pid * TOTAL_SLOTS + offs_s, mask=s_mask, other=0.0)
    m_global = tl.max(ms, axis=0)
    w = tl.where(s_mask, tl.math.exp2(ms - m_global), 0.0)
    l_total = tl.sum(ls * w, axis=0)
    inv_l = 1.0 / tl.maximum(l_total, 1e-12)

    o_all = tl.load(
        OutPart + (pid * TOTAL_SLOTS + offs_s[:, None]) * 128 + offs_d[None, :],
        mask=s_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    o_acc = tl.sum(o_all * w[:, None], axis=0)
    tl.store(
        Out + b * stride_out_b + h * stride_out_h + offs_d,
        (o_acc * inv_l).to(tl.bfloat16),
    )


class _I8KVScratch:
    """Device-scoped static scratchpad shared across all 42 layers (CUDAGraph-safe)."""

    def __init__(
        self, device: torch.device, max_b: int = 64, max_quads: int = _MAX_QUADS_STATIC
    ):
        self.device = device
        self.max_b = max_b
        self.max_quads = max_quads
        self.p_quad = torch.empty(
            (max_b, 2, max_quads, 8, 64), dtype=torch.int8, device=device
        )
        self.tm_quad = torch.empty(
            (max_b, 2, max_quads, 16), dtype=torch.float32, device=device
        )
        self.tl_quad = torch.empty(
            (max_b, 2, max_quads, 16), dtype=torch.float32, device=device
        )
        self.vm_quad = torch.empty(
            (max_b, 2, max_quads), dtype=torch.float32, device=device
        )
        self.split_max = torch.empty(
            (max_b, 2, _NUM_SPLITS, 16), dtype=torch.float32, device=device
        )
        self.out_part = torch.empty(
            (max_b, 16, _TOTAL_SLOTS, 128), dtype=torch.bfloat16, device=device
        )
        self.max_part = torch.empty(
            (max_b, 16, _TOTAL_SLOTS), dtype=torch.float32, device=device
        )
        self.sum_part = torch.empty(
            (max_b, 16, _TOTAL_SLOTS), dtype=torch.float32, device=device
        )
        self.q_i8_tmp = torch.empty((max_b, 16, 128), dtype=torch.int8, device=device)
        self.q_sc_tmp = torch.empty((max_b, 16), dtype=torch.float32, device=device)


_SCRATCH_BY_DEVICE: dict[int, _I8KVScratch] = {}


def _get_scratch(
    device: torch.device, batch_size: int, needed_quads: int
) -> _I8KVScratch:
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    sc = _SCRATCH_BY_DEVICE.get(dev_idx)
    if sc is None:
        sc = _I8KVScratch(
            device,
            max_b=max(64, batch_size),
            max_quads=max(_MAX_QUADS_STATIC, needed_quads),
        )
        _SCRATCH_BY_DEVICE[dev_idx] = sc
    elif (
        batch_size > sc.max_b or needed_quads > sc.max_quads
    ) and not torch.cuda.is_current_stream_capturing():
        sc = _I8KVScratch(
            device,
            max_b=max(sc.max_b, batch_size),
            max_quads=max(sc.max_quads, needed_quads),
        )
        _SCRATCH_BY_DEVICE[dev_idx] = sc
    return sc


def _kv_cache_views(key_cache: torch.Tensor, value_cache: torch.Tensor):
    k_i8 = key_cache.view(torch.int8)
    k_f16 = key_cache.view(torch.float16)
    v_i8 = value_cache.view(torch.int8)
    v_f16 = value_cache.view(torch.float16)
    sb_k_f16 = key_cache.stride(0)
    sb_k_i8 = sb_k_f16 * 2
    sb_v_f16 = value_cache.stride(0)
    sb_v_i8 = sb_v_f16 * 2
    use_i64 = (key_cache.shape[0] * sb_k_i8 >= (1 << 31)) or (
        value_cache.shape[0] * sb_v_i8 >= (1 << 31)
    )
    return k_i8, k_f16, v_i8, v_f16, sb_k_i8, sb_k_f16, sb_v_i8, sb_v_f16, use_i64


def write_i8_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Write [M, 2, 128] BF16 K/V into V7 Dual-Buffer in-place 8192B INT8 KV cache (race-free)."""
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return
    k_i8, k_f16, v_i8, _, sb_k_i8, sb_k_f16, sb_v_i8, _, use_i64 = _kv_cache_views(
        key_cache, value_cache
    )
    with torch_device_fn.device(key.device):
        _write_i8_kv_cache_4w_kernel[(num_tokens,)](
            key,
            value,
            k_i8,
            k_f16,
            v_i8,
            slot_mapping,
            num_tokens,
            key_cache.shape[0],
            key.stride(0),
            value.stride(0),
            slot_mapping.stride(0),
            STRIDE_KB_I8=sb_k_i8,
            STRIDE_KB_F16=sb_k_f16,
            STRIDE_VB_I8=sb_v_i8,
            USE_I64=use_i64,
            num_warps=4,
            num_stages=1,
        )
        _finalize_v_t_from_slots_kernel[(num_tokens, 2)](
            v_i8,
            slot_mapping,
            num_tokens,
            value_cache.shape[0],
            slot_mapping.stride(0),
            STRIDE_VB_I8=sb_v_i8,
            USE_I64=use_i64,
            num_warps=1,
            num_stages=1,
        )


def compact_prefill_kv_i8(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    total_kv_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize and compact paged V7 INT8 KV cache into contiguous BF16 [total_kv_tokens, 2, 128]."""
    pk = torch.empty(
        (total_kv_tokens, 2, 128), dtype=torch.bfloat16, device=key_cache.device
    )
    pv = torch.empty(
        (total_kv_tokens, 2, 128), dtype=torch.bfloat16, device=key_cache.device
    )
    if total_kv_tokens == 0 or max_seqlen_k == 0:
        return pk, pv
    k_i8, k_f16, v_i8, _, sb_k_i8, sb_k_f16, sb_v_i8, _, use_i64 = _kv_cache_views(
        key_cache, value_cache
    )
    num_reqs = block_table.shape[0]
    with torch_device_fn.device(key_cache.device):
        _compact_prefill_kv_i8_kernel[(triton.cdiv(max_seqlen_k, 16), num_reqs)](
            k_i8,
            k_f16,
            v_i8,
            block_table,
            cu_seqlens_k,
            pk,
            pv,
            block_table.stride(0),
            STRIDE_KB_I8=sb_k_i8,
            STRIDE_KB_F16=sb_k_f16,
            STRIDE_VB_I8=sb_v_i8,
            USE_I64=use_i64,
            num_warps=4,
            num_stages=1,
        )
    return pk, pv


def int8_kv_flash_decode(
    decode_query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    sm_scale: float,
    q_i8: torch.Tensor | None = None,
    q_sc: torch.Tensor | None = None,
    max_seq_len_hint: int = 0,
) -> torch.Tensor:
    """Pure FlagGems Triton V7 INT8 KV Flash-Decoding for MiniCPM5-2B (Hq=16, Hkv=2, D=128, BLK=16)."""
    B = decode_query.shape[0]
    if B == 0:
        return output
    if B > 64:
        for b0 in range(0, B, 64):
            b1 = min(b0 + 64, B)
            int8_kv_flash_decode(
                decode_query[b0:b1],
                key_cache,
                value_cache,
                block_table[b0:b1],
                seq_lens[b0:b1],
                output[b0:b1],
                sm_scale,
                q_i8=None if q_i8 is None else q_i8[b0:b1],
                q_sc=None if q_sc is None else q_sc[b0:b1],
                max_seq_len_hint=max_seq_len_hint,
            )
        return output

    needed_quads = (
        max(1, max_seq_len_hint // 64) if max_seq_len_hint > 0 else _MAX_QUADS_STATIC
    )
    scratch = _get_scratch(decode_query.device, B, needed_quads)

    k_i8, k_f16, v_i8, _, sb_k_i8, sb_k_f16, sb_v_i8, _, use_i64 = _kv_cache_views(
        key_cache, value_cache
    )
    sm_scale_log2 = sm_scale * _LOG2E

    with torch_device_fn.device(decode_query.device):
        if q_i8 is None or q_sc is None:
            q_i8 = scratch.q_i8_tmp[:B]
            q_sc = scratch.q_sc_tmp[:B]
            _quant_q_i8_4w_kernel[(B,)](
                decode_query,
                q_i8,
                q_sc,
                decode_query.stride(0),
                num_warps=4,
                num_stages=1,
            )

        p_quad = scratch.p_quad
        tm_quad = scratch.tm_quad
        tl_quad = scratch.tl_quad
        vm_quad = scratch.vm_quad
        split_max = scratch.split_max
        out_part = scratch.out_part
        max_part = scratch.max_part
        sum_part = scratch.sum_part

        _c500_i8kv_k1_quad64_kernel[(B * 2, _NUM_SPLITS)](
            q_i8,
            q_sc,
            k_i8,
            k_f16,
            block_table,
            seq_lens,
            p_quad,
            tm_quad,
            tl_quad,
            vm_quad,
            split_max,
            sm_scale_log2,
            block_table.stride(0),
            p_quad.stride(0),
            p_quad.stride(1),
            tm_quad.stride(0),
            tm_quad.stride(1),
            vm_quad.stride(0),
            vm_quad.stride(1),
            STRIDE_KB_I8=sb_k_i8,
            STRIDE_KB_F16=sb_k_f16,
            NUM_SPLITS=_NUM_SPLITS,
            USE_I64=use_i64,
            num_warps=1,
            num_stages=2,
            pipeline="cpasync",
            scenario="disable_int8_opt",
        )
        _c500_i8kv_k2_tailfirst_kernel[(B * 2, _TOTAL_SLOTS)](
            q_i8,
            q_sc,
            k_i8,
            k_f16,
            p_quad,
            tm_quad,
            tl_quad,
            vm_quad,
            split_max,
            v_i8,
            block_table,
            seq_lens,
            out_part,
            max_part,
            sum_part,
            sm_scale_log2,
            block_table.stride(0),
            p_quad.stride(0),
            p_quad.stride(1),
            tm_quad.stride(0),
            tm_quad.stride(1),
            vm_quad.stride(0),
            vm_quad.stride(1),
            STRIDE_KB_I8=sb_k_i8,
            STRIDE_KB_F16=sb_k_f16,
            STRIDE_VB_I8=sb_v_i8,
            NUM_SPLITS=_NUM_SPLITS,
            TOTAL_SLOTS=_TOTAL_SLOTS,
            USE_I64=use_i64,
            num_warps=2,
            num_stages=2,
            pipeline="cpasync",
        )
        _c500_i8kv_reduce_kernel[(B * 16,)](
            out_part,
            max_part,
            sum_part,
            output,
            output.stride(0),
            output.stride(1),
            TOTAL_SLOTS=_TOTAL_SLOTS,
            PAD_SLOTS=_PAD_SLOTS,
            num_warps=1,
            num_stages=1,
        )
    return output
