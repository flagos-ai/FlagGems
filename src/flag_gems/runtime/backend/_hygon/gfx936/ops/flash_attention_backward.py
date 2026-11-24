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
import math
import struct

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

__all__ = ["flash_attention_backward"]


@triton.jit
def _hygon_dropout(
    RNG,
    RNG_OFFSET,
    batch,
    head,
    m,
    n,
    HQ: tl.constexpr,
    threshold,
    drop_scale,
    DIM: tl.constexpr = 64,
):
    seed = tl.load(RNG).to(tl.uint64)
    offset = tl.load(RNG_OFFSET).to(tl.uint64)
    offset += ((batch * HQ + head) * 32).to(tl.uint64)
    c0 = tl.full((m.shape[0], n.shape[0]), 0, tl.uint32) + offset.to(tl.uint32)
    c1 = tl.full((m.shape[0], n.shape[0]), 0, tl.uint32) + (offset >> 32).to(tl.uint32)
    if DIM == 128:
        c2 = tl.broadcast_to((m[:, None] // 16 * 16).to(tl.uint32), c0.shape)
        c3 = tl.broadcast_to(
            (n[None, :] // 64 * 64 + n[None, :] % 16 // 4 * 4).to(tl.uint32), c0.shape
        )
    else:
        c2 = tl.broadcast_to(m[:, None].to(tl.uint32), c0.shape)
        c3 = tl.broadcast_to(n[None, :].to(tl.uint32), c0.shape)
    k0, k1 = seed.to(tl.uint32), (seed >> 32).to(tl.uint32)
    for _ in tl.static_range(7):
        hi0 = tl.umulhi(c0, 0xD2511F53)
        hi1 = tl.umulhi(c2, 0xCD9E8D57)
        lo0 = c0 * 0xD2511F53
        lo1 = c2 * 0xCD9E8D57
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0 = (k0 + 0x9E3779B9).to(tl.uint32)
        k1 = (k1 + 0xBB67AE85).to(tl.uint32)
    if DIM == 128:
        word = n[None, :] // 16 % 4
        bits = tl.where(
            word == 0, c0, tl.where(word == 1, c1, tl.where(word == 2, c2, c3))
        )
        random_byte = (bits >> (8 * (n[None, :] % 4)).to(tl.uint32)) & 255
    else:
        random_byte = c0 & 255
    return (random_byte <= threshold).to(tl.float32) * drop_scale


@triton.jit
def _aten_flash_probability(
    q,
    k,
    lse,
    m,
    n,
    sq,
    sk,
    scale,
    scale_log2,
    LSE_SCALE: tl.constexpr,
    WL: tl.constexpr,
    WR: tl.constexpr,
    TOP_LEFT_CAUSAL: tl.constexpr,
    FUSE_FMA: tl.constexpr = False,
):
    scores = tl.dot(q, tl.trans(k))
    offset = 0 if TOP_LEFT_CAUSAL else sk - sq
    distance = m[:, None] + offset - n[None, :]
    valid = (m[:, None] < sq) & (n[None, :] < sk)
    if WL >= 0:
        valid &= distance <= WL
    if WR >= 0:
        valid &= distance >= -WR
    # Native forward uses +inf for empty rows; mathematical oracles may use -inf.
    valid &= (lse[:, None] != float("inf")) & (lse[:, None] != -float("inf"))
    if scale_log2 is None:
        probability = tl.exp(scores * scale - lse[:, None])
    else:
        # DTK fuses <=128 heads, D256 dropout, and D192 p0.
        # Other larger heads use separately rounded operations. The host
        # rounds scale*log2(e) to FP32 before either reconstruction.
        if q.shape[1] <= 128 or FUSE_FMA:
            probability = tl.exp2(tl.fma(scores, scale_log2, -lse[:, None] * LSE_SCALE))
        else:
            probability = tl.exp2(scores * scale_log2 - lse[:, None] * LSE_SCALE)
    return tl.where(valid, probability, 0.0)


@triton.jit
def _hygon_delta_chunks128(x, BM: tl.constexpr):
    x = tl.reshape(x, (BM, 2, 8, 4, 2))
    lo, hi = tl.split(tl.trans(x, (0, 2, 3, 4, 1)))
    loe, loo = tl.split(lo)
    loe04, loe26 = tl.split(tl.reshape(loe, (BM, 8, 2, 2)))
    loo04, loo26 = tl.split(tl.reshape(loo, (BM, 8, 2, 2)))
    lo0, lo4 = tl.split(loe04)
    lo2, lo6 = tl.split(loe26)
    lo1, lo5 = tl.split(loo04)
    lo3, lo7 = tl.split(loo26)
    hie, hio = tl.split(hi)
    hie04, hie26 = tl.split(tl.reshape(hie, (BM, 8, 2, 2)))
    hio04, hio26 = tl.split(tl.reshape(hio, (BM, 8, 2, 2)))
    hi0, hi4 = tl.split(hie04)
    hi2, hi6 = tl.split(hie26)
    hi1, hi5 = tl.split(hio04)
    hi3, hi7 = tl.split(hio26)
    return (
        lo0,
        lo1,
        lo2,
        lo3,
        lo4,
        lo5,
        lo6,
        lo7,
        hi0,
        hi1,
        hi2,
        hi3,
        hi4,
        hi5,
        hi6,
        hi7,
    )


@triton.jit
def _aten_flash_delta(
    o,
    do,
    OUT_PTR,
    DO_PTR,
    base,
    rows,
    DIM: tl.constexpr,
    BM: tl.constexpr,
    DROPOUT: tl.constexpr = False,
):
    if DIM == 128 and do.dtype == tl.float16 and DROPOUT:
        # Reuse the loaded rows and preserve DTK's native lane accumulation.
        ov = _hygon_delta_chunks128(o, BM)
        dv = _hygon_delta_chunks128(do, BM)
        acc = ov[1].to(tl.float32) * dv[1].to(tl.float32)
        acc = tl.fma(ov[0].to(tl.float32), dv[0].to(tl.float32), acc)
        ap = ov[3].to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32) | (
            ov[2].to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32) << 16
        )
        bp = dv[3].to(tl.uint16, bitcast=True).to(tl.uint32) | (
            dv[2].to(tl.uint16, bitcast=True).to(tl.uint32) << 16
        )
        acc = tl.inline_asm_elementwise(
            "v_dot2_f32_f16 $0, $1, $2, $3;",
            "=v,v,v,v",
            [ap, bp, acc],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        acc = tl.fma(ov[5].to(tl.float32), dv[5].to(tl.float32), acc)
        acc = tl.fma(ov[9].to(tl.float32), dv[9].to(tl.float32), acc)
        acc = tl.fma(ov[4].to(tl.float32), dv[4].to(tl.float32), acc)
        acc = tl.fma(ov[8].to(tl.float32), dv[8].to(tl.float32), acc)
        acc = tl.fma(ov[11].to(tl.float32), dv[11].to(tl.float32), acc)
        acc = tl.fma(ov[10].to(tl.float32), dv[10].to(tl.float32), acc)
        acc = tl.fma(ov[13].to(tl.float32), dv[13].to(tl.float32), acc)
        acc = tl.fma(ov[12].to(tl.float32), dv[12].to(tl.float32), acc)
        acc = tl.fma(ov[14].to(tl.float32), dv[14].to(tl.float32), acc)
        acc = tl.fma(ov[6].to(tl.float32), dv[6].to(tl.float32), acc)
        acc = tl.fma(ov[15].to(tl.float32), dv[15].to(tl.float32), acc)
        acc = tl.fma(ov[7].to(tl.float32), dv[7].to(tl.float32), acc)
        lane = tl.arange(0, 8)
        for step in tl.static_range(3):
            other = tl.broadcast_to((lane ^ (4 >> step))[None, :], (BM, 8))
            acc += tl.gather(acc, other, 1)
        return tl.sum(tl.where(lane[None, :] == 0, acc, 0.0), 1)
    elif DIM == 64 and do.dtype == tl.float16 and not DROPOUT:
        # gfx936 native D64 preprocessor: eight contiguous features per lane.
        lane = tl.arange(0, 8)
        idx = base[:, None] + lane[None, :] * 8
        a = tl.load(OUT_PTR + idx, rows[:, None], 0).to(tl.float32)
        b = tl.load(DO_PTR + idx, rows[:, None], 0).to(tl.float32)
        acc = a * b
        OFFSETS: tl.constexpr = (1, 3, 2, 5, 4)
        for i in tl.static_range(5):
            off = OFFSETS[i]
            a = tl.load(OUT_PTR + idx + off, rows[:, None], 0).to(tl.float32)
            b = tl.load(DO_PTR + idx + off, rows[:, None], 0).to(tl.float32)
            acc += a * b
        for off in tl.static_range(6, 8):
            a = tl.load(OUT_PTR + idx + off, rows[:, None], 0).to(tl.float32)
            b = tl.load(DO_PTR + idx + off, rows[:, None], 0).to(tl.float32)
            acc = tl.fma(a, b, acc)
        for step in tl.static_range(3):
            other = tl.broadcast_to((lane ^ (4 >> step))[None, :], (BM, 8))
            acc += tl.gather(acc, other, 1)
        return tl.sum(tl.where(lane[None, :] == 0, acc, 0.0), 1)
    elif DIM == 192 and do.dtype == tl.bfloat16 and not DROPOUT:
        # gfx936 BF16 D192 preprocess: native FMA chain per eight-feature lane.
        lane = tl.arange(0, 8)
        idx = base[:, None] + lane[None, :] * 8
        a = tl.load(OUT_PTR + idx + 128, rows[:, None], 0).to(tl.float32)
        b = tl.load(DO_PTR + idx + 128, rows[:, None], 0).to(tl.float32)
        acc = a * b
        a = tl.load(OUT_PTR + idx + 131, rows[:, None], 0).to(tl.float32)
        b = tl.load(DO_PTR + idx + 131, rows[:, None], 0).to(tl.float32)
        second = a * b
        FIRST: tl.constexpr = (1, 0, 3, 2, 5, 4, 65, 64, 67, 66, 129, 69, 68)
        for i in tl.static_range(13):
            off = FIRST[i]
            a = tl.load(OUT_PTR + idx + off, rows[:, None], 0).to(tl.float32)
            b = tl.load(DO_PTR + idx + off, rows[:, None], 0).to(tl.float32)
            acc = tl.fma(a, b, acc)
        acc += second
        LAST: tl.constexpr = (130, 133, 132, 134, 135, 70, 71, 6, 7)
        for i in tl.static_range(9):
            off = LAST[i]
            a = tl.load(OUT_PTR + idx + off, rows[:, None], 0).to(tl.float32)
            b = tl.load(DO_PTR + idx + off, rows[:, None], 0).to(tl.float32)
            acc = tl.fma(a, b, acc)
        for step in tl.static_range(3):
            other = tl.broadcast_to((lane ^ (4 >> step))[None, :], (BM, 8))
            acc += tl.gather(acc, other, 1)
        return tl.sum(tl.where(lane[None, :] == 0, acc, 0.0), 1)
    elif DIM == 192 and do.dtype == tl.float16 and not DROPOUT:
        # DTK D192 FP16 preprocess: eight lanes, packed dot2 groups, then XOR 4/2/1.
        lane = tl.arange(0, 8)
        idx = base[:, None] + lane[None, :] * 8
        a = tl.load(OUT_PTR + idx + 1, rows[:, None], 0).to(tl.float32)
        b = tl.load(DO_PTR + idx + 1, rows[:, None], 0).to(tl.float32)
        acc = a * b
        a = tl.load(OUT_PTR + idx, rows[:, None], 0).to(tl.float32)
        b = tl.load(DO_PTR + idx, rows[:, None], 0).to(tl.float32)
        acc += a * b
        PAIRS: tl.constexpr = (3, 5, 65, 67)
        for i in tl.static_range(4):
            off = PAIRS[i]
            al = tl.load(OUT_PTR + idx + off, rows[:, None], 0)
            ah = tl.load(OUT_PTR + idx + off - 1, rows[:, None], 0)
            bl = tl.load(DO_PTR + idx + off, rows[:, None], 0)
            bh = tl.load(DO_PTR + idx + off - 1, rows[:, None], 0)
            ap = al.to(tl.uint16, bitcast=True).to(tl.uint32) | (
                ah.to(tl.uint16, bitcast=True).to(tl.uint32) << 16
            )
            bp = bl.to(tl.uint16, bitcast=True).to(tl.uint32) | (
                bh.to(tl.uint16, bitcast=True).to(tl.uint32) << 16
            )
            acc = tl.inline_asm_elementwise(
                "v_dot2_f32_f16 $0, $1, $2, $3;",
                "=v,v,v,v",
                [ap, bp, acc],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        OFFSETS: tl.constexpr = (
            69,
            129,
            68,
            128,
            131,
            130,
            133,
            132,
            134,
            70,
            135,
            6,
            71,
            7,
        )
        for i in tl.static_range(14):
            off = OFFSETS[i]
            a = tl.load(OUT_PTR + idx + off, rows[:, None], 0).to(tl.float32)
            b = tl.load(DO_PTR + idx + off, rows[:, None], 0).to(tl.float32)
            acc += a * b
        for step in tl.static_range(3):
            other = tl.broadcast_to((lane ^ (4 >> step))[None, :], (BM, 8))
            acc += tl.gather(acc, other, 1)
        return tl.sum(tl.where(lane[None, :] == 0, acc, 0.0), 1)
    else:
        return tl.sum(o * do.to(tl.float32), 1)


@triton.jit
def _aten_flash_bwd_dq(
    Q,
    K,
    V,
    O,
    DO,
    LSE,
    DELTA,
    DQ,
    CQ,
    CK,
    RNG,
    RNG_OFFSET,
    threshold,
    drop_scale,
    keep_prob,
    sq_max,
    sk_max,
    total_q,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    DIM: tl.constexpr,
    BD: tl.constexpr,
    scale,
    scale_log2,
    LSE_SCALE: tl.constexpr,
    LB: tl.constexpr,
    LH: tl.constexpr,
    LT: tl.constexpr,
    WL: tl.constexpr,
    WR: tl.constexpr,
    TOP_LEFT_CAUSAL: tl.constexpr,
    VARLEN: tl.constexpr,
    PADDED_LSE: tl.constexpr,
    DROPOUT: tl.constexpr,
    WIDE_SEQUENCE_INDICES: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    WRITE_DELTA: tl.constexpr = True,
    ASCEND_LEGACY_COMPILER: tl.constexpr = False,
    FULL_PRECISION: tl.constexpr = False,
):
    block = tl.program_id(0)
    if WIDE_SEQUENCE_INDICES:
        block = block.to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    head = tl.program_id(2).to(tl.int64)
    if VARLEN:
        q0 = tl.load(CQ + batch).to(tl.int64)
        k0 = tl.load(CK + batch).to(tl.int64)
        sq = tl.load(CQ + batch + 1).to(tl.int64) - q0
        sk = tl.load(CK + batch + 1).to(tl.int64) - k0
        lbase = head * LH + q0 * LT
        if PADDED_LSE:
            lbase = batch * LB + head * LH
    else:
        sq, sk = sq_max, sk_max
        if WIDE_SEQUENCE_INDICES:
            sq, sk = tl.cast(sq, tl.int64), tl.cast(sk, tl.int64)
        q0, k0 = batch * sq, batch * sk
        lbase = batch * LB + head * LH
    m = (block * BM + tl.arange(0, BM)).to(tl.int64)
    n = tl.arange(0, BN).to(tl.int64)
    d = tl.arange(0, BD)
    qi = ((q0 + m[:, None]) * HQ + head) * DIM + d[None, :]
    qm = (m[:, None] < sq) & (d[None, :] < DIM)
    q = tl.load(Q + qi, qm, 0)
    do = tl.load(DO + qi, qm, 0)
    o = tl.load(O + qi, qm, 0).to(tl.float32)
    delta = (
        _aten_flash_delta(
            o, do, O, DO, ((q0 + m) * HQ + head) * DIM, m < sq, DIM, BM, DROPOUT
        )
        * keep_prob
    )
    if WRITE_DELTA:
        tl.store(DELTA + head * total_q + q0 + m, delta, m < sq)
    lse = tl.load(LSE + lbase + m * LT, m < sq, float("inf"))
    dq = tl.full((BM, BD), 0, tl.float32)
    lo = 0
    hi = sk
    offset = 0 if TOP_LEFT_CAUSAL else sk - sq
    if not ASCEND_LEGACY_COMPILER:
        if WL >= 0:
            lo = tl.maximum(0, block * BM + offset - WL)
        if WR >= 0:
            hi = tl.minimum(sk, (block + 1) * BM + offset + WR)
    for tile in range(lo // BN, tl.cdiv(tl.maximum(hi, 0), BN)):
        kn = tile * BN + n
        ki = ((k0 + kn[:, None]) * HK + head // (HQ // HK)) * DIM + d[None, :]
        km = (kn[:, None] < sk) & (d[None, :] < DIM)
        k = tl.load(K + ki, km, 0)
        v = tl.load(V + ki, km, 0)
        p = _aten_flash_probability(
            q,
            k,
            lse,
            m,
            kn,
            sq,
            sk,
            scale,
            scale_log2,
            LSE_SCALE,
            WL,
            WR,
            TOP_LEFT_CAUSAL,
            (DROPOUT and DIM == 256) or (DIM == 192 and not DROPOUT),
        )
        dp = tl.dot(do, tl.trans(v))
        # Some NPU dot lowerings leave out-of-bounds tile lanes undefined.
        # A zero probability does not suppress a NaN in those lanes.
        dp_valid = (m[:, None] < sq) & (kn[None, :] < sk)
        if ASCEND_LEGACY_COMPILER:
            dp_valid |= dp == 0.0
        dp = tl.where(dp_valid, dp, 0.0)
        if DROPOUT:
            dp *= _hygon_dropout(
                RNG, RNG_OFFSET, batch, head, m, kn, HQ, threshold, 1.0, DIM
            )
        ds = p * (dp - delta[:, None])
        ds_valid = (m[:, None] < sq) & (kn[None, :] < sk)
        if ASCEND_LEGACY_COMPILER:
            ds_valid |= ds == 0.0
        ds = tl.where(ds_valid, ds, 0.0)
        if FULL_PRECISION:
            dq += tl.dot(ds, k.to(tl.float32), input_precision="ieee")
        else:
            dq += tl.dot(ds.to(q.dtype, fp_downcast_rounding="rtz"), k)
    tl.store(
        DQ + qi,
        tl.where(sk > 0, dq * (scale * drop_scale), 0.0).to(
            DQ.dtype.element_ty,
            fp_downcast_rounding=(
                "rtz"
                if DQ.dtype.element_ty == tl.bfloat16
                or DIM <= 32
                or (DIM == 192 and not DROPOUT)
                or (DIM > 128 and DIM <= 160)
                or (DIM > 192 and DIM <= 224)
                else "rtne"
            ),
        ),
        qm,
    )


@triton.jit
def _aten_flash_bwd_dkv(
    Q,
    K,
    V,
    O,
    DO,
    LSE,
    DELTA,
    DK,
    DV,
    CQ,
    CK,
    RNG,
    RNG_OFFSET,
    threshold,
    drop_scale,
    keep_prob,
    sq_max,
    sk_max,
    total_q,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    DIM: tl.constexpr,
    BD: tl.constexpr,
    scale,
    scale_log2,
    LSE_SCALE: tl.constexpr,
    LB: tl.constexpr,
    LH: tl.constexpr,
    LT: tl.constexpr,
    WL: tl.constexpr,
    WR: tl.constexpr,
    TOP_LEFT_CAUSAL: tl.constexpr,
    VARLEN: tl.constexpr,
    PADDED_LSE: tl.constexpr,
    DROPOUT: tl.constexpr,
    WIDE_SEQUENCE_INDICES: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HEAD_OFFSET: tl.constexpr = 0,
    COMPUTE_DELTA: tl.constexpr = False,
    ASCEND_LEGACY_COMPILER: tl.constexpr = False,
    FULL_PRECISION: tl.constexpr = False,
):
    block = tl.program_id(0)
    if WIDE_SEQUENCE_INDICES:
        block = block.to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    head_k = tl.program_id(2).to(tl.int64) - HEAD_OFFSET
    if VARLEN:
        q0 = tl.load(CQ + batch).to(tl.int64)
        k0 = tl.load(CK + batch).to(tl.int64)
        sq = tl.load(CQ + batch + 1).to(tl.int64) - q0
        sk = tl.load(CK + batch + 1).to(tl.int64) - k0
        lbase = q0 * LT
        if PADDED_LSE:
            lbase = batch * LB
    else:
        sq, sk = sq_max, sk_max
        if WIDE_SEQUENCE_INDICES:
            sq, sk = tl.cast(sq, tl.int64), tl.cast(sk, tl.int64)
        q0, k0 = batch * sq, batch * sk
        lbase = batch * LB
    m = tl.arange(0, BM).to(tl.int64)
    n = (block * BN + tl.arange(0, BN)).to(tl.int64)
    d = tl.arange(0, BD)
    ki = ((k0 + n[:, None]) * HK + head_k) * DIM + d[None, :]
    km = (n[:, None] < sk) & (d[None, :] < DIM)
    k = tl.load(K + ki, km, 0)
    v = tl.load(V + ki, km, 0)
    dk = tl.full((BN, BD), 0, tl.float32)
    dv = tl.full((BN, BD), 0, tl.float32)
    lo = 0
    hi = sq
    offset = 0 if TOP_LEFT_CAUSAL else sk - sq
    if not ASCEND_LEGACY_COMPILER:
        if WR >= 0:
            lo = tl.maximum(0, block * BN - offset - WR)
        if WL >= 0:
            hi = tl.minimum(sq, (block + 1) * BN - offset + WL)
    # Native stores each query head's contribution in the input dtype before
    # reducing the GQA group. Keep that rounding boundary, with a fixed-order
    # FP32 group reduction and no expanded KV workspace or atomics.
    for group in range(HQ // HK):
        head = head_k * (HQ // HK) + group
        dk_head = tl.full((BN, BD), 0, tl.float32)
        dv_head = tl.full((BN, BD), 0, tl.float32)
        for tile in range(lo // BM, tl.cdiv(tl.maximum(hi, 0), BM)):
            query_tile = tile
            if DIM == 192 and not DROPOUT and BM == 32:
                # DTK's D192 kernel accumulates 32-query blocks in reverse.
                query_tile = lo // BM + tl.cdiv(tl.maximum(hi, 0), BM) - 1 - tile
            qm = query_tile * BM + m
            qi = ((q0 + qm[:, None]) * HQ + head) * DIM + d[None, :]
            mask = (qm[:, None] < sq) & (d[None, :] < DIM)
            q = tl.load(Q + qi, mask, 0)
            do = tl.load(DO + qi, mask, 0)
            if COMPUTE_DELTA:
                o = tl.load(O + qi, mask, 0).to(tl.float32)
                delta = (
                    _aten_flash_delta(
                        o,
                        do,
                        O,
                        DO,
                        ((q0 + qm) * HQ + head) * DIM,
                        qm < sq,
                        DIM,
                        BM,
                        DROPOUT,
                    )
                    * keep_prob
                )
            else:
                delta = tl.load(DELTA + head * total_q + q0 + qm, qm < sq, 0)
            lse = tl.load(LSE + lbase + head * LH + qm * LT, qm < sq, float("inf"))
            p = _aten_flash_probability(
                q,
                k,
                lse,
                qm,
                n,
                sq,
                sk,
                scale,
                scale_log2,
                LSE_SCALE,
                WL,
                WR,
                TOP_LEFT_CAUSAL,
                (DROPOUT and DIM == 256) or (DIM == 192 and not DROPOUT),
            )
            dp = tl.dot(do, tl.trans(v))
            dp_valid = (qm[:, None] < sq) & (n[None, :] < sk)
            if ASCEND_LEGACY_COMPILER:
                dp_valid |= dp == 0.0
            dp = tl.where(dp_valid, dp, 0.0)
            p_drop = p
            if DROPOUT:
                keep = _hygon_dropout(
                    RNG, RNG_OFFSET, batch, head, qm, n, HQ, threshold, 1.0, DIM
                )
                dp *= keep
                p_drop = p * keep
            ds = p * (dp - delta[:, None])
            ds_valid = (qm[:, None] < sq) & (n[None, :] < sk)
            if ASCEND_LEGACY_COMPILER:
                ds_valid |= ds == 0.0
            ds = tl.where(ds_valid, ds, 0.0)
            if FULL_PRECISION:
                dk_head += tl.dot(
                    tl.trans(ds), q.to(tl.float32), input_precision="ieee"
                )
                dv_head += tl.dot(
                    tl.trans(p_drop), do.to(tl.float32), input_precision="ieee"
                )
            else:
                if DIM == 192 and q.dtype == tl.float16 and not DROPOUT and BM != 32:
                    dk_head += tl.trans(
                        tl.dot(tl.trans(q), ds.to(q.dtype, fp_downcast_rounding="rtz"))
                    )
                else:
                    dk_head += tl.dot(
                        tl.trans(ds.to(q.dtype, fp_downcast_rounding="rtz")), q
                    )
                if (DROPOUT and DIM == 128) or (
                    not DROPOUT and DIM == 80 and q.dtype == tl.float16
                ):
                    # Match native D128 dropout and FP16 D80 p0 dV orientation.
                    dv_head += tl.trans(
                        tl.dot(
                            tl.trans(do), p_drop.to(q.dtype, fp_downcast_rounding="rtz")
                        )
                    )
                else:
                    dv_head += tl.dot(
                        tl.trans(p_drop.to(q.dtype, fp_downcast_rounding="rtz")), do
                    )
        if FULL_PRECISION:
            dk += dk_head * (scale * drop_scale)
            dv += dv_head * drop_scale
        else:
            dk += (
                (dk_head * (scale * drop_scale))
                .to(
                    k.dtype,
                    fp_downcast_rounding=(
                        "rtz"
                        if k.dtype == tl.bfloat16
                        or DIM <= 32
                        or (DIM == 192 and not DROPOUT)
                        or (DIM > 128 and DIM <= 160)
                        or (DIM > 192 and DIM <= 224)
                        else "rtne"
                    ),
                )
                .to(tl.float32)
            )
            dv += (
                (dv_head * drop_scale)
                .to(
                    v.dtype,
                    fp_downcast_rounding=(
                        "rtz"
                        if v.dtype == tl.bfloat16
                        or DIM <= 32
                        or (DIM == 192 and not DROPOUT)
                        or (DIM > 128 and DIM <= 160)
                        or (DIM > 192 and DIM <= 224)
                        else "rtne"
                    ),
                )
                .to(tl.float32)
            )
    tl.store(DK + ki, tl.where(sq > 0, dk, 0.0), km)
    tl.store(DV + ki, tl.where(sq > 0, dv, 0.0), km)


@triton.jit
def _hygon_dk_sequential(
    ds, q, acc, BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr
):
    # Preserve query order before DTK casts each BF16 head contribution.
    for r in range(BM):
        dsrow = (
            tl.gather(ds, tl.full((1, BN), r, tl.int32), axis=0)
            .reshape(BN)
            .to(tl.float32)
        )
        qrow = (
            tl.gather(q, tl.full((1, BD), r, tl.int32), axis=0)
            .reshape(BD)
            .to(tl.float32)
        )
        acc = acc + dsrow[:, None] * qrow[None, :]
    return acc


@triton.jit
def _hygon_bwd_dkv_per_head(
    Q,
    K,
    V,
    O,
    DO,
    LSE,
    DELTA,
    DK,
    DV,
    CQ,
    CK,
    RNG,
    RNG_OFFSET,
    threshold,
    drop_scale,
    keep_prob,
    sq_max,
    sk_max,
    total_q,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    DIM: tl.constexpr,
    BD: tl.constexpr,
    scale,
    scale_log2,
    LSE_SCALE: tl.constexpr,
    LB: tl.constexpr,
    LH: tl.constexpr,
    LT: tl.constexpr,
    WL: tl.constexpr,
    WR: tl.constexpr,
    TOP_LEFT_CAUSAL: tl.constexpr,
    VARLEN: tl.constexpr,
    PADDED_LSE: tl.constexpr,
    DROPOUT: tl.constexpr,
    WIDE_SEQUENCE_INDICES: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HEAD_OFFSET: tl.constexpr = 0,
    COMPUTE_DELTA: tl.constexpr = False,
    ASCEND_LEGACY_COMPILER: tl.constexpr = False,
    FULL_PRECISION: tl.constexpr = False,
):
    block = tl.program_id(0)
    if WIDE_SEQUENCE_INDICES:
        block = block.to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    head_q = tl.program_id(2).to(tl.int64)
    head_k = head_q // (HQ // HK)
    if VARLEN:
        q0 = tl.load(CQ + batch).to(tl.int64)
        k0 = tl.load(CK + batch).to(tl.int64)
        sq = tl.load(CQ + batch + 1).to(tl.int64) - q0
        sk = tl.load(CK + batch + 1).to(tl.int64) - k0
        lbase = q0 * LT
        if PADDED_LSE:
            lbase = batch * LB
    else:
        sq, sk = sq_max, sk_max
        if WIDE_SEQUENCE_INDICES:
            sq, sk = tl.cast(sq, tl.int64), tl.cast(sk, tl.int64)
        q0, k0 = batch * sq, batch * sk
        lbase = batch * LB
    m = tl.arange(0, BM).to(tl.int64)
    n = (block * BN + tl.arange(0, BN)).to(tl.int64)
    d = tl.arange(0, BD)
    ki = ((k0 + n[:, None]) * HK + head_k) * DIM + d[None, :]
    km = (n[:, None] < sk) & (d[None, :] < DIM)
    k = tl.load(K + ki, km, 0)
    v = tl.load(V + ki, km, 0)
    dk = tl.full((BN, BD), 0, tl.float32)
    dv = tl.full((BN, BD), 0, tl.float32)
    lo = 0
    hi = sq
    offset = 0 if TOP_LEFT_CAUSAL else sk - sq
    if not ASCEND_LEGACY_COMPILER:
        if WR >= 0:
            lo = tl.maximum(0, block * BN - offset - WR)
        if WL >= 0:
            hi = tl.minimum(sq, (block + 1) * BN - offset + WL)
    # Native stores each query head's contribution in the input dtype before
    # reducing the GQA group. Keep that rounding boundary, with a fixed-order
    # FP32 group reduction and no expanded KV workspace or atomics.
    for group in range(1):
        head = head_q
        dk_head = tl.full((BN, BD), 0, tl.float32)
        dv_head = tl.full((BN, BD), 0, tl.float32)
        for tile in range(lo // BM, tl.cdiv(tl.maximum(hi, 0), BM)):
            qm = tile * BM + m
            qi = ((q0 + qm[:, None]) * HQ + head) * DIM + d[None, :]
            mask = (qm[:, None] < sq) & (d[None, :] < DIM)
            q = tl.load(Q + qi, mask, 0)
            do = tl.load(DO + qi, mask, 0)
            if COMPUTE_DELTA:
                o = tl.load(O + qi, mask, 0).to(tl.float32)
                delta = (
                    _aten_flash_delta(
                        o,
                        do,
                        O,
                        DO,
                        ((q0 + qm) * HQ + head) * DIM,
                        qm < sq,
                        DIM,
                        BM,
                        DROPOUT,
                    )
                    * keep_prob
                )
            else:
                delta = tl.load(DELTA + head * total_q + q0 + qm, qm < sq, 0)
            lse = tl.load(LSE + lbase + head * LH + qm * LT, qm < sq, float("inf"))
            p = _aten_flash_probability(
                q,
                k,
                lse,
                qm,
                n,
                sq,
                sk,
                scale,
                scale_log2,
                LSE_SCALE,
                WL,
                WR,
                TOP_LEFT_CAUSAL,
                (DROPOUT and DIM == 256) or (DIM == 192 and not DROPOUT),
            )
            dp = tl.dot(do, tl.trans(v))
            dp_valid = (qm[:, None] < sq) & (n[None, :] < sk)
            if ASCEND_LEGACY_COMPILER:
                dp_valid |= dp == 0.0
            dp = tl.where(dp_valid, dp, 0.0)
            p_drop = p
            if DROPOUT:
                keep = _hygon_dropout(
                    RNG, RNG_OFFSET, batch, head, qm, n, HQ, threshold, 1.0, DIM
                )
                dp *= keep
                p_drop = p * keep
            ds = p * (dp - delta[:, None])
            ds_valid = (qm[:, None] < sq) & (n[None, :] < sk)
            if ASCEND_LEGACY_COMPILER:
                ds_valid |= ds == 0.0
            ds = tl.where(ds_valid, ds, 0.0)
            if FULL_PRECISION:
                dk_head += tl.dot(
                    tl.trans(ds), q.to(tl.float32), input_precision="ieee"
                )
                dv_head += tl.dot(
                    tl.trans(p_drop), do.to(tl.float32), input_precision="ieee"
                )
            else:
                if DIM == 80 and q.dtype == tl.bfloat16 and DROPOUT:
                    dk_head = _hygon_dk_sequential(
                        ds.to(q.dtype, fp_downcast_rounding="rtz"),
                        q,
                        dk_head,
                        BM,
                        BN,
                        BD,
                    )
                else:
                    if DIM == 192 and q.dtype == tl.float16 and not DROPOUT:
                        dk_head += tl.trans(
                            tl.dot(
                                tl.trans(q), ds.to(q.dtype, fp_downcast_rounding="rtz")
                            )
                        )
                    else:
                        if DIM == 80 and q.dtype == tl.float16 and DROPOUT:
                            # Native parity needs each query tile's MMA to start at zero.
                            # The identity keeps the FP32 outer add from fusing into MMA.
                            dk_part = tl.dot(
                                tl.trans(ds.to(q.dtype, fp_downcast_rounding="rtz")), q
                            )
                            dk_part = tl.inline_asm_elementwise(
                                "v_mov_b32 $0, $1;",
                                "=v,v",
                                [dk_part],
                                dtype=tl.float32,
                                is_pure=True,
                                pack=1,
                            )
                            dk_head += dk_part
                        else:
                            dk_head += tl.dot(
                                tl.trans(ds.to(q.dtype, fp_downcast_rounding="rtz")), q
                            )
                if (DROPOUT and DIM == 128) or (
                    not DROPOUT and DIM == 80 and q.dtype == tl.float16
                ):
                    # Match native D128 dropout and FP16 D80 p0 dV orientation.
                    dv_head += tl.trans(
                        tl.dot(
                            tl.trans(do), p_drop.to(q.dtype, fp_downcast_rounding="rtz")
                        )
                    )
                else:
                    if DIM == 80 and q.dtype == tl.float16 and DROPOUT:
                        dv_part = tl.dot(
                            tl.trans(p_drop.to(q.dtype, fp_downcast_rounding="rtz")), do
                        )
                        dv_part = tl.inline_asm_elementwise(
                            "v_mov_b32 $0, $1;",
                            "=v,v",
                            [dv_part],
                            dtype=tl.float32,
                            is_pure=True,
                            pack=1,
                        )
                        dv_head += dv_part
                    else:
                        dv_head += tl.dot(
                            tl.trans(p_drop.to(q.dtype, fp_downcast_rounding="rtz")), do
                        )
        if FULL_PRECISION:
            dk += dk_head * (scale * drop_scale)
            dv += dv_head * drop_scale
        else:
            dk += (
                (dk_head * (scale * drop_scale))
                .to(
                    k.dtype,
                    fp_downcast_rounding=(
                        "rtz"
                        if k.dtype == tl.bfloat16
                        or DIM <= 32
                        or (DIM == 192 and not DROPOUT)
                        or (DIM > 128 and DIM <= 160)
                        or (DIM > 192 and DIM <= 224)
                        else "rtne"
                    ),
                )
                .to(tl.float32)
            )
            dv += (
                (dv_head * drop_scale)
                .to(
                    v.dtype,
                    fp_downcast_rounding=(
                        "rtz"
                        if v.dtype == tl.bfloat16
                        or DIM <= 32
                        or (DIM == 192 and not DROPOUT)
                        or (DIM > 128 and DIM <= 160)
                        or (DIM > 192 and DIM <= 224)
                        else "rtne"
                    ),
                )
                .to(tl.float32)
            )
    oi = ((k0 + n[:, None]) * HQ + head_q) * DIM + d[None, :]
    tl.store(DK + oi, tl.where(sq > 0, dk, 0.0), km)
    tl.store(DV + oi, tl.where(sq > 0, dv, 0.0), km)


@triton.jit
def _hygon_reduce_gqa(
    DKH,
    DVH,
    DK,
    DV,
    N: tl.constexpr,
    HK: tl.constexpr,
    DIM: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // (HK * DIM)
    head = (i // DIM) % HK
    col = i % DIM
    base = (row * HK * GROUP + head * GROUP) * DIM + col
    dk = tl.full((BLOCK,), 0, tl.float32)
    dv = tl.full((BLOCK,), 0, tl.float32)
    for g in tl.static_range(GROUP):
        dk += tl.load(DKH + base + g * DIM, i < N, 0).to(tl.float32)
        dv += tl.load(DVH + base + g * DIM, i < N, 0).to(tl.float32)
    tl.store(DK + i, dk, i < N)
    tl.store(DV + i, dv, i < N)


@triton.jit
def _aten_flash_bwd_small(
    Q,
    K,
    V,
    O,
    DO,
    LSE,
    DQ,
    DK,
    DV,
    CQ,
    CK,
    RNG,
    RNG_OFFSET,
    threshold,
    drop_scale,
    keep_prob,
    sq_max,
    sk_max,
    total_q,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    DIM: tl.constexpr,
    BD: tl.constexpr,
    scale,
    scale_log2,
    LSE_SCALE: tl.constexpr,
    LB: tl.constexpr,
    LH: tl.constexpr,
    LT: tl.constexpr,
    WL: tl.constexpr,
    WR: tl.constexpr,
    TOP_LEFT_CAUSAL: tl.constexpr,
    VARLEN: tl.constexpr,
    PADDED_LSE: tl.constexpr,
    DROPOUT: tl.constexpr,
    WIDE_SEQUENCE_INDICES: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    FULL_PRECISION: tl.constexpr = False,
):
    block = tl.program_id(0)
    if WIDE_SEQUENCE_INDICES:
        block = block.to(tl.int64)
    # Disjoint programs own Q or KV outputs. KV programs recompute their small
    # delta tiles, removing the inter-kernel dependency and delta allocation.
    if tl.program_id(2) < HQ:
        if block * BM < sq_max:
            _aten_flash_bwd_dq(
                Q,
                K,
                V,
                O,
                DO,
                LSE,
                Q,
                DQ,
                CQ,
                CK,
                RNG,
                RNG_OFFSET,
                threshold,
                drop_scale,
                keep_prob,
                sq_max,
                sk_max,
                total_q,
                HQ,
                HK,
                DIM,
                BD,
                scale,
                scale_log2,
                LSE_SCALE,
                LB,
                LH,
                LT,
                WL,
                WR,
                TOP_LEFT_CAUSAL,
                VARLEN,
                PADDED_LSE,
                DROPOUT,
                WIDE_SEQUENCE_INDICES,
                BM,
                BN,
                WRITE_DELTA=False,
                FULL_PRECISION=FULL_PRECISION,
            )
    else:
        if block * BN < sk_max:
            _aten_flash_bwd_dkv(
                Q,
                K,
                V,
                O,
                DO,
                LSE,
                Q,
                DK,
                DV,
                CQ,
                CK,
                RNG,
                RNG_OFFSET,
                threshold,
                drop_scale,
                keep_prob,
                sq_max,
                sk_max,
                total_q,
                HQ,
                HK,
                DIM,
                BD,
                scale,
                scale_log2,
                LSE_SCALE,
                LB,
                LH,
                LT,
                WL,
                WR,
                TOP_LEFT_CAUSAL,
                VARLEN,
                PADDED_LSE,
                DROPOUT,
                WIDE_SEQUENCE_INDICES,
                BM,
                BN,
                HEAD_OFFSET=HQ,
                COMPUTE_DELTA=True,
                FULL_PRECISION=FULL_PRECISION,
            )


def _aten_flash_backward(
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    rng_state,
    unused,
    *,
    scale=None,
    window_size_left=None,
    window_size_right=None,
    _tile_override=None,
    _warps_override=None,
):
    # Validate metadata without reading device values or synchronizing the stream.
    if query.ndim != 4 or any(t.ndim != 4 for t in (key, value, out, grad_out)):
        raise ValueError("Hygon dense flash_attention_backward expects BSHD tensors")
    if any(
        t.dtype != query.dtype for t in (key, value, out, grad_out)
    ) or query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "flash_attention_backward requires matching float16 or bfloat16 tensors"
        )
    if any(t.device != query.device for t in (key, value, out, grad_out, logsumexp)):
        raise ValueError("flash_attention_backward tensors must be on the same device")
    if (
        query.shape != out.shape
        or query.shape != grad_out.shape
        or key.shape != value.shape
    ):
        raise ValueError("incompatible query/output/gradient or key/value shapes")
    dim, hq, hk = query.shape[-1], query.shape[-2], key.shape[-2]
    if dim != key.shape[-1] or dim <= 0 or dim > 256 or dim % 8:
        raise ValueError("head dimension must match and be a multiple of 8 in [8, 256]")
    if hq <= 0 or hk <= 0 or hq % hk:
        raise ValueError("query heads must be a positive multiple of KV heads")
    if any(t.stride(-1) != 1 for t in (query, key, value, out)):
        raise ValueError(
            "query, key, value and out must have contiguous last dimensions"
        )
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0, 1)")
    threshold, drop_scale, keep32 = 255, 1.0, 1.0
    if dropout_p:
        p32 = struct.unpack("f", struct.pack("f", dropout_p))[0]
        if p32 >= 1.0:
            raise ValueError("dropout_p must remain below 1 after float32 conversion")
        keep32 = struct.unpack("f", struct.pack("f", 1.0 - p32))[0]
        threshold = int(keep32 * 255.0)
        drop_scale = struct.unpack("f", struct.pack("f", 1.0 / keep32))[0]
        for state in (rng_state, unused):
            if (
                state is None
                or state.dtype != torch.int64
                or state.device != query.device
                or state.shape != ()
                or not state.is_contiguous()
            ):
                raise ValueError(
                    "Hygon dropout requires device int64 scalar seed and offset"
                )
    wl = -1 if window_size_left is None else window_size_left
    wr = -1 if window_size_right is None else window_size_right
    wl, wr = max(-1, wl), max(-1, wr)
    if is_causal:
        wr = 0
    if max_q < 0 or max_k < 0:
        raise ValueError("maximum sequence lengths must be nonnegative")
    if cum_seq_q is not None or cum_seq_k is not None:
        raise ValueError("dense tensors require undefined cumulative sequence arrays")
    batch, sq = query.shape[:2]
    if key.shape[0] != batch:
        raise ValueError("query and key batch sizes must match")
    max_q, max_k = sq, key.shape[1]
    total_q = batch * sq
    expected_lse = (batch, hq, sq)
    lb, lh, lt = logsumexp.stride() if logsumexp.ndim == 3 else (0, 0, 0)
    if logsumexp.dtype != torch.float32 or tuple(logsumexp.shape) != expected_lse:
        raise ValueError(f"logsumexp must be float32 with shape {expected_lse}")
    # FlashAttention canonicalizes windows using the public K length bound.
    # In particular, a wide right window can unmask early rows when Sq > Sk.
    if wl >= max_k:
        wl = -1
    if wr >= max_k:
        wr = -1
    scale = dim**-0.5 if scale is None else scale
    scale_log2 = None
    lse_scale = math.log2(math.e)
    launch_options = {
        "matrix_instr_nonkdim": 16,
        "kpack": (1 if dim == 192 and not dropout_p else 2),
        "enable_fp_fusion": False,
    }
    try:
        scale32 = struct.unpack("f", struct.pack("f", scale))[0]
        scale_log2 = struct.unpack("f", struct.pack("f", scale32 * math.log2(math.e)))[
            0
        ]
    except OverflowError:
        scale_log2 = math.copysign(math.inf, scale)
    q, k, v, o, do = (t.contiguous() for t in (query, key, value, out, grad_out))
    dq, dk, dv = (torch.empty_like(t) for t in (q, k, v))
    if batch == 0:
        if q.numel() or k.numel():
            raise ValueError("zero sequences require empty packed inputs")
        return dq, dk, dv
    # Below 2**29, all tile/window expressions (at most three lengths plus
    # a tile <= 128) fit int32. Pointers still widen before head/D products.
    # Large length bounds use int64 throughout.
    wide_sequence_indices = max(max_q, max_k) >= 2**29
    common = (
        rng_state if dropout_p else q,
        unused if dropout_p else q,
        threshold,
        drop_scale,
        keep32,
        max_q,
        max_k,
        total_q,
        hq,
        hk,
        dim,
        max(16, triton.next_power_of_2(dim)),
        scale,
        scale_log2,
        lse_scale,
        lb,
        lh,
        lt,
        wl,
        wr,
        False,
        False,
        False,
        dropout_p > 0,
        wide_sequence_indices,
    )
    tile = (
        (64, 32)
        if max(max_q, max_k) >= 1536
        else (64 if max(max_q, max_k) >= 256 else 32)
    )
    fused_dropout = dropout_p > 0 and dim in (96, 128) and max(max_q, max_k) <= 256
    if fused_dropout:
        tile = 32
    num_warps = 4
    if _tile_override is not None:
        tile = _tile_override
    bm, bn = tile if isinstance(tile, tuple) else (tile, tile)
    if _warps_override is not None:
        num_warps = _warps_override
    # Short dropout workloads also benefit from one disjoint Q/KV launch.
    if max(max_q, max_k) <= 128 or fused_dropout:
        if max_q or max_k:
            _aten_flash_bwd_small[
                (max(triton.cdiv(max_q, bm), triton.cdiv(max_k, bn)), batch, hq + hk)
            ](
                q,
                k,
                v,
                o,
                do,
                logsumexp,
                dq,
                dk,
                dv,
                cum_seq_q,
                cum_seq_k,
                *common,
                BM=bm,
                BN=bn,
                num_warps=(
                    8
                    if dim == 192
                    and q.dtype == torch.float16
                    and not dropout_p
                    and _warps_override is None
                    else num_warps
                ),
                num_stages=1,
                **launch_options,
            )
        return dq, dk, dv
    delta = torch.empty((hq, total_q), dtype=torch.float32, device=q.device)
    # Fixed-order, disjoint writes: deterministic in strict/warn/off modes.
    dq_bm = (
        128
        if (
            dim == 128
            and max_q == max_k
            and max_q >= 1536
            and not dropout_p
            and wl == -1
            and wr in (-1, 0)
            and _tile_override is None
        )
        else bm
    )
    if max_q:
        _aten_flash_bwd_dq[(triton.cdiv(max_q, dq_bm), batch, hq)](
            q,
            k,
            v,
            o,
            do,
            logsumexp,
            delta,
            dq,
            cum_seq_q,
            cum_seq_k,
            *common,
            BM=dq_bm,
            BN=bn,
            num_warps=num_warps,
            num_stages=1,
            **launch_options,
        )
    split_gqa = hq > hk and max(max_q, max_k) >= 256
    if max_k and split_gqa:
        dkh = torch.empty((*k.shape[:-2], hq, dim), dtype=k.dtype, device=k.device)
        dvh = torch.empty_like(dkh)
        _hygon_bwd_dkv_per_head[(triton.cdiv(max_k, bn), batch, hq)](
            q,
            k,
            v,
            o,
            do,
            logsumexp,
            delta,
            dkh,
            dvh,
            cum_seq_q,
            cum_seq_k,
            *common,
            BM=(
                32
                if dim == 80
                and q.dtype == torch.bfloat16
                and dropout_p
                and _tile_override is None
                else bm
            ),
            BN=bn,
            num_warps=(
                8
                if (
                    (
                        dim == 80
                        and (
                            (
                                q.dtype == torch.float16
                                and (not dropout_p or (wl == -1 and wr == 0))
                            )
                            or (q.dtype == torch.bfloat16 and dropout_p)
                        )
                    )
                    or (dim == 192 and q.dtype == torch.float16 and not dropout_p)
                )
                and _tile_override is None
                and _warps_override is None
                else num_warps
            ),
            num_stages=1,
            **launch_options,
        )
        _hygon_reduce_gqa[(triton.cdiv(k.numel(), 256),)](
            dkh,
            dvh,
            dk,
            dv,
            k.numel(),
            hk,
            dim,
            hq // hk,
            256,
            num_warps=4,
        )
    elif max_k:
        _aten_flash_bwd_dkv[(triton.cdiv(max_k, bn), batch, hk)](
            q,
            k,
            v,
            o,
            do,
            logsumexp,
            delta,
            dk,
            dv,
            cum_seq_q,
            cum_seq_k,
            *common,
            BM=bm,
            BN=bn,
            num_warps=(
                8
                if dim == 192
                and not dropout_p
                and bm == 32
                and _tile_override is None
                and _warps_override is None
                else num_warps
            ),
            num_stages=1,
            **launch_options,
        )
    return dq, dk, dv


def flash_attention_backward(
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    rng_state,
    unused,
    *,
    scale=None,
    window_size_left=None,
    window_size_right=None,
):
    logger.debug("GEMS FLASH_ATTENTION_BACKWARD")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0, 1)")
    if dropout_p and (query.ndim != 4 or (scale is not None and scale <= 0)):
        raise NotImplementedError(
            "Hygon dropout currently requires dense positive-scale inputs"
        )
    if dropout_p and query.shape[-1] not in (64, 80, 96, 128, 256):
        raise NotImplementedError(
            "Hygon dropout gradient validation is pending for this head dimension"
        )
    implementation = _aten_flash_backward
    extra = {}
    if query.ndim != 4 or (scale is not None and scale <= 0):
        from flag_gems.ops.flash_attention_backward import (
            _aten_flash_backward as implementation,
        )

        extra = {"_full_precision": True}
    return implementation(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        rng_state,
        unused,
        scale=scale,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        **extra,
    )
