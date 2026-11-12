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

from flag_gems.ops.flash_attention_backward import _aten_flash_delta
from flag_gems.utils import tl_extra_shim as libdevice

logger = logging.getLogger(__name__)


@triton.jit
def _corex_flash_dropout(
    RNG,
    RNG_OFFSET,
    batch,
    head,
    m,
    n,
    sk,
    HQ: tl.constexpr,
    threshold,
    drop_scale,
    POSITION_WRAP32: tl.constexpr = False,
):
    # CoreX 4.4 native probes: Philox7, four words assigned to four-row groups.
    seed = tl.load(RNG).to(tl.uint64)
    offset = tl.load(RNG_OFFSET).to(tl.uint64)
    sequence = offset + tl.cast(batch * HQ + head, tl.uint64)
    position = ((m[:, None] // 16 * 16 + m[:, None] % 4) * sk + n[None, :]).to(
        tl.uint64
    )
    # CoreX 4.4 D128 native IR forms the position with i32 mul/add, then zext.
    # Keep the independent seed and sequence/offset at their original 64 bits.
    if POSITION_WRAP32:
        position = position.to(tl.uint32).to(tl.uint64)
    c0 = tl.full(position.shape, 0, tl.uint32) + sequence.to(tl.uint32)
    c1 = tl.full(position.shape, 0, tl.uint32) + (sequence >> 32).to(tl.uint32)
    c2, c3 = position.to(tl.uint32), (position >> 32).to(tl.uint32)
    k0, k1 = seed.to(tl.uint32), (seed >> 32).to(tl.uint32)
    for _ in tl.static_range(7):
        hi0, hi1 = tl.umulhi(c0, 0xD2511F53), tl.umulhi(c2, 0xCD9E8D57)
        lo0, lo1 = c0 * 0xD2511F53, c2 * 0xCD9E8D57
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0, k1 = (k0 + 0x9E3779B9).to(tl.uint32), (k1 + 0xBB67AE85).to(tl.uint32)
    slot = (m[:, None] % 16) // 4
    word = tl.where(slot == 0, c0, tl.where(slot == 1, c1, tl.where(slot == 2, c2, c3)))
    uniform = word.to(tl.float32) * 2.3283064365386963e-10
    return (uniform > threshold).to(tl.float32) * drop_scale


@triton.jit
def _corex_flash_probability(
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
    NATIVE_D128: tl.constexpr = False,
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
    if NATIVE_D128:
        probability = libdevice.exp2(
            (scores * scale) * 1.4426950408889634 + lse[:, None]
        )
    elif scale_log2 is None:
        probability = tl.exp(scores * scale - lse[:, None])
    else:
        # PyTorch FlashAttention builds with UNFUSE_FMA and rounds the
        # host-computed scale*log2(e) once before reconstructing probabilities.
        probability = (
            libdevice.exp(scores * scale + lse[:, None])
            if LSE_SCALE < -1.1
            else libdevice.exp2(scores * scale_log2 + lse[:, None])
        )
    return tl.where(valid, probability, 0.0)


@triton.jit
def _corex_rowsum128_grouped(OUT, DO, base, rows, BM: tl.constexpr):
    # The native MR RowSum reduces each 16-column group with XOR 8/4/2/1,
    # then adds the eight group sums in order. Preserve these FP32 boundaries.
    lane = tl.arange(0, 16)
    delta = tl.full((BM,), 0, tl.float32)
    for group in tl.static_range(0, 8):
        col = group * 16 + lane
        a = tl.load(OUT + base[:, None] + col[None, :], rows[:, None], 0).to(tl.float32)
        b = tl.load(DO + base[:, None] + col[None, :], rows[:, None], 0).to(tl.float32)
        acc = a * b
        acc8 = tl.sum(tl.reshape(acc, (BM, 2, 8)), 1)
        acc4 = tl.sum(tl.reshape(acc8, (BM, 2, 4)), 1)
        acc2 = tl.sum(tl.reshape(acc4, (BM, 2, 2)), 1)
        delta = delta + tl.sum(acc2, 1)
    return delta


@triton.jit
def _corex_flash_bwd_dq(
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
    if DIM == 128 and not DROPOUT:
        delta = _corex_rowsum128_grouped(
            O, DO, ((q0 + m) * HQ + head) * DIM, m < sq, BM
        )
    else:
        delta = _aten_flash_delta(
            o, do, O, DO, ((q0 + m) * HQ + head) * DIM, m < sq, DIM, BM
        ) * (1.0 if DROPOUT else keep_prob)
    if WRITE_DELTA:
        tl.store(DELTA + head * total_q + q0 + m, delta, m < sq)
    lse = tl.load(LSE + lbase + m * LT, m < sq, float("inf"))
    dq = tl.full((BM, BD), 0, tl.float32)
    dq_partial = tl.full((BM, BD), 0, tl.float32)
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
        p = _corex_flash_probability(
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
            DIM == 128,
        )
        dp = tl.dot(do, tl.trans(v))
        # Some NPU dot lowerings leave out-of-bounds tile lanes undefined.
        # A zero probability does not suppress a NaN in those lanes.
        dp_valid = (m[:, None] < sq) & (kn[None, :] < sk)
        if ASCEND_LEGACY_COMPILER:
            dp_valid |= dp == 0.0
        dp = tl.where(dp_valid, dp, 0.0)
        if DROPOUT:
            dp *= _corex_flash_dropout(
                RNG, RNG_OFFSET, batch, head, m, kn, sk, HQ, threshold, 1.0, DIM == 128
            )
        if DROPOUT:
            dp *= drop_scale
        ds = p * (dp - delta[:, None])
        ds_valid = (m[:, None] < sq) & (kn[None, :] < sk)
        if ASCEND_LEGACY_COMPILER:
            ds_valid |= ds == 0.0
        ds = tl.where(ds_valid, ds, 0.0)
        if (DIM == 128 or (DIM == 80 and not DROPOUT)) and not FULL_PRECISION:
            # Native MR stores each scaled 128-key partial in the input dtype
            # before adding it to the input-dtype running dQ.
            dq_partial += tl.dot(ds.to(q.dtype), k)
            if ((tile + 1) * BN) % 128 == 0 or tile + 1 == tl.cdiv(
                tl.maximum(hi, 0), BN
            ):
                scaled_partial = (dq_partial * scale).to(q.dtype).to(tl.float32)
                dq = (dq + scaled_partial).to(q.dtype).to(tl.float32)
                dq_partial = tl.full((BM, BD), 0, tl.float32)
        elif FULL_PRECISION:
            dq += tl.dot(ds, k.to(tl.float32), input_precision="ieee")
        else:
            dq += tl.dot(ds.to(q.dtype), k)
    if (DIM == 128 or (DIM == 80 and not DROPOUT)) and not FULL_PRECISION:
        tl.store(DQ + qi, tl.where(sk > 0, dq, 0.0), qm)
    else:
        tl.store(
            DQ + qi,
            tl.where(sk > 0, dq * (scale if DROPOUT else scale * drop_scale), 0.0),
            qm,
        )


@triton.jit
def _corex_flash_bwd_dkv(
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
    PER_HEAD: tl.constexpr = False,
    WRITE_DK: tl.constexpr = True,
    WRITE_DV: tl.constexpr = True,
):
    block = tl.program_id(0)
    if WIDE_SEQUENCE_INDICES:
        block = block.to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    head_id = tl.program_id(2).to(tl.int64) - HEAD_OFFSET
    head_k = head_id // (HQ // HK) if PER_HEAD else head_id
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
    # Reverse 16-row subtiles to preserve native CoreX dK accumulation order.
    row = tl.arange(0, BM)
    if WRITE_DV:
        m = ((BM // 16 - 1 - row // 16) * 16 + row % 16).to(tl.int64)
    else:
        # Match the native D128 p0 dS register query positions within each 16 rows.
        if Q.dtype.element_ty == tl.float16:
            m = ((row // 8) * 8 + (row % 2) * 4 + (row % 8) // 2).to(tl.int64)
        else:
            m = row.to(tl.int64)
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
    GROUPS: tl.constexpr = 1 if PER_HEAD else HQ // HK
    for group in range(GROUPS):
        head = head_id if PER_HEAD else head_k * (HQ // HK) + group
        dk_head = tl.full((BN, BD), 0, tl.float32)
        dv_head = tl.full((BN, BD), 0, tl.float32)
        dv_partial = tl.full((BN, BD), 0, tl.float32)
        dk_partial = tl.full((BN, BD), 0, tl.float32)
        for tile in range(lo // BM, tl.cdiv(tl.maximum(hi, 0), BM)):
            qm = tile * BM + m
            qi = ((q0 + qm[:, None]) * HQ + head) * DIM + d[None, :]
            mask = (qm[:, None] < sq) & (d[None, :] < DIM)
            q = tl.load(Q + qi, mask, 0)
            do = tl.load(DO + qi, mask, 0)
            if COMPUTE_DELTA:
                o = tl.load(O + qi, mask, 0).to(tl.float32)
                delta = _aten_flash_delta(
                    o, do, O, DO, ((q0 + qm) * HQ + head) * DIM, qm < sq, DIM, BM
                ) * (1.0 if DROPOUT else keep_prob)
            else:
                delta = tl.load(DELTA + head * total_q + q0 + qm, qm < sq, 0)
            lse = tl.load(LSE + lbase + head * LH + qm * LT, qm < sq, float("inf"))
            p = _corex_flash_probability(
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
                DIM == 128,
            )
            dp = tl.dot(do, tl.trans(v))
            dp_valid = (qm[:, None] < sq) & (n[None, :] < sk)
            if ASCEND_LEGACY_COMPILER:
                dp_valid |= dp == 0.0
            dp = tl.where(dp_valid, dp, 0.0)
            p_drop = p
            if DROPOUT:
                keep = _corex_flash_dropout(
                    RNG,
                    RNG_OFFSET,
                    batch,
                    head,
                    qm,
                    n,
                    sk,
                    HQ,
                    threshold,
                    1.0,
                    DIM == 128,
                )
                dp *= keep
                p_drop = (p * keep) * drop_scale
            if DROPOUT:
                dp *= drop_scale
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
                if DIM == 80 and not DROPOUT:
                    # Native rounds each scaled 128-query dK partial before
                    # adding it to the input-dtype per-head accumulator.
                    dk_partial += tl.dot(tl.trans(ds.to(q.dtype)), q)
                    if (tile + 1) * BM % 128 == 0 or tile + 1 == tl.cdiv(
                        tl.maximum(hi, 0), BM
                    ):
                        dk_head = (
                            (dk_head + (dk_partial * scale).to(k.dtype).to(tl.float32))
                            .to(k.dtype)
                            .to(tl.float32)
                        )
                        dk_partial = tl.full((BN, BD), 0, tl.float32)
                else:
                    dk_head += tl.dot(tl.trans(ds.to(q.dtype)), q)
                if DIM == 80 and not DROPOUT:
                    # The native common kernel rounds each 128-query dV partial
                    # before adding it to the input-dtype per-head accumulator.
                    dv_partial += tl.dot(tl.trans(p_drop.to(q.dtype)), do)
                    if (tile + 1) * BM % 128 == 0 or tile + 1 == tl.cdiv(
                        tl.maximum(hi, 0), BM
                    ):
                        dv_head = (
                            (dv_head + dv_partial.to(v.dtype).to(tl.float32))
                            .to(v.dtype)
                            .to(tl.float32)
                        )
                        dv_partial = tl.full((BN, BD), 0, tl.float32)
                else:
                    dv_head += tl.dot(tl.trans(p_drop.to(q.dtype)), do)
        if PER_HEAD:
            # Only the D80 p0 path launches this mode. Its per-head partials
            # have already been rounded to the input dtype at each boundary.
            oi = ((k0 + n[:, None]) * HQ + head) * DIM + d[None, :]
            if WRITE_DK:
                tl.store(DK + oi, tl.where(sq > 0, dk_head, 0.0), km)
            if WRITE_DV:
                tl.store(DV + oi, tl.where(sq > 0, dv_head, 0.0), km)
        else:
            if FULL_PRECISION:
                dk += dk_head * (scale if DROPOUT else scale * drop_scale)
                dv += dv_head
            else:
                if DIM == 80 and not DROPOUT:
                    dk += dk_head
                else:
                    dk += (
                        (dk_head * (scale if DROPOUT else scale * drop_scale))
                        .to(k.dtype)
                        .to(tl.float32)
                    )
                dv += dv_head.to(v.dtype).to(tl.float32)
    if not PER_HEAD:
        if WRITE_DK:
            tl.store(DK + ki, tl.where(sq > 0, dk, 0.0), km)
        if WRITE_DV:
            tl.store(DV + ki, tl.where(sq > 0, dv, 0.0), km)


@triton.jit
def _corex_flash_bwd_dkv_phase(
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
    PER_HEAD: tl.constexpr = False,
    WRITE_DK: tl.constexpr = True,
    WRITE_DV: tl.constexpr = True,
):
    # D128 p0 GQA only: runtime phase shares the final contraction site.
    block = tl.program_id(0)
    if WIDE_SEQUENCE_INDICES:
        block = block.to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    phase = tl.program_id(2) >= HK
    head_k = tl.program_id(2).to(tl.int64) % HK
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
    row = tl.arange(0, BM)
    if phase:
        m = ((BM // 16 - 1 - row // 16) * 16 + row % 16).to(tl.int64)
    else:
        if Q.dtype.element_ty == tl.float16:
            m = ((row // 8) * 8 + (row % 2) * 4 + (row % 8) // 2).to(tl.int64)
        else:
            m = row.to(tl.int64)
    n = (block * BN + tl.arange(0, BN)).to(tl.int64)
    d = tl.arange(0, BD)
    ki = ((k0 + n[:, None]) * HK + head_k) * DIM + d[None, :]
    km = (n[:, None] < sk) & (d[None, :] < DIM)
    k = tl.load(K + ki, km, 0)
    v = tl.load(V + ki, km, 0)
    acc = tl.full((BN, BD), 0, tl.float32)
    lo = 0
    hi = sq
    offset = 0 if TOP_LEFT_CAUSAL else sk - sq
    if WR >= 0:
        lo = tl.maximum(0, block * BN - offset - WR)
    if WL >= 0:
        hi = tl.minimum(sq, (block + 1) * BN - offset + WL)
    for group in range(HQ // HK):
        head = head_k * (HQ // HK) + group
        acc_head = tl.full((BN, BD), 0, tl.float32)
        for tile in range(lo // BM, tl.cdiv(tl.maximum(hi, 0), BM)):
            qm = tile * BM + m
            qi = ((q0 + qm[:, None]) * HQ + head) * DIM + d[None, :]
            mask = (qm[:, None] < sq) & (d[None, :] < DIM)
            q = tl.load(Q + qi, mask, 0)
            do = tl.load(DO + qi, mask, 0)
            lse = tl.load(LSE + lbase + head * LH + qm * LT, qm < sq, float("inf"))
            p = _corex_flash_probability(
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
                DIM == 128,
            )
            if phase:
                left = p.to(q.dtype)
                right = do
            else:
                delta = tl.load(DELTA + head * total_q + q0 + qm, qm < sq, 0)
                dp = tl.dot(do, tl.trans(v))
                valid = (qm[:, None] < sq) & (n[None, :] < sk)
                dp = tl.where(valid, dp, 0.0)
                ds = p * (dp - delta[:, None])
                ds = tl.where(valid, ds, 0.0)
                left = ds.to(q.dtype)
                right = q
            acc_head += tl.dot(tl.trans(left), right)
        if phase:
            acc += acc_head.to(v.dtype).to(tl.float32)
        else:
            acc += (acc_head * (scale * drop_scale)).to(k.dtype).to(tl.float32)
    if phase:
        tl.store(DV + ki, tl.where(sq > 0, acc, 0.0), km)
    else:
        tl.store(DK + ki, tl.where(sq > 0, acc, 0.0), km)


@triton.jit
def _corex_reduce_d80_heads(
    TK,
    TV,
    DK,
    DV,
    N: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = i < N
    key_row = i // (HK * DIM)
    key_head = (i // DIM) % HK
    feature = i % DIM
    dk = tl.full((BLOCK,), 0.0, tl.float32)
    dv = tl.full((BLOCK,), 0.0, tl.float32)
    for group in range(HQ // HK):
        head = key_head * (HQ // HK) + group
        offset = (key_row * HQ + head) * DIM + feature
        dk += tl.load(TK + offset, mask, 0).to(tl.float32)
        dv += tl.load(TV + offset, mask, 0).to(tl.float32)
    tl.store(DK + i, dk, mask)
    tl.store(DV + i, dv, mask)


@triton.jit
def _corex_flash_bwd_small(
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
            _corex_flash_bwd_dq(
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
            _corex_flash_bwd_dkv(
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


@triton.jit
def _corex_flash_bwd_dkv_split_grid(
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
    PER_HEAD: tl.constexpr = False,
    WRITE_DK: tl.constexpr = True,
    WRITE_DV: tl.constexpr = True,
):
    # Separate programs preserve the existing dK and dV contraction orders.
    if tl.program_id(2) < HK:
        _corex_flash_bwd_dkv(
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
            WRITE_DV=False,
            FULL_PRECISION=FULL_PRECISION,
        )
    else:
        _corex_flash_bwd_dkv(
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
            HEAD_OFFSET=HK,
            WRITE_DK=False,
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
):
    # Validate metadata without reading device values or synchronizing the stream.
    varlen = query.ndim == 3
    rank = 3 if varlen else 4
    if query.ndim not in (3, 4) or any(
        t.ndim != rank for t in (key, value, out, grad_out)
    ):
        raise ValueError("flash_attention_backward expects BSHD or packed THD tensors")
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
        for state in (rng_state, unused):
            if (
                not isinstance(state, torch.Tensor)
                or state.dtype != torch.int64
                or state.numel() != 1
                or state.device not in (torch.device("cpu"), query.device)
            ):
                raise ValueError(
                    "CoreX dropout state requires scalar CPU or query-device int64 seed and offset"
                )
        threshold = struct.unpack("f", struct.pack("f", dropout_p))[0]
        if threshold >= 1.0:
            raise ValueError("dropout_p must remain less than one in float32")
        keep32 = struct.unpack("f", struct.pack("f", 1.0 - threshold))[0]
        drop_scale = struct.unpack("f", struct.pack("f", 1.0 / keep32))[0]
        # Native eager returns CPU values; graph forward returns live device values.
        rng_state = rng_state.to(query.device, non_blocking=True)
        unused = unused.to(query.device, non_blocking=True)
    wl = -1 if window_size_left is None else window_size_left
    wr = -1 if window_size_right is None else window_size_right
    wl, wr = max(-1, wl), max(-1, wr)
    if is_causal:
        wr = 0
    if max_q < 0 or max_k < 0:
        raise ValueError("maximum sequence lengths must be nonnegative")
    # CoreX native forward returns packed HT or padded BHS LSE.
    padded_lse = logsumexp.ndim == 3
    if varlen:
        if cum_seq_q is None or cum_seq_k is None:
            raise ValueError("packed tensors require both cumulative sequence arrays")
        for cu in (cum_seq_q, cum_seq_k):
            if (
                cu.dtype != torch.int32
                or cu.device != query.device
                or cu.ndim != 1
                or not cu.is_contiguous()
                or cu.numel() < 1
            ):
                raise ValueError("cu_seqlens must be contiguous device int32 vectors")
        if cum_seq_q.shape != cum_seq_k.shape:
            raise ValueError("cumulative sequence arrays must have equal lengths")
        batch = cum_seq_q.numel() - 1
        total_q = query.shape[0]
        expected_lse = (hq, total_q)
        lb, lh, lt = (
            0,
            logsumexp.stride(0) if logsumexp.ndim == 2 else 0,
            logsumexp.stride(1) if logsumexp.ndim == 2 else 0,
        )
        if padded_lse:
            expected_lse = (batch, hq, max_q)
            lb, lh, lt = logsumexp.stride() if logsumexp.ndim == 3 else (0, 0, 0)
        # Monotone offsets, endpoints and per-sequence maxima are caller preconditions,
        # as in native FlashAttention. Checking their values would synchronize here.
        if (max_q == 0 and total_q) or (max_k == 0 and key.shape[0]):
            raise ValueError("nonempty packed inputs need nonzero maximum lengths")
    else:
        if cum_seq_q is not None or cum_seq_k is not None:
            raise ValueError(
                "dense tensors require undefined cumulative sequence arrays"
            )
        batch, sq = query.shape[:2]
        if key.shape[0] != batch:
            raise ValueError("query and key batch sizes must match")
        max_q, max_k = sq, key.shape[1]
        total_q = batch * sq
        expected_lse = (batch, hq, sq)
        lb, lh, lt = logsumexp.stride() if logsumexp.ndim == 3 else (0, 0, 0)
    if logsumexp.dtype != torch.float32 or tuple(logsumexp.shape) != expected_lse:
        raise ValueError(f"logsumexp must be float32 with shape {expected_lse}")
    # CoreX 4.4 causal and dropout modes ignore local windows for dense /
    # uniform packed BHS, while preserving causality. Native UNPAD-local is
    # unsupported and keeps its explicit mathematical window here.
    if (dropout_p > 0 or is_causal) and (
        not varlen
        or (padded_lse and total_q == batch * max_q and key.shape[0] == batch * max_k)
    ):
        wl, wr = -1, 0 if is_causal else -1
    # FlashAttention canonicalizes windows using the public K length bound.
    # In particular, a wide right window can unmask early rows when Sq > Sk.
    if wl >= max_k:
        wl = -1
    if wr >= max_k:
        wr = -1
    scale = dim**-0.5 if scale is None else scale
    scale_log2 = None
    lse_scale = math.log2(math.e)
    # Native kernels save negative natural-log or negative log2 LSE by D.
    lse_scale = -math.log2(math.e) if dim in (32, 96, 160, 192, 256) else -1.0
    try:
        scale32 = struct.unpack("f", struct.pack("f", scale))[0]
        scale_log2 = struct.unpack("f", struct.pack("f", scale32 * math.log2(math.e)))[
            0
        ]
    except OverflowError:
        scale_log2 = math.copysign(math.inf, scale)
    # Native mixed-length batches use HT LSE, and native UNPAD kernels reject
    # local windows. Uniform packed BHS inputs accept native local windows;
    # preserve their native low-dtype rounding along with dense inputs.
    math_packed = varlen and (
        (padded_lse and batch > 1 and total_q != batch * max_q)
        or (not padded_lse and (wl >= 0 or (wr >= 0 and not is_causal)))
    )
    launch_options = {"FULL_PRECISION": math_packed, "enable_fp_fusion": False}
    q, k, v, o, do = (t.contiguous() for t in (query, key, value, out, grad_out))
    dq, dk, dv = (torch.empty_like(t) for t in (q, k, v))
    if batch == 0:
        if q.numel() or k.numel():
            raise ValueError("zero sequences require empty packed inputs")
        return dq, dk, dv
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
        varlen,
        varlen and padded_lse,
        dropout_p > 0,
        wide_sequence_indices,
    )
    tile, num_warps = 32, 4
    if not varlen and dim in (64, 128) and hq == hk and min(max_q, max_k) >= 256:
        tile = 64
        if (
            dim == 64
            and min(max_q, max_k) >= 1024
            and not is_causal
            and wl < 0
            and wr < 0
        ):
            tile, num_warps = 128, 8
    if max(max_q, max_k) <= 128:
        if max_q or max_k:
            _corex_flash_bwd_small[
                (triton.cdiv(max(max_q, max_k), tile), batch, hq + hk)
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
                BM=tile,
                BN=tile,
                num_warps=num_warps,
                num_stages=1,
                **launch_options,
            )
        return dq, dk, dv
    delta = torch.empty((hq, total_q), dtype=torch.float32, device=q.device)
    # Fixed-order, disjoint writes: deterministic in strict/warn/off modes.
    if max_q:
        _corex_flash_bwd_dq[(triton.cdiv(max_q, tile), batch, hq)](
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
            BM=tile,
            BN=tile,
            num_warps=num_warps,
            num_stages=1,
            ASCEND_LEGACY_COMPILER=False,
            **launch_options,
        )
    # Bound temporary memory independently of sequence length. Each query
    # head owns its KV partials; a final ordered reduction preserves GQA sums.
    split_heads = (
        not varlen
        and dim == 80
        and dropout_p == 0
        and hq > hk
        and not math_packed
        and 2 * batch * max_k * hq * dim * k.element_size() <= 64 * 1024 * 1024
    )
    if split_heads:
        tk = torch.empty((batch * max_k, hq, dim), dtype=k.dtype, device=k.device)
        tv = torch.empty_like(tk)
    else:
        tk, tv = dk, dv
    # Isolate D128 p0 contractions without altering D80 per-head routing.
    separate_dkv = dim == 128 and dropout_p == 0 and hq > hk and not math_packed
    if max_k:
        uniform_packed_dkv = (
            separate_dkv
            and varlen
            and padded_lse
            and total_q == batch * max_q
            and key.shape[0] == batch * max_k
        )
        if uniform_packed_dkv:
            dkv_kernel = _corex_flash_bwd_dkv_split_grid
        elif separate_dkv:
            dkv_kernel = _corex_flash_bwd_dkv_phase
        else:
            dkv_kernel = _corex_flash_bwd_dkv
        dkv_kernel[
            (
                triton.cdiv(max_k, tile),
                batch,
                2 * hk if separate_dkv else (hq if split_heads else hk),
            )
        ](
            q,
            k,
            v,
            o,
            do,
            logsumexp,
            delta,
            tk,
            tv,
            cum_seq_q,
            cum_seq_k,
            *common,
            BM=(
                32
                if not varlen
                and dropout_p == 0.0
                and dim == 128
                and hq == hk
                and min(max_q, max_k) >= 256
                else tile
            ),
            BN=tile,
            num_warps=num_warps,
            num_stages=1,
            ASCEND_LEGACY_COMPILER=False,
            PER_HEAD=split_heads,
            **launch_options,
        )
        if split_heads:
            _corex_reduce_d80_heads[(triton.cdiv(dk.numel(), 256),)](
                tk,
                tv,
                dk,
                dv,
                dk.numel(),
                hq,
                hk,
                dim,
                BLOCK=256,
                num_warps=4,
                enable_fp_fusion=False,
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
    logger.debug("GEMS COREX FLASH_ATTENTION_BACKWARD")
    return _aten_flash_backward(
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
    )
