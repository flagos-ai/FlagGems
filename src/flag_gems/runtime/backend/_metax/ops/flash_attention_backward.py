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


"""MetaX FlashAttention backward with split P/dS and FP32 GQA accumulation."""

import logging

import triton
import triton.language as tl
from triton.language.extra import libdevice

from flag_gems.ops.flash_attention_backward import (
    _aten_flash_backward,
    _aten_flash_dropout,
)

logger = logging.getLogger(__name__)


@triton.jit
def _metax_two_sum(ah, al, bh, bl):
    total = ah + bh
    part = total - ah
    error = (ah - (total - part)) + (bh - part)
    error = error + (al + bl)
    high = total + error
    part2 = high - total
    low = (total - (high - part2)) + (error - part2)
    return high, low


@triton.jit
def _metax_delta(
    o, do, OUT_PTR, DO_PTR, base, rows, DIM: tl.constexpr, BM: tl.constexpr
):
    product = o * do.to(tl.float32)
    high, low = tl.reduce(
        (product, tl.full(product.shape, 0, tl.float32)), 1, _metax_two_sum
    )
    return high + low


@triton.jit
def _metax_accurate_dp(DO, V, qbase, vbase, qmask, vmask, DIM: tl.constexpr):
    high = tl.full((qbase.shape[0], vbase.shape[0]), 0, tl.float32)
    low = tl.full((qbase.shape[0], vbase.shape[0]), 0, tl.float32)
    for block in range(tl.cdiv(DIM, 16)):
        d = block * 16 + tl.arange(0, 16)
        a = tl.load(
            DO + qbase[:, None] + d[None, :], qmask[:, None] & (d[None, :] < DIM), 0
        ).to(tl.float32)
        b = tl.load(
            V + vbase[:, None] + d[None, :], vmask[:, None] & (d[None, :] < DIM), 0
        ).to(tl.float32)
        product = a[:, None, :] * b[None, :, :]
        ph, pl = tl.reduce(
            (product, tl.full(product.shape, 0, tl.float32)), 2, _metax_two_sum
        )
        high, low = _metax_two_sum(high, low, ph, pl)
    return high + low


@triton.jit
def _metax_probability(
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
        x = scores * scale
        y = -lse[:, None]
        argument = x + y
        part = argument - x
        error = (x - (argument - part)) + (y - part)
        value = libdevice.exp(argument)
        probability = tl.fma(value, error, value)
    else:
        # PyTorch FlashAttention builds with UNFUSE_FMA and rounds the
        # host-computed scale*log2(e) once before reconstructing probabilities.
        probability = tl.exp2(scores * scale_log2 - lse[:, None] * LSE_SCALE)
    return tl.where(valid, probability, 0.0)


@triton.jit
def _metax_log_scale_dot(a, b):
    if b.dtype == tl.float16:
        # Normalize each row by an exact power of two before conversion.
        # Normalize the residual independently to preserve small products even
        # when the original row spans more than the FP16 exponent range.
        peak = tl.maximum(tl.max(tl.abs(a), 1), 1.0e-30)
        exponent = tl.floor(tl.log2(peak)) - 14.0
        factor = tl.exp2(-exponent)
        hi = (a * factor[:, None]).to(tl.float16)
        residual = a - hi.to(tl.float32) * tl.exp2(exponent)[:, None]
        low_peak = tl.maximum(tl.max(tl.abs(residual), 1), 1.0e-30)
        low_exponent = tl.floor(tl.log2(low_peak)) - 14.0
        lo = (residual * tl.exp2(-low_exponent)[:, None]).to(tl.float16)
        result = tl.dot(hi, b) * tl.exp2(exponent)[:, None]
        result += tl.dot(lo, b) * tl.exp2(low_exponent)[:, None]
    else:
        hi = a.to(b.dtype)
        lo = (a - hi.to(tl.float32)).to(b.dtype)
        residual = ((a - hi.to(tl.float32)) - lo.to(tl.float32)).to(b.dtype)
        result = tl.dot(hi, b) + tl.dot(lo, b) + tl.dot(residual, b)
    return result


@triton.jit
def _metax_split_dot(a, b, LOG_SCALE: tl.constexpr):
    if LOG_SCALE:
        return _metax_log_scale_dot(a, b)
    if b.dtype == tl.float16:
        # Normalize each row by an exact power of two before conversion.
        # Normalize the residual independently to preserve small products even
        # when the original row spans more than the FP16 exponent range.
        peak = tl.maximum(tl.max(tl.abs(a), 1), 1.0e-30)
        # peak is a finite positive normal float after the 1e-30 clamp.
        # Read its unbiased binary exponent directly, then build exact normal
        # powers of two. This removes log2/exp2 from the normalization path.
        exponent = ((peak.to(tl.int32, bitcast=True) >> 23) & 255) - 141
        factor = ((127 - exponent) << 23).to(tl.float32, bitcast=True)
        inverse = ((127 + exponent) << 23).to(tl.float32, bitcast=True)
        hi = (a * factor[:, None]).to(tl.float16)
        residual = a - hi.to(tl.float32) * inverse[:, None]
        low_peak = tl.maximum(tl.max(tl.abs(residual), 1), 1.0e-30)
        low_exponent = ((low_peak.to(tl.int32, bitcast=True) >> 23) & 255) - 141
        low_factor = ((127 - low_exponent) << 23).to(tl.float32, bitcast=True)
        low_inverse = ((127 + low_exponent) << 23).to(tl.float32, bitcast=True)
        lo = (residual * low_factor[:, None]).to(tl.float16)
        result = tl.dot(hi, b) * inverse[:, None]
        result += tl.dot(lo, b) * low_inverse[:, None]
    else:
        hi = a.to(b.dtype)
        lo = (a - hi.to(tl.float32)).to(b.dtype)
        residual = ((a - hi.to(tl.float32)) - lo.to(tl.float32)).to(b.dtype)
        result = tl.dot(hi, b) + tl.dot(lo, b) + tl.dot(residual, b)
    return result


@triton.jit
def _metax_flash_bwd_dq(
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
        _metax_delta(o, do, O, DO, ((q0 + m) * HQ + head) * DIM, m < sq, DIM, BM)
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
        p = _metax_probability(
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
        )
        if DIM == 256:
            dp = _metax_accurate_dp(
                DO,
                V,
                ((q0 + m) * HQ + head) * DIM,
                ((k0 + kn) * HK + head // (HQ // HK)) * DIM,
                m < sq,
                kn < sk,
                DIM,
            )
        else:
            dp = tl.dot(do, tl.trans(v))
        # Clear invalid lanes before subsequent products.
        dp_valid = (m[:, None] < sq) & (kn[None, :] < sk)
        if ASCEND_LEGACY_COMPILER:
            dp_valid |= dp == 0.0
        dp = tl.where(dp_valid, dp, 0.0)
        if DROPOUT:
            dp *= _aten_flash_dropout(RNG, batch, head, m, kn, HQ, threshold, 1.0)
        ds = p * (dp - delta[:, None])
        ds_valid = (m[:, None] < sq) & (kn[None, :] < sk)
        if ASCEND_LEGACY_COMPILER:
            ds_valid |= ds == 0.0
        ds = tl.where(ds_valid, ds, 0.0)
        dq += _metax_split_dot(ds, k, DIM == 128 and WR == 0)
    tl.store(DQ + qi, tl.where(sk > 0, dq * (scale * drop_scale), 0.0), qm)


@triton.jit
def _metax_flash_bwd_dkv(
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
    # Keep all query-head contributions in FP32 through the GQA reduction,
    # in fixed order without expanded KV workspace or atomics.
    for group in range(HQ // HK):
        head = head_k * (HQ // HK) + group
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
                    _metax_delta(
                        o, do, O, DO, ((q0 + qm) * HQ + head) * DIM, qm < sq, DIM, BM
                    )
                    * keep_prob
                )
            else:
                delta = tl.load(DELTA + head * total_q + q0 + qm, qm < sq, 0)
            lse = tl.load(LSE + lbase + head * LH + qm * LT, qm < sq, float("inf"))
            p = _metax_probability(
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
            )
            if DIM == 256:
                dp = _metax_accurate_dp(
                    DO,
                    V,
                    ((q0 + qm) * HQ + head) * DIM,
                    ((k0 + n) * HK + head_k) * DIM,
                    qm < sq,
                    n < sk,
                    DIM,
                )
            else:
                dp = tl.dot(do, tl.trans(v))
            dp_valid = (qm[:, None] < sq) & (n[None, :] < sk)
            if ASCEND_LEGACY_COMPILER:
                dp_valid |= dp == 0.0
            dp = tl.where(dp_valid, dp, 0.0)
            p_drop = p
            if DROPOUT:
                keep = _aten_flash_dropout(RNG, batch, head, qm, n, HQ, threshold, 1.0)
                dp *= keep
                p_drop = p * keep
            ds = p * (dp - delta[:, None])
            ds_valid = (qm[:, None] < sq) & (n[None, :] < sk)
            if ASCEND_LEGACY_COMPILER:
                ds_valid |= ds == 0.0
            ds = tl.where(ds_valid, ds, 0.0)
            dk_head += _metax_split_dot(tl.trans(ds), q, DIM == 128 and WR == 0)
            dv_head += _metax_split_dot(tl.trans(p_drop), do, DIM == 128 and WR == 0)
        dk += dk_head * (scale * drop_scale)
        dv += dv_head * drop_scale
    tl.store(DK + ki, tl.where(sq > 0, dk, 0.0), km)
    tl.store(DV + ki, tl.where(sq > 0, dv, 0.0), km)


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
        # All D > 128 pad to 256: the split dKV product with tile 32 needs
        # 96 KiB shared memory, exceeding C550's 64 KiB limit.
        _tile_override=16 if query.ndim in (3, 4) and query.shape[-1] > 128 else None,
        _kernel_override=(_metax_flash_bwd_dq, _metax_flash_bwd_dkv),
    )
