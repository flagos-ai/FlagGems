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

import inspect
import logging
import math
import struct

import torch
import triton
import triton.language as tl

try:
    from triton.experimental.tle import language as tle

    _HAS_ASYNC_LOAD = "is_async" in inspect.signature(tle.load).parameters
except (ImportError, AttributeError, TypeError, ValueError):
    _HAS_ASYNC_LOAD = False

logger = logging.getLogger(__name__)


# Keep the established load path when the optional compiler extension is absent.
if _HAS_ASYNC_LOAD:

    @triton.jit
    def _musa_matrix_load(
        ptr,
        mask,
        FULL_PRECISION: tl.constexpr,
        DIM: tl.constexpr,
        ASYNC_LOAD: tl.constexpr,
    ):
        if FULL_PRECISION:
            value = tl.load(ptr, mask, 0)
        elif DIM == 128 and ASYNC_LOAD:
            value = tle.load(ptr, mask, 0, is_async=True)
        else:
            value = tl.load(ptr, mask, 0)
        return value

else:

    @triton.jit
    def _musa_matrix_load(
        ptr,
        mask,
        FULL_PRECISION: tl.constexpr,
        DIM: tl.constexpr,
        ASYNC_LOAD: tl.constexpr,
    ):
        return tl.load(ptr, mask, 0)


@triton.jit
def _musa_flash_delta_serial8(
    o, do, BM: tl.constexpr, BD: tl.constexpr, DIM: tl.constexpr
):
    # Preserve the observed native reduction order: serial groups of eight.
    products = o.to(tl.float32) * do.to(tl.float32)
    even, odd = tl.split(tl.reshape(products, (BM, BD // 8, 4, 2)))
    even04, even26 = tl.split(tl.reshape(even, (BM, BD // 8, 2, 2)))
    odd15, odd37 = tl.split(tl.reshape(odd, (BM, BD // 8, 2, 2)))
    x0, x4 = tl.split(even04)
    x2, x6 = tl.split(even26)
    x1, x5 = tl.split(odd15)
    x3, x7 = tl.split(odd37)
    groups = tl.full((BM, BD // 8), 0, tl.float32) + x0
    groups = groups + x1
    groups = groups + x2
    groups = groups + x3
    groups = groups + x4
    groups = groups + x5
    groups = groups + x6
    groups = groups + x7
    result = tl.full((BM,), 0, tl.float32)
    for i in tl.static_range(0, triton.cdiv(DIM, 8)):
        result = result + tl.gather(groups, tl.full((BM, 1), i, tl.int32), 1).reshape(
            (BM,)
        )
    return result


@triton.jit
def _musa_dropout_scalar(
    seed, rng_offset, batch, head, m, n, sq, sk, HQ: tl.constexpr, DROP_P: tl.constexpr
):
    row_start = (
        (batch.to(tl.uint64) * HQ + head.to(tl.uint64)) * sq + m[:, None].to(tl.uint64)
    ) * sk
    vector_start = row_start + (n[None, :].to(tl.uint64) // 4) * 4
    vector_start = tl.where(vector_start % 8 < 4, vector_start // 4 * 4, vector_start)
    index = vector_start + n[None, :].to(tl.uint64) % 4
    counter = rng_offset.to(tl.uint64) // 4 + (index % 8) // 4
    subsequence = index // 8
    c0, c1 = counter.to(tl.uint32), (counter >> 32).to(tl.uint32)
    c2, c3 = subsequence.to(tl.uint32), (subsequence >> 32).to(tl.uint32)
    seed64 = seed.to(tl.uint64)
    k0, k1 = seed64.to(tl.uint32), (seed64 >> 32).to(tl.uint32)
    for _ in tl.static_range(10):
        hi0, hi1 = tl.umulhi(c0, 0xD2511F53), tl.umulhi(c2, 0xCD9E8D57)
        lo0, lo1 = c0 * 0xD2511F53, c2 * 0xCD9E8D57
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0, k1 = (k0 + 0x9E3779B9).to(tl.uint32), (k1 + 0xBB67AE85).to(tl.uint32)
    lane = index % 4
    bits = tl.where(lane == 0, c0, tl.where(lane == 1, c1, tl.where(lane == 2, c2, c3)))
    uniform = bits.to(tl.float32) * 2.3283064365386963e-10
    return (uniform >= tl.full((), DROP_P, tl.float32)).to(tl.float32)


@triton.jit
def _musa_dropout_group4(
    seed, rng_offset, batch, head, m, n, sq, sk, HQ: tl.constexpr, DROP_P: tl.constexpr
):
    # The host selects this helper only for dense Sk divisible by four.
    # Every kernel key vector is contiguous and begins at a multiple of four.
    N: tl.constexpr = n.shape[0]
    base_n = tl.min(tl.reshape(n, (N // 4, 4)), 1).to(tl.uint64)
    row_start = (
        (batch.to(tl.uint64) * HQ + head.to(tl.uint64)) * sq + m[:, None].to(tl.uint64)
    ) * sk
    index0 = row_start + base_n[None, :]
    counter = rng_offset.to(tl.uint64) // 4 + (index0 % 8) // 4
    subsequence = index0 // 8
    c0, c1 = counter.to(tl.uint32), (counter >> 32).to(tl.uint32)
    c2, c3 = subsequence.to(tl.uint32), (subsequence >> 32).to(tl.uint32)
    seed64 = seed.to(tl.uint64)
    k0, k1 = seed64.to(tl.uint32), (seed64 >> 32).to(tl.uint32)
    for _ in tl.static_range(10):
        hi0, hi1 = tl.umulhi(c0, 0xD2511F53), tl.umulhi(c2, 0xCD9E8D57)
        lo0, lo1 = c0 * 0xD2511F53, c2 * 0xCD9E8D57
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0, k1 = (k0 + 0x9E3779B9).to(tl.uint32), (k1 + 0xBB67AE85).to(tl.uint32)
    # Nested join adds the last axis: join(02,13) flattens to 0,1,2,3.
    words = tl.join(tl.join(c0, c2), tl.join(c1, c3))
    bits = tl.reshape(words, (m.shape[0], N))
    uniform = bits.to(tl.float32) * 2.3283064365386963e-10
    return (uniform >= tl.full((), DROP_P, tl.float32)).to(tl.float32)


@triton.jit
def _musa_dropout(
    seed,
    rng_offset,
    batch,
    head,
    m,
    n,
    sq,
    sk,
    HQ: tl.constexpr,
    DROP_P: tl.constexpr,
    GROUP4: tl.constexpr = False,
):
    if GROUP4:
        return _musa_dropout_group4(
            seed, rng_offset, batch, head, m, n, sq, sk, HQ, DROP_P
        )
    else:
        return _musa_dropout_scalar(
            seed, rng_offset, batch, head, m, n, sq, sk, HQ, DROP_P
        )


@triton.jit
def _musa_flash_probability(
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
    FULL_TILES: tl.constexpr = False,
):
    scores = tl.dot(q, tl.trans(k))
    offset = 0 if TOP_LEFT_CAUSAL else sk - sq
    distance = m[:, None] + offset - n[None, :]
    if FULL_TILES:
        # Tiles are fully in-bounds: m < sq and n < sk hold for every element,
        # so the original shape mask is replaced by a compile-time all-true
        # mask of the same (BLOCK_M, BLOCK_N) boolean shape.
        valid = tl.full(scores.shape, 1, tl.int1)
    else:
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
        # MuDNN probabilities require the score term to be fused with the
        # rounded LSE term before the base-two exponential.
        probability = tl.exp2(tl.fma(scores, scale_log2, -lse[:, None] * LSE_SCALE))
    return tl.where(valid, probability, 0.0)


@triton.jit
def _musa_bf16_product(a, b):
    # Move rounded FP32 midpoints toward the exact product before BF16 RNE.
    product = a * b
    bits = product.to(tl.uint32, bitcast=True)
    residual = tl.fma(a, b, -product)
    residual_bits = residual.to(tl.uint32, bitcast=True)
    grow = ((bits ^ residual_bits) & 0x80000000) == 0
    adjusted = bits + tl.where(grow, 1, -1).to(tl.uint32)
    midpoint = ((bits & 0xFFFF) == 0x8000) & (residual != 0)
    bits = tl.where(midpoint, adjusted, bits)
    return bits.to(tl.float32, bitcast=True).to(tl.bfloat16)


@triton.jit
def _musa_round_probability(p, DTYPE: tl.constexpr):
    if DTYPE == tl.bfloat16:
        # MUSA removes a plain BF16->FP32 cast roundtrip in the DQ kernel.
        # Preserve the native high-D probability rounding explicitly (RNE).
        bits = p.to(tl.uint32, bitcast=True)
        rounded = (bits + 32767 + ((bits >> 16) & 1)) & 0xFFFF0000
        rounded = tl.where((bits & 0x7F800000) == 0x7F800000, bits, rounded)
        return rounded.to(tl.float32, bitcast=True)
    return p.to(DTYPE).to(tl.float32)


@triton.jit(do_not_specialize=["seed", "rng_offset", "offset_intragraph"])
def _musa_flash_bwd_dq(
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
    WIDE_SEQUENCE_INDICES: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    FULL_PRECISION: tl.constexpr = False,
    ASYNC_LOAD: tl.constexpr = False,
    WRITE_DELTA: tl.constexpr = True,
    seed=0,
    rng_offset=0,
    DROP_P: tl.constexpr = 0.0,
    DROP_KEEP: tl.constexpr = 1.0,
    DROP_GROUP4: tl.constexpr = False,
    offset_intragraph=0,
    RNG_DEVICE: tl.constexpr = False,
    OUT_HEADS: tl.constexpr = 0,
    ALIGNED_S: tl.constexpr = 0,
):
    if DROP_P > 0 and RNG_DEVICE:
        seed = tl.load(tl.cast(seed, tl.pointer_type(tl.uint64)))
        rng_offset = tl.load(
            tl.cast(rng_offset, tl.pointer_type(tl.uint64))
        ) + offset_intragraph.to(tl.uint64)
    block = tl.program_id(0)
    if WIDE_SEQUENCE_INDICES:
        block = block.to(tl.int64)
    batch = tl.program_id(1).to(tl.int64 if WIDE_SEQUENCE_INDICES else tl.int32)
    head = tl.program_id(2).to(tl.int64 if WIDE_SEQUENCE_INDICES else tl.int32)
    if VARLEN:
        q0 = tl.load(CQ + batch).to(tl.int64 if WIDE_SEQUENCE_INDICES else tl.int32)
        k0 = tl.load(CK + batch).to(tl.int64 if WIDE_SEQUENCE_INDICES else tl.int32)
        sq = (
            tl.load(CQ + batch + 1).to(tl.int64 if WIDE_SEQUENCE_INDICES else tl.int32)
            - q0
        )
        sk = (
            tl.load(CK + batch + 1).to(tl.int64 if WIDE_SEQUENCE_INDICES else tl.int32)
            - k0
        )
        lbase = head * LH + q0 * LT
        if PADDED_LSE:
            lbase = batch * LB + head * LH
    else:
        sq, sk = sq_max, sk_max
        if WIDE_SEQUENCE_INDICES:
            sq, sk = tl.cast(sq, tl.int64), tl.cast(sk, tl.int64)
        q0, k0 = batch * sq, batch * sk
        lbase = batch * LB + head * LH
    m = (block * BM + tl.arange(0, BM)).to(
        tl.int64 if WIDE_SEQUENCE_INDICES else tl.int32
    )
    n = tl.arange(0, BN).to(tl.int64 if WIDE_SEQUENCE_INDICES else tl.int32)
    d = tl.arange(0, BD)
    qi = ((q0 + m[:, None]) * HQ + head) * DIM + d[None, :]
    if ALIGNED_S:
        # Confirmed dense square D64 tile: BM divides sq, BN divides sk and
        # DIM == BD, so every row/feature lane of the tile is in bounds.
        qm = tl.full((BM, BD), 1, tl.int1)
    else:
        qm = (m[:, None] < sq) & (d[None, :] < DIM)
    q = _musa_matrix_load(Q + qi, qm, FULL_PRECISION, DIM, ASYNC_LOAD)
    do = _musa_matrix_load(DO + qi, qm, FULL_PRECISION, DIM, ASYNC_LOAD)
    if OUT_HEADS:
        # Native O keeps BSHD storage while independent heads are folded into B.
        oi = (
            (batch // OUT_HEADS * sq + m[:, None]) * OUT_HEADS + batch % OUT_HEADS
        ) * DIM + d[None, :]
    else:
        oi = qi
    if ALIGNED_S:
        o = tl.load(O + oi).to(tl.float32)
    else:
        o = tl.load(O + oi, qm, 0).to(tl.float32)
    if FULL_PRECISION:
        delta = tl.sum(o * do.to(tl.float32), 1)
    elif DIM <= 128:
        # The native preprocessing kernel uses this order for both low dtypes.
        delta = _musa_flash_delta_serial8(o, do, BM, BD, DIM)
    else:
        delta_matrix = tl.dot(o.to(do.dtype), tl.trans(do))
        diagonal = tl.arange(0, BM)
        delta = tl.sum(
            tl.where(diagonal[:, None] == diagonal[None, :], delta_matrix, 0.0), 1
        )
    if WRITE_DELTA:
        if ALIGNED_S:
            tl.store(DELTA + head * total_q + q0 + m, delta)
        else:
            tl.store(DELTA + head * total_q + q0 + m, delta, m < sq)
    if ALIGNED_S:
        lse = tl.load(LSE + lbase + m * LT)
        delta_scale = -(delta * scale)
    else:
        lse = tl.load(LSE + lbase + m * LT, m < sq, float("inf"))
    dq = tl.full((BM, BD), 0, tl.float32)
    lo = 0
    hi = sk
    offset = 0 if TOP_LEFT_CAUSAL else sk - sq
    if WL >= 0:
        lo = tl.maximum(0, block * BM + offset - WL)
    if WR >= 0:
        hi = tl.minimum(sk, (block + 1) * BM + offset + WR)
    if ALIGNED_S:
        # Dense square rows: the tile window is already an exact multiple of BN.
        start = lo // BN
        stop = hi // BN
    else:
        start = lo // BN
        stop = tl.cdiv(tl.maximum(hi, 0), BN)
    for tile in range(start, stop):
        kn = tile * BN + n
        ki = ((k0 + kn[:, None]) * HK + head // (HQ // HK)) * DIM + d[None, :]
        if ALIGNED_S:
            km = tl.full((BN, BD), 1, tl.int1)
        else:
            km = (kn[:, None] < sk) & (d[None, :] < DIM)
        k = _musa_matrix_load(K + ki, km, FULL_PRECISION, DIM, ASYNC_LOAD)
        v = _musa_matrix_load(V + ki, km, FULL_PRECISION, DIM, ASYNC_LOAD)
        p = _musa_flash_probability(
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
            FULL_TILES=ALIGNED_S,
        )
        dp = tl.dot(do, tl.trans(v))
        # Some NPU dot lowerings leave out-of-bounds tile lanes undefined.
        # A zero probability does not suppress a NaN in those lanes.
        if not ALIGNED_S:
            dp_valid = (m[:, None] < sq) & (kn[None, :] < sk)
            dp = tl.where(dp_valid, dp, 0.0)
        if DROP_P > 0:
            dp *= _musa_dropout(
                seed, rng_offset, batch, head, m, kn, sq, sk, HQ, DROP_P, DROP_GROUP4
            )
        # MuDNN dropout and D224/D256 MHA round P before forming dS.
        if not FULL_PRECISION and (DROP_P > 0 or DIM == 224 or DIM == 256):
            p = _musa_round_probability(p, q.dtype)
        ds = p * (dp - delta[:, None])
        if DROP_P > 0:
            # Match measured MuDNN backward saved-state semantics: forward O
            # already includes dropout scaling, while backward scales the
            # entire masked-dP minus dot(O,dO) difference once more.
            if DROP_KEEP == 0.0:
                ds *= 0.0
            else:
                ds *= 1.0 / DROP_KEEP
        if not ALIGNED_S:
            ds_valid = (m[:, None] < sq) & (kn[None, :] < sk)
            ds = tl.where(ds_valid, ds, 0.0)
        # MuDNN scales dS before its low-precision tensor-core operand cast.
        if FULL_PRECISION:
            dq += tl.dot(ds * scale, k.to(tl.float32), input_precision="ieee")
        else:
            if q.dtype == tl.bfloat16 and DROP_P == 0 and DIM <= 128:
                if ALIGNED_S:
                    ds_scaled = _musa_bf16_product(
                        p, tl.fma(dp, scale, delta_scale[:, None])
                    )
                else:
                    ds_scaled = _musa_bf16_product(
                        p, tl.fma(dp, scale, -delta[:, None] * scale)
                    )
                    ds_scaled = tl.where(ds_valid, ds_scaled, 0.0)
            elif q.dtype == tl.float16 and DROP_P == 0:
                # Preserve a fused scaled difference to align FP16 native rounding.
                if ALIGNED_S:
                    ds_scaled = p * tl.fma(dp, scale, delta_scale[:, None])
                else:
                    ds_scaled = p * tl.fma(dp, scale, -delta[:, None] * scale)
                    ds_scaled = tl.where(ds_valid, ds_scaled, 0.0)
            else:
                ds_scaled = ds * scale
            dq += tl.dot(ds_scaled.to(q.dtype), k)
    if ALIGNED_S:
        tl.store(DQ + qi, dq)
    else:
        tl.store(DQ + qi, tl.where(sk > 0, dq, 0.0), qm)


@triton.jit(do_not_specialize=["seed", "rng_offset", "offset_intragraph"])
def _musa_flash_bwd_dkv(
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
    WIDE_SEQUENCE_INDICES: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    FULL_PRECISION: tl.constexpr = False,
    ASYNC_LOAD: tl.constexpr = False,
    HEAD_OFFSET: tl.constexpr = 0,
    COMPUTE_DELTA: tl.constexpr = False,
    seed=0,
    rng_offset=0,
    DROP_P: tl.constexpr = 0.0,
    DROP_KEEP: tl.constexpr = 1.0,
    DROP_GROUP4: tl.constexpr = False,
    offset_intragraph=0,
    RNG_DEVICE: tl.constexpr = False,
    ALIGNED_S: tl.constexpr = 0,
):
    if DROP_P > 0 and RNG_DEVICE:
        seed = tl.load(tl.cast(seed, tl.pointer_type(tl.uint64)))
        rng_offset = tl.load(
            tl.cast(rng_offset, tl.pointer_type(tl.uint64))
        ) + offset_intragraph.to(tl.uint64)
    block = tl.program_id(0)
    if WIDE_SEQUENCE_INDICES:
        block = block.to(tl.int64)
    batch = tl.program_id(1).to(tl.int64 if WIDE_SEQUENCE_INDICES else tl.int32)
    head_k = (
        tl.program_id(2).to(tl.int64 if WIDE_SEQUENCE_INDICES else tl.int32)
        - HEAD_OFFSET
    )
    if VARLEN:
        q0 = tl.load(CQ + batch).to(tl.int64 if WIDE_SEQUENCE_INDICES else tl.int32)
        k0 = tl.load(CK + batch).to(tl.int64 if WIDE_SEQUENCE_INDICES else tl.int32)
        sq = (
            tl.load(CQ + batch + 1).to(tl.int64 if WIDE_SEQUENCE_INDICES else tl.int32)
            - q0
        )
        sk = (
            tl.load(CK + batch + 1).to(tl.int64 if WIDE_SEQUENCE_INDICES else tl.int32)
            - k0
        )
        lbase = q0 * LT
        if PADDED_LSE:
            lbase = batch * LB
    else:
        sq, sk = sq_max, sk_max
        if WIDE_SEQUENCE_INDICES:
            sq, sk = tl.cast(sq, tl.int64), tl.cast(sk, tl.int64)
        q0, k0 = batch * sq, batch * sk
        lbase = batch * LB
    m = tl.arange(0, BM).to(tl.int64 if WIDE_SEQUENCE_INDICES else tl.int32)
    n = (block * BN + tl.arange(0, BN)).to(
        tl.int64 if WIDE_SEQUENCE_INDICES else tl.int32
    )
    d = tl.arange(0, BD)
    ki = ((k0 + n[:, None]) * HK + head_k) * DIM + d[None, :]
    if ALIGNED_S:
        # D64, S==1024 aligned dense tiles: every K/V row and feature index is
        # in bounds, so the qm/n/d masks are provably all-true here.
        k = tl.load(K + ki)
        v = tl.load(V + ki)
    else:
        km = (n[:, None] < sk) & (d[None, :] < DIM)
        k = _musa_matrix_load(K + ki, km, FULL_PRECISION, DIM, ASYNC_LOAD)
        v = _musa_matrix_load(V + ki, km, FULL_PRECISION, DIM, ASYNC_LOAD)
    dk = tl.full((BN, BD), 0, tl.float32)
    dv = tl.full((BN, BD), 0, tl.float32)
    lo = 0
    hi = sq
    offset = 0 if TOP_LEFT_CAUSAL else sk - sq
    if WR >= 0:
        lo = tl.maximum(0, block * BN - offset - WR)
    if WL >= 0:
        hi = tl.minimum(sq, (block + 1) * BN - offset + WL)
    # MuDNN accumulates the query-head contributions in FP32 before the final
    # GQA output cast. Keep disjoint writes and fixed-order group reduction.
    for group in range(HQ // HK):
        head = head_k * (HQ // HK) + group
        dk_head = tl.full((BN, BD), 0, tl.float32)
        dv_head = tl.full((BN, BD), 0, tl.float32)
        for tile in range(lo // BM, tl.cdiv(tl.maximum(hi, 0), BM)):
            qm = tile * BM + m
            qi = ((q0 + qm[:, None]) * HQ + head) * DIM + d[None, :]
            if ALIGNED_S:
                q = tl.load(Q + qi)
                do = tl.load(DO + qi)
            else:
                mask = (qm[:, None] < sq) & (d[None, :] < DIM)
                q = _musa_matrix_load(Q + qi, mask, FULL_PRECISION, DIM, ASYNC_LOAD)
                do = _musa_matrix_load(DO + qi, mask, FULL_PRECISION, DIM, ASYNC_LOAD)
            if COMPUTE_DELTA:
                if ALIGNED_S:
                    o = tl.load(O + qi).to(tl.float32)
                else:
                    o = tl.load(O + qi, (qm[:, None] < sq) & (d[None, :] < DIM), 0).to(
                        tl.float32
                    )
                if FULL_PRECISION:
                    delta = tl.sum(o * do.to(tl.float32), 1)
                elif DIM <= 128:
                    delta = _musa_flash_delta_serial8(o, do, BM, BD, DIM)
                else:
                    delta_matrix = tl.dot(o.to(do.dtype), tl.trans(do))
                    diagonal = tl.arange(0, BM)
                    delta = tl.sum(
                        tl.where(
                            diagonal[:, None] == diagonal[None, :], delta_matrix, 0.0
                        ),
                        1,
                    )
            else:
                if ALIGNED_S:
                    delta = tl.load(DELTA + head * total_q + q0 + qm)
                else:
                    delta = tl.load(DELTA + head * total_q + q0 + qm, qm < sq, 0)
            if ALIGNED_S:
                lse = tl.load(LSE + lbase + head * LH + qm * LT)
            else:
                lse = tl.load(LSE + lbase + head * LH + qm * LT, qm < sq, float("inf"))
            p = _musa_flash_probability(
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
                FULL_TILES=ALIGNED_S,
            )
            dp = tl.dot(do, tl.trans(v))
            if not ALIGNED_S:
                dp = tl.where((qm[:, None] < sq) & (n[None, :] < sk), dp, 0.0)
            p_drop = p
            if DROP_P > 0:
                keep = _musa_dropout(
                    seed,
                    rng_offset,
                    batch,
                    head,
                    qm,
                    n,
                    sq,
                    sk,
                    HQ,
                    DROP_P,
                    DROP_GROUP4,
                )
                dp *= keep
                p_drop *= keep
            # MuDNN dropout and D224/D256 MHA round P before forming dS.
            if not FULL_PRECISION and (DROP_P > 0 or DIM == 224 or DIM == 256):
                p = _musa_round_probability(p, q.dtype)
            ds = p * (dp - delta[:, None])
            if DROP_P > 0:
                if DROP_KEEP == 0.0:
                    ds *= 0.0
                else:
                    ds *= 1.0 / DROP_KEEP
            if not ALIGNED_S:
                ds = tl.where((qm[:, None] < sq) & (n[None, :] < sk), ds, 0.0)
            if FULL_PRECISION:
                dk_head += tl.dot(
                    tl.trans(ds * scale), q.to(tl.float32), input_precision="ieee"
                )
                dv_head += tl.dot(
                    tl.trans(p_drop), do.to(tl.float32), input_precision="ieee"
                )
            else:
                if q.dtype == tl.bfloat16 and DROP_P == 0 and DIM <= 128:
                    ds_scaled = _musa_bf16_product(
                        p, tl.fma(dp, scale, -delta[:, None] * scale)
                    )
                    if not ALIGNED_S:
                        ds_scaled = tl.where(
                            (qm[:, None] < sq) & (n[None, :] < sk), ds_scaled, 0.0
                        )
                elif q.dtype == tl.float16 and DROP_P == 0:
                    ds_scaled = p * tl.fma(dp, scale, -delta[:, None] * scale)
                    if not ALIGNED_S:
                        ds_scaled = tl.where(
                            (qm[:, None] < sq) & (n[None, :] < sk), ds_scaled, 0.0
                        )
                else:
                    ds_scaled = ds * scale
                dk_head += tl.dot(tl.trans(ds_scaled.to(q.dtype)), q)
                dv_operand = p_drop
                if DROP_P > 0:
                    dv_operand = _musa_round_probability(p_drop, q.dtype)
                    if DROP_KEEP == 0.0:
                        dv_operand *= 0.0
                    else:
                        dv_operand /= DROP_KEEP
                dv_head += tl.dot(tl.trans(dv_operand.to(q.dtype)), do)
        dk += dk_head
        dv += dv_head
    if ALIGNED_S:
        tl.store(DK + ki, tl.where(sq > 0, dk, 0.0))
        tl.store(DV + ki, tl.where(sq > 0, dv, 0.0))
    else:
        tl.store(DK + ki, tl.where(sq > 0, dk, 0.0), km)
        tl.store(DV + ki, tl.where(sq > 0, dv, 0.0), km)


@triton.jit
def _musa_flash_bwd_small(
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
    WIDE_SEQUENCE_INDICES: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    FULL_PRECISION: tl.constexpr = False,
    ASYNC_LOAD: tl.constexpr = False,
):
    block = tl.program_id(0)
    if WIDE_SEQUENCE_INDICES:
        block = block.to(tl.int64)
    # Disjoint programs own Q or KV outputs. KV programs recompute their small
    # delta tiles, removing the inter-kernel dependency and delta allocation.
    if tl.program_id(2) < HQ:
        if block * BM < sq_max:
            _musa_flash_bwd_dq(
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
                WIDE_SEQUENCE_INDICES,
                BM,
                BN,
                WRITE_DELTA=False,
                FULL_PRECISION=FULL_PRECISION,
                ASYNC_LOAD=ASYNC_LOAD,
            )
    else:
        if block * BN < sk_max:
            _musa_flash_bwd_dkv(
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
                WIDE_SEQUENCE_INDICES,
                BM,
                BN,
                HEAD_OFFSET=HQ,
                COMPUTE_DELTA=True,
                FULL_PRECISION=FULL_PRECISION,
                ASYNC_LOAD=ASYNC_LOAD,
            )


@triton.jit
def _musa_flash_tail_metadata(CQ, CK, FLAG, batch, BLOCK: tl.constexpr):
    # One program initializes the sole flag, using bounded metadata tiles.
    lane = tl.arange(0, BLOCK).to(tl.int64)
    invalid = tl.full((), False, tl.int1)
    for start in range(tl.cdiv(batch, BLOCK)):
        index = start * BLOCK + lane
        valid = index < batch
        sq = tl.load(CQ + index + 1, valid, 0).to(tl.int64) - tl.load(
            CQ + index, valid, 0
        ).to(tl.int64)
        sk = tl.load(CK + index + 1, valid, 0).to(tl.int64) - tl.load(
            CK + index, valid, 0
        ).to(tl.int64)
        invalid = invalid | (
            tl.sum((valid & (sk > tl.cdiv(sq, 128) * 128)).to(tl.int32), 0) > 0
        )
    tl.store(FLAG, invalid.to(tl.int32))


@triton.jit
def _musa_flash_bwd_dq_tail(
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
    WIDE_SEQUENCE_INDICES: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    TAIL_FLAG,
    TAIL_MATH: tl.constexpr,
    FULL_PRECISION: tl.constexpr = False,
    ASYNC_LOAD: tl.constexpr = False,
    WRITE_DELTA: tl.constexpr = True,
):
    if tl.load(TAIL_FLAG) == TAIL_MATH:
        _musa_flash_bwd_dq(
            Q=Q,
            K=K,
            V=V,
            O=O,
            DO=DO,
            LSE=LSE,
            DELTA=DELTA,
            DQ=DQ,
            CQ=CQ,
            CK=CK,
            sq_max=sq_max,
            sk_max=sk_max,
            total_q=total_q,
            HQ=HQ,
            HK=HK,
            DIM=DIM,
            BD=BD,
            scale=scale,
            scale_log2=None if TAIL_MATH else scale_log2,
            LSE_SCALE=LSE_SCALE,
            LB=LB,
            LH=LH,
            LT=LT,
            WL=WL,
            WR=WR,
            TOP_LEFT_CAUSAL=TOP_LEFT_CAUSAL,
            VARLEN=VARLEN,
            PADDED_LSE=PADDED_LSE,
            WIDE_SEQUENCE_INDICES=WIDE_SEQUENCE_INDICES,
            BM=BM,
            BN=BN,
            FULL_PRECISION=TAIL_MATH or FULL_PRECISION,
            ASYNC_LOAD=ASYNC_LOAD,
            WRITE_DELTA=WRITE_DELTA,
        )


@triton.jit
def _musa_flash_bwd_dkv_tail(
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
    WIDE_SEQUENCE_INDICES: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    TAIL_FLAG,
    TAIL_MATH: tl.constexpr,
    FULL_PRECISION: tl.constexpr = False,
    ASYNC_LOAD: tl.constexpr = False,
    HEAD_OFFSET: tl.constexpr = 0,
    COMPUTE_DELTA: tl.constexpr = False,
):
    if tl.load(TAIL_FLAG) == TAIL_MATH:
        _musa_flash_bwd_dkv(
            Q=Q,
            K=K,
            V=V,
            O=O,
            DO=DO,
            LSE=LSE,
            DELTA=DELTA,
            DK=DK,
            DV=DV,
            CQ=CQ,
            CK=CK,
            sq_max=sq_max,
            sk_max=sk_max,
            total_q=total_q,
            HQ=HQ,
            HK=HK,
            DIM=DIM,
            BD=BD,
            scale=scale,
            scale_log2=None if TAIL_MATH else scale_log2,
            LSE_SCALE=LSE_SCALE,
            LB=LB,
            LH=LH,
            LT=LT,
            WL=WL,
            WR=WR,
            TOP_LEFT_CAUSAL=TOP_LEFT_CAUSAL,
            VARLEN=VARLEN,
            PADDED_LSE=PADDED_LSE,
            WIDE_SEQUENCE_INDICES=WIDE_SEQUENCE_INDICES,
            BM=BM,
            BN=BN,
            FULL_PRECISION=TAIL_MATH or FULL_PRECISION,
            ASYNC_LOAD=ASYNC_LOAD,
            HEAD_OFFSET=HEAD_OFFSET,
            COMPUTE_DELTA=COMPUTE_DELTA,
        )


@triton.jit
def _musa_flash_bwd_persistent_delta(
    O,
    DO,
    DELTA,
    DQA,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    OB: tl.constexpr,
    OH: tl.constexpr,
    OS: tl.constexpr,
    BM: tl.constexpr,
):
    bh = tl.program_id(1)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    d = tl.arange(0, D)
    do = tl.load(DO + (bh * S + m[:, None]) * D + d[None, :], m[:, None] < S, 0)
    o = tl.load(
        O + (bh // H) * OB + (bh % H) * OH + m[:, None] * OS + d[None, :],
        m[:, None] < S,
        0,
    )
    delta = _musa_flash_delta_serial8(o, do, BM, D, D)
    tl.store(DELTA + bh * S + m, delta, m < S)
    tl.store(DQA + (bh * S + m[:, None]) * D + d[None, :], 0.0, m[:, None] < S)


@triton.jit
def _musa_flash_bwd_persistent_ds(
    score,
    dp,
    LSE,
    DELTA,
    m,
    n,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    SCALE,
    SCALE2,
    DTYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    lse = tl.load(LSE + m)
    delta = tl.load(DELTA + m)
    p = tl.exp2(tl.fma(score, SCALE2, -lse[:, None] * 1.4426950408889634))
    if CAUSAL:
        p = tl.where(m[:, None] % SQ >= n[None, :] % SK, p, 0.0)
    if DTYPE == tl.bfloat16:
        ds = _musa_bf16_product(p, tl.fma(dp, SCALE, -delta[:, None] * SCALE))
    else:
        ds = (p * tl.fma(dp, SCALE, -delta[:, None] * SCALE)).to(DTYPE)
    return p.to(DTYPE), ds


@triton.jit
def _musa_flash_bwd_persistent(
    Q,
    DO,
    K,
    V,
    DQA,
    DK,
    DV,
    LSE,
    DELTA,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    BH: tl.constexpr,
    NUM_SMS: tl.constexpr,
    CAUSAL: tl.constexpr,
    SCALE,
    SCALE2,
):
    qtiles: tl.constexpr = tl.cdiv(SQ, BM)
    pid = tl.program_id(0)
    for job in range(pid, BH * (SK // BN), NUM_SMS):
        bh = job % BH
        block = job // BH
        n = bh * SK + block * BN + tl.arange(0, BN)
        d = tl.arange(0, BD)
        kv = n[:, None] * D + d[None, :]
        if CAUSAL:
            k = tl.load(K + n[:, None] * D + d[None, :], d[None, :] < D, 0)
            v = tl.load(V + n[:, None] * D + d[None, :], d[None, :] < D, 0)
        else:
            k = tl.load(K + kv, eviction_policy="evict_last")
            v = tl.load(V + kv, eviction_policy="evict_last")
        ak = tl.zeros((BN, BD), tl.float32)
        av = tl.zeros((BN, BD), tl.float32)
        start = 0
        if CAUSAL:
            start = block * BN // BM
        qcount = qtiles - start
        shift = block % qcount
        rows = tl.arange(0, BM)
        for it in range(qcount):
            t = start + it + shift
            t = tl.where(t >= qtiles, t - qcount, t)
            m = bh * SQ + t * BM + rows
            qp = m[:, None] * D + d[None, :]
            if CAUSAL:
                q = tl.load(Q + m[:, None] * D + d[None, :], d[None, :] < D, 0)
                do = tl.load(DO + m[:, None] * D + d[None, :], d[None, :] < D, 0)
            else:
                q = tl.load(Q + qp, eviction_policy="evict_first")
                do = tl.load(DO + qp, eviction_policy="evict_first")
            score = tl.dot(q, tl.trans(k))
            dp = tl.dot(do, tl.trans(v))
            if CAUSAL:
                # Native masked path, unchanged arithmetic for the causal case.
                p, ds = _musa_flash_bwd_persistent_ds(
                    score,
                    dp,
                    LSE,
                    DELTA,
                    m,
                    n,
                    SQ,
                    SK,
                    SCALE,
                    SCALE2,
                    DK.dtype.element_ty,
                    CAUSAL,
                )
            else:
                lse = tl.load(LSE + m)
                delta = tl.load(DELTA + m)
                p = tl.exp2(tl.fma(score, SCALE2, -lse[:, None] * 1.4426950408889634))
                if DK.dtype.element_ty == tl.bfloat16:
                    ds = _musa_bf16_product(
                        p, tl.fma(dp, SCALE, -delta[:, None] * SCALE)
                    )
                else:
                    ds = (p * tl.fma(dp, SCALE, -delta[:, None] * SCALE)).to(
                        DK.dtype.element_ty
                    )
                p = p.to(DK.dtype.element_ty)
            # Same FP32 reduction and rotated tile order as the reference.
            ak = tl.dot(tl.trans(ds), q, ak)
            av = tl.dot(tl.trans(p), do, av)
            dq = tl.dot(ds, k)
            if CAUSAL:
                tl.atomic_add(
                    DQA + m[:, None] * D + d[None, :],
                    dq,
                    mask=d[None, :] < D,
                    sem="relaxed",
                )
            else:
                tl.atomic_add(DQA + qp, dq, sem="relaxed")
        if CAUSAL:
            idx = n[:, None] * D + d[None, :]
            tl.store(DK + idx, ak.to(DK.dtype.element_ty), d[None, :] < D)
            tl.store(DV + idx, av.to(DV.dtype.element_ty), d[None, :] < D)
        else:
            tl.store(DK + kv, ak.to(DK.dtype.element_ty))
            tl.store(DV + kv, av.to(DV.dtype.element_ty))
        tl.debug_barrier()


@triton.jit
def _musa_flash_bwd_persistent_cast(DQA, DQ, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(DQ + i, tl.load(DQA + i, i < N, 0).to(DQ.dtype.element_ty), i < N)


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
    logger.debug("GEMS_MTHREADS FLASH_ATTENTION_BACKWARD")
    # torch-musa 2.9.1 forwards the dense ATen call directly to MuDNN's
    # BHSD SDPA entry. Normalize that producer ABI before the shared kernel.
    musa_dense = query.ndim == 4
    # Independent head/batch controls expose MuDNN D160 dQ dependence on a
    # neighboring logical K segment in an incomplete 128-key tile. Only tensor
    # metadata selects the same-saved-state math path below; no device read or
    # native call participates in implementation dispatch.
    dense_key_contiguous = musa_dense and key.is_contiguous()
    if musa_dense:
        query, key, value, out, grad_out = (
            t.transpose(1, 2) for t in (query, key, value, out, grad_out)
        )
    # Validate metadata without reading device values or synchronizing the stream.
    varlen = query.ndim == 3
    rank = 3 if varlen else 4
    if query.ndim not in (3, 4) or any(
        t.ndim != rank for t in (key, value, out, grad_out)
    ):
        raise ValueError("flash_attention_backward expects BHSD or packed THD tensors")
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
    seed = rng_offset = offset_intragraph = 0
    rng_device = False
    if dropout_p:
        # MuDNN PH1 backward subtracts float32(p) from one in float32,
        # although its host descriptor accepts double. Keep the mask unchanged.
        dropout_keep = struct.unpack(
            "f",
            struct.pack("f", 1.0 - struct.unpack("f", struct.pack("f", dropout_p))[0]),
        )[0]
        # MuDNN PH1 host lowering selects a zero reciprocal when p32 >= 1.
        # Preserve the device arithmetic path instead of rejecting legal p < 1.
        if window_size_left is not None or window_size_right is not None:
            raise NotImplementedError("native MUSA dropout has no local-window ABI")
        if dim in (136, 144, 152, 192):
            raise NotImplementedError("native MUSA dropout dimension is not validated")
        if not musa_dense or hq != hk:
            raise NotImplementedError(
                "native MUSA dropout currently requires dense MHA"
            )
        if not (
            isinstance(rng_state, torch.Tensor)
            and rng_state.device.type == "cpu"
            and rng_state.dtype == torch.uint64
            and rng_state.ndim == 1
            and rng_state.numel() in (2, 3)
            and rng_state.is_contiguous()
        ):
            raise NotImplementedError(
                "expected native CPU uint64[2] or uint64[3] RNG state"
            )
        # MuDNN stores ordinary seed/offset on CPU. In graph mode its three
        # CPU words hold two device addresses and an intragraph offset; the
        # device kernel dereferences those addresses without a host readback.
        state_values = rng_state.tolist()
        seed, rng_offset = state_values[:2]
        rng_device = len(state_values) == 3
        if rng_device:
            offset_intragraph = state_values[2]
    wl = -1 if window_size_left is None else window_size_left
    wr = -1 if window_size_right is None else window_size_right
    wl, wr = max(-1, wl), max(-1, wr)
    if is_causal:
        wr = 0
    if max_q < 0 or max_k < 0:
        raise ValueError("maximum sequence lengths must be nonnegative")
    # MuDNN saves dense and packed forward LSE in padded BHS layout.
    padded_lse = True
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
    # Explicit windows use the mathematical mask; MuDNN has no window ABI.
    # MuDNN treats zero as its default-scale sentinel for its supported ABI.
    # Explicit windows are a Gems extension and retain mathematical zero.
    native_default_scale = scale is None or (
        scale == 0 and window_size_left is None and window_size_right is None
    )
    # Round the reciprocal square root once in the kernel argument. Rounding
    # sqrt(D) first changes the D24/D96 coefficient by one FP32 ULP.
    scale = 1.0 / math.sqrt(dim) if native_default_scale else scale
    scale_log2 = None
    lse_scale = math.log2(math.e)
    launch_options = {
        # This optional load route has device evidence only on S5000/mp31.
        "ASYNC_LOAD": (
            dropout_p == 0
            and _HAS_ASYNC_LOAD
            and dim == 128
            and "S5000" in torch.musa.get_device_name(query.device)
        ),
        "FULL_PRECISION": (
            window_size_left is not None
            or window_size_right is not None
            or (
                dropout_p == 0
                and not varlen
                and is_causal
                and dim <= 160
                and max_k > triton.cdiv(max_q, 128) * 128
            )
            or dim in (136, 144, 152, 192)
            or (varlen and dim == 160)
            or (
                dropout_p == 0
                and not varlen
                and dim == 160
                and max_k % 128 != 0
                and (batch > 1 or (dense_key_contiguous and hk > 1))
                and (not is_causal or max_q > max_k)
            )
            or (hq > hk and dim in (224, 256))
        ),
    }
    if dropout_p:
        launch_options.update(
            seed=seed,
            rng_offset=rng_offset,
            DROP_P=dropout_p,
            DROP_KEEP=dropout_keep,
            DROP_GROUP4=max_k % 4 == 0,
            offset_intragraph=offset_intragraph,
            RNG_DEVICE=rng_device,
        )
    if not launch_options["FULL_PRECISION"]:
        try:
            scale32 = struct.unpack("f", struct.pack("f", scale))[0]
            scale_log2 = struct.unpack(
                "f", struct.pack("f", scale32 * math.log2(math.e))
            )[0]
        except OverflowError:
            scale_log2 = math.copysign(math.inf, scale)
    fold_heads = (
        musa_dense
        and dropout_p == 0
        and dim in (64, 128)
        and hq == hk
        and hq > 1
        and max_q == max_k
        and max_q >= 256
        and native_default_scale
        and window_size_left is None
        and window_size_right is None
        and logsumexp.is_contiguous()
        and all(
            t.transpose(1, 2).is_contiguous() for t in (query, key, value, grad_out)
        )
    )
    if (
        fold_heads
        and batch > 0
        and dim == 128
        and max_q >= 1024
        and max_q % 128 == 0
        and query.numel() < 2**31
        and (out.is_contiguous() or out.transpose(1, 2).is_contiguous())
        and not torch.are_deterministic_algorithms_enabled()
        and "S5000" in torch.musa.get_device_name(query.device)
    ):
        # Share QK/dP work between all gradients; dQ accumulates in FP32.
        # Strict and warn-only determinism keep the disjoint-write path below.
        pq, pk, pv, pdo = (
            tensor.transpose(1, 2) for tensor in (query, key, value, grad_out)
        )
        po = out.transpose(1, 2)
        pdq, pdk, pdv = (torch.empty_like(tensor) for tensor in (pq, pk, pv))
        pdelta = torch.empty(
            (batch * hq * max_q,), dtype=torch.float32, device=query.device
        )
        pdqa = torch.empty_like(pq, dtype=torch.float32)
        _musa_flash_bwd_persistent_delta[(triton.cdiv(max_q, 32), batch * hq)](
            po,
            pdo,
            pdelta,
            pdqa,
            max_q,
            hq,
            dim,
            *po.stride()[:3],
            32,
            num_warps=4,
            num_stages=1,
        )
        sms = torch.musa.get_device_properties(query.device.index).multi_processor_count
        _musa_flash_bwd_persistent[(min(sms, batch * hq * (max_k // 128)),)](
            pq,
            pdo,
            pk,
            pv,
            pdqa,
            pdk,
            pdv,
            logsumexp,
            pdelta,
            max_q,
            max_k,
            dim,
            64,
            128,
            128,
            batch * hq,
            sms,
            is_causal,
            scale,
            scale_log2,
            num_warps=16,
            num_stages=1,
        )
        _musa_flash_bwd_persistent_cast[(triton.cdiv(pq.numel(), 256),)](
            pdqa,
            pdq,
            pq.numel(),
            256,
            num_warps=4,
        )
        return pdq, pdk, pdv
    fold_out = fold_heads and out.is_contiguous()
    out_heads = hq if fold_out else 0
    if fold_heads:
        # Dense MHA heads are independent when dropout is disabled.
        batch *= hq
        hq = hk = 1
        total_q = batch * max_q
        lb, lh, lt = max_q, max_q, 1
        logsumexp = logsumexp.view(batch, 1, max_q)
        q, k, v, do = (t.transpose(1, 2) for t in (query, key, value, grad_out))
        o = out.transpose(1, 2)
        if not fold_out:
            o = o.contiguous()
    else:
        q, k, v, o, do = (t.contiguous() for t in (query, key, value, out, grad_out))
    dq, dk, dv = (torch.empty_like(t) for t in (q, k, v))
    if batch == 0:
        if q.numel() or k.numel():
            raise ValueError("zero sequences require empty packed inputs")
        return (
            tuple(t.transpose(1, 2) for t in (dq, dk, dv))
            if musa_dense and not fold_heads
            else (dq, dk, dv)
        )
    tile = 32
    num_warps = 4
    dense_s5000 = fold_heads and "S5000" in torch.musa.get_device_name(query.device)
    if dense_s5000:
        # Contiguous S5000 rows favor synchronous loads and wider FP16 tiles.
        launch_options["ASYNC_LOAD"] = False
        if query.dtype == torch.float16 and not is_causal:
            tile, num_warps = 64, 8
    # Bound padded masked-lane addresses as well as valid elements. LSE may
    # have nonstandard positive strides; its storage span, not numel, matters.
    lse_span = 1 + sum(
        (size - 1) * stride for size, stride in zip(logsumexp.shape, logsumexp.stride())
    )
    wide_sequence_indices = (
        max(
            (max_q + tile - 1) * batch * hq * dim,
            (max_k + tile - 1) * batch * hk * dim,
            lse_span + (tile - 1) * lt,
        )
        >= 2**31
    )
    common = (
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
        is_causal,
        varlen,
        varlen and padded_lse,
        wide_sequence_indices,
    )
    if (
        musa_dense
        and dropout_p == 0
        and hq == hk
        and dim == 64
        and max_q == max_k
        and max_q <= 128
        and not is_causal
        and wl < 0
        and wr < 0
    ):
        if max_q or max_k:
            _musa_flash_bwd_small[
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
        return (
            tuple(t.transpose(1, 2) for t in (dq, dk, dv))
            if musa_dense and not fold_heads
            else (dq, dk, dv)
        )
    dq_kernel = _musa_flash_bwd_dq
    dkv_kernel = _musa_flash_bwd_dkv
    # Independent backward-only zero-QK controls on S5000/MuDNN 3400
    # establish unreliable masked tails beyond the actual Q extent rounded
    # to 128 for the supported <=128 and dense-160 dimension families.
    # Unsupported dimensions above already select the mathematical path.
    # Device metadata selects one precision contract for the entire packed
    # call; no device value is read by the host and references are not mixed
    # between sequences. Declared max bounds do not decide this condition.
    batch_tail_check = (
        varlen and is_causal and dim <= 160 and not launch_options["FULL_PRECISION"]
    )
    if batch_tail_check:
        tail_flag = torch.empty((), dtype=torch.int32, device=q.device)
        _musa_flash_tail_metadata[(1,)](
            cum_seq_q, cum_seq_k, tail_flag, batch, BLOCK=256
        )
        dq_kernel = _musa_flash_bwd_dq_tail
        dkv_kernel = _musa_flash_bwd_dkv_tail
    delta = torch.empty((hq, total_q), dtype=torch.float32, device=q.device)
    dq_extra = {"OUT_HEADS": out_heads} if fold_out else {}
    # Burst combining changes native Delta/dS rounding. Preserve dQ's order
    # and use the full optimization set only for the BF16 dKV consumer.
    dkv_extra = {}
    aligned_s = (
        dense_s5000
        and dim == 64
        and max_q == 1024
        and max_k == 1024
        and max_q % tile == 0
        and max_k % tile == 0
        and query.numel() > 0
        and query.numel() < 2**31
        and (out.is_contiguous() or out.transpose(1, 2).is_contiguous())
        and not torch.are_deterministic_algorithms_enabled()
    )
    if aligned_s:
        dq_extra["ALIGNED_S"] = 1
        dkv_extra["ALIGNED_S"] = 1
    if dense_s5000:
        dq_extra.update(
            enable_backend_opt=False,
            enable_llc_opt=False,
            llc_options=(
                "-mtgpu-opt-level=1 -mtgpu-combine-instr-with-burst=0 "
                "-mtgpu-combine-fop-instr=1 -misched=mtgpu-max-ilp"
            ),
        )
        dkv_extra.update(
            enable_backend_opt=query.dtype == torch.bfloat16,
            enable_llc_opt=False,
        )
    for tail_math in [False, True] if batch_tail_check else [None]:
        branch_options = dict(launch_options)
        if tail_math is not None:
            branch_options["TAIL_MATH"] = tail_math
            branch_options["TAIL_FLAG"] = tail_flag
        # Fixed-order, disjoint writes: deterministic in strict/warn/off modes.
        if max_q:
            dq_kernel[(triton.cdiv(max_q, tile), batch, hq)](
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
                **branch_options,
                **dq_extra,
            )
        if max_k:
            dkv_kernel[(triton.cdiv(max_k, tile), batch, hk)](
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
                BM=tile,
                BN=tile,
                num_warps=num_warps,
                num_stages=1,
                **branch_options,
                **dkv_extra,
            )
    return (
        tuple(t.transpose(1, 2) for t in (dq, dk, dv))
        if musa_dense and not fold_heads
        else (dq, dk, dv)
    )
