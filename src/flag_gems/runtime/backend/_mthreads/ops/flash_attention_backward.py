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

import ctypes
import fcntl
import functools
import hashlib
import inspect
import json
import logging
import math
import os
import shutil
import struct
import subprocess
import tempfile
import threading
from pathlib import Path

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


@triton.jit
def _musa_split_dot(x, y):
    hi = x.to(y.dtype)
    # Preserve FP16 residuals during the cast even at small probabilities.
    lo = ((x - hi.to(tl.float32)) * 4096.0).to(y.dtype)
    return tl.dot(hi, y) + tl.dot(lo, y) * (1.0 / 4096.0)


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
    math_dropout: tl.constexpr = DROP_P > 0 and HQ != HK
    drop_scale: tl.constexpr = 0.0 if DROP_KEEP == 0.0 else 1.0 / DROP_KEEP
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
        # MuDNN MHA dropout and D224/D256 MHA round P before forming dS.
        if (
            not FULL_PRECISION
            and not math_dropout
            and (DROP_P > 0 or DIM == 224 or DIM == 256)
        ):
            p = _musa_round_probability(p, q.dtype)
        # In the mathematical dropout path, delay the reciprocal until
        # after accumulation to avoid widening the dot operands' range.
        ds = p * (dp - delta[:, None] * (DROP_KEEP if math_dropout else 1.0))
        if DROP_P > 0 and not math_dropout:
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
        elif math_dropout:
            dq += _musa_split_dot(ds * scale, k)
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
    if math_dropout:
        dq *= drop_scale
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
    math_dropout: tl.constexpr = DROP_P > 0 and HQ != HK
    drop_scale: tl.constexpr = 0.0 if DROP_KEEP == 0.0 else 1.0 / DROP_KEEP
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
            if (
                not FULL_PRECISION
                and not math_dropout
                and (DROP_P > 0 or DIM == 224 or DIM == 256)
            ):
                p = _musa_round_probability(p, q.dtype)
            # In the mathematical dropout path, delay the reciprocal until
            # after accumulation to avoid widening the dot operands' range.
            ds = p * (dp - delta[:, None] * (DROP_KEEP if math_dropout else 1.0))
            if DROP_P > 0 and not math_dropout:
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
            elif math_dropout:
                dk_head += _musa_split_dot(tl.trans(ds * scale), q)
                dv_head += _musa_split_dot(tl.trans(p_drop), do)
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
    if math_dropout:
        dk *= drop_scale
        dv *= drop_scale
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
    LSE=None,
    SCALE: tl.constexpr = 1.0,
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
    offset = bh * S + m
    if LSE is None:
        tl.store(DELTA + offset, delta, m < S)
    else:
        # The SDK consumes a pair of already scaled row statistics.
        lse = tl.load(LSE + offset, m < S, other=float("inf"))
        lse = tl.where(tl.abs(lse) < float("inf"), lse, float(1 << 30))
        stats = tl.interleave(lse * 1.4426950408889634, delta * SCALE)
        pair = 2 * tl.program_id(0) * BM + tl.arange(0, 2 * BM)
        tl.store(DELTA + 2 * bh * S + pair, stats, pair < 2 * S)

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
        if not musa_dense:
            raise NotImplementedError("native MUSA dropout requires dense inputs")
        if hq != hk:
            from torch_musa.core import mudnn

            sq, sk = query.shape[1], key.shape[1]
            # The native forward mask has been verified for aligned causal D80
            # GQA on S5000 with MuDNN 3400. Compute its mathematical backward.
            if not (
                is_causal
                and (scale is None or scale >= 0.0)
                and dim == 80
                and 0 < sq <= sk
                and sq % 128 == 0
                and sk % 128 == 0
                and mudnn.version() == 3400
                and "S5000" in torch.musa.get_device_name(query.device)
            ):
                raise NotImplementedError(
                    "MUSA GQA dropout requires aligned causal D80 on S5000/MuDNN 3400 and nonnegative scale"
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
        if hq != hk and rng_state.numel() != 2:
            raise NotImplementedError("MUSA GQA dropout requires eager Philox state")
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
    default_scale = 1.0 / math.sqrt(dim)
    scale = default_scale if native_default_scale else scale
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
    can_fold_heads = (
        musa_dense
        and dropout_p == 0
        and dim in (64, 128)
        and hq == hk
        and hq > 1
        and max_q == max_k
        and max_q >= 256
        and window_size_left is None
        and window_size_right is None
        and logsumexp.is_contiguous()
        and all(
            t.transpose(1, 2).is_contiguous() for t in (query, key, value, grad_out)
        )
    )
    # Keep explicit default scale limited to the existing aligned D64 route.
    explicit_aligned_d64 = (
        can_fold_heads
        and not native_default_scale
        and scale == default_scale
        and dim == 64
        and max_q == 1024
        and 0 < query.numel() < 2**31
        and (out.is_contiguous() or out.transpose(1, 2).is_contiguous())
        and not torch.are_deterministic_algorithms_enabled()  # allow-torch-compute: reads a runtime flag
        and "S5000" in torch.musa.get_device_name(query.device)
    )
    fold_heads = can_fold_heads and (native_default_scale or explicit_aligned_d64)
    if (
        explicit_aligned_d64 or (can_fold_heads and native_default_scale and dim == 64)
    ) and (
        (batch, hq, max_q) == (4, 16, 1024)
        and (out.is_contiguous() or out.transpose(1, 2).is_contiguous())
        and not torch.are_deterministic_algorithms_enabled()  # allow-torch-compute: reads a runtime flag
        and "S5000" in torch.musa.get_device_name(query.device)
    ):
        result = _musa_flash_bwd_d64_sdk(
            query, key, value, out, grad_out, logsumexp, scale32, scale_log2, is_causal
        )
        if result is not None:
            return result
    if (
        can_fold_heads
        and (native_default_scale or scale == default_scale)
        and batch > 0
        and dim == 128
        and max_q >= 1024
        and max_q % 128 == 0
        and query.numel() < 2**31
        and (out.is_contiguous() or out.transpose(1, 2).is_contiguous())
        and not torch.are_deterministic_algorithms_enabled()  # allow-torch-compute: reads a runtime flag
        and "S5000" in torch.musa.get_device_name(query.device)
    ):
        # Share QK/dP work between all gradients; dQ accumulates in FP32.
        # Strict and warn-only determinism keep the disjoint-write path below.
        pq, pk, pv, pdo = (
            tensor.transpose(1, 2) for tensor in (query, key, value, grad_out)
        )
        po = out.transpose(1, 2)
        pdq, pdk, pdv = (torch.empty_like(tensor) for tensor in (pq, pk, pv))
        d128_handle = _d128_prepare(pq, pk, pv, pdo, logsumexp, is_causal)
        if d128_handle is not None:
            # The MP31 main kernel writes dK/dV directly and
            # accumulates dQ in a BHSD fp32 buffer.  The delta kernel zeroes
            # that buffer while computing delta because its DQA stores cover
            # every flat element exactly once.
            dq_accum = torch.empty(
                (batch, hq, max_q, dim), dtype=torch.float32, device=query.device
            )
            row_stats = torch.empty(
                (batch, hq, max_q, 2), dtype=torch.float32, device=query.device
            )
            _musa_flash_bwd_persistent_delta[(triton.cdiv(max_q, 32), batch * hq)](
                po,
                pdo,
                row_stats,
                dq_accum,
                max_q,
                hq,
                dim,
                *po.stride()[:3],
                32,
                num_warps=4,
                num_stages=1,
                LSE=logsumexp,
                SCALE=scale32,
            )
            _d128_launch(
                d128_handle,
                pq,
                pk,
                pv,
                pdo,
                row_stats,
                dq_accum,
                pdk,
                pdv,
                scale32,
                scale_log2,
            )
            _musa_flash_bwd_persistent_cast[(triton.cdiv(pdq.numel(), 256),)](
                dq_accum, pdq, pdq.numel(), 256, num_warps=4
            )
            return pdq, pdk, pdv
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
        and not torch.are_deterministic_algorithms_enabled()  # allow-torch-compute: reads a runtime flag
    )
    if aligned_s:
        dq_extra["ALIGNED_S"] = 1
        dkv_extra["ALIGNED_S"] = 1
    if dense_s5000:
        # Select the compiler's SQMMA scheduling bundle. Older MUSA compilers
        # cannot append custom llc_options to the unoptimized SQMMA flag set.
        dq_extra.update(enable_backend_opt=True, enable_llc_opt=False)
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


def _d128_compiler_env():
    # Exclude include/library injection and ambient compiler overrides.
    env = os.environ.copy()
    for key in (
        "CPATH",
        "CPLUS_INCLUDE_PATH",
        "C_INCLUDE_PATH",
        "MUSA_INCLUDE_PATH",
        "LIBRARY_PATH",
        "COMPILER_PATH",
        "GCC_EXEC_PREFIX",
        "CCC_OVERRIDE_OPTIONS",
    ):
        env.pop(key, None)
    return env


@functools.lru_cache(maxsize=8)
def _d128_toolchain(mcc, cxx, sdk):
    env = _d128_compiler_env()
    common = dict(env=env, stderr=subprocess.STDOUT, timeout=15)
    version = subprocess.check_output([mcc, "--version"], **common)
    if b"mcc version 5.2.0" not in version:
        raise OSError("D128 SDK path is qualified for mcc 5.2.0")
    cxx_version = subprocess.check_output([cxx, "--version"], **common)
    gcc_major = subprocess.check_output([cxx, "-dumpversion"], **common)
    gcc_major = gcc_major.decode().strip().split(".", 1)[0]
    cxx_include = Path("/usr/include/c++") / gcc_major
    arch_include = Path("/usr/include/x86_64-linux-gnu/c++") / gcc_major
    if not (cxx_include / "cmath").is_file() or not arch_include.is_dir():
        raise OSError("D128 SDK path requires the system GCC C++ headers")
    includes = (
        "--gcc-toolchain=/usr",
        "-isystem",
        str(cxx_include),
        "-isystem",
        str(arch_include),
        "-isystem",
        str(cxx_include / "backward"),
    )
    sdk_hash = hashlib.sha256()
    for name in (
        "musa.h",
        "musa_runtime.h",
        "musa_robust.h",
        "musa_burst.h",
        "musa_bf16.h",
        "musa_fp16.h",
    ):
        sdk_hash.update((sdk / "include" / name).read_bytes())
    return includes, (version, cxx_version, sdk_hash.hexdigest().encode())


def _d128_compile(cache, mcc, cxx, sdk, bf16, causal, *, head_dim=128):
    device_cpp, host_cpp = (
        (_D64_DEVICE_CPP, _D64_HOST_CPP)
        if head_dim == 64
        else (_D128_DEVICE_CPP, _D128_HOST_CPP)
    )
    includes, tool_ids = _d128_toolchain(mcc, cxx, sdk)
    dtype_flags = ("-DFG_D128_BF16=1",) if bf16 else ()
    if causal:
        dtype_flags += ("-DFG_D128_CAUSAL=1",)
    flags = _D128_MCC_FLAGS + includes + dtype_flags
    host_flags = ("-std=c++17", "-O2", "-shared", "-fPIC", f"-I{sdk / 'include'}")
    key = hashlib.sha256()
    for item in (
        *tool_ids,
        *(os.fsencode(x) for x in (mcc, cxx, sdk)),
        repr((flags, host_flags)).encode(),
        device_cpp.encode(),
        host_cpp.encode(),
    ):
        key.update(len(item).to_bytes(8, "little"))
        key.update(item)
    cache.mkdir(parents=True, mode=0o700, exist_ok=True)
    destination = cache / key.hexdigest()
    with destination.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (destination / "kernel.mubin").is_file() and (
            destination / "launcher.so"
        ).is_file():
            return destination
        with tempfile.TemporaryDirectory(prefix=".d128-", dir=cache) as temporary:
            stage = Path(temporary) / "artifact"
            stage.mkdir(mode=0o700)
            (stage / "kernel.mu").write_text(device_cpp)
            (stage / "launcher.cc").write_text(host_cpp)
            commands = (
                [
                    mcc,
                    *flags,
                    "-MD",
                    "-MF",
                    "kernel.d",
                    "kernel.mu",
                    "-o",
                    "kernel.mubin",
                ],
                [
                    cxx,
                    *host_flags,
                    "launcher.cc",
                    f"-L{sdk / 'lib'}",
                    "-l:libmusa.so.1",
                    "-o",
                    "launcher.so",
                ],
            )
            env = _d128_compiler_env()
            for command in commands:
                result = subprocess.run(
                    command,
                    cwd=stage,
                    env=env,
                    timeout=120,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                if result.returncode:
                    raise OSError(
                        f"{Path(command[0]).name} failed ({result.returncode}): "
                        f"{result.stdout[-2500:]}"
                    )
            (stage / "build.json").write_text(json.dumps(commands))
            if destination.exists():
                # Recover a partial directory left by manual cache maintenance.
                shutil.rmtree(destination)
            stage.rename(destination)
    return destination


_D128_HANDLE_LOCK = threading.Lock()
_D128_HANDLES = {}
_D128_UNAVAILABLE = set()


def _d128_prepare(q, k, v, dout, lse, causal, *, head_dim=128):
    if os.getenv(f"FLAGGEMS_MUSA_BWD_D{head_dim}", "auto") not in ("auto", "cpp"):
        return None
    if q.device.type != "musa" or q.dtype not in (torch.float16, torch.bfloat16):
        return None
    # The SDK tiles, including their packed row-statistics loads, assume this shape.
    if any(
        t.shape != (4, 16, 1024, head_dim)
        or t.device != q.device
        or t.dtype != q.dtype
        or not t.is_contiguous()
        for t in (q, k, v, dout)
    ):
        return None
    if (
        lse.shape != (4, 16, 1024)
        or lse.device != q.device
        or lse.dtype != torch.float32
        or not lse.is_contiguous()
    ):
        return None
    if any(t.data_ptr() % 16 for t in (q, k, v, dout, lse)):
        return None
    with torch.musa.device(q.device):
        properties = torch.musa.get_device_properties(q.device)
        if (int(properties.major), int(properties.minor)) != (3, 1):
            return None
        is_capturing = getattr(torch.musa, "is_current_stream_capturing", None)
        if is_capturing is not None and is_capturing():
            return None
        sdk = Path(os.getenv("MUSA_HOME", "/usr/local/musa"))
        mcc = shutil.which("mcc", path=f"{sdk / 'bin'}:{os.defpath}")
        cxx = shutil.which("g++", path=os.defpath)
        if not mcc or not cxx:
            return None
        cache = Path(
            os.getenv(f"FLAGGEMS_MUSA_BWD_D{head_dim}_CACHE_DIR")
            or Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
            / "flaggems"
            / f"musa-flash-bwd-d{head_dim}"
        )
        bf16 = q.dtype == torch.bfloat16
        key = (
            os.getpid(),
            q.device.index,
            mcc,
            cxx,
            sdk,
            cache,
            bf16,
            bool(causal),
            head_dim,
        )
        with _D128_HANDLE_LOCK:
            if key in _D128_HANDLES:
                return _D128_HANDLES[key]
            if key in _D128_UNAVAILABLE:
                return None
            try:
                artifact = _d128_compile(
                    cache, mcc, cxx, sdk, bf16, bool(causal), head_dim=head_dim
                )
                lib = ctypes.CDLL(os.fspath(artifact / "launcher.so"))
                ptr = ctypes.c_void_p
                lib.mp31_load.argtypes = [ctypes.c_char_p, ctypes.POINTER(ptr)]
                lib.mp31_load.restype = ctypes.c_int
                lib.mp31_launch.argtypes = [
                    ptr,
                    ptr,
                    ctypes.c_int,
                    *([ptr] * 8),
                    ctypes.c_float,
                    ctypes.c_float,
                ]
                lib.mp31_launch.restype = ctypes.c_int
                function = ptr()
                status = lib.mp31_load(
                    os.fsencode(artifact / "kernel.mubin"), ctypes.byref(function)
                )
                if status or not function.value:
                    raise OSError(
                        f"MUSA module load/shared-memory opt-in failed ({status})"
                    )
            except (OSError, subprocess.SubprocessError) as error:
                _D128_UNAVAILABLE.add(key)
                logger.debug("D%s SDK route unavailable: %s", head_dim, error)
                return None
            handle = lib, function, artifact
            _D128_HANDLES[key] = handle
            return handle


def _d128_launch(handle, q, k, v, dout, row_stats, dq_accum, dk, dv, scale, scale_log2):
    lib, function, _ = handle
    tensors = q, k, v, dout, row_stats, dq_accum, dk, dv
    with torch.musa.device(q.device):
        stream = torch.musa.current_stream(q.device)
        status = lib.mp31_launch(
            function,
            stream.musa_stream,
            int(q.dtype == torch.bfloat16),
            *(tensor.data_ptr() for tensor in tensors),
            scale,
            scale_log2,
        )
        if status:
            raise RuntimeError(f"D128 SDK kernel launch failed ({status})")
        for tensor in tensors:
            tensor.record_stream(stream)


_D128_MCC_FLAGS = (
    "--cuda-device-only",
    "--cuda-gpu-arch=mp_31",
    "-std=c++17",
    "-Wno-error=address-of-temporary",
    "-Wno-error=c++11-narrowing",
    "-D__MUSA_ARCH_LIST__=310",
    "-mllvm",
    "-mtgpu-alloc-shared-memory-from-zero=1",
    "-fmusa-flush-denormals-to-zero",
    "-fno-signed-zeros",
    "-fno-strict-aliasing",
    "-mllvm",
    "-misched=mtgpu-max-ilp",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-misched-recompute-slotindex=1",
    "-mllvm",
    "-mtgpu-if-convert=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
)


# D128 MP31 device body generated by TileLang-MUSA 0.1.12+musa.5 from
# FlagGems 218608685, adapted from MooreThreads/mate at
# 5ab49c437638caf499b6f8b2480e4a1fa0da7dfe
# (mate/flash_attention/tilelang/_flash_attention_bwd_unsplit.py).
# Copyright (c) Moore Threads Technology Co., Ltd. Apache-2.0 (root LICENSE).
# This adaptation replaces the generated helper calls with fixed MP31 SDK
# primitives for BM64/BN128, and constructs the same launch ABI with the SDK
# Driver API. It preserves the generated computation and async schedule.
# The following notices cover the TileLang code generation and the adapted
# helper semantics from TileLang and MUTLASS/MuTe (MUTLASS d31e365170c9f).
#     MIT License
#
#     Copyright (c) Tile-AI.
#     **During the period from December 1, 2024, to Mar 14, 2025, this project is
#     subject to additional collaboration terms with Microsoft Corporation.**
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included in all
#     copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#     SOFTWARE
#
# The MUTLASS from Moore Threads is licensed under the BSD-3-Clause listed below.
# Copyright (c) 2024-2024 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Terms of the BSD-3-Clause
# -------------------------------------------------------------------------
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# -------------------------------------------------------------------------
#
# The following copyright statements and licenses apply to various open source software/model
# packages (or portions thereof) that are distributed with this MUTLASS. MUTLASS that
# includes this file does not necessarily use all the open source software packages referred
# to below and may also only use portions of a given package. Some open source software
# packages referred to below may have been modified by Moore Threads Technology Co., Ltd
# -------------------------------------------------------------------------
# cutlass
# Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -------------------------------------------------------------------------
# googletest
# Copyright 2008, Google Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the name of Google Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Escaped line breaks wrap the Python source without changing the SDK payload.
_D128_DEVICE_CPP = """
// Derived from the MP31 TileLang code generation for the MATE backward kernel.
// See the copyright and dependency notices in the enclosing Python source.
#include <cstdint>
#include <musa.h>
#include <musa_runtime.h>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_burst.h>
#include <musa_robust.h>
#include <math_constants.h>

#define FG_DEVICE __device__ __forceinline__
#define FG_GLOBAL_AS __attribute__((address_space(1)))
#define FG_SHARED_AS __attribute__((address_space(3)))

using uchar = unsigned char;
using half_t = __half;
using bfloat16_t = __mt_bfloat16;
using tl_f4 = float __attribute__((ext_vector_type(4)));
using tl_h4 = decltype(__half_raw{}.data) __attribute__((ext_vector_type(4)));
using tl_h8 = decltype(__half_raw{}.data) __attribute__((ext_vector_type(8)));
using tl_bf4 = decltype(__mt_bfloat16_raw{}.data) __attribute__((ext_vector_type(4)));
using tl_bf8 = decltype(__mt_bfloat16_raw{}.data) __attribute__((ext_vector_type(8)));

FG_DEVICE void AtomicAdd(float& dst, float value) {
  atomicAdd(&dst, value);
}

namespace d128 {
using robust_param_desc_t = __musa::robust_data_t;
using Tile = int32_t __attribute__((vector_size(16)));
static_assert(sizeof(robust_param_desc_t) == 16);

FG_DEVICE bool block_leader() { return threadIdx.x == 0; }
FG_DEVICE void fence_barrier_init() { asm volatile("" ::: "memory"); }
FG_DEVICE void lma_wait() { __musa_lma_wait(); }
FG_DEVICE void finish_robust_copies() { __musa_memcpy_g2s_wait(); }
FG_DEVICE void warpgroup_commit_batch() { __musa_tce_commit_group(); }
template <int Pending> FG_DEVICE void warpgroup_wait() { __musa_tce_wait_group(Pending); }
FG_DEVICE void wait_sqmma() { __musa_sqmma_wait(); }

template <int Bytes>
FG_DEVICE void cp_async_gs_robust(const void* dst, const void* src,
                                 robust_param_desc_t bounds) {
  __musa_memcpy_g2s_robust_v4((void FG_SHARED_AS *)dst,
                             (void FG_GLOBAL_AS *)src, Bytes, bounds, 0);
}

enum class TmaGranularity : int32_t { B16 = 1, B32 = 2 };
template <TmaGranularity Granularity>
FG_DEVICE void tma_load(const MUtensorDescriptor& descriptor, uint32_t barrier,
                        void* dst, int x, int h, int s, int b,
                        int tile_x, int tile_h, int tile_s, int tile_b) {
  using SharedPointer = void FG_SHARED_AS *;
  const auto shared = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  // MP31: stride=256 B, line=256 B, no prefetch, normal inner/outer cache.
  __musa_tme_ld_tile_4d(barrier, reinterpret_cast<SharedPointer>(shared),
      reinterpret_cast<uint64_t>(&descriptor), Tile{tile_x, tile_h, tile_s, tile_b},
      Tile{x, h, s, b}, static_cast<int>(Granularity), 3, 1, 0, 2, 2, 0);
}

// Fuse scores with the previously rounded LSE term, as in the Triton path.
FG_DEVICE tl_f4 fma4(tl_f4 a, float b, float c) {
  return tl_f4{fmaf(a[0], b, c), fmaf(a[1], b, c),
               fmaf(a[2], b, c), fmaf(a[3], b, c)};
}
FG_DEVICE tl_f4 vec_exp2_f4(tl_f4 x) {
  const float4 y = __exp2f(float4{x[0], x[1], x[2], x[3]});
  return tl_f4{y.x, y.y, y.z, y.w};
}
using half4_t = __half4;
using bfloat164_t = __mt_bfloat164;
FG_DEVICE half4_t cvt_float_to_half_x4(float4 x) { return __float42half4_rn(x); }
FG_DEVICE bfloat164_t cvt_float_to_bfloat16_x4(float4 x) { return __float42bfloat164_rn(x); }

#ifdef FG_D128_BF16
using Scalar = bfloat16_t;
using Vec4 = tl_bf4;
using Vec8 = tl_bf8;
using Packed4 = bfloat164_t;
FG_DEVICE Packed4 round_to_scalar4(float4 x) { return cvt_float_to_bfloat16_x4(x); }
#else
using Scalar = half_t;
using Vec4 = tl_h4;
using Vec8 = tl_h8;
using Packed4 = half4_t;
FG_DEVICE Packed4 round_to_scalar4(float4 x) { return cvt_float_to_half_x4(x); }
#endif

// p is the nonnegative exp2 result. The product bounds are adjacent
// FP32 values (or identical for an exact product). Preserve the sign of
// the truncated bound and use the other bound's low bit to round to odd
// before the FP16/BF16 conversion.
FG_DEVICE float scale_ds(float p, float diff) {
  const float truncated = __fmul_rz(p, diff);
  const float away = __fmul_ru(p, fabsf(diff));
  const uint32_t bits = __builtin_bit_cast(uint32_t, truncated);
  const uint32_t sticky = __builtin_bit_cast(uint32_t, away) & 1u;
  return __builtin_bit_cast(float, bits | sticky);
}

enum class Product { QK, DOV, DV, DK, DQ };
// Fields in an MP31 SQMMA descriptor: 18-bit shared byte address, then
// 3-bit leading stride and 2-bit swizzle granularity. Only the three
// layout combinations emitted by the frozen BM64/BN128 kernel are used.
enum DescriptorField {
  kKMajorSwizzled = (6 << 18) | (1 << 21),
  kMNMajorSwizzled = (5 << 18) | (2 << 21),
  kMNMajorWide = (6 << 18) | (2 << 21),
};

FG_DEVICE int32_t sqmma_desc(const void* ptr, DescriptorField layout) {
  return (static_cast<uint32_t>(__cvta_generic_to_shared(ptr)) & 0x3ffff) | layout;
}

template <Product P, typename T>
FG_DEVICE void gemm(T* a, T* b, float* accum, int first_thread) {
  constexpr int dtype = __is_same(T, __mt_bfloat16) ? 3 : 2;
  const int squad = ((static_cast<int>(threadIdx.x) - first_thread) / 128) % 2;
  auto* d = reinterpret_cast<int32_t*>(accum);
  if constexpr (P == Product::DK) {
    const int32_t da = sqmma_desc(a, kMNMajorWide);
    const int32_t db = sqmma_desc(b + squad * 4096, kMNMajorSwizzled);
    __musa_sqmma_m128n64_mma(d, &da, &db, d, 1, 0, 0, 0, 1, 0, 64, dtype);
  } else if constexpr (P == Product::DV) {
    const int32_t da = sqmma_desc(a + squad * 4096, kMNMajorSwizzled);
    const int32_t db = sqmma_desc(b, kMNMajorWide);
    __musa_sqmma_m64n128_mma(d, &da, &db, d, 1, 0, 0, 0, 1, 0, 64, dtype);
  } else {
    constexpr bool dq = P == Product::DQ;
    const int32_t da = sqmma_desc(a, kKMajorSwizzled);
    const int32_t db = sqmma_desc(b + squad * 8192, dq ? kMNMajorSwizzled : kKMajorSwizzled);
    __musa_sqmma_m64n64_mma(d, &da, &db, d, 0, !dq, 0, 0,
                            P == Product::QK, 0, 64, dtype);
    const int32_t da2 = da + 128;
    const int32_t db2 = db + (dq ? 8192 : 128);
    __musa_sqmma_m64n64_mma(d, &da2, &db2, d, 0, !dq, 0, 0, 1, 0, 64, dtype);
  }
}
}  // namespace d128

#ifdef FG_D128_CAUSAL
extern "C" __global__ void flashattn_bwd_ws_unsplit_kernel_kernel(const float2* __restrict__ RowStats,\
 const d128::Scalar* __restrict__ K, __attribute__((grid_constant)) const MUtensorDescriptor K_desc_0,\
 __attribute__((grid_constant)) const d128::robust_param_desc_t K_robust_desc_0, __attribute__((grid_constant))\
 const MUtensorDescriptor Q_desc_1, __attribute__((grid_constant)) const MUtensorDescriptor Q_desc_4,\
 const d128::Scalar* __restrict__ V, __attribute__((grid_constant)) const d128::robust_param_desc_t V_robust_desc_1,\
 d128::Scalar* __restrict__ dK, __attribute__((grid_constant)) const MUtensorDescriptor dO_desc_2,\
 __attribute__((grid_constant)) const MUtensorDescriptor dO_desc_3, float* __restrict__ dQ_accum,\
 d128::Scalar* __restrict__ dV, int batch, int dk_stride_b, int dk_stride_h, int dk_stride_s, int dv_stride_b,\
 int dv_stride_h, int dv_stride_s, int heads_q, int k_stride_b, int k_stride_h, int k_stride_s, int max_seq_kv,\
 int max_seq_q, float rln2_scale, int seqlen_k, int seqlen_q, float smscale, int v_stride_b, int v_stride_h,\
 int v_stride_s);
extern "C" __global__ void __launch_bounds__(640, 1) flashattn_bwd_ws_unsplit_kernel_kernel(const float2* __restrict__\
 RowStats, const d128::Scalar* __restrict__ K, __attribute__((grid_constant)) const MUtensorDescriptor K_desc_0,\
 __attribute__((grid_constant)) const d128::robust_param_desc_t K_robust_desc_0, __attribute__((grid_constant))\
 const MUtensorDescriptor Q_desc_1, __attribute__((grid_constant)) const MUtensorDescriptor Q_desc_4,\
 const d128::Scalar* __restrict__ V, __attribute__((grid_constant)) const d128::robust_param_desc_t V_robust_desc_1,\
 d128::Scalar* __restrict__ dK, __attribute__((grid_constant)) const MUtensorDescriptor dO_desc_2,\
 __attribute__((grid_constant)) const MUtensorDescriptor dO_desc_3, float* __restrict__ dQ_accum,\
 d128::Scalar* __restrict__ dV, int batch, int dk_stride_b, int dk_stride_h, int dk_stride_s, int dv_stride_b,\
 int dv_stride_h, int dv_stride_s, int heads_q, int k_stride_b, int k_stride_h, int k_stride_s, int max_seq_kv,\
 int max_seq_q, float rln2_scale, int seqlen_k, int seqlen_q, float smscale, int v_stride_b, int v_stride_h,\
 int v_stride_s) {
  __musa_async_bar_record(28);
  extern __shared__ __align__(4096) uchar buf_dyn_shmem[];
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Kt_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* Vt_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  void* Qb_shared = ((void*)((char*)buf_dyn_shmem + 98304));
  void* dOb_shared = ((void*)((char*)buf_dyn_shmem + 114688));
  void* Pt_shared = ((void*)((char*)buf_dyn_shmem + 131072));
  void* dOa_shared = ((void*)((char*)buf_dyn_shmem + 147456));
  void* dS_shared = ((void*)((char*)buf_dyn_shmem + 163840));
  void* dSt_shared = ((void*)((char*)buf_dyn_shmem + 180224));
  int kv_group = 0;
  int begin_seq_q = 0;
  int end_seq_q = 0;
  int begin_seq_kv = 0;
  int end_seq_kv = 0;
  int stage_producer = 0;
  int phase_producer = 0;
  int prev_stage_producer = 0;
  int prev_phase_producer = 0;
  int warprole = 0;
  int producer_causal_offset = 0;
  int producer_causal_q_local_start = 0;
  int producer_causal_q_start = 0;
  int stage_consumer1 = 0;
  int phase_consumer1 = 0;
  float dk_accum[64];
  float dq_accum[32];
  float dq_store_buffer[32];
  int consumer1_causal_offset = 0;
  int consumer1_causal_q_local_start = 0;
  int consumer1_causal_q_start = 0;
  int local_q_start = 0;
  d128::Scalar dk_cast[64];
  int stage_consumer0 = 0;
  int phase_consumer0 = 0;
  int prev_stage_consumer0 = 0;
  int prev_phase_consumer0 = 0;
  float dv_accum[64];
  int consumer0_causal_offset = 0;
  int consumer0_causal_q_local_start = 0;
  int consumer0_causal_q_start = 0;
  int consumer0_causal_q_local_mask_end = 0;
  int consumer0_causal_q_mask_end = 0;
  d128::Scalar dv_cast[64];
  float lse_buffer[4];
  float delta_buffer[4];
  float accs_accum[32];
  float dp_accum[32];
  d128::Scalar accs_cast[32];
  if (d128::block_leader()) {
    __musa_async_init_arrival(1, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(2, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(3, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(4, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(5, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(6, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(7, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(8, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(9, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(10, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(11, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(12, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(13, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(14, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(15, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(16, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(17, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(18, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(19, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(20, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(21, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(22, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(23, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(24, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(25, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(26, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(27, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(28, ((512 + 31) / 32), 0);
  }
  d128::fence_barrier_init();
  __syncthreads_lm();
  kv_group = ((int)blockIdx.y);
  begin_seq_q = (((int)blockIdx.z) * seqlen_q);
  end_seq_q = (begin_seq_q + seqlen_q);
  begin_seq_kv = (((int)blockIdx.z) * seqlen_k);
  end_seq_kv = (begin_seq_kv + seqlen_k);
  __syncthreads_lm();
  if (((((int)blockIdx.x) * 128) + begin_seq_kv) < end_seq_kv) {
    if (512 <= ((int)threadIdx.x)) {
      stage_producer = 0;
      phase_producer = 0;
      prev_stage_producer = 1;
      prev_phase_producer = 1;
      warprole = ((((int)threadIdx.x) >> 5) - 16);
      if (((int)threadIdx.x) == 512) {
        __musa_async_add_trans(1, 32768);
        d128::tma_load<d128::TmaGranularity::B32>(K_desc_0, 1, (&(((d128::Scalar*)K_shared)[0])), 0, kv_group, (((int)\
blockIdx.x) * 128), ((int)blockIdx.z), 64, 1, 128, 1);
        d128::tma_load<d128::TmaGranularity::B32>(K_desc_0, 1, (&(((d128::Scalar*)K_shared)[8192])), 64, kv_group,\
 (((int)blockIdx.x) * 128), ((int)blockIdx.z), 64, 1, 128, 1);
      }
      __musa_async_arrive(1);
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        d128::cp_async_gs_robust<16>((&(((d128::Scalar*)Kt_shared)[((((i * 1024) + ((((int)threadIdx.x) >> 4) * 128))\
 + (((((int)threadIdx.x) & 15) ^ (((i * 8) + (((int)threadIdx.x) >> 4)) & 15)) * 8)) - 4096)])), (&(K[(((((((int)\
threadIdx.x) & 15) * 8) + (((int)blockIdx.z) * k_stride_b)) + (((((((int)blockIdx.x) * 128) + ((i >> 3) * 64))\
 + (((((int)threadIdx.x) & 127) >> 4) * 8)) + (i & 7)) * k_stride_s)) + (kv_group * k_stride_h))])), K_robust_desc_0);
      }
      #pragma unroll
      for (int i_1 = 0; i_1 < 16; ++i_1) {
        d128::cp_async_gs_robust<16>((&(((d128::Scalar*)Vt_shared)[((((i_1 * 1024) + ((((int)threadIdx.x) >> 4)\
 * 128)) + (((((int)threadIdx.x) & 15) ^ (((i_1 * 8) + (((int)threadIdx.x) >> 4)) & 15)) * 8)) - 4096)])),\
 (&(V[(((((((int)threadIdx.x) & 15) * 8) + (((int)blockIdx.z) * v_stride_b)) + (((((((int)blockIdx.x) * 128)\
 + ((i_1 >> 3) * 64)) + (((((int)threadIdx.x) & 127) >> 4) * 8)) + (i_1 & 7)) * v_stride_s))\
 + (kv_group * v_stride_h))])), V_robust_desc_1);
      }
      d128::finish_robust_copies();
      __musa_async_arrive(2);
      __musa_async_arrive(3);
      producer_causal_offset = (((end_seq_kv + begin_seq_q) - begin_seq_kv) - end_seq_q);
      producer_causal_q_local_start = ((((int)blockIdx.x) * 128) - producer_causal_offset);
      int condval;
      if ((0 < producer_causal_q_local_start)) {
        condval = producer_causal_q_local_start;
      } else {
        condval = 0;
      }
      producer_causal_q_local_start = condval;
      producer_causal_q_start = (((producer_causal_q_local_start >> 6) * 64) + begin_seq_q);
      for (int _tmp = 0; _tmp < (((end_seq_q + 63) - producer_causal_q_start) >> 6); ++_tmp) {
        int q_start = ((_tmp * 64) + producer_causal_q_start);
        if (warprole == 0) {
          __musa_async_wait((prev_stage_producer + 12), prev_phase_producer);
          if (((int)threadIdx.x) == 512) {
            __musa_async_add_trans((stage_producer + 4), 16384);
            d128::tma_load<d128::TmaGranularity::B16>(Q_desc_1, (stage_producer + 4), (&(((d128::Scalar*)\
Pt_shared)[0])), 0, ((int)blockIdx.y), (q_start - begin_seq_q), ((int)blockIdx.z), 128, 1, 64, 1);
          }
          __musa_async_arrive((stage_producer + 4));
        }
        if (warprole == 1) {
          __musa_async_wait((prev_stage_producer + 16), prev_phase_producer);
          if (((int)threadIdx.x) == 544) {
            __musa_async_add_trans((stage_producer + 8), 16384);
            d128::tma_load<d128::TmaGranularity::B16>(dO_desc_2, (stage_producer + 8), (&(((d128::Scalar*)\
dOa_shared)[0])), 0, ((int)blockIdx.y), (q_start - begin_seq_q), ((int)blockIdx.z), 128, 1, 64, 1);
          }
          __musa_async_arrive((stage_producer + 8));
        }
        if (warprole == 2) {
          __musa_async_wait((prev_stage_producer + 18), prev_phase_producer);
          if (((int)threadIdx.x) == 576) {
            __musa_async_add_trans((stage_producer + 10), 16384);
            d128::tma_load<d128::TmaGranularity::B32>(dO_desc_3, (stage_producer + 10), (&(((d128::Scalar*)\
dOb_shared)[0])), 0, ((int)blockIdx.y), (q_start - begin_seq_q), ((int)blockIdx.z), 128, 1, 64, 1);
          }
          __musa_async_arrive((stage_producer + 10));
        }
        if (warprole == 3) {
          __musa_async_wait((prev_stage_producer + 14), prev_phase_producer);
          if (((int)threadIdx.x) == 608) {
            __musa_async_add_trans((stage_producer + 6), 16384);
            d128::tma_load<d128::TmaGranularity::B32>(Q_desc_4, (stage_producer + 6), (&(((d128::Scalar*)\
Qb_shared)[0])), 0, ((int)blockIdx.y), (q_start - begin_seq_q), ((int)blockIdx.z), 64, 1, 64, 1);
            d128::tma_load<d128::TmaGranularity::B32>(Q_desc_4, (stage_producer + 6), (&(((d128::Scalar*)\
Qb_shared)[4096])), 64, ((int)blockIdx.y), (q_start - begin_seq_q), ((int)blockIdx.z), 64, 1, 64, 1);
          }
          __musa_async_arrive((stage_producer + 6));
        }
        prev_stage_producer = stage_producer;
        prev_phase_producer = phase_producer;
        phase_producer = (phase_producer ^ stage_producer);
        stage_producer = (stage_producer ^ 1);
      }
      __musa_async_wait(28, 0);
    } else {
      if (256 <= ((int)threadIdx.x)) {
        stage_consumer1 = 0;
        phase_consumer1 = 0;
        __musa_async_wait(1, 0);
        __musa_async_wait(2, 0);
        __musa_async_wait(3, 0);
        #pragma unroll
        for (int i_2 = 0; i_2 < 16; ++i_2) {
          float broadcast_var = 0x0p+0f/*0.000000e+00*/;
          *(tl_f4*)(dk_accum + (i_2 * 4)) = (tl_f4{static_cast<float>(broadcast_var),\
 static_cast<float>(broadcast_var), static_cast<float>(broadcast_var), static_cast<float>(broadcast_var)});
        }
        __musa_async_wait((stage_consumer1 + 6), phase_consumer1);
        __musa_async_wait((stage_consumer1 + 26), phase_consumer1);
        d128::gemm<d128::Product::DK>((&(((d128::Scalar*)dSt_shared)[0])), (&(((d128::Scalar*)Qb_shared)[0])),\
 (&(dk_accum[0])), 256);
        d128::wait_sqmma();
        __musa_async_arrive((stage_consumer1 + 14));
        __musa_async_wait((stage_consumer1 + 24), phase_consumer1);
        d128::gemm<d128::Product::DQ>((&(((d128::Scalar*)dS_shared)[0])), (&(((d128::Scalar*)K_shared)[0])),\
 (&(dq_accum[0])), 256);
        d128::wait_sqmma();
        __musa_async_arrive((stage_consumer1 + 22));
        #pragma unroll
        for (int i_3 = 0; i_3 < 8; ++i_3) {
          *(tl_f4*)(dq_store_buffer + (i_3 * 4)) = *(tl_f4*)(dq_accum + (i_3 * 4));
        }
        phase_consumer1 = (phase_consumer1 ^ stage_consumer1);
        stage_consumer1 = (stage_consumer1 ^ 1);
        consumer1_causal_offset = (((end_seq_kv + begin_seq_q) - begin_seq_kv) - end_seq_q);
        consumer1_causal_q_local_start = ((((int)blockIdx.x) * 128) - consumer1_causal_offset);
        int condval_1;
        if ((0 < consumer1_causal_q_local_start)) {
          condval_1 = consumer1_causal_q_local_start;
        } else {
          condval_1 = 0;
        }
        consumer1_causal_q_local_start = condval_1;
        consumer1_causal_q_start = (((consumer1_causal_q_local_start >> 6) * 64) + begin_seq_q);
        int dq_loop_start = consumer1_causal_q_start;
        local_q_start = (dq_loop_start - begin_seq_q);
        for (int _tmp_1 = 0; _tmp_1 < (((end_seq_q - dq_loop_start) - 1) >> 6); ++_tmp_1) {
          #pragma unroll
          for (int i_4 = 0; i_4 < 32; ++i_4) {
            AtomicAdd(*((&(dQ_accum[((((((((((i_4 >> 3) * 2048) + (((((int)threadIdx.x) & 127) >> 3) * 128))\
 + (local_q_start * 128)) + (((((int)blockIdx.z) * max_seq_q) * heads_q) * 128)) + ((((int)blockIdx.y) * max_seq_q)\
 * 128)) + ((((int)threadIdx.x) >> 7) * 64)) + ((i_4 & 7) * 8)) + (((int)threadIdx.x) & 7)) - 128)]))),\
 dq_store_buffer[i_4]);
          }
          __musa_async_wait((stage_consumer1 + 6), phase_consumer1);
          __musa_async_wait((stage_consumer1 + 26), phase_consumer1);
          d128::gemm<d128::Product::DK>((&(((d128::Scalar*)dSt_shared)[0])), (&(((d128::Scalar*)Qb_shared)[0])),\
 (&(dk_accum[0])), 256);
          d128::wait_sqmma();
          __musa_async_wait((stage_consumer1 + 24), phase_consumer1);
          d128::gemm<d128::Product::DQ>((&(((d128::Scalar*)dS_shared)[0])), (&(((d128::Scalar*)K_shared)[0])),\
 (&(dq_accum[0])), 256);
          d128::wait_sqmma();
          __musa_async_arrive((stage_consumer1 + 14));
          __musa_async_arrive((stage_consumer1 + 22));
          #pragma unroll
          for (int i_5 = 0; i_5 < 8; ++i_5) {
            *(tl_f4*)(dq_store_buffer + (i_5 * 4)) = *(tl_f4*)(dq_accum + (i_5 * 4));
          }
          phase_consumer1 = (phase_consumer1 ^ stage_consumer1);
          stage_consumer1 = (stage_consumer1 ^ 1);
          local_q_start = (local_q_start + 64);
        }
        #pragma unroll
        for (int i_6 = 0; i_6 < 32; ++i_6) {
          AtomicAdd(*((&(dQ_accum[((((((((((i_6 >> 3) * 2048) + (((((int)threadIdx.x) & 127) >> 3) * 128))\
 + (local_q_start * 128)) + (((((int)blockIdx.z) * max_seq_q) * heads_q) * 128)) + ((((int)blockIdx.y) * max_seq_q)\
 * 128)) + ((((int)threadIdx.x) >> 7) * 64)) + ((i_6 & 7) * 8)) + (((int)threadIdx.x) & 7)) - 128)]))),\
 dq_store_buffer[i_6]);
        }
        #pragma unroll
        for (int i_7 = 0; i_7 < 16; ++i_7) {
          d128::Vec4 __1;
          tl_f4 v_ = *(tl_f4*)(dk_accum + (i_7 * 4));
          ((d128::Packed4*)(&__1))[0] = d128::round_to_scalar4(((float4*)(&v_))[0]);
          *(d128::Vec4*)(dk_cast + (i_7 * 4)) = __1;
        }
        if (end_seq_kv < (((((int)blockIdx.x) * 128) + begin_seq_kv) + 128)) {
          #pragma unroll
          for (int i_8 = 0; i_8 < 64; ++i_8) {
            if (((((((int)blockIdx.x) * 128) + ((i_8 >> 3) * 16)) + ((((int)threadIdx.x) & 127) >> 3)) + begin_seq_kv)\
 < end_seq_kv) {
              dK[((((((((((int)threadIdx.x) >> 7) * 64) + ((i_8 & 7) * 8)) + (((int)blockIdx.z) * dk_stride_b))\
 + ((((((int)blockIdx.x) * 128) + ((i_8 >> 3) * 16)) + ((((int)threadIdx.x) & 127) >> 3)) * dk_stride_s)) + (((int)\
blockIdx.y) * dk_stride_h)) + (((int)threadIdx.x) & 7)) - 128)] = dk_cast[i_8];
            }
          }
        } else {
          #pragma unroll
          for (int i_9 = 0; i_9 < 64; ++i_9) {
            dK[((((((((((int)threadIdx.x) >> 7) * 64) + ((i_9 & 7) * 8)) + (((int)blockIdx.z) * dk_stride_b))\
 + ((((((int)blockIdx.x) * 128) + ((i_9 >> 3) * 16)) + ((((int)threadIdx.x) & 127) >> 3)) * dk_stride_s)) + (((int)\
blockIdx.y) * dk_stride_h)) + (((int)threadIdx.x) & 7)) - 128)] = dk_cast[i_9];
          }
        }
        __musa_async_arrive(28);
      } else {
        stage_consumer0 = 0;
        phase_consumer0 = 0;
        prev_stage_consumer0 = 1;
        prev_phase_consumer0 = 1;
        __musa_async_wait(1, 0);
        __musa_async_wait(2, 0);
        __musa_async_wait(3, 0);
        #pragma unroll
        for (int i_10 = 0; i_10 < 16; ++i_10) {
          float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
          *(tl_f4*)(dv_accum + (i_10 * 4)) = (tl_f4{static_cast<float>(broadcast_var_1),\
 static_cast<float>(broadcast_var_1), static_cast<float>(broadcast_var_1), static_cast<float>(broadcast_var_1)});
        }
        consumer0_causal_offset = (((end_seq_kv + begin_seq_q) - begin_seq_kv) - end_seq_q);
        consumer0_causal_q_local_start = ((((int)blockIdx.x) * 128) - consumer0_causal_offset);
        int condval_2;
        if ((0 < consumer0_causal_q_local_start)) {
          condval_2 = consumer0_causal_q_local_start;
        } else {
          condval_2 = 0;
        }
        consumer0_causal_q_local_start = condval_2;
        consumer0_causal_q_start = (((consumer0_causal_q_local_start >> 6) * 64) + begin_seq_q);
        consumer0_causal_q_local_mask_end = (((((int)blockIdx.x) * 128) + 128) - consumer0_causal_offset);
        int condval_3;
        if ((0 < consumer0_causal_q_local_mask_end)) {
          condval_3 = consumer0_causal_q_local_mask_end;
        } else {
          condval_3 = 0;
        }
        consumer0_causal_q_local_mask_end = condval_3;
        consumer0_causal_q_mask_end = ((((consumer0_causal_q_local_mask_end + 63) >> 6) * 64) + begin_seq_q);
        int condval_4;
        if ((consumer0_causal_q_mask_end < end_seq_q)) {
          condval_4 = consumer0_causal_q_mask_end;
        } else {
          condval_4 = end_seq_q;
        }
        consumer0_causal_q_mask_end = condval_4;
        for (int _tmp_2 = 0; _tmp_2 < (((end_seq_q + 63) - consumer0_causal_q_start) >> 6); ++_tmp_2) {
          int q_start_1 = ((_tmp_2 * 64) + consumer0_causal_q_start);
          #pragma unroll
          for (int i_11 = 0; i_11 < 4; ++i_11) {
            const int row = i_11 * 16 + ((int(threadIdx.x) & 127) >> 3)
                + (int(blockIdx.z) * heads_q + kv_group) * seqlen_q
                + q_start_1 - begin_seq_q;
            const float2 stats = RowStats[row];
            lse_buffer[i_11] = stats.x;
            delta_buffer[i_11] = stats.y;
          }
          if (end_seq_kv < (((((int)blockIdx.x) * 128) + begin_seq_kv) + 128)) {
            #pragma unroll
            for (int i_14 = 0; i_14 < 32; ++i_14) {
              float condval_6;
              if (((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 7) * 64)) + ((((int)threadIdx.x) & 7) * 8))\
 + begin_seq_kv) + (i_14 & 7)) < end_seq_kv)) {
                condval_6 = 0x0p+0f/*0.000000e+00*/;
              } else {
                condval_6 = -MUSART_INF_F;
              }
              accs_accum[i_14] = condval_6;
            }
          } else {
            #pragma unroll
            for (int i_15 = 0; i_15 < 8; ++i_15) {
              float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
              *(tl_f4*)(accs_accum + (i_15 * 4)) = (tl_f4{static_cast<float>(broadcast_var_2),\
 static_cast<float>(broadcast_var_2), static_cast<float>(broadcast_var_2), static_cast<float>(broadcast_var_2)});
            }
          }
          if (q_start_1 < consumer0_causal_q_mask_end) {
            #pragma unroll
            for (int i_16 = 0; i_16 < 32; ++i_16) {
              int q_idx = (((((i_16 >> 3) * 16) + ((((int)threadIdx.x) & 127) >> 3)) + q_start_1) - begin_seq_q);
              bool valid = (((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 7) * 64)) + ((((int)threadIdx.x)\
 & 7) * 8)) + (i_16 & 7)) <= (q_idx + consumer0_causal_offset));
              float condval_7;
              if (valid) {
                condval_7 = accs_accum[i_16];
              } else {
                condval_7 = -MUSART_INF_F;
              }
              accs_accum[i_16] = condval_7;
            }
          }
          __musa_async_wait((stage_consumer0 + 4), phase_consumer0);
          d128::gemm<d128::Product::QK>((&(((d128::Scalar*)Pt_shared)[0])), (&(((d128::Scalar*)Kt_shared)[0])),\
 (&(accs_accum[0])), 0);
          d128::warpgroup_commit_batch();
          __musa_async_wait((stage_consumer0 + 8), phase_consumer0);
          d128::gemm<d128::Product::DOV>((&(((d128::Scalar*)dOa_shared)[0])), (&(((d128::Scalar*)Vt_shared)[0])),\
 (&(dp_accum[0])), 0);
          d128::warpgroup_commit_batch();
          d128::warpgroup_wait<1>();
          int stage = stage_consumer0;
          __musa_async_arrive((stage + 20));
          #pragma unroll
          for (int i_21 = 0; i_21 < 8; ++i_21) {
            tl_f4 v__1 = d128::fma4(*(tl_f4*)(accs_accum + (i_21 * 4)), rln2_scale, -lse_buffer[(i_21 >> 1)]);
            *(tl_f4*)(accs_accum + (i_21 * 4)) = d128::vec_exp2_f4(v__1);
          }
          #pragma unroll
          for (int i_22 = 0; i_22 < 8; ++i_22) {
            d128::Vec4 __2;
            tl_f4 v__2 = *(tl_f4*)(accs_accum + (i_22 * 4));
            ((d128::Packed4*)(&__2))[0] = d128::round_to_scalar4(((float4*)(&v__2))[0]);
            *(d128::Vec4*)(accs_cast + (i_22 * 4)) = __2;
          }
          __musa_async_wait((stage_consumer0 + 20), phase_consumer0);
          #pragma unroll
          for (int i_23 = 0; i_23 < 4; ++i_23) {
            *(d128::Vec8*)(((d128::Scalar*)Pt_shared) + ((((((((int)threadIdx.x) >> 7) * 4096) + (i_23 * 1024))\
 + (((((int)threadIdx.x) & 127) >> 4) * 128)) + ((((((int)threadIdx.x) & 15) >> 1) ^ ((((int)threadIdx.x) & 127)\
 >> 4)) * 16)) + ((((int)threadIdx.x) & 1) * 8))) = *(d128::Vec8*)(accs_cast + (i_23 * 8));
          }
          d128::lma_wait();
          __musa_async_wait((stage_consumer0 + 10), phase_consumer0);
          d128::gemm<d128::Product::DV>((&(((d128::Scalar*)Pt_shared)[0])), (&(((d128::Scalar*)dOb_shared)[0])),\
 (&(dv_accum[0])), 0);
          d128::warpgroup_commit_batch();
          d128::warpgroup_wait<1>();
          int stage_1 = stage_consumer0;
          __musa_async_arrive((stage_1 + 16));
          #pragma unroll
          for (int i_25 = 0; i_25 < 8; ++i_25) {
            d128::Vec4 __3;
            const tl_f4 p = *(tl_f4*)(accs_accum + (i_25 * 4));
            const tl_f4 dp = *(tl_f4*)(dp_accum + (i_25 * 4));
            const tl_f4 d = d128::fma4(dp, smscale, -delta_buffer[i_25 >> 1]);
            const tl_f4 v__3 = {
                d128::scale_ds(p[0], d[0]), d128::scale_ds(p[1], d[1]),
                d128::scale_ds(p[2], d[2]), d128::scale_ds(p[3], d[3])};
            ((d128::Packed4*)(&__3))[0] = d128::round_to_scalar4(((float4*)(&v__3))[0]);
            *(d128::Vec4*)(accs_cast + (i_25 * 4)) = __3;
          }
          __musa_async_wait((prev_stage_consumer0 + 22), prev_phase_consumer0);
          #pragma unroll
          for (int i_26 = 0; i_26 < 4; ++i_26) {
            *(d128::Vec8*)(((d128::Scalar*)dSt_shared) + ((((i_26 * 2048) + (((((int)threadIdx.x) & 127) >> 3) * 128))\
 + (((((((int)threadIdx.x) >> 7) * 4) + ((((int)threadIdx.x) & 7) >> 1)) ^ ((((int)threadIdx.x) & 63) >> 3)) * 16))\
 + ((((int)threadIdx.x) & 1) * 8))) = *(d128::Vec8*)(accs_cast + (i_26 * 8));
          }
          d128::lma_wait();
          __musa_async_arrive((stage_consumer0 + 26));
          d128::warpgroup_wait<0>();
          __musa_async_arrive((stage_consumer0 + 12));
          __musa_async_arrive((stage_consumer0 + 18));
          #pragma unroll
          for (int i_27 = 0; i_27 < 4; ++i_27) {
            *(d128::Vec8*)(((d128::Scalar*)dS_shared) + (((i_27 * 2048) + (((((int)threadIdx.x) & 127) >> 3) * 128))\
 + (((((((int)threadIdx.x) >> 7) * 8) + (((int)threadIdx.x) & 7)) ^ ((((int)threadIdx.x) & 127) >> 3)) * 8)))\
 = *(d128::Vec8*)(accs_cast + (i_27 * 8));
          }
          d128::lma_wait();
          __musa_async_arrive((stage_consumer0 + 24));
          prev_stage_consumer0 = stage_consumer0;
          prev_phase_consumer0 = phase_consumer0;
          phase_consumer0 = (phase_consumer0 ^ stage_consumer0);
          stage_consumer0 = (stage_consumer0 ^ 1);
        }
        #pragma unroll
        for (int i_28 = 0; i_28 < 16; ++i_28) {
          d128::Vec4 __4;
          tl_f4 v__4 = *(tl_f4*)(dv_accum + (i_28 * 4));
          ((d128::Packed4*)(&__4))[0] = d128::round_to_scalar4(((float4*)(&v__4))[0]);
          *(d128::Vec4*)(dv_cast + (i_28 * 4)) = __4;
        }
        if (end_seq_kv < (((((int)blockIdx.x) * 128) + begin_seq_kv) + 128)) {
          #pragma unroll
          for (int i_29 = 0; i_29 < 64; ++i_29) {
            if ((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 7) * 64)) + ((i_29 >> 4) * 16)) + ((((int)\
threadIdx.x) & 127) >> 3)) + begin_seq_kv) < end_seq_kv) {
              dV[((((((i_29 & 15) * 8) + (((int)blockIdx.z) * dv_stride_b)) + (((((((int)blockIdx.x) * 128) + ((((int)\
threadIdx.x) >> 7) * 64)) + ((i_29 >> 4) * 16)) + ((((int)threadIdx.x) & 127) >> 3)) * dv_stride_s)) + (((int)\
blockIdx.y) * dv_stride_h)) + (((int)threadIdx.x) & 7))] = dv_cast[i_29];
            }
          }
        } else {
          #pragma unroll
          for (int i_30 = 0; i_30 < 64; ++i_30) {
            dV[((((((i_30 & 15) * 8) + (((int)blockIdx.z) * dv_stride_b)) + (((((((int)blockIdx.x) * 128) + ((((int)\
threadIdx.x) >> 7) * 64)) + ((i_30 >> 4) * 16)) + ((((int)threadIdx.x) & 127) >> 3)) * dv_stride_s)) + (((int)\
blockIdx.y) * dv_stride_h)) + (((int)threadIdx.x) & 7))] = dv_cast[i_30];
          }
        }
        __musa_async_arrive(28);
      }
    }
  }
}

#else
extern "C" __global__ void flashattn_bwd_ws_unsplit_kernel_kernel(const float2* __restrict__ RowStats,\
 const d128::Scalar* __restrict__ K, __attribute__((grid_constant)) const MUtensorDescriptor K_desc_0,\
 __attribute__((grid_constant)) const d128::robust_param_desc_t K_robust_desc_0, __attribute__((grid_constant))\
 const MUtensorDescriptor Q_desc_1, __attribute__((grid_constant)) const MUtensorDescriptor Q_desc_4,\
 const d128::Scalar* __restrict__ V, __attribute__((grid_constant)) const d128::robust_param_desc_t V_robust_desc_1,\
 d128::Scalar* __restrict__ dK, __attribute__((grid_constant)) const MUtensorDescriptor dO_desc_2,\
 __attribute__((grid_constant)) const MUtensorDescriptor dO_desc_3, float* __restrict__ dQ_accum,\
 d128::Scalar* __restrict__ dV, int batch, int dk_stride_b, int dk_stride_h, int dk_stride_s, int dv_stride_b,\
 int dv_stride_h, int dv_stride_s, int heads_q, int k_stride_b, int k_stride_h, int k_stride_s, int max_seq_kv,\
 int max_seq_q, float rln2_scale, int seqlen_k, int seqlen_q, float smscale, int v_stride_b, int v_stride_h,\
 int v_stride_s);
extern "C" __global__ void __launch_bounds__(640, 1) flashattn_bwd_ws_unsplit_kernel_kernel(const float2* __restrict__\
 RowStats, const d128::Scalar* __restrict__ K, __attribute__((grid_constant)) const MUtensorDescriptor K_desc_0,\
 __attribute__((grid_constant)) const d128::robust_param_desc_t K_robust_desc_0, __attribute__((grid_constant))\
 const MUtensorDescriptor Q_desc_1, __attribute__((grid_constant)) const MUtensorDescriptor Q_desc_4,\
 const d128::Scalar* __restrict__ V, __attribute__((grid_constant)) const d128::robust_param_desc_t V_robust_desc_1,\
 d128::Scalar* __restrict__ dK, __attribute__((grid_constant)) const MUtensorDescriptor dO_desc_2,\
 __attribute__((grid_constant)) const MUtensorDescriptor dO_desc_3, float* __restrict__ dQ_accum,\
 d128::Scalar* __restrict__ dV, int batch, int dk_stride_b, int dk_stride_h, int dk_stride_s, int dv_stride_b,\
 int dv_stride_h, int dv_stride_s, int heads_q, int k_stride_b, int k_stride_h, int k_stride_s, int max_seq_kv,\
 int max_seq_q, float rln2_scale, int seqlen_k, int seqlen_q, float smscale, int v_stride_b, int v_stride_h,\
 int v_stride_s) {
  __musa_async_bar_record(28);
  extern __shared__ __align__(4096) uchar buf_dyn_shmem[];
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Kt_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* Vt_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  void* Qb_shared = ((void*)((char*)buf_dyn_shmem + 98304));
  void* dOb_shared = ((void*)((char*)buf_dyn_shmem + 114688));
  void* Pt_shared = ((void*)((char*)buf_dyn_shmem + 131072));
  void* dOa_shared = ((void*)((char*)buf_dyn_shmem + 147456));
  void* dS_shared = ((void*)((char*)buf_dyn_shmem + 163840));
  void* dSt_shared = ((void*)((char*)buf_dyn_shmem + 180224));
  int kv_group = 0;
  int begin_seq_q = 0;
  int end_seq_q = 0;
  int begin_seq_kv = 0;
  int end_seq_kv = 0;
  int stage_producer = 0;
  int phase_producer = 0;
  int prev_stage_producer = 0;
  int prev_phase_producer = 0;
  int warprole = 0;
  int stage_consumer1 = 0;
  int phase_consumer1 = 0;
  float dk_accum[64];
  float dq_accum[32];
  float dq_store_buffer[32];
  int local_q_start = 0;
  d128::Scalar dk_cast[64];
  int stage_consumer0 = 0;
  int phase_consumer0 = 0;
  int prev_stage_consumer0 = 0;
  int prev_phase_consumer0 = 0;
  float dv_accum[64];
  d128::Scalar dv_cast[64];
  float lse_buffer[4];
  float delta_buffer[4];
  float accs_accum[32];
  float dp_accum[32];
  d128::Scalar accs_cast[32];
  if (d128::block_leader()) {
    __musa_async_init_arrival(1, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(2, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(3, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(4, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(5, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(6, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(7, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(8, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(9, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(10, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(11, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(12, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(13, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(14, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(15, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(16, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(17, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(18, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(19, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(20, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(21, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(22, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(23, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(24, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(25, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(26, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(27, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(28, ((512 + 31) / 32), 0);
  }
  d128::fence_barrier_init();
  __syncthreads_lm();
  kv_group = ((int)blockIdx.y);
  begin_seq_q = (((int)blockIdx.z) * seqlen_q);
  end_seq_q = (begin_seq_q + seqlen_q);
  begin_seq_kv = (((int)blockIdx.z) * seqlen_k);
  end_seq_kv = (begin_seq_kv + seqlen_k);
  __syncthreads_lm();
  if (((((int)blockIdx.x) * 128) + begin_seq_kv) < end_seq_kv) {
    if (512 <= ((int)threadIdx.x)) {
      stage_producer = 0;
      phase_producer = 0;
      prev_stage_producer = 1;
      prev_phase_producer = 1;
      warprole = ((((int)threadIdx.x) >> 5) - 16);
      if (((int)threadIdx.x) == 512) {
        __musa_async_add_trans(1, 32768);
        d128::tma_load<d128::TmaGranularity::B32>(K_desc_0, 1, (&(((d128::Scalar*)K_shared)[0])), 0, kv_group, (((int)\
blockIdx.x) * 128), ((int)blockIdx.z), 64, 1, 128, 1);
        d128::tma_load<d128::TmaGranularity::B32>(K_desc_0, 1, (&(((d128::Scalar*)K_shared)[8192])), 64, kv_group,\
 (((int)blockIdx.x) * 128), ((int)blockIdx.z), 64, 1, 128, 1);
      }
      __musa_async_arrive(1);
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        d128::cp_async_gs_robust<16>((&(((d128::Scalar*)Kt_shared)[((((i * 1024) + ((((int)threadIdx.x) >> 4) * 128))\
 + (((((int)threadIdx.x) & 15) ^ (((i * 8) + (((int)threadIdx.x) >> 4)) & 15)) * 8)) - 4096)])), (&(K[(((((((int)\
threadIdx.x) & 15) * 8) + (((int)blockIdx.z) * k_stride_b)) + (((((((int)blockIdx.x) * 128) + ((i >> 3) * 64))\
 + (((((int)threadIdx.x) & 127) >> 4) * 8)) + (i & 7)) * k_stride_s)) + (kv_group * k_stride_h))])), K_robust_desc_0);
      }
      #pragma unroll
      for (int i_1 = 0; i_1 < 16; ++i_1) {
        d128::cp_async_gs_robust<16>((&(((d128::Scalar*)Vt_shared)[((((i_1 * 1024) + ((((int)threadIdx.x) >> 4)\
 * 128)) + (((((int)threadIdx.x) & 15) ^ (((i_1 * 8) + (((int)threadIdx.x) >> 4)) & 15)) * 8)) - 4096)])),\
 (&(V[(((((((int)threadIdx.x) & 15) * 8) + (((int)blockIdx.z) * v_stride_b)) + (((((((int)blockIdx.x) * 128)\
 + ((i_1 >> 3) * 64)) + (((((int)threadIdx.x) & 127) >> 4) * 8)) + (i_1 & 7)) * v_stride_s))\
 + (kv_group * v_stride_h))])), V_robust_desc_1);
      }
      d128::finish_robust_copies();
      __musa_async_arrive(2);
      __musa_async_arrive(3);
      for (int _tmp = 0; _tmp < (((end_seq_q + 63) - begin_seq_q) >> 6); ++_tmp) {
        int q_start = ((_tmp * 64) + begin_seq_q);
        if (warprole == 0) {
          __musa_async_wait((prev_stage_producer + 12), prev_phase_producer);
          if (((int)threadIdx.x) == 512) {
            __musa_async_add_trans((stage_producer + 4), 16384);
            d128::tma_load<d128::TmaGranularity::B16>(Q_desc_1, (stage_producer + 4), (&(((d128::Scalar*)\
Pt_shared)[0])), 0, ((int)blockIdx.y), (q_start - begin_seq_q), ((int)blockIdx.z), 128, 1, 64, 1);
          }
          __musa_async_arrive((stage_producer + 4));
        }
        if (warprole == 1) {
          __musa_async_wait((prev_stage_producer + 16), prev_phase_producer);
          if (((int)threadIdx.x) == 544) {
            __musa_async_add_trans((stage_producer + 8), 16384);
            d128::tma_load<d128::TmaGranularity::B16>(dO_desc_2, (stage_producer + 8), (&(((d128::Scalar*)\
dOa_shared)[0])), 0, ((int)blockIdx.y), (q_start - begin_seq_q), ((int)blockIdx.z), 128, 1, 64, 1);
          }
          __musa_async_arrive((stage_producer + 8));
        }
        if (warprole == 2) {
          __musa_async_wait((prev_stage_producer + 18), prev_phase_producer);
          if (((int)threadIdx.x) == 576) {
            __musa_async_add_trans((stage_producer + 10), 16384);
            d128::tma_load<d128::TmaGranularity::B32>(dO_desc_3, (stage_producer + 10), (&(((d128::Scalar*)\
dOb_shared)[0])), 0, ((int)blockIdx.y), (q_start - begin_seq_q), ((int)blockIdx.z), 128, 1, 64, 1);
          }
          __musa_async_arrive((stage_producer + 10));
        }
        if (warprole == 3) {
          __musa_async_wait((prev_stage_producer + 14), prev_phase_producer);
          if (((int)threadIdx.x) == 608) {
            __musa_async_add_trans((stage_producer + 6), 16384);
            d128::tma_load<d128::TmaGranularity::B32>(Q_desc_4, (stage_producer + 6), (&(((d128::Scalar*)\
Qb_shared)[0])), 0, ((int)blockIdx.y), (q_start - begin_seq_q), ((int)blockIdx.z), 64, 1, 64, 1);
            d128::tma_load<d128::TmaGranularity::B32>(Q_desc_4, (stage_producer + 6), (&(((d128::Scalar*)\
Qb_shared)[4096])), 64, ((int)blockIdx.y), (q_start - begin_seq_q), ((int)blockIdx.z), 64, 1, 64, 1);
          }
          __musa_async_arrive((stage_producer + 6));
        }
        prev_stage_producer = stage_producer;
        prev_phase_producer = phase_producer;
        phase_producer = (phase_producer ^ stage_producer);
        stage_producer = (stage_producer ^ 1);
      }
      __musa_async_wait(28, 0);
    } else {
      if (256 <= ((int)threadIdx.x)) {
        stage_consumer1 = 0;
        phase_consumer1 = 0;
        __musa_async_wait(1, 0);
        __musa_async_wait(2, 0);
        __musa_async_wait(3, 0);
        #pragma unroll
        for (int i_2 = 0; i_2 < 16; ++i_2) {
          float broadcast_var = 0x0p+0f/*0.000000e+00*/;
          *(tl_f4*)(dk_accum + (i_2 * 4)) = (tl_f4{static_cast<float>(broadcast_var),\
 static_cast<float>(broadcast_var), static_cast<float>(broadcast_var), static_cast<float>(broadcast_var)});
        }
        __musa_async_wait((stage_consumer1 + 6), phase_consumer1);
        __musa_async_wait((stage_consumer1 + 26), phase_consumer1);
        d128::gemm<d128::Product::DK>((&(((d128::Scalar*)dSt_shared)[0])), (&(((d128::Scalar*)Qb_shared)[0])),\
 (&(dk_accum[0])), 256);
        d128::wait_sqmma();
        __musa_async_arrive((stage_consumer1 + 14));
        __musa_async_wait((stage_consumer1 + 24), phase_consumer1);
        d128::gemm<d128::Product::DQ>((&(((d128::Scalar*)dS_shared)[0])), (&(((d128::Scalar*)K_shared)[0])),\
 (&(dq_accum[0])), 256);
        d128::wait_sqmma();
        __musa_async_arrive((stage_consumer1 + 22));
        #pragma unroll
        for (int i_3 = 0; i_3 < 8; ++i_3) {
          *(tl_f4*)(dq_store_buffer + (i_3 * 4)) = *(tl_f4*)(dq_accum + (i_3 * 4));
        }
        phase_consumer1 = (phase_consumer1 ^ stage_consumer1);
        stage_consumer1 = (stage_consumer1 ^ 1);
        int dq_loop_start = begin_seq_q;
        local_q_start = (dq_loop_start - begin_seq_q);
        for (int _tmp_1 = 0; _tmp_1 < (((end_seq_q - dq_loop_start) - 1) >> 6); ++_tmp_1) {
          #pragma unroll
          for (int i_4 = 0; i_4 < 32; ++i_4) {
            AtomicAdd(*((&(dQ_accum[((((((((((i_4 >> 3) * 2048) + (((((int)threadIdx.x) & 127) >> 3) * 128))\
 + (local_q_start * 128)) + (((((int)blockIdx.z) * max_seq_q) * heads_q) * 128)) + ((((int)blockIdx.y) * max_seq_q)\
 * 128)) + ((((int)threadIdx.x) >> 7) * 64)) + ((i_4 & 7) * 8)) + (((int)threadIdx.x) & 7)) - 128)]))),\
 dq_store_buffer[i_4]);
          }
          __musa_async_wait((stage_consumer1 + 6), phase_consumer1);
          __musa_async_wait((stage_consumer1 + 26), phase_consumer1);
          d128::gemm<d128::Product::DK>((&(((d128::Scalar*)dSt_shared)[0])), (&(((d128::Scalar*)Qb_shared)[0])),\
 (&(dk_accum[0])), 256);
          d128::wait_sqmma();
          __musa_async_wait((stage_consumer1 + 24), phase_consumer1);
          d128::gemm<d128::Product::DQ>((&(((d128::Scalar*)dS_shared)[0])), (&(((d128::Scalar*)K_shared)[0])),\
 (&(dq_accum[0])), 256);
          d128::wait_sqmma();
          __musa_async_arrive((stage_consumer1 + 14));
          __musa_async_arrive((stage_consumer1 + 22));
          #pragma unroll
          for (int i_5 = 0; i_5 < 8; ++i_5) {
            *(tl_f4*)(dq_store_buffer + (i_5 * 4)) = *(tl_f4*)(dq_accum + (i_5 * 4));
          }
          phase_consumer1 = (phase_consumer1 ^ stage_consumer1);
          stage_consumer1 = (stage_consumer1 ^ 1);
          local_q_start = (local_q_start + 64);
        }
        #pragma unroll
        for (int i_6 = 0; i_6 < 32; ++i_6) {
          AtomicAdd(*((&(dQ_accum[((((((((((i_6 >> 3) * 2048) + (((((int)threadIdx.x) & 127) >> 3) * 128))\
 + (local_q_start * 128)) + (((((int)blockIdx.z) * max_seq_q) * heads_q) * 128)) + ((((int)blockIdx.y) * max_seq_q)\
 * 128)) + ((((int)threadIdx.x) >> 7) * 64)) + ((i_6 & 7) * 8)) + (((int)threadIdx.x) & 7)) - 128)]))),\
 dq_store_buffer[i_6]);
        }
        #pragma unroll
        for (int i_7 = 0; i_7 < 16; ++i_7) {
          d128::Vec4 __1;
          tl_f4 v_ = *(tl_f4*)(dk_accum + (i_7 * 4));
          ((d128::Packed4*)(&__1))[0] = d128::round_to_scalar4(((float4*)(&v_))[0]);
          *(d128::Vec4*)(dk_cast + (i_7 * 4)) = __1;
        }
        if (end_seq_kv < (((((int)blockIdx.x) * 128) + begin_seq_kv) + 128)) {
          #pragma unroll
          for (int i_8 = 0; i_8 < 64; ++i_8) {
            if (((((((int)blockIdx.x) * 128) + ((i_8 >> 3) * 16)) + ((((int)threadIdx.x) & 127) >> 3)) + begin_seq_kv)\
 < end_seq_kv) {
              dK[((((((((((int)threadIdx.x) >> 7) * 64) + ((i_8 & 7) * 8)) + (((int)blockIdx.z) * dk_stride_b))\
 + ((((((int)blockIdx.x) * 128) + ((i_8 >> 3) * 16)) + ((((int)threadIdx.x) & 127) >> 3)) * dk_stride_s)) + (((int)\
blockIdx.y) * dk_stride_h)) + (((int)threadIdx.x) & 7)) - 128)] = dk_cast[i_8];
            }
          }
        } else {
          #pragma unroll
          for (int i_9 = 0; i_9 < 64; ++i_9) {
            dK[((((((((((int)threadIdx.x) >> 7) * 64) + ((i_9 & 7) * 8)) + (((int)blockIdx.z) * dk_stride_b))\
 + ((((((int)blockIdx.x) * 128) + ((i_9 >> 3) * 16)) + ((((int)threadIdx.x) & 127) >> 3)) * dk_stride_s)) + (((int)\
blockIdx.y) * dk_stride_h)) + (((int)threadIdx.x) & 7)) - 128)] = dk_cast[i_9];
          }
        }
        __musa_async_arrive(28);
      } else {
        stage_consumer0 = 0;
        phase_consumer0 = 0;
        prev_stage_consumer0 = 1;
        prev_phase_consumer0 = 1;
        __musa_async_wait(1, 0);
        __musa_async_wait(2, 0);
        __musa_async_wait(3, 0);
        #pragma unroll
        for (int i_10 = 0; i_10 < 16; ++i_10) {
          float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
          *(tl_f4*)(dv_accum + (i_10 * 4)) = (tl_f4{static_cast<float>(broadcast_var_1),\
 static_cast<float>(broadcast_var_1), static_cast<float>(broadcast_var_1), static_cast<float>(broadcast_var_1)});
        }
        for (int _tmp_2 = 0; _tmp_2 < (((end_seq_q + 63) - begin_seq_q) >> 6); ++_tmp_2) {
          int q_start_1 = ((_tmp_2 * 64) + begin_seq_q);
          #pragma unroll
          for (int i_11 = 0; i_11 < 4; ++i_11) {
            const int row = i_11 * 16 + ((int(threadIdx.x) & 127) >> 3)
                + (int(blockIdx.z) * heads_q + kv_group) * seqlen_q
                + q_start_1 - begin_seq_q;
            const float2 stats = RowStats[row];
            lse_buffer[i_11] = stats.x;
            delta_buffer[i_11] = stats.y;
          }
          if (end_seq_kv < (((((int)blockIdx.x) * 128) + begin_seq_kv) + 128)) {
            #pragma unroll
            for (int i_14 = 0; i_14 < 32; ++i_14) {
              float condval_1;
              if (((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 7) * 64)) + ((((int)threadIdx.x) & 7) * 8))\
 + begin_seq_kv) + (i_14 & 7)) < end_seq_kv)) {
                condval_1 = 0x0p+0f/*0.000000e+00*/;
              } else {
                condval_1 = -MUSART_INF_F;
              }
              accs_accum[i_14] = condval_1;
            }
          } else {
            #pragma unroll
            for (int i_15 = 0; i_15 < 8; ++i_15) {
              float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
              *(tl_f4*)(accs_accum + (i_15 * 4)) = (tl_f4{static_cast<float>(broadcast_var_2),\
 static_cast<float>(broadcast_var_2), static_cast<float>(broadcast_var_2), static_cast<float>(broadcast_var_2)});
            }
          }
          __musa_async_wait((stage_consumer0 + 4), phase_consumer0);
          d128::gemm<d128::Product::QK>((&(((d128::Scalar*)Pt_shared)[0])), (&(((d128::Scalar*)Kt_shared)[0])),\
 (&(accs_accum[0])), 0);
          d128::warpgroup_commit_batch();
          __musa_async_wait((stage_consumer0 + 8), phase_consumer0);
          d128::gemm<d128::Product::DOV>((&(((d128::Scalar*)dOa_shared)[0])), (&(((d128::Scalar*)Vt_shared)[0])),\
 (&(dp_accum[0])), 0);
          d128::warpgroup_commit_batch();
          d128::warpgroup_wait<1>();
          int stage = stage_consumer0;
          __musa_async_arrive((stage + 20));
          #pragma unroll
          for (int i_20 = 0; i_20 < 8; ++i_20) {
            tl_f4 v__1 = d128::fma4(*(tl_f4*)(accs_accum + (i_20 * 4)), rln2_scale, -lse_buffer[(i_20 >> 1)]);
            *(tl_f4*)(accs_accum + (i_20 * 4)) = d128::vec_exp2_f4(v__1);
          }
          #pragma unroll
          for (int i_21 = 0; i_21 < 8; ++i_21) {
            d128::Vec4 __2;
            tl_f4 v__2 = *(tl_f4*)(accs_accum + (i_21 * 4));
            ((d128::Packed4*)(&__2))[0] = d128::round_to_scalar4(((float4*)(&v__2))[0]);
            *(d128::Vec4*)(accs_cast + (i_21 * 4)) = __2;
          }
          __musa_async_wait((stage_consumer0 + 20), phase_consumer0);
          #pragma unroll
          for (int i_22 = 0; i_22 < 4; ++i_22) {
            *(d128::Vec8*)(((d128::Scalar*)Pt_shared) + ((((((((int)threadIdx.x) >> 7) * 4096) + (i_22 * 1024))\
 + (((((int)threadIdx.x) & 127) >> 4) * 128)) + ((((((int)threadIdx.x) & 15) >> 1) ^ ((((int)threadIdx.x) & 127)\
 >> 4)) * 16)) + ((((int)threadIdx.x) & 1) * 8))) = *(d128::Vec8*)(accs_cast + (i_22 * 8));
          }
          d128::lma_wait();
          __musa_async_wait((stage_consumer0 + 10), phase_consumer0);
          d128::gemm<d128::Product::DV>((&(((d128::Scalar*)Pt_shared)[0])), (&(((d128::Scalar*)dOb_shared)[0])),\
 (&(dv_accum[0])), 0);
          d128::warpgroup_commit_batch();
          d128::warpgroup_wait<1>();
          int stage_1 = stage_consumer0;
          __musa_async_arrive((stage_1 + 16));
          #pragma unroll
          for (int i_24 = 0; i_24 < 8; ++i_24) {
            d128::Vec4 __3;
            const tl_f4 p = *(tl_f4*)(accs_accum + (i_24 * 4));
            const tl_f4 dp = *(tl_f4*)(dp_accum + (i_24 * 4));
            const tl_f4 d = d128::fma4(dp, smscale, -delta_buffer[i_24 >> 1]);
            const tl_f4 v__3 = {
                d128::scale_ds(p[0], d[0]), d128::scale_ds(p[1], d[1]),
                d128::scale_ds(p[2], d[2]), d128::scale_ds(p[3], d[3])};
            ((d128::Packed4*)(&__3))[0] = d128::round_to_scalar4(((float4*)(&v__3))[0]);
            *(d128::Vec4*)(accs_cast + (i_24 * 4)) = __3;
          }
          __musa_async_wait((prev_stage_consumer0 + 22), prev_phase_consumer0);
          #pragma unroll
          for (int i_25 = 0; i_25 < 4; ++i_25) {
            *(d128::Vec8*)(((d128::Scalar*)dSt_shared) + ((((i_25 * 2048) + (((((int)threadIdx.x) & 127) >> 3) * 128))\
 + (((((((int)threadIdx.x) >> 7) * 4) + ((((int)threadIdx.x) & 7) >> 1)) ^ ((((int)threadIdx.x) & 63) >> 3)) * 16))\
 + ((((int)threadIdx.x) & 1) * 8))) = *(d128::Vec8*)(accs_cast + (i_25 * 8));
          }
          d128::lma_wait();
          __musa_async_arrive((stage_consumer0 + 26));
          d128::warpgroup_wait<0>();
          __musa_async_arrive((stage_consumer0 + 12));
          __musa_async_arrive((stage_consumer0 + 18));
          #pragma unroll
          for (int i_26 = 0; i_26 < 4; ++i_26) {
            *(d128::Vec8*)(((d128::Scalar*)dS_shared) + (((i_26 * 2048) + (((((int)threadIdx.x) & 127) >> 3) * 128))\
 + (((((((int)threadIdx.x) >> 7) * 8) + (((int)threadIdx.x) & 7)) ^ ((((int)threadIdx.x) & 127) >> 3)) * 8)))\
 = *(d128::Vec8*)(accs_cast + (i_26 * 8));
          }
          d128::lma_wait();
          __musa_async_arrive((stage_consumer0 + 24));
          prev_stage_consumer0 = stage_consumer0;
          prev_phase_consumer0 = phase_consumer0;
          phase_consumer0 = (phase_consumer0 ^ stage_consumer0);
          stage_consumer0 = (stage_consumer0 ^ 1);
        }
        #pragma unroll
        for (int i_27 = 0; i_27 < 16; ++i_27) {
          d128::Vec4 __4;
          tl_f4 v__4 = *(tl_f4*)(dv_accum + (i_27 * 4));
          ((d128::Packed4*)(&__4))[0] = d128::round_to_scalar4(((float4*)(&v__4))[0]);
          *(d128::Vec4*)(dv_cast + (i_27 * 4)) = __4;
        }
        if (end_seq_kv < (((((int)blockIdx.x) * 128) + begin_seq_kv) + 128)) {
          #pragma unroll
          for (int i_28 = 0; i_28 < 64; ++i_28) {
            if ((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 7) * 64)) + ((i_28 >> 4) * 16)) + ((((int)\
threadIdx.x) & 127) >> 3)) + begin_seq_kv) < end_seq_kv) {
              dV[((((((i_28 & 15) * 8) + (((int)blockIdx.z) * dv_stride_b)) + (((((((int)blockIdx.x) * 128) + ((((int)\
threadIdx.x) >> 7) * 64)) + ((i_28 >> 4) * 16)) + ((((int)threadIdx.x) & 127) >> 3)) * dv_stride_s)) + (((int)\
blockIdx.y) * dv_stride_h)) + (((int)threadIdx.x) & 7))] = dv_cast[i_28];
            }
          }
        } else {
          #pragma unroll
          for (int i_29 = 0; i_29 < 64; ++i_29) {
            dV[((((((i_29 & 15) * 8) + (((int)blockIdx.z) * dv_stride_b)) + (((((((int)blockIdx.x) * 128) + ((((int)\
threadIdx.x) >> 7) * 64)) + ((i_29 >> 4) * 16)) + ((((int)threadIdx.x) & 127) >> 3)) * dv_stride_s)) + (((int)\
blockIdx.y) * dv_stride_h)) + (((int)threadIdx.x) & 7))] = dv_cast[i_29];
          }
        }
        __musa_async_arrive(28);
      }
    }
  }
}

#endif
"""

_D128_HOST_CPP = r"""
// SPDX-License-Identifier: Apache-2.0
// Driver adapter for the checked, fixed-shape FlagGems D128 codegen ABI.
// Tensor and robust descriptors reconstructed from the archived TVM/MUSA
// host ABI. The generated device code retains MATE/TileLang attribution.
#include <musa.h>

#include <cstdint>

namespace {
constexpr int kBatch = 4;
constexpr int kHeads = 16;
constexpr int kSeq = 1024;
constexpr int kHeadDim = 128;
constexpr int kHeadStride = kSeq * kHeadDim;
constexpr int kBatchStride = kHeads * kHeadStride;
constexpr int kSharedBytes = 192 * 1024;
constexpr int kThreads = 640;
constexpr char kFunction[] = "flashattn_bwd_ws_unsplit_kernel_kernel";

struct alignas(16) RobustDescriptor {
  std::uint64_t base;
  std::uint64_t size;
};
static_assert(sizeof(RobustDescriptor) == 16, "MP31 robust descriptor ABI");

RobustDescriptor make_robust(const void* base, std::uint64_t bytes) {
  return {reinterpret_cast<std::uintptr_t>(base), bytes};
}

MUresult encode(void* address, MUtensorDescriptorDataType type,
                MUtensorDescriptor* result) {
  const muuint64_t shape[] = {kHeadDim, kHeads, kSeq, kBatch};
  const muuint64_t strides[] = {2 * kHeadStride, 2 * kHeadDim, 2 * kBatchStride};
  return muTensorDescriptorEncode(result, type, 4, address, shape, strides,
                                 MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE, 0);
}
}  // namespace

extern "C" int mp31_load(const char* path, MUfunction* function) {
  MUmodule module = nullptr;
  MUresult rc = muModuleLoad(&module, path);
  if (rc != MUSA_SUCCESS) return rc;
  rc = muModuleGetFunction(function, module, kFunction);
  if (rc == MUSA_SUCCESS) {
    rc = muFuncSetAttribute(*function,
                            MU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            kSharedBytes);
  }
  if (rc != MUSA_SUCCESS) muModuleUnload(module);
  // Successfully loaded modules live for the process/context lifetime.
  return rc;
}

extern "C" int mp31_context(MUcontext* result) {
  return muCtxGetCurrent(result);
}

extern "C" int mp31_resources(MUfunction function, int* values) {
  const MUfunction_attribute attributes[] = {
      MU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, MU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
      MU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, MU_FUNC_ATTRIBUTE_NUM_REGS,
      MU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES};
  for (unsigned i = 0; i < sizeof(attributes) / sizeof(attributes[0]); ++i) {
    MUresult rc = muFuncGetAttribute(values + i, attributes[i], function);
    if (rc != MUSA_SUCCESS) return rc;
  }
  return MUSA_SUCCESS;
}

extern "C" int mp31_launch(MUfunction function, MUstream stream, int bf16,
                            void* q, void* k, void* v, void* dout,
                            void* row_stats, void* dqa, void* dk, void* dv,
                            float scale, float scale_log2) {
  const MUtensorDescriptorDataType type =
      bf16 ? MU_TENSOR_DESCRIPTOR_DATA_TYPE_BFLOAT16 : MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT16;
  const auto tensor_bytes = static_cast<std::uint64_t>(kBatch) * kBatchStride * 2;
  auto k_bounds = make_robust(k, tensor_bytes);
  auto v_bounds = make_robust(v, tensor_bytes);
  alignas(128) MUtensorDescriptor k_desc{}, q_desc{}, dout_desc{};
  MUresult rc = encode(k, type, &k_desc);
  if (rc != MUSA_SUCCESS) return rc;
  rc = encode(q, type, &q_desc);
  if (rc != MUSA_SUCCESS) return rc;
  rc = encode(dout, type, &dout_desc);
  if (rc != MUSA_SUCCESS) return rc;
  // Tensor map tile box/swizzle are instruction operands on this SDK. The
  // two Q uses share equal encoded global descriptors, as do the two dO uses.
  int batch = kBatch, heads = kHeads, seq = kSeq;
  int bs = kBatchStride, hs = kHeadStride, ss = kHeadDim;
  void* parameters[] = {
      &row_stats, &k, &k_desc, &k_bounds,
      &q_desc, &q_desc, &v, &v_bounds, &dk, &dout_desc, &dout_desc, &dqa, &dv,
      &batch, &bs, &hs, &ss, &bs, &hs, &ss, &heads, &bs, &hs, &ss, &seq,
      &seq, &scale_log2, &seq, &seq, &scale, &bs, &hs, &ss};
  static_assert(sizeof(parameters) / sizeof(parameters[0]) == 33, "kernel argument count");
  return muLaunchKernel(function, kSeq / 128, kHeads, kBatch, kThreads, 1, 1,
                        kSharedBytes, stream, parameters, nullptr);
}
"""


def _d64_launch(handle, q, k, v, dout, row_stats, dq_accum, dk, dv, scale, scale_log2):
    _d128_launch(handle, q, k, v, dout, row_stats, dq_accum, dk, dv, scale, scale_log2)


def _musa_flash_bwd_d64_sdk(q, k, v, o, dout, lse, scale, scale_log2, causal):
    pq, pk, pv, pdo, po = (t.transpose(1, 2) for t in (q, k, v, dout, o))
    handle = _d128_prepare(pq, pk, pv, pdo, lse, causal, head_dim=64)
    if handle is None:
        return None
    batch, heads, seq, dim = pq.shape
    dq = torch.empty_like(pq)
    dk = torch.empty_like(pk)
    dv = torch.empty_like(pv)
    dq_accum = torch.empty_like(pq, dtype=torch.float32)
    row_stats = torch.empty(
        (batch, heads, seq, 2), device=q.device, dtype=torch.float32
    )
    _musa_flash_bwd_persistent_delta[(triton.cdiv(seq, 32), batch * heads)](
        po,
        pdo,
        row_stats,
        dq_accum,
        seq,
        heads,
        dim,
        *po.stride()[:3],
        32,
        num_warps=4,
        num_stages=1,
        LSE=lse,
        SCALE=scale,
    )
    _d64_launch(handle, pq, pk, pv, pdo, row_stats, dq_accum, dk, dv, scale, scale_log2)
    _musa_flash_bwd_persistent_cast[(triton.cdiv(dq.numel(), 256),)](
        dq_accum, dq, dq.numel(), 256, num_warps=4
    )
    return dq, dk, dv


# D64 schedule from the same licensed FlagGems/MATE/TileLang adaptation above,
# using the verified 128 KiB layout and SDK-only per-product descriptors.
# Share the SDK preamble and host ABI without changing D128's source bytes.
_D64_DEVICE_CPP = _D128_DEVICE_CPP.split("\n#ifdef FG_D128_CAUSAL\n", 1)[0] + """
// Fixed BM64/BN128 D64 products from the validated generated MP31 schedule.
// Reuse SDK representations above. SDK instruction encodes the BF16/FP16
// operand type and the per-product transpose modes as immediate arguments.
namespace d64 {
template <d128::Product P, typename T>
FG_DEVICE void gemm(T* a, T* b, float* accum, int first_thread) {
  constexpr int dtype = __is_same(T, __mt_bfloat16) ? 3 : 2;
  const int squad = ((static_cast<int>(threadIdx.x) - first_thread) / 128) % 2;
  auto* d = reinterpret_cast<int32_t*>(accum);
  // K-major 128-byte pitch, and MN-major 64-byte pitch for the 32-column
  // TMA placements of K/Q. Both use the layout's recorded swizzle.
  constexpr auto kKMajor128 = static_cast<d128::DescriptorField>((5 << 18) | (1 << 21));
  constexpr auto kMNMajor64 = static_cast<d128::DescriptorField>((4 << 18) | (2 << 21));
  if constexpr (P == d128::Product::DK) {
    const int32_t da = d128::sqmma_desc(a, d128::kMNMajorWide);
    const int32_t db = d128::sqmma_desc(b + squad * 2048, kMNMajor64);
    __musa_sqmma_m128n32_mma(d, &da, &db, d, 1, 0, 0, 0, 1, 0, 64, dtype);
  } else if constexpr (P == d128::Product::DV) {
    const int32_t da = d128::sqmma_desc(a + squad * 4096, d128::kMNMajorSwizzled);
    const int32_t db = d128::sqmma_desc(b, d128::kMNMajorSwizzled);
    __musa_sqmma_m64n64_mma(d, &da, &db, d, 1, 0, 0, 0, 1, 0, 64, dtype);
  } else if constexpr (P == d128::Product::DQ) {
    const int32_t da = d128::sqmma_desc(a, d128::kKMajorSwizzled);
    const int32_t db = d128::sqmma_desc(b + squad * 4096, kMNMajor64);
    __musa_sqmma_m64n32_mma(d, &da, &db, d, 0, 0, 0, 0, 0, 0, 64, dtype);
    const int32_t da2 = da + 128;
    const int32_t db2 = db + 4096;
    __musa_sqmma_m64n32_mma(d, &da2, &db2, d, 0, 0, 0, 0, 1, 0, 64, dtype);
  } else {
    const int32_t da = d128::sqmma_desc(a, kKMajor128);
    const int32_t db = d128::sqmma_desc(b + squad * 4096, kKMajor128);
    __musa_sqmma_m64n64_mma(d, &da, &db, d, 0, 1, 0, 0,
                            P == d128::Product::QK, 0, 64, dtype);
  }
}
}  // namespace d64

#ifdef FG_D128_CAUSAL
extern "C" __global__ void flashattn_bwd_ws_unsplit_kernel_kernel(const float2* __restrict__ RowStats,\
 const d128::Scalar* __restrict__ K, __attribute__((grid_constant)) const MUtensorDescriptor K_desc_0,\
 __attribute__((grid_constant)) const d128::robust_param_desc_t K_robust_desc_0, __attribute__((grid_constant))\
 const MUtensorDescriptor Q_desc_1, __attribute__((grid_constant)) const MUtensorDescriptor Q_desc_4,\
 const d128::Scalar* __restrict__ V, __attribute__((grid_constant)) const d128::robust_param_desc_t V_robust_desc_1,\
 d128::Scalar* __restrict__ dK, __attribute__((grid_constant)) const MUtensorDescriptor dO_desc_2,\
 __attribute__((grid_constant)) const MUtensorDescriptor dO_desc_3, float* __restrict__ dQ_accum,\
 d128::Scalar* __restrict__ dV, int batch, int dk_stride_b, int dk_stride_h, int dk_stride_s, int dv_stride_b,\
 int dv_stride_h, int dv_stride_s, int heads_q, int k_stride_b, int k_stride_h, int k_stride_s, int max_seq_kv,\
 int max_seq_q, float rln2_scale, int seqlen_k, int seqlen_q, float smscale, int v_stride_b, int v_stride_h,\
 int v_stride_s);
extern "C" __global__ void __launch_bounds__(640, 1) flashattn_bwd_ws_unsplit_kernel_kernel(const float2* __restrict__\
 RowStats, const d128::Scalar* __restrict__ K, __attribute__((grid_constant)) const MUtensorDescriptor K_desc_0,\
 __attribute__((grid_constant)) const d128::robust_param_desc_t K_robust_desc_0, __attribute__((grid_constant))\
 const MUtensorDescriptor Q_desc_1, __attribute__((grid_constant)) const MUtensorDescriptor Q_desc_4,\
 const d128::Scalar* __restrict__ V, __attribute__((grid_constant)) const d128::robust_param_desc_t V_robust_desc_1,\
 d128::Scalar* __restrict__ dK, __attribute__((grid_constant)) const MUtensorDescriptor dO_desc_2,\
 __attribute__((grid_constant)) const MUtensorDescriptor dO_desc_3, float* __restrict__ dQ_accum,\
 d128::Scalar* __restrict__ dV, int batch, int dk_stride_b, int dk_stride_h, int dk_stride_s, int dv_stride_b,\
 int dv_stride_h, int dv_stride_s, int heads_q, int k_stride_b, int k_stride_h, int k_stride_s, int max_seq_kv,\
 int max_seq_q, float rln2_scale, int seqlen_k, int seqlen_q, float smscale, int v_stride_b, int v_stride_h,\
 int v_stride_s) {
  __musa_async_bar_record(28);
  extern __shared__ __align__(4096) uchar buf_dyn_shmem[];
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Kt_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* Vt_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* Qb_shared = ((void*)((char*)buf_dyn_shmem + 49152));
  void* dOb_shared = ((void*)((char*)buf_dyn_shmem + 57344));
  void* Pt_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  void* dS_shared = ((void*)((char*)buf_dyn_shmem + 81920));
  void* dSt_shared = ((void*)((char*)buf_dyn_shmem + 98304));
  void* Qa_shared = ((void*)((char*)buf_dyn_shmem + 114688));
  void* dOa_shared = ((void*)((char*)buf_dyn_shmem + 122880));
  int kv_group = 0;
  int begin_seq_q = 0;
  int end_seq_q = 0;
  int begin_seq_kv = 0;
  int end_seq_kv = 0;
  int stage_producer = 0;
  int phase_producer = 0;
  int prev_stage_producer = 0;
  int prev_phase_producer = 0;
  int warprole = 0;
  int producer_causal_offset = 0;
  int producer_causal_q_local_start = 0;
  int producer_causal_q_start = 0;
  int stage_consumer1 = 0;
  int phase_consumer1 = 0;
  float dk_accum[32];
  float dq_accum[16];
  float dq_store_buffer[16];
  int consumer1_causal_offset = 0;
  int consumer1_causal_q_local_start = 0;
  int consumer1_causal_q_start = 0;
  int local_q_start = 0;
  d128::Scalar dk_cast[32];
  int stage_consumer0 = 0;
  int phase_consumer0 = 0;
  int prev_stage_consumer0 = 0;
  int prev_phase_consumer0 = 0;
  float dv_accum[32];
  int consumer0_causal_offset = 0;
  int consumer0_causal_q_local_start = 0;
  int consumer0_causal_q_start = 0;
  int consumer0_causal_q_local_mask_end = 0;
  int consumer0_causal_q_mask_end = 0;
  d128::Scalar dv_cast[32];
  float lse_buffer[4];
  float delta_buffer[4];
  float accs_accum[32];
  float dp_accum[32];
  d128::Scalar accs_cast[32];
  if (d128::block_leader()) {
    __musa_async_init_arrival(1, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(2, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(3, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(4, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(5, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(6, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(7, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(8, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(9, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(10, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(11, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(12, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(13, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(14, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(15, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(16, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(17, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(18, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(19, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(20, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(21, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(22, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(23, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(24, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(25, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(26, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(27, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(28, ((512 + 31) / 32), 0);
  }
  d128::fence_barrier_init();
  __syncthreads_lm();
  kv_group = ((int)blockIdx.y);
  begin_seq_q = (((int)blockIdx.z) * seqlen_q);
  end_seq_q = (begin_seq_q + seqlen_q);
  begin_seq_kv = (((int)blockIdx.z) * seqlen_k);
  end_seq_kv = (begin_seq_kv + seqlen_k);
  __syncthreads_lm();
  if (((((int)blockIdx.x) * 128) + begin_seq_kv) < end_seq_kv) {
    if (512 <= ((int)threadIdx.x)) {
      stage_producer = 0;
      phase_producer = 0;
      prev_stage_producer = 1;
      prev_phase_producer = 1;
      warprole = ((((int)threadIdx.x) >> 5) - 16);
      if (((int)threadIdx.x) == 512) {
        __musa_async_add_trans(1, 16384);
        d128::tma_load<d128::TmaGranularity::B32>(K_desc_0, 1, (&(((d128::Scalar*)K_shared)[0])), 0, kv_group, (((int)\
blockIdx.x) * 128), ((int)blockIdx.z), 32, 1, 128, 1);
        d128::tma_load<d128::TmaGranularity::B32>(K_desc_0, 1, (&(((d128::Scalar*)K_shared)[4096])), 32, kv_group,\
 (((int)blockIdx.x) * 128), ((int)blockIdx.z), 32, 1, 128, 1);
      }
      __musa_async_arrive(1);
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        d128::cp_async_gs_robust<16>((&(((d128::Scalar*)Kt_shared)[((((i * 1024) + ((((int)threadIdx.x) >> 4) * 128))\
 + (((((int)threadIdx.x) & 15) ^ ((((i * 16) + (((int)threadIdx.x) >> 3)) & 31) >> 1)) * 8)) - 4096)])),\
 (&(K[(((((((int)threadIdx.x) & 7) * 8) + (((int)blockIdx.z) * k_stride_b)) + (((((((int)blockIdx.x) * 128)\
 + ((i >> 2) * 64)) + (((((int)threadIdx.x) & 63) >> 3) * 8)) + (((i * 2) + (((int)threadIdx.x) >> 6)) & 7))\
 * k_stride_s)) + (kv_group * k_stride_h))])), K_robust_desc_0);
      }
      #pragma unroll
      for (int i_1 = 0; i_1 < 8; ++i_1) {
        d128::cp_async_gs_robust<16>((&(((d128::Scalar*)Vt_shared)[((((i_1 * 1024) + ((((int)threadIdx.x) >> 4)\
 * 128)) + (((((int)threadIdx.x) & 15) ^ ((((i_1 * 16) + (((int)threadIdx.x) >> 3)) & 31) >> 1)) * 8)) - 4096)])),\
 (&(V[(((((((int)threadIdx.x) & 7) * 8) + (((int)blockIdx.z) * v_stride_b)) + (((((((int)blockIdx.x) * 128)\
 + ((i_1 >> 2) * 64)) + (((((int)threadIdx.x) & 63) >> 3) * 8)) + (((i_1 * 2) + (((int)threadIdx.x) >> 6)) & 7))\
 * v_stride_s)) + (kv_group * v_stride_h))])), V_robust_desc_1);
      }
      d128::finish_robust_copies();
      __musa_async_arrive(2);
      __musa_async_arrive(3);
      producer_causal_offset = (((end_seq_kv + begin_seq_q) - begin_seq_kv) - end_seq_q);
      producer_causal_q_local_start = ((((int)blockIdx.x) * 128) - producer_causal_offset);
      int condval;
      if ((0 < producer_causal_q_local_start)) {
        condval = producer_causal_q_local_start;
      } else {
        condval = 0;
      }
      producer_causal_q_local_start = condval;
      producer_causal_q_start = (((producer_causal_q_local_start >> 6) * 64) + begin_seq_q);
      for (int _tmp = 0; _tmp < (((end_seq_q + 63) - producer_causal_q_start) >> 6); ++_tmp) {
        int q_start = ((_tmp * 64) + producer_causal_q_start);
        if (warprole == 0) {
          __musa_async_wait((prev_stage_producer + 12), prev_phase_producer);
          if (((int)threadIdx.x) == 512) {
            __musa_async_add_trans((stage_producer + 4), 8192);
            d128::tma_load<d128::TmaGranularity::B16>(Q_desc_1, (stage_producer + 4), (&(((d128::Scalar*)\
Qa_shared)[0])), 0, ((int)blockIdx.y), (q_start - begin_seq_q), ((int)blockIdx.z), 64, 1, 64, 1);
          }
          __musa_async_arrive((stage_producer + 4));
        }
        if (warprole == 1) {
          __musa_async_wait((prev_stage_producer + 16), prev_phase_producer);
          if (((int)threadIdx.x) == 544) {
            __musa_async_add_trans((stage_producer + 8), 8192);
            d128::tma_load<d128::TmaGranularity::B16>(dO_desc_2, (stage_producer + 8), (&(((d128::Scalar*)\
dOa_shared)[0])), 0, ((int)blockIdx.y), (q_start - begin_seq_q), ((int)blockIdx.z), 64, 1, 64, 1);
          }
          __musa_async_arrive((stage_producer + 8));
        }
        if (warprole == 2) {
          __musa_async_wait((prev_stage_producer + 18), prev_phase_producer);
          if (((int)threadIdx.x) == 576) {
            __musa_async_add_trans((stage_producer + 10), 8192);
            d128::tma_load<d128::TmaGranularity::B32>(dO_desc_3, (stage_producer + 10), (&(((d128::Scalar*)\
dOb_shared)[0])), 0, ((int)blockIdx.y), (q_start - begin_seq_q), ((int)blockIdx.z), 64, 1, 64, 1);
          }
          __musa_async_arrive((stage_producer + 10));
        }
        if (warprole == 3) {
          __musa_async_wait((prev_stage_producer + 14), prev_phase_producer);
          if (((int)threadIdx.x) == 608) {
            __musa_async_add_trans((stage_producer + 6), 8192);
            d128::tma_load<d128::TmaGranularity::B32>(Q_desc_4, (stage_producer + 6), (&(((d128::Scalar*)\
Qb_shared)[0])), 0, ((int)blockIdx.y), (q_start - begin_seq_q), ((int)blockIdx.z), 32, 1, 64, 1);
            d128::tma_load<d128::TmaGranularity::B32>(Q_desc_4, (stage_producer + 6), (&(((d128::Scalar*)\
Qb_shared)[2048])), 32, ((int)blockIdx.y), (q_start - begin_seq_q), ((int)blockIdx.z), 32, 1, 64, 1);
          }
          __musa_async_arrive((stage_producer + 6));
        }
        prev_stage_producer = stage_producer;
        prev_phase_producer = phase_producer;
        phase_producer = (phase_producer ^ stage_producer);
        stage_producer = (stage_producer ^ 1);
      }
      __musa_async_wait(28, 0);
    } else {
      if (256 <= ((int)threadIdx.x)) {
        stage_consumer1 = 0;
        phase_consumer1 = 0;
        __musa_async_wait(1, 0);
        __musa_async_wait(2, 0);
        __musa_async_wait(3, 0);
        #pragma unroll
        for (int i_2 = 0; i_2 < 8; ++i_2) {
          float broadcast_var = 0x0p+0f/*0.000000e+00*/;
          *(tl_f4*)(dk_accum + (i_2 * 4)) = (tl_f4{static_cast<float>(broadcast_var),\
 static_cast<float>(broadcast_var), static_cast<float>(broadcast_var), static_cast<float>(broadcast_var)});
        }
        __musa_async_wait((stage_consumer1 + 6), phase_consumer1);
        __musa_async_wait((stage_consumer1 + 26), phase_consumer1);
        d64::gemm<d128::Product::DK>((&(((d128::Scalar*)dSt_shared)[0])), (&(((d128::Scalar*)Qb_shared)[0])),\
 (&(dk_accum[0])), 256);
        d128::wait_sqmma();
        __musa_async_arrive((stage_consumer1 + 14));
        __musa_async_wait((stage_consumer1 + 24), phase_consumer1);
        d64::gemm<d128::Product::DQ>((&(((d128::Scalar*)dS_shared)[0])), (&(((d128::Scalar*)K_shared)[0])),\
 (&(dq_accum[0])), 256);
        d128::wait_sqmma();
        __musa_async_arrive((stage_consumer1 + 22));
        #pragma unroll
        for (int i_3 = 0; i_3 < 4; ++i_3) {
          *(tl_f4*)(dq_store_buffer + (i_3 * 4)) = *(tl_f4*)(dq_accum + (i_3 * 4));
        }
        phase_consumer1 = (phase_consumer1 ^ stage_consumer1);
        stage_consumer1 = (stage_consumer1 ^ 1);
        consumer1_causal_offset = (((end_seq_kv + begin_seq_q) - begin_seq_kv) - end_seq_q);
        consumer1_causal_q_local_start = ((((int)blockIdx.x) * 128) - consumer1_causal_offset);
        int condval_1;
        if ((0 < consumer1_causal_q_local_start)) {
          condval_1 = consumer1_causal_q_local_start;
        } else {
          condval_1 = 0;
        }
        consumer1_causal_q_local_start = condval_1;
        consumer1_causal_q_start = (((consumer1_causal_q_local_start >> 6) * 64) + begin_seq_q);
        int dq_loop_start = consumer1_causal_q_start;
        local_q_start = (dq_loop_start - begin_seq_q);
        for (int _tmp_1 = 0; _tmp_1 < (((end_seq_q - dq_loop_start) - 1) >> 6); ++_tmp_1) {
          #pragma unroll
          for (int i_4 = 0; i_4 < 16; ++i_4) {
            AtomicAdd(*((&(dQ_accum[((((((((((i_4 >> 2) * 1024) + (((((int)threadIdx.x) & 127) >> 3) * 64))\
 + (local_q_start * 64)) + (((((int)blockIdx.z) * max_seq_q) * heads_q) * 64)) + ((((int)blockIdx.y) * max_seq_q)\
 * 64)) + ((((int)threadIdx.x) >> 7) * 32)) + ((i_4 & 3) * 8)) + (((int)threadIdx.x) & 7)) - 64)]))),\
 dq_store_buffer[i_4]);
          }
          __musa_async_wait((stage_consumer1 + 6), phase_consumer1);
          __musa_async_wait((stage_consumer1 + 26), phase_consumer1);
          d64::gemm<d128::Product::DK>((&(((d128::Scalar*)dSt_shared)[0])), (&(((d128::Scalar*)Qb_shared)[0])),\
 (&(dk_accum[0])), 256);
          d128::wait_sqmma();
          __musa_async_wait((stage_consumer1 + 24), phase_consumer1);
          d64::gemm<d128::Product::DQ>((&(((d128::Scalar*)dS_shared)[0])), (&(((d128::Scalar*)K_shared)[0])),\
 (&(dq_accum[0])), 256);
          d128::wait_sqmma();
          __musa_async_arrive((stage_consumer1 + 14));
          __musa_async_arrive((stage_consumer1 + 22));
          #pragma unroll
          for (int i_5 = 0; i_5 < 4; ++i_5) {
            *(tl_f4*)(dq_store_buffer + (i_5 * 4)) = *(tl_f4*)(dq_accum + (i_5 * 4));
          }
          phase_consumer1 = (phase_consumer1 ^ stage_consumer1);
          stage_consumer1 = (stage_consumer1 ^ 1);
          local_q_start = (local_q_start + 64);
        }
        #pragma unroll
        for (int i_6 = 0; i_6 < 16; ++i_6) {
          AtomicAdd(*((&(dQ_accum[((((((((((i_6 >> 2) * 1024) + (((((int)threadIdx.x) & 127) >> 3) * 64))\
 + (local_q_start * 64)) + (((((int)blockIdx.z) * max_seq_q) * heads_q) * 64)) + ((((int)blockIdx.y) * max_seq_q)\
 * 64)) + ((((int)threadIdx.x) >> 7) * 32)) + ((i_6 & 3) * 8)) + (((int)threadIdx.x) & 7)) - 64)]))),\
 dq_store_buffer[i_6]);
        }
        #pragma unroll
        for (int i_7 = 0; i_7 < 8; ++i_7) {
          d128::Vec4 __1;
          tl_f4 v_ = *(tl_f4*)(dk_accum + (i_7 * 4));
          ((d128::Packed4*)(&__1))[0] = d128::round_to_scalar4(((float4*)(&v_))[0]);
          *(d128::Vec4*)(dk_cast + (i_7 * 4)) = __1;
        }
        if (end_seq_kv < (((((int)blockIdx.x) * 128) + begin_seq_kv) + 128)) {
          #pragma unroll
          for (int i_8 = 0; i_8 < 32; ++i_8) {
            if (((((((int)blockIdx.x) * 128) + ((i_8 >> 2) * 16)) + ((((int)threadIdx.x) & 127) >> 3)) + begin_seq_kv)\
 < end_seq_kv) {
              dK[((((((((((int)threadIdx.x) >> 7) * 32) + ((i_8 & 3) * 8)) + (((int)blockIdx.z) * dk_stride_b))\
 + ((((((int)blockIdx.x) * 128) + ((i_8 >> 2) * 16)) + ((((int)threadIdx.x) & 127) >> 3)) * dk_stride_s)) + (((int)\
blockIdx.y) * dk_stride_h)) + (((int)threadIdx.x) & 7)) - 64)] = dk_cast[i_8];
            }
          }
        } else {
          #pragma unroll
          for (int i_9 = 0; i_9 < 32; ++i_9) {
            dK[((((((((((int)threadIdx.x) >> 7) * 32) + ((i_9 & 3) * 8)) + (((int)blockIdx.z) * dk_stride_b))\
 + ((((((int)blockIdx.x) * 128) + ((i_9 >> 2) * 16)) + ((((int)threadIdx.x) & 127) >> 3)) * dk_stride_s)) + (((int)\
blockIdx.y) * dk_stride_h)) + (((int)threadIdx.x) & 7)) - 64)] = dk_cast[i_9];
          }
        }
        __musa_async_arrive(28);
      } else {
        stage_consumer0 = 0;
        phase_consumer0 = 0;
        prev_stage_consumer0 = 1;
        prev_phase_consumer0 = 1;
        __musa_async_wait(1, 0);
        __musa_async_wait(2, 0);
        __musa_async_wait(3, 0);
        #pragma unroll
        for (int i_10 = 0; i_10 < 8; ++i_10) {
          float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
          *(tl_f4*)(dv_accum + (i_10 * 4)) = (tl_f4{static_cast<float>(broadcast_var_1),\
 static_cast<float>(broadcast_var_1), static_cast<float>(broadcast_var_1), static_cast<float>(broadcast_var_1)});
        }
        consumer0_causal_offset = (((end_seq_kv + begin_seq_q) - begin_seq_kv) - end_seq_q);
        consumer0_causal_q_local_start = ((((int)blockIdx.x) * 128) - consumer0_causal_offset);
        int condval_2;
        if ((0 < consumer0_causal_q_local_start)) {
          condval_2 = consumer0_causal_q_local_start;
        } else {
          condval_2 = 0;
        }
        consumer0_causal_q_local_start = condval_2;
        consumer0_causal_q_start = (((consumer0_causal_q_local_start >> 6) * 64) + begin_seq_q);
        consumer0_causal_q_local_mask_end = (((((int)blockIdx.x) * 128) + 128) - consumer0_causal_offset);
        int condval_3;
        if ((0 < consumer0_causal_q_local_mask_end)) {
          condval_3 = consumer0_causal_q_local_mask_end;
        } else {
          condval_3 = 0;
        }
        consumer0_causal_q_local_mask_end = condval_3;
        consumer0_causal_q_mask_end = ((((consumer0_causal_q_local_mask_end + 63) >> 6) * 64) + begin_seq_q);
        int condval_4;
        if ((consumer0_causal_q_mask_end < end_seq_q)) {
          condval_4 = consumer0_causal_q_mask_end;
        } else {
          condval_4 = end_seq_q;
        }
        consumer0_causal_q_mask_end = condval_4;
        for (int _tmp_2 = 0; _tmp_2 < (((end_seq_q + 63) - consumer0_causal_q_start) >> 6); ++_tmp_2) {
          int q_start_1 = ((_tmp_2 * 64) + consumer0_causal_q_start);
          #pragma unroll
          for (int i_11 = 0; i_11 < 4; ++i_11) {
            const int row = i_11 * 16 + ((int(threadIdx.x) & 127) >> 3)
                + (int(blockIdx.z) * heads_q + kv_group) * seqlen_q
                + q_start_1 - begin_seq_q;
            const float2 stats = RowStats[row];
            lse_buffer[i_11] = stats.x;
            delta_buffer[i_11] = stats.y;
          }
          if (end_seq_kv < (((((int)blockIdx.x) * 128) + begin_seq_kv) + 128)) {
            #pragma unroll
            for (int i_14 = 0; i_14 < 32; ++i_14) {
              float condval_6;
              if (((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 7) * 64)) + ((((int)threadIdx.x) & 7) * 8))\
 + begin_seq_kv) + (i_14 & 7)) < end_seq_kv)) {
                condval_6 = 0x0p+0f/*0.000000e+00*/;
              } else {
                condval_6 = -MUSART_INF_F;
              }
              accs_accum[i_14] = condval_6;
            }
          } else {
            #pragma unroll
            for (int i_15 = 0; i_15 < 8; ++i_15) {
              float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
              *(tl_f4*)(accs_accum + (i_15 * 4)) = (tl_f4{static_cast<float>(broadcast_var_2),\
 static_cast<float>(broadcast_var_2), static_cast<float>(broadcast_var_2), static_cast<float>(broadcast_var_2)});
            }
          }
          if (q_start_1 < consumer0_causal_q_mask_end) {
            #pragma unroll
            for (int i_16 = 0; i_16 < 32; ++i_16) {
              int q_idx = (((((i_16 >> 3) * 16) + ((((int)threadIdx.x) & 127) >> 3)) + q_start_1) - begin_seq_q);
              bool valid = (((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 7) * 64)) + ((((int)threadIdx.x)\
 & 7) * 8)) + (i_16 & 7)) <= (q_idx + consumer0_causal_offset));
              float condval_7;
              if (valid) {
                condval_7 = accs_accum[i_16];
              } else {
                condval_7 = -MUSART_INF_F;
              }
              accs_accum[i_16] = condval_7;
            }
          }
          __musa_async_wait((stage_consumer0 + 4), phase_consumer0);
          d64::gemm<d128::Product::QK>((&(((d128::Scalar*)Qa_shared)[0])), (&(((d128::Scalar*)Kt_shared)[0])),\
 (&(accs_accum[0])), 0);
          d128::warpgroup_commit_batch();
          __musa_async_wait((stage_consumer0 + 8), phase_consumer0);
          d64::gemm<d128::Product::DOV>((&(((d128::Scalar*)dOa_shared)[0])), (&(((d128::Scalar*)Vt_shared)[0])),\
 (&(dp_accum[0])), 0);
          d128::warpgroup_commit_batch();
          d128::warpgroup_wait<1>();
          int stage = stage_consumer0;
          __musa_async_arrive((stage + 20));
          #pragma unroll
          for (int i_21 = 0; i_21 < 8; ++i_21) {
            tl_f4 v__1 = d128::fma4(*(tl_f4*)(accs_accum + (i_21 * 4)), rln2_scale, -lse_buffer[(i_21 >> 1)]);
            *(tl_f4*)(accs_accum + (i_21 * 4)) = d128::vec_exp2_f4(v__1);
          }
          #pragma unroll
          for (int i_22 = 0; i_22 < 8; ++i_22) {
            d128::Vec4 __2;
            tl_f4 v__2 = *(tl_f4*)(accs_accum + (i_22 * 4));
            ((d128::Packed4*)(&__2))[0] = d128::round_to_scalar4(((float4*)(&v__2))[0]);
            *(d128::Vec4*)(accs_cast + (i_22 * 4)) = __2;
          }
          __musa_async_wait((stage_consumer0 + 20), phase_consumer0);
          #pragma unroll
          for (int i_23 = 0; i_23 < 4; ++i_23) {
            *(d128::Vec8*)(((d128::Scalar*)Pt_shared) + ((((((((int)threadIdx.x) >> 7) * 4096) + (i_23 * 1024))\
 + (((((int)threadIdx.x) & 127) >> 4) * 128)) + ((((((int)threadIdx.x) & 15) >> 1) ^ ((((int)threadIdx.x) & 127)\
 >> 4)) * 16)) + ((((int)threadIdx.x) & 1) * 8))) = *(d128::Vec8*)(accs_cast + (i_23 * 8));
          }
          d128::lma_wait();
          __musa_async_wait((stage_consumer0 + 10), phase_consumer0);
          d64::gemm<d128::Product::DV>((&(((d128::Scalar*)Pt_shared)[0])), (&(((d128::Scalar*)dOb_shared)[0])),\
 (&(dv_accum[0])), 0);
          d128::warpgroup_commit_batch();
          d128::warpgroup_wait<1>();
          int stage_1 = stage_consumer0;
          __musa_async_arrive((stage_1 + 16));
          #pragma unroll
          for (int i_25 = 0; i_25 < 8; ++i_25) {
            d128::Vec4 __3;
            const tl_f4 p = *(tl_f4*)(accs_accum + (i_25 * 4));
            const tl_f4 dp = *(tl_f4*)(dp_accum + (i_25 * 4));
            const tl_f4 d = d128::fma4(dp, smscale, -delta_buffer[i_25 >> 1]);
            const tl_f4 v__3 = {
                d128::scale_ds(p[0], d[0]), d128::scale_ds(p[1], d[1]),
                d128::scale_ds(p[2], d[2]), d128::scale_ds(p[3], d[3])};
            ((d128::Packed4*)(&__3))[0] = d128::round_to_scalar4(((float4*)(&v__3))[0]);
            *(d128::Vec4*)(accs_cast + (i_25 * 4)) = __3;
          }
          __musa_async_wait((prev_stage_consumer0 + 22), prev_phase_consumer0);
          #pragma unroll
          for (int i_26 = 0; i_26 < 4; ++i_26) {
            *(d128::Vec8*)(((d128::Scalar*)dSt_shared) + ((((i_26 * 2048) + (((((int)threadIdx.x) & 127) >> 3) * 128))\
 + (((((((int)threadIdx.x) >> 7) * 4) + ((((int)threadIdx.x) & 7) >> 1)) ^ ((((int)threadIdx.x) & 63) >> 3)) * 16))\
 + ((((int)threadIdx.x) & 1) * 8))) = *(d128::Vec8*)(accs_cast + (i_26 * 8));
          }
          d128::lma_wait();
          __musa_async_arrive((stage_consumer0 + 26));
          d128::warpgroup_wait<0>();
          __musa_async_arrive((stage_consumer0 + 12));
          __musa_async_arrive((stage_consumer0 + 18));
          #pragma unroll
          for (int i_27 = 0; i_27 < 4; ++i_27) {
            *(d128::Vec8*)(((d128::Scalar*)dS_shared) + (((i_27 * 2048) + (((((int)threadIdx.x) & 127) >> 3) * 128))\
 + (((((((int)threadIdx.x) >> 7) * 8) + (((int)threadIdx.x) & 7)) ^ ((((int)threadIdx.x) & 127) >> 3)) * 8)))\
 = *(d128::Vec8*)(accs_cast + (i_27 * 8));
          }
          d128::lma_wait();
          __musa_async_arrive((stage_consumer0 + 24));
          prev_stage_consumer0 = stage_consumer0;
          prev_phase_consumer0 = phase_consumer0;
          phase_consumer0 = (phase_consumer0 ^ stage_consumer0);
          stage_consumer0 = (stage_consumer0 ^ 1);
        }
        #pragma unroll
        for (int i_28 = 0; i_28 < 8; ++i_28) {
          d128::Vec4 __4;
          tl_f4 v__4 = *(tl_f4*)(dv_accum + (i_28 * 4));
          ((d128::Packed4*)(&__4))[0] = d128::round_to_scalar4(((float4*)(&v__4))[0]);
          *(d128::Vec4*)(dv_cast + (i_28 * 4)) = __4;
        }
        if (end_seq_kv < (((((int)blockIdx.x) * 128) + begin_seq_kv) + 128)) {
          #pragma unroll
          for (int i_29 = 0; i_29 < 32; ++i_29) {
            if ((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 7) * 64)) + ((i_29 >> 3) * 16)) + ((((int)\
threadIdx.x) & 127) >> 3)) + begin_seq_kv) < end_seq_kv) {
              dV[((((((i_29 & 7) * 8) + (((int)blockIdx.z) * dv_stride_b)) + (((((((int)blockIdx.x) * 128) + ((((int)\
threadIdx.x) >> 7) * 64)) + ((i_29 >> 3) * 16)) + ((((int)threadIdx.x) & 127) >> 3)) * dv_stride_s)) + (((int)\
blockIdx.y) * dv_stride_h)) + (((int)threadIdx.x) & 7))] = dv_cast[i_29];
            }
          }
        } else {
          #pragma unroll
          for (int i_30 = 0; i_30 < 32; ++i_30) {
            dV[((((((i_30 & 7) * 8) + (((int)blockIdx.z) * dv_stride_b)) + (((((((int)blockIdx.x) * 128) + ((((int)\
threadIdx.x) >> 7) * 64)) + ((i_30 >> 3) * 16)) + ((((int)threadIdx.x) & 127) >> 3)) * dv_stride_s)) + (((int)\
blockIdx.y) * dv_stride_h)) + (((int)threadIdx.x) & 7))] = dv_cast[i_30];
          }
        }
        __musa_async_arrive(28);
      }
    }
  }
}

#else
extern "C" __global__ void flashattn_bwd_ws_unsplit_kernel_kernel(const float2* __restrict__ RowStats,\
 const d128::Scalar* __restrict__ K, __attribute__((grid_constant)) const MUtensorDescriptor K_desc_0,\
 __attribute__((grid_constant)) const d128::robust_param_desc_t K_robust_desc_0, __attribute__((grid_constant))\
 const MUtensorDescriptor Q_desc_1, __attribute__((grid_constant)) const MUtensorDescriptor Q_desc_4,\
 const d128::Scalar* __restrict__ V, __attribute__((grid_constant)) const d128::robust_param_desc_t V_robust_desc_1,\
 d128::Scalar* __restrict__ dK, __attribute__((grid_constant)) const MUtensorDescriptor dO_desc_2,\
 __attribute__((grid_constant)) const MUtensorDescriptor dO_desc_3, float* __restrict__ dQ_accum,\
 d128::Scalar* __restrict__ dV, int batch, int dk_stride_b, int dk_stride_h, int dk_stride_s, int dv_stride_b,\
 int dv_stride_h, int dv_stride_s, int heads_q, int k_stride_b, int k_stride_h, int k_stride_s, int max_seq_kv,\
 int max_seq_q, float rln2_scale, int seqlen_k, int seqlen_q, float smscale, int v_stride_b, int v_stride_h,\
 int v_stride_s);
extern "C" __global__ void __launch_bounds__(640, 1) flashattn_bwd_ws_unsplit_kernel_kernel(const float2* __restrict__\
 RowStats, const d128::Scalar* __restrict__ K, __attribute__((grid_constant)) const MUtensorDescriptor K_desc_0,\
 __attribute__((grid_constant)) const d128::robust_param_desc_t K_robust_desc_0, __attribute__((grid_constant))\
 const MUtensorDescriptor Q_desc_1, __attribute__((grid_constant)) const MUtensorDescriptor Q_desc_4,\
 const d128::Scalar* __restrict__ V, __attribute__((grid_constant)) const d128::robust_param_desc_t V_robust_desc_1,\
 d128::Scalar* __restrict__ dK, __attribute__((grid_constant)) const MUtensorDescriptor dO_desc_2,\
 __attribute__((grid_constant)) const MUtensorDescriptor dO_desc_3, float* __restrict__ dQ_accum,\
 d128::Scalar* __restrict__ dV, int batch, int dk_stride_b, int dk_stride_h, int dk_stride_s, int dv_stride_b,\
 int dv_stride_h, int dv_stride_s, int heads_q, int k_stride_b, int k_stride_h, int k_stride_s, int max_seq_kv,\
 int max_seq_q, float rln2_scale, int seqlen_k, int seqlen_q, float smscale, int v_stride_b, int v_stride_h,\
 int v_stride_s) {
  __musa_async_bar_record(28);
  extern __shared__ __align__(4096) uchar buf_dyn_shmem[];
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Kt_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* Vt_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* Qb_shared = ((void*)((char*)buf_dyn_shmem + 49152));
  void* dOb_shared = ((void*)((char*)buf_dyn_shmem + 57344));
  void* Pt_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  void* dS_shared = ((void*)((char*)buf_dyn_shmem + 81920));
  void* dSt_shared = ((void*)((char*)buf_dyn_shmem + 98304));
  void* Qa_shared = ((void*)((char*)buf_dyn_shmem + 114688));
  void* dOa_shared = ((void*)((char*)buf_dyn_shmem + 122880));
  int kv_group = 0;
  int begin_seq_q = 0;
  int end_seq_q = 0;
  int begin_seq_kv = 0;
  int end_seq_kv = 0;
  int stage_producer = 0;
  int phase_producer = 0;
  int prev_stage_producer = 0;
  int prev_phase_producer = 0;
  int warprole = 0;
  int stage_consumer1 = 0;
  int phase_consumer1 = 0;
  float dk_accum[32];
  float dq_accum[16];
  float dq_store_buffer[16];
  int local_q_start = 0;
  d128::Scalar dk_cast[32];
  int stage_consumer0 = 0;
  int phase_consumer0 = 0;
  int prev_stage_consumer0 = 0;
  int prev_phase_consumer0 = 0;
  float dv_accum[32];
  d128::Scalar dv_cast[32];
  float lse_buffer[4];
  float delta_buffer[4];
  float accs_accum[32];
  float dp_accum[32];
  d128::Scalar accs_cast[32];
  if (d128::block_leader()) {
    __musa_async_init_arrival(1, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(2, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(3, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(4, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(5, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(6, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(7, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(8, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(9, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(10, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(11, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(12, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(13, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(14, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(15, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(16, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(17, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(18, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(19, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(20, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(21, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(22, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(23, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(24, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(25, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(26, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(27, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(28, ((512 + 31) / 32), 0);
  }
  d128::fence_barrier_init();
  __syncthreads_lm();
  kv_group = ((int)blockIdx.y);
  begin_seq_q = (((int)blockIdx.z) * seqlen_q);
  end_seq_q = (begin_seq_q + seqlen_q);
  begin_seq_kv = (((int)blockIdx.z) * seqlen_k);
  end_seq_kv = (begin_seq_kv + seqlen_k);
  __syncthreads_lm();
  if (((((int)blockIdx.x) * 128) + begin_seq_kv) < end_seq_kv) {
    if (512 <= ((int)threadIdx.x)) {
      stage_producer = 0;
      phase_producer = 0;
      prev_stage_producer = 1;
      prev_phase_producer = 1;
      warprole = ((((int)threadIdx.x) >> 5) - 16);
      if (((int)threadIdx.x) == 512) {
        __musa_async_add_trans(1, 16384);
        d128::tma_load<d128::TmaGranularity::B32>(K_desc_0, 1, (&(((d128::Scalar*)K_shared)[0])), 0, kv_group, (((int)\
blockIdx.x) * 128), ((int)blockIdx.z), 32, 1, 128, 1);
        d128::tma_load<d128::TmaGranularity::B32>(K_desc_0, 1, (&(((d128::Scalar*)K_shared)[4096])), 32, kv_group,\
 (((int)blockIdx.x) * 128), ((int)blockIdx.z), 32, 1, 128, 1);
      }
      __musa_async_arrive(1);
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        d128::cp_async_gs_robust<16>((&(((d128::Scalar*)Kt_shared)[((((i * 1024) + ((((int)threadIdx.x) >> 4) * 128))\
 + (((((int)threadIdx.x) & 15) ^ ((((i * 16) + (((int)threadIdx.x) >> 3)) & 31) >> 1)) * 8)) - 4096)])),\
 (&(K[(((((((int)threadIdx.x) & 7) * 8) + (((int)blockIdx.z) * k_stride_b)) + (((((((int)blockIdx.x) * 128)\
 + ((i >> 2) * 64)) + (((((int)threadIdx.x) & 63) >> 3) * 8)) + (((i * 2) + (((int)threadIdx.x) >> 6)) & 7))\
 * k_stride_s)) + (kv_group * k_stride_h))])), K_robust_desc_0);
      }
      #pragma unroll
      for (int i_1 = 0; i_1 < 8; ++i_1) {
        d128::cp_async_gs_robust<16>((&(((d128::Scalar*)Vt_shared)[((((i_1 * 1024) + ((((int)threadIdx.x) >> 4)\
 * 128)) + (((((int)threadIdx.x) & 15) ^ ((((i_1 * 16) + (((int)threadIdx.x) >> 3)) & 31) >> 1)) * 8)) - 4096)])),\
 (&(V[(((((((int)threadIdx.x) & 7) * 8) + (((int)blockIdx.z) * v_stride_b)) + (((((((int)blockIdx.x) * 128)\
 + ((i_1 >> 2) * 64)) + (((((int)threadIdx.x) & 63) >> 3) * 8)) + (((i_1 * 2) + (((int)threadIdx.x) >> 6)) & 7))\
 * v_stride_s)) + (kv_group * v_stride_h))])), V_robust_desc_1);
      }
      d128::finish_robust_copies();
      __musa_async_arrive(2);
      __musa_async_arrive(3);
      for (int _tmp = 0; _tmp < (((end_seq_q + 63) - begin_seq_q) >> 6); ++_tmp) {
        int q_start = ((_tmp * 64) + begin_seq_q);
        if (warprole == 0) {
          __musa_async_wait((prev_stage_producer + 12), prev_phase_producer);
          if (((int)threadIdx.x) == 512) {
            __musa_async_add_trans((stage_producer + 4), 8192);
            d128::tma_load<d128::TmaGranularity::B16>(Q_desc_1, (stage_producer + 4), (&(((d128::Scalar*)\
Qa_shared)[0])), 0, ((int)blockIdx.y), (q_start - begin_seq_q), ((int)blockIdx.z), 64, 1, 64, 1);
          }
          __musa_async_arrive((stage_producer + 4));
        }
        if (warprole == 1) {
          __musa_async_wait((prev_stage_producer + 16), prev_phase_producer);
          if (((int)threadIdx.x) == 544) {
            __musa_async_add_trans((stage_producer + 8), 8192);
            d128::tma_load<d128::TmaGranularity::B16>(dO_desc_2, (stage_producer + 8), (&(((d128::Scalar*)\
dOa_shared)[0])), 0, ((int)blockIdx.y), (q_start - begin_seq_q), ((int)blockIdx.z), 64, 1, 64, 1);
          }
          __musa_async_arrive((stage_producer + 8));
        }
        if (warprole == 2) {
          __musa_async_wait((prev_stage_producer + 18), prev_phase_producer);
          if (((int)threadIdx.x) == 576) {
            __musa_async_add_trans((stage_producer + 10), 8192);
            d128::tma_load<d128::TmaGranularity::B32>(dO_desc_3, (stage_producer + 10), (&(((d128::Scalar*)\
dOb_shared)[0])), 0, ((int)blockIdx.y), (q_start - begin_seq_q), ((int)blockIdx.z), 64, 1, 64, 1);
          }
          __musa_async_arrive((stage_producer + 10));
        }
        if (warprole == 3) {
          __musa_async_wait((prev_stage_producer + 14), prev_phase_producer);
          if (((int)threadIdx.x) == 608) {
            __musa_async_add_trans((stage_producer + 6), 8192);
            d128::tma_load<d128::TmaGranularity::B32>(Q_desc_4, (stage_producer + 6), (&(((d128::Scalar*)\
Qb_shared)[0])), 0, ((int)blockIdx.y), (q_start - begin_seq_q), ((int)blockIdx.z), 32, 1, 64, 1);
            d128::tma_load<d128::TmaGranularity::B32>(Q_desc_4, (stage_producer + 6), (&(((d128::Scalar*)\
Qb_shared)[2048])), 32, ((int)blockIdx.y), (q_start - begin_seq_q), ((int)blockIdx.z), 32, 1, 64, 1);
          }
          __musa_async_arrive((stage_producer + 6));
        }
        prev_stage_producer = stage_producer;
        prev_phase_producer = phase_producer;
        phase_producer = (phase_producer ^ stage_producer);
        stage_producer = (stage_producer ^ 1);
      }
      __musa_async_wait(28, 0);
    } else {
      if (256 <= ((int)threadIdx.x)) {
        stage_consumer1 = 0;
        phase_consumer1 = 0;
        __musa_async_wait(1, 0);
        __musa_async_wait(2, 0);
        __musa_async_wait(3, 0);
        #pragma unroll
        for (int i_2 = 0; i_2 < 8; ++i_2) {
          float broadcast_var = 0x0p+0f/*0.000000e+00*/;
          *(tl_f4*)(dk_accum + (i_2 * 4)) = (tl_f4{static_cast<float>(broadcast_var),\
 static_cast<float>(broadcast_var), static_cast<float>(broadcast_var), static_cast<float>(broadcast_var)});
        }
        __musa_async_wait((stage_consumer1 + 6), phase_consumer1);
        __musa_async_wait((stage_consumer1 + 26), phase_consumer1);
        d64::gemm<d128::Product::DK>((&(((d128::Scalar*)dSt_shared)[0])), (&(((d128::Scalar*)Qb_shared)[0])),\
 (&(dk_accum[0])), 256);
        d128::wait_sqmma();
        __musa_async_arrive((stage_consumer1 + 14));
        __musa_async_wait((stage_consumer1 + 24), phase_consumer1);
        d64::gemm<d128::Product::DQ>((&(((d128::Scalar*)dS_shared)[0])), (&(((d128::Scalar*)K_shared)[0])),\
 (&(dq_accum[0])), 256);
        d128::wait_sqmma();
        __musa_async_arrive((stage_consumer1 + 22));
        #pragma unroll
        for (int i_3 = 0; i_3 < 4; ++i_3) {
          *(tl_f4*)(dq_store_buffer + (i_3 * 4)) = *(tl_f4*)(dq_accum + (i_3 * 4));
        }
        phase_consumer1 = (phase_consumer1 ^ stage_consumer1);
        stage_consumer1 = (stage_consumer1 ^ 1);
        int dq_loop_start = begin_seq_q;
        local_q_start = (dq_loop_start - begin_seq_q);
        for (int _tmp_1 = 0; _tmp_1 < (((end_seq_q - dq_loop_start) - 1) >> 6); ++_tmp_1) {
          #pragma unroll
          for (int i_4 = 0; i_4 < 16; ++i_4) {
            AtomicAdd(*((&(dQ_accum[((((((((((i_4 >> 2) * 1024) + (((((int)threadIdx.x) & 127) >> 3) * 64))\
 + (local_q_start * 64)) + (((((int)blockIdx.z) * max_seq_q) * heads_q) * 64)) + ((((int)blockIdx.y) * max_seq_q)\
 * 64)) + ((((int)threadIdx.x) >> 7) * 32)) + ((i_4 & 3) * 8)) + (((int)threadIdx.x) & 7)) - 64)]))),\
 dq_store_buffer[i_4]);
          }
          __musa_async_wait((stage_consumer1 + 6), phase_consumer1);
          __musa_async_wait((stage_consumer1 + 26), phase_consumer1);
          d64::gemm<d128::Product::DK>((&(((d128::Scalar*)dSt_shared)[0])), (&(((d128::Scalar*)Qb_shared)[0])),\
 (&(dk_accum[0])), 256);
          d128::wait_sqmma();
          __musa_async_wait((stage_consumer1 + 24), phase_consumer1);
          d64::gemm<d128::Product::DQ>((&(((d128::Scalar*)dS_shared)[0])), (&(((d128::Scalar*)K_shared)[0])),\
 (&(dq_accum[0])), 256);
          d128::wait_sqmma();
          __musa_async_arrive((stage_consumer1 + 14));
          __musa_async_arrive((stage_consumer1 + 22));
          #pragma unroll
          for (int i_5 = 0; i_5 < 4; ++i_5) {
            *(tl_f4*)(dq_store_buffer + (i_5 * 4)) = *(tl_f4*)(dq_accum + (i_5 * 4));
          }
          phase_consumer1 = (phase_consumer1 ^ stage_consumer1);
          stage_consumer1 = (stage_consumer1 ^ 1);
          local_q_start = (local_q_start + 64);
        }
        #pragma unroll
        for (int i_6 = 0; i_6 < 16; ++i_6) {
          AtomicAdd(*((&(dQ_accum[((((((((((i_6 >> 2) * 1024) + (((((int)threadIdx.x) & 127) >> 3) * 64))\
 + (local_q_start * 64)) + (((((int)blockIdx.z) * max_seq_q) * heads_q) * 64)) + ((((int)blockIdx.y) * max_seq_q)\
 * 64)) + ((((int)threadIdx.x) >> 7) * 32)) + ((i_6 & 3) * 8)) + (((int)threadIdx.x) & 7)) - 64)]))),\
 dq_store_buffer[i_6]);
        }
        #pragma unroll
        for (int i_7 = 0; i_7 < 8; ++i_7) {
          d128::Vec4 __1;
          tl_f4 v_ = *(tl_f4*)(dk_accum + (i_7 * 4));
          ((d128::Packed4*)(&__1))[0] = d128::round_to_scalar4(((float4*)(&v_))[0]);
          *(d128::Vec4*)(dk_cast + (i_7 * 4)) = __1;
        }
        if (end_seq_kv < (((((int)blockIdx.x) * 128) + begin_seq_kv) + 128)) {
          #pragma unroll
          for (int i_8 = 0; i_8 < 32; ++i_8) {
            if (((((((int)blockIdx.x) * 128) + ((i_8 >> 2) * 16)) + ((((int)threadIdx.x) & 127) >> 3)) + begin_seq_kv)\
 < end_seq_kv) {
              dK[((((((((((int)threadIdx.x) >> 7) * 32) + ((i_8 & 3) * 8)) + (((int)blockIdx.z) * dk_stride_b))\
 + ((((((int)blockIdx.x) * 128) + ((i_8 >> 2) * 16)) + ((((int)threadIdx.x) & 127) >> 3)) * dk_stride_s)) + (((int)\
blockIdx.y) * dk_stride_h)) + (((int)threadIdx.x) & 7)) - 64)] = dk_cast[i_8];
            }
          }
        } else {
          #pragma unroll
          for (int i_9 = 0; i_9 < 32; ++i_9) {
            dK[((((((((((int)threadIdx.x) >> 7) * 32) + ((i_9 & 3) * 8)) + (((int)blockIdx.z) * dk_stride_b))\
 + ((((((int)blockIdx.x) * 128) + ((i_9 >> 2) * 16)) + ((((int)threadIdx.x) & 127) >> 3)) * dk_stride_s)) + (((int)\
blockIdx.y) * dk_stride_h)) + (((int)threadIdx.x) & 7)) - 64)] = dk_cast[i_9];
          }
        }
        __musa_async_arrive(28);
      } else {
        stage_consumer0 = 0;
        phase_consumer0 = 0;
        prev_stage_consumer0 = 1;
        prev_phase_consumer0 = 1;
        __musa_async_wait(1, 0);
        __musa_async_wait(2, 0);
        __musa_async_wait(3, 0);
        #pragma unroll
        for (int i_10 = 0; i_10 < 8; ++i_10) {
          float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
          *(tl_f4*)(dv_accum + (i_10 * 4)) = (tl_f4{static_cast<float>(broadcast_var_1),\
 static_cast<float>(broadcast_var_1), static_cast<float>(broadcast_var_1), static_cast<float>(broadcast_var_1)});
        }
        for (int _tmp_2 = 0; _tmp_2 < (((end_seq_q + 63) - begin_seq_q) >> 6); ++_tmp_2) {
          int q_start_1 = ((_tmp_2 * 64) + begin_seq_q);
          #pragma unroll
          for (int i_11 = 0; i_11 < 4; ++i_11) {
            const int row = i_11 * 16 + ((int(threadIdx.x) & 127) >> 3)
                + (int(blockIdx.z) * heads_q + kv_group) * seqlen_q
                + q_start_1 - begin_seq_q;
            const float2 stats = RowStats[row];
            lse_buffer[i_11] = stats.x;
            delta_buffer[i_11] = stats.y;
          }
          if (end_seq_kv < (((((int)blockIdx.x) * 128) + begin_seq_kv) + 128)) {
            #pragma unroll
            for (int i_14 = 0; i_14 < 32; ++i_14) {
              float condval_1;
              if (((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 7) * 64)) + ((((int)threadIdx.x) & 7) * 8))\
 + begin_seq_kv) + (i_14 & 7)) < end_seq_kv)) {
                condval_1 = 0x0p+0f/*0.000000e+00*/;
              } else {
                condval_1 = -MUSART_INF_F;
              }
              accs_accum[i_14] = condval_1;
            }
          } else {
            #pragma unroll
            for (int i_15 = 0; i_15 < 8; ++i_15) {
              float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
              *(tl_f4*)(accs_accum + (i_15 * 4)) = (tl_f4{static_cast<float>(broadcast_var_2),\
 static_cast<float>(broadcast_var_2), static_cast<float>(broadcast_var_2), static_cast<float>(broadcast_var_2)});
            }
          }
          __musa_async_wait((stage_consumer0 + 4), phase_consumer0);
          d64::gemm<d128::Product::QK>((&(((d128::Scalar*)Qa_shared)[0])), (&(((d128::Scalar*)Kt_shared)[0])),\
 (&(accs_accum[0])), 0);
          d128::warpgroup_commit_batch();
          __musa_async_wait((stage_consumer0 + 8), phase_consumer0);
          d64::gemm<d128::Product::DOV>((&(((d128::Scalar*)dOa_shared)[0])), (&(((d128::Scalar*)Vt_shared)[0])),\
 (&(dp_accum[0])), 0);
          d128::warpgroup_commit_batch();
          d128::warpgroup_wait<1>();
          int stage = stage_consumer0;
          __musa_async_arrive((stage + 20));
          #pragma unroll
          for (int i_20 = 0; i_20 < 8; ++i_20) {
            tl_f4 v__1 = d128::fma4(*(tl_f4*)(accs_accum + (i_20 * 4)), rln2_scale, -lse_buffer[(i_20 >> 1)]);
            *(tl_f4*)(accs_accum + (i_20 * 4)) = d128::vec_exp2_f4(v__1);
          }
          #pragma unroll
          for (int i_21 = 0; i_21 < 8; ++i_21) {
            d128::Vec4 __2;
            tl_f4 v__2 = *(tl_f4*)(accs_accum + (i_21 * 4));
            ((d128::Packed4*)(&__2))[0] = d128::round_to_scalar4(((float4*)(&v__2))[0]);
            *(d128::Vec4*)(accs_cast + (i_21 * 4)) = __2;
          }
          __musa_async_wait((stage_consumer0 + 20), phase_consumer0);
          #pragma unroll
          for (int i_22 = 0; i_22 < 4; ++i_22) {
            *(d128::Vec8*)(((d128::Scalar*)Pt_shared) + ((((((((int)threadIdx.x) >> 7) * 4096) + (i_22 * 1024))\
 + (((((int)threadIdx.x) & 127) >> 4) * 128)) + ((((((int)threadIdx.x) & 15) >> 1) ^ ((((int)threadIdx.x) & 127)\
 >> 4)) * 16)) + ((((int)threadIdx.x) & 1) * 8))) = *(d128::Vec8*)(accs_cast + (i_22 * 8));
          }
          d128::lma_wait();
          __musa_async_wait((stage_consumer0 + 10), phase_consumer0);
          d64::gemm<d128::Product::DV>((&(((d128::Scalar*)Pt_shared)[0])), (&(((d128::Scalar*)dOb_shared)[0])),\
 (&(dv_accum[0])), 0);
          d128::warpgroup_commit_batch();
          d128::warpgroup_wait<1>();
          int stage_1 = stage_consumer0;
          __musa_async_arrive((stage_1 + 16));
          #pragma unroll
          for (int i_24 = 0; i_24 < 8; ++i_24) {
            d128::Vec4 __3;
            const tl_f4 p = *(tl_f4*)(accs_accum + (i_24 * 4));
            const tl_f4 dp = *(tl_f4*)(dp_accum + (i_24 * 4));
            const tl_f4 d = d128::fma4(dp, smscale, -delta_buffer[i_24 >> 1]);
            const tl_f4 v__3 = {
                d128::scale_ds(p[0], d[0]), d128::scale_ds(p[1], d[1]),
                d128::scale_ds(p[2], d[2]), d128::scale_ds(p[3], d[3])};
            ((d128::Packed4*)(&__3))[0] = d128::round_to_scalar4(((float4*)(&v__3))[0]);
            *(d128::Vec4*)(accs_cast + (i_24 * 4)) = __3;
          }
          __musa_async_wait((prev_stage_consumer0 + 22), prev_phase_consumer0);
          #pragma unroll
          for (int i_25 = 0; i_25 < 4; ++i_25) {
            *(d128::Vec8*)(((d128::Scalar*)dSt_shared) + ((((i_25 * 2048) + (((((int)threadIdx.x) & 127) >> 3) * 128))\
 + (((((((int)threadIdx.x) >> 7) * 4) + ((((int)threadIdx.x) & 7) >> 1)) ^ ((((int)threadIdx.x) & 63) >> 3)) * 16))\
 + ((((int)threadIdx.x) & 1) * 8))) = *(d128::Vec8*)(accs_cast + (i_25 * 8));
          }
          d128::lma_wait();
          __musa_async_arrive((stage_consumer0 + 26));
          d128::warpgroup_wait<0>();
          __musa_async_arrive((stage_consumer0 + 12));
          __musa_async_arrive((stage_consumer0 + 18));
          #pragma unroll
          for (int i_26 = 0; i_26 < 4; ++i_26) {
            *(d128::Vec8*)(((d128::Scalar*)dS_shared) + (((i_26 * 2048) + (((((int)threadIdx.x) & 127) >> 3) * 128))\
 + (((((((int)threadIdx.x) >> 7) * 8) + (((int)threadIdx.x) & 7)) ^ ((((int)threadIdx.x) & 127) >> 3)) * 8)))\
 = *(d128::Vec8*)(accs_cast + (i_26 * 8));
          }
          d128::lma_wait();
          __musa_async_arrive((stage_consumer0 + 24));
          prev_stage_consumer0 = stage_consumer0;
          prev_phase_consumer0 = phase_consumer0;
          phase_consumer0 = (phase_consumer0 ^ stage_consumer0);
          stage_consumer0 = (stage_consumer0 ^ 1);
        }
        #pragma unroll
        for (int i_27 = 0; i_27 < 8; ++i_27) {
          d128::Vec4 __4;
          tl_f4 v__4 = *(tl_f4*)(dv_accum + (i_27 * 4));
          ((d128::Packed4*)(&__4))[0] = d128::round_to_scalar4(((float4*)(&v__4))[0]);
          *(d128::Vec4*)(dv_cast + (i_27 * 4)) = __4;
        }
        if (end_seq_kv < (((((int)blockIdx.x) * 128) + begin_seq_kv) + 128)) {
          #pragma unroll
          for (int i_28 = 0; i_28 < 32; ++i_28) {
            if ((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 7) * 64)) + ((i_28 >> 3) * 16)) + ((((int)\
threadIdx.x) & 127) >> 3)) + begin_seq_kv) < end_seq_kv) {
              dV[((((((i_28 & 7) * 8) + (((int)blockIdx.z) * dv_stride_b)) + (((((((int)blockIdx.x) * 128) + ((((int)\
threadIdx.x) >> 7) * 64)) + ((i_28 >> 3) * 16)) + ((((int)threadIdx.x) & 127) >> 3)) * dv_stride_s)) + (((int)\
blockIdx.y) * dv_stride_h)) + (((int)threadIdx.x) & 7))] = dv_cast[i_28];
            }
          }
        } else {
          #pragma unroll
          for (int i_29 = 0; i_29 < 32; ++i_29) {
            dV[((((((i_29 & 7) * 8) + (((int)blockIdx.z) * dv_stride_b)) + (((((((int)blockIdx.x) * 128) + ((((int)\
threadIdx.x) >> 7) * 64)) + ((i_29 >> 3) * 16)) + ((((int)threadIdx.x) & 127) >> 3)) * dv_stride_s)) + (((int)\
blockIdx.y) * dv_stride_h)) + (((int)threadIdx.x) & 7))] = dv_cast[i_29];
          }
        }
        __musa_async_arrive(28);
      }
    }
  }
}

#endif
"""

_D64_HOST_CPP = _D128_HOST_CPP.replace("kHeadDim = 128;", "kHeadDim = 64;").replace(
    "kSharedBytes = 192", "kSharedBytes = 128"
)
