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

from flag_gems.ops.flash_attention_backward import (
    _aten_flash_bwd_dkv,
    _aten_flash_bwd_dq,
    _aten_flash_bwd_small,
    _aten_flash_delta,
    _aten_flash_dropout,
    _aten_flash_probability,
)

logger = logging.getLogger(__name__)


@triton.jit
def _split_dkv(
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
    head_q = tl.program_id(2) - HEAD_OFFSET
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
        # Native uses 128-row query blocks for D <= 64 on Hopper and
        # 64-row blocks for larger heads. Blocks descend, MMA subtiles ascend.
        NATIVE_BM: tl.constexpr = 128 if DIM <= 64 else 64
        for iteration in range(
            (tl.cdiv(tl.maximum(hi, 0), NATIVE_BM) - lo // NATIVE_BM)
            * (NATIVE_BM // BM)
        ):
            tile = (
                tl.cdiv(tl.maximum(hi, 0), NATIVE_BM)
                - 1
                - iteration // (NATIVE_BM // BM)
            ) * (NATIVE_BM // BM) + iteration % (NATIVE_BM // BM)
            qm = tile * BM + m
            qi = ((q0 + qm[:, None]) * HQ + head) * DIM + d[None, :]
            mask = (qm[:, None] < sq) & (d[None, :] < DIM)
            q = tl.load(Q + qi, mask, 0)
            do = tl.load(DO + qi, mask, 0)
            if COMPUTE_DELTA:
                o = tl.load(O + qi, mask, 0).to(tl.float32)
                delta = (
                    _aten_flash_delta(
                        o, do, O, DO, ((q0 + qm) * HQ + head) * DIM, qm < sq, DIM, BM
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
            )
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
            dk_head = tl.dot(tl.trans(ds.to(q.dtype)), q, dk_head)
            dv_head = tl.dot(tl.trans(p_drop.to(q.dtype)), do, dv_head)
        dk += (dk_head * (scale * drop_scale)).to(k.dtype).to(tl.float32)
        dv += (dv_head * drop_scale).to(v.dtype).to(tl.float32)
    oi = ((k0 + n[:, None]) * HQ + head_q) * DIM + d[None, :]
    tl.store(DK + oi, tl.where(sq > 0, dk, 0.0), km)
    tl.store(DV + oi, tl.where(sq > 0, dv, 0.0), km)


@triton.jit
def _split_small(
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
            )
    else:
        if block * BN < sk_max:
            _split_dkv(
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
            )


@triton.jit
def _reduce_gqa(
    KS,
    VS,
    K,
    V,
    TK: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = i < TK * HK * D
    d = i % D
    h = (i // D) % HK
    t = i // (HK * D)
    k = tl.full((BLOCK,), 0, tl.float32)
    v = tl.full((BLOCK,), 0, tl.float32)
    for g in tl.static_range(HQ // HK):
        j = (t * HQ + h * (HQ // HK) + g) * D + d
        k += tl.load(KS + j, mask, 0).to(tl.float32)
        v += tl.load(VS + j, mask, 0).to(tl.float32)
    tl.store(K + i, k, mask)
    tl.store(V + i, v, mask)


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
    varlen = query.ndim == 3
    rank = 3 if varlen else 4
    if query.ndim not in (3, 4) or any(
        (t.ndim != rank for t in (key, value, out, grad_out))
    ):
        raise ValueError("flash_attention_backward expects BSHD or packed THD tensors")
    if any(
        (t.dtype != query.dtype for t in (key, value, out, grad_out))
    ) or query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "flash_attention_backward requires matching float16 or bfloat16 tensors"
        )
    if any((t.device != query.device for t in (key, value, out, grad_out, logsumexp))):
        raise ValueError("flash_attention_backward tensors must be on the same device")
    if (
        query.shape != out.shape
        or query.shape != grad_out.shape
        or key.shape != value.shape
    ):
        raise ValueError("incompatible query/output/gradient or key/value shapes")
    dim, hq, hk = (query.shape[-1], query.shape[-2], key.shape[-2])
    if dim != key.shape[-1] or dim <= 0 or dim > 256 or dim % 8:
        raise ValueError("head dimension must match and be a multiple of 8 in [8, 256]")
    if hq <= 0 or hk <= 0 or hq % hk:
        raise ValueError("query heads must be a positive multiple of KV heads")
    if any((t.stride(-1) != 1 for t in (query, key, value, out))):
        raise ValueError(
            "query, key, value and out must have contiguous last dimensions"
        )
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0, 1)")
    threshold, drop_scale, keep32 = (255, 1.0, 1.0)
    if dropout_p:
        p32 = struct.unpack("f", struct.pack("f", dropout_p))[0]
        if p32 >= 1.0:
            raise ValueError("dropout_p must remain below 1 after float32 conversion")
        keep32 = struct.unpack("f", struct.pack("f", 1.0 - p32))[0]
        threshold = int(keep32 * 255.0)
        drop_scale = struct.unpack("f", struct.pack("f", 1.0 / keep32))[0]
        if torch.version.cuda is None:
            raise NotImplementedError(
                "native dropout ABI is only implemented for NVIDIA CUDA"
            )
        schema = torch.ops.aten._flash_attention_backward.default._schema
        if schema.arguments[12].name != "rng_state":
            raise NotImplementedError(
                "legacy seed/offset dropout ABI is not implemented"
            )
        if (
            rng_state is None
            or rng_state.dtype != torch.uint64
            or rng_state.device != query.device
            or (rng_state.shape != (2,))
            or (not rng_state.is_contiguous())
        ):
            raise ValueError(
                "CUDA FlashAttention requires a contiguous device uint64[2] saved rng_state"
            )
    wl = -1 if window_size_left is None else window_size_left
    wr = -1 if window_size_right is None else window_size_right
    wl, wr = (max(-1, wl), max(-1, wr))
    if is_causal:
        wr = 0
    if max_q < 0 or max_k < 0:
        raise ValueError("maximum sequence lengths must be nonnegative")
    if varlen:
        if cum_seq_q is None or cum_seq_k is None:
            raise ValueError("packed tensors require both cumulative sequence arrays")
        for cu in (cum_seq_q, cum_seq_k):
            if (
                cu.dtype != torch.int32
                or cu.device != query.device
                or cu.ndim != 1
                or (not cu.is_contiguous())
                or (cu.numel() < 1)
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
        if max_q == 0 and total_q or (max_k == 0 and key.shape[0]):
            raise ValueError("nonempty packed inputs need nonzero maximum lengths")
    else:
        if cum_seq_q is not None or cum_seq_k is not None:
            raise ValueError(
                "dense tensors require undefined cumulative sequence arrays"
            )
        batch, sq = query.shape[:2]
        if key.shape[0] != batch:
            raise ValueError("query and key batch sizes must match")
        max_q, max_k = (sq, key.shape[1])
        total_q = batch * sq
        expected_lse = (batch, hq, sq)
        lb, lh, lt = logsumexp.stride() if logsumexp.ndim == 3 else (0, 0, 0)
    if logsumexp.dtype != torch.float32 or tuple(logsumexp.shape) != expected_lse:
        raise ValueError(f"logsumexp must be float32 with shape {expected_lse}")
    if wl >= max_k:
        wl = -1
    if wr >= max_k:
        wr = -1
    scale = dim ** (-0.5) if scale is None else scale
    # The native masked-score path is not a finite reference for nonpositive
    # scales. Evaluate valid saved state with FP32 P/dS and GQA accumulation.
    full_precision = scale <= 0.0
    scale_log2 = None
    lse_scale = math.log2(math.e)
    launch_options = {}
    try:
        scale32 = struct.unpack("f", struct.pack("f", scale))[0]
        scale_log2 = struct.unpack("f", struct.pack("f", scale32 * math.log2(math.e)))[
            0
        ]
    except OverflowError:
        scale_log2 = math.copysign(math.inf, scale)
    launch_options["enable_fp_fusion"] = False
    q, k, v, o, do = (t.contiguous() for t in (query, key, value, out, grad_out))
    dq, dk, dv = (torch.empty_like(t) for t in (q, k, v))
    if batch == 0:
        if q.numel() or k.numel():
            raise ValueError("zero sequences require empty packed inputs")
        return (dq, dk, dv)
    wide_sequence_indices = max(max_q, max_k) >= 2**29
    common = (
        rng_state if dropout_p else q,
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
        False,
        dropout_p > 0,
        wide_sequence_indices,
    )
    tile = 32
    num_warps = 4
    if dim <= 128 and hq == hk and (min(max_q, max_k) >= 256):
        tile = 64
    if hq == hk and max(max_q, max_k) <= 256:
        if max_q or max_k:
            _aten_flash_bwd_small[triton.cdiv(max(max_q, max_k), tile), batch, hq + hk](
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
                FULL_PRECISION=full_precision,
                num_warps=num_warps,
                num_stages=1,
                **launch_options,
            )
        return (dq, dk, dv)
    if hq != hk and max(max_q, max_k) <= 256 and not full_precision:
        tile = 16
        total_k = k.numel() // (hk * dim)
        # Two input-dtype workspaces: 2 * total_k * hq * dim elements.
        # Every valid element is written before the fixed-order reduction.
        ks = torch.empty((total_k, hq, dim), dtype=k.dtype, device=k.device)
        vs = torch.empty_like(ks)
        if max_q or max_k:
            _split_small[triton.cdiv(max(max_q, max_k), tile), batch, 2 * hq](
                q,
                k,
                v,
                o,
                do,
                logsumexp,
                dq,
                ks,
                vs,
                cum_seq_q,
                cum_seq_k,
                *common,
                BM=tile,
                BN=tile,
                num_warps=4,
                num_stages=1,
                **launch_options,
            )
        if total_k:
            _reduce_gqa[(triton.cdiv(total_k * hk * dim, 256),)](
                ks, vs, dk, dv, total_k, hq, hk, dim, 256
            )
        return dq, dk, dv
    delta = torch.empty((hq, total_q), dtype=torch.float32, device=q.device)
    if max_q:
        _aten_flash_bwd_dq[triton.cdiv(max_q, tile), batch, hq](
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
            FULL_PRECISION=full_precision,
            **launch_options,
        )
    if max_k:
        _aten_flash_bwd_dkv[triton.cdiv(max_k, tile), batch, hk](
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
            FULL_PRECISION=full_precision,
            num_warps=num_warps,
            num_stages=1,
            ASCEND_LEGACY_COMPILER=False,
            **launch_options,
        )
    return (dq, dk, dv)
