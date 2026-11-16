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

import torch
import triton
import triton.language as tl


@triton.jit
def _mask_scores(
    s,
    Mask,
    Drop,
    pbh,
    offs_m,
    offs_n,
    L,
    S,
    HAS_MASK: tl.constexpr,
    MASK_BOOL: tl.constexpr,
    HAS_DROP: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    valid = offs_n[None, :] < S
    if IS_CAUSAL:
        valid = valid & (offs_n[None, :] <= offs_m[:, None] + (S - L))
    if HAS_MASK:
        m_ptr = Mask + pbh * L * S + offs_m[:, None] * S + offs_n[None, :]
        m_ok = (offs_m[:, None] < L) & (offs_n[None, :] < S)
        if MASK_BOOL:
            valid = valid & tl.load(m_ptr, mask=m_ok, other=False)
        else:
            s = s + tl.load(m_ptr, mask=m_ok, other=0.0).to(tl.float32)
    if HAS_DROP:
        d_ptr = Drop + pbh * L * S + offs_m[:, None] * S + offs_n[None, :]
        d_ok = (offs_m[:, None] < L) & (offs_n[None, :] < S)
        valid = valid & tl.load(d_ptr, mask=d_ok, other=False)
    return tl.where(valid, s, float("-inf"))


@triton.jit
def _qk_tile(
    q,
    K,
    kv,
    n0,
    offs_d,
    D,
    S,
    scale,
    Mask,
    Drop,
    pbh,
    offs_m,
    L,
    HAS_MASK: tl.constexpr,
    MASK_BOOL: tl.constexpr,
    HAS_DROP: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IEEE: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    offs_n = n0 + tl.arange(0, BN)
    k_ptr = K + kv * S * D + offs_n[:, None] * D + offs_d[None, :]
    k = tl.load(k_ptr, mask=(offs_n[:, None] < S) & (offs_d[None, :] < D), other=0.0)
    if IEEE:
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32, input_precision="ieee")
    else:
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
    s = s * scale
    return _mask_scores(
        s,
        Mask,
        Drop,
        pbh,
        offs_m,
        offs_n,
        L,
        S,
        HAS_MASK,
        MASK_BOOL,
        HAS_DROP,
        IS_CAUSAL,
    )


@triton.jit
def _pv_tile(
    p,
    V,
    kv,
    n0,
    offs_dv,
    Dv,
    S,
    IEEE: tl.constexpr,
    BN: tl.constexpr,
    BDV: tl.constexpr,
):
    offs_n = n0 + tl.arange(0, BN)
    v_ptr = V + kv * S * Dv + offs_n[:, None] * Dv + offs_dv[None, :]
    v = tl.load(v_ptr, mask=(offs_n[:, None] < S) & (offs_dv[None, :] < Dv), other=0.0)
    if IEEE:
        return tl.dot(p.to(v.dtype), v, out_dtype=tl.float32, input_precision="ieee")
    return tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)


@triton.jit
def _sdpa_math_fwd(
    Q,
    K,
    V,
    Out,
    AW,
    Mask,
    Drop,
    scale,
    L,
    S,
    D,
    Dv,
    Hq,
    Hk,
    GQA: tl.constexpr,
    HAS_MASK: tl.constexpr,
    MASK_BOOL: tl.constexpr,
    HAS_DROP: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IEEE: tl.constexpr,
    NCHUNK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    BDV: tl.constexpr,
):
    # Q/K/V/Out/AW are the caller's dense contiguous tensors, so every stride
    # is derived from L/S/D/Dv here instead of being a launch argument.
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    if GQA:
        kv = (pid_bh // Hq) * Hk + (pid_bh % Hq) // (Hq // Hk)
    else:
        kv = pid_bh

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, BD)
    offs_dv = tl.arange(0, BDV)

    q_ptr = Q + pid_bh * L * D + offs_m[:, None] * D + offs_d[None, :]
    q = tl.load(q_ptr, mask=(offs_m[:, None] < L) & (offs_d[None, :] < D), other=0.0)
    w_base = AW + pid_bh * L * S + offs_m[:, None] * S
    m_ok = offs_m[:, None] < L
    offs_n0 = tl.arange(0, BN)

    if NCHUNK > 0:
        # The whole key range is covered by NCHUNK tiles that stay live in
        # registers, so every QK product is evaluated once instead of twice.
        if IS_CAUSAL:
            hi = (pid_m + 1) * BM + (S - L)
            hi = tl.minimum(hi, S)
            hi = tl.maximum(hi, 0)
        else:
            hi = S

        if NCHUNK == 2 and IS_CAUSAL and hi <= BN:
            # Causal rows in this block reach only the first key tile, so the
            # second tile is entirely masked: skip its QK product and value
            # product and just zero-fill those weight columns.
            s0 = _qk_tile(
                q,
                K,
                kv,
                0,
                offs_d,
                D,
                S,
                scale,
                Mask,
                Drop,
                pid_bh,
                offs_m,
                L,
                HAS_MASK,
                MASK_BOOL,
                HAS_DROP,
                IS_CAUSAL,
                IEEE,
                BN,
                BD,
            )
            m_i = tl.max(s0, 1)
            if HAS_MASK or HAS_DROP:
                safe = m_i != float("-inf")
                p0 = tl.where(safe[:, None], tl.exp(s0 - m_i[:, None]), 0.0)
                l_i = tl.sum(p0, 1)
                inv_l = tl.where(safe & (l_i != 0.0), 1.0 / l_i, 0.0)
            else:
                p0 = tl.exp(s0 - m_i[:, None])
                inv_l = 1.0 / tl.sum(p0, 1)
            p0 = p0 * inv_l[:, None]
            tl.store(
                w_base + offs_n0[None, :],
                p0.to(AW.dtype.element_ty),
                mask=m_ok & (offs_n0[None, :] < S),
            )
            offs_n1 = BN + offs_n0
            tl.store(
                w_base + offs_n1[None, :],
                tl.zeros([BM, BN], AW.dtype.element_ty),
                mask=m_ok & (offs_n1[None, :] < S),
            )
            acc = _pv_tile(p0, V, kv, 0, offs_dv, Dv, S, IEEE, BN, BDV)
        else:
            s0 = _qk_tile(
                q,
                K,
                kv,
                0,
                offs_d,
                D,
                S,
                scale,
                Mask,
                Drop,
                pid_bh,
                offs_m,
                L,
                HAS_MASK,
                MASK_BOOL,
                HAS_DROP,
                IS_CAUSAL,
                IEEE,
                BN,
                BD,
            )
            if NCHUNK == 1:
                m_i = tl.max(s0, 1)
                if HAS_MASK or HAS_DROP:
                    safe = m_i != float("-inf")
                    p0 = tl.where(safe[:, None], tl.exp(s0 - m_i[:, None]), 0.0)
                    l_i = tl.sum(p0, 1)
                    inv_l = tl.where(safe & (l_i != 0.0), 1.0 / l_i, 0.0)
                else:
                    p0 = tl.exp(s0 - m_i[:, None])
                    inv_l = 1.0 / tl.sum(p0, 1)
                p0 = p0 * inv_l[:, None]
                tl.store(
                    w_base + offs_n0[None, :],
                    p0.to(AW.dtype.element_ty),
                    mask=m_ok & (offs_n0[None, :] < S),
                )
                acc = _pv_tile(p0, V, kv, 0, offs_dv, Dv, S, IEEE, BN, BDV)
            else:
                s1 = _qk_tile(
                    q,
                    K,
                    kv,
                    BN,
                    offs_d,
                    D,
                    S,
                    scale,
                    Mask,
                    Drop,
                    pid_bh,
                    offs_m,
                    L,
                    HAS_MASK,
                    MASK_BOOL,
                    HAS_DROP,
                    IS_CAUSAL,
                    IEEE,
                    BN,
                    BD,
                )
                m_i = tl.maximum(tl.max(s0, 1), tl.max(s1, 1))
                if HAS_MASK or HAS_DROP:
                    safe = m_i != float("-inf")
                    p0 = tl.where(safe[:, None], tl.exp(s0 - m_i[:, None]), 0.0)
                    p1 = tl.where(safe[:, None], tl.exp(s1 - m_i[:, None]), 0.0)
                    l_i = tl.sum(p0, 1) + tl.sum(p1, 1)
                    inv_l = tl.where(safe & (l_i != 0.0), 1.0 / l_i, 0.0)
                else:
                    p0 = tl.exp(s0 - m_i[:, None])
                    p1 = tl.exp(s1 - m_i[:, None])
                    inv_l = 1.0 / (tl.sum(p0, 1) + tl.sum(p1, 1))
                p0 = p0 * inv_l[:, None]
                p1 = p1 * inv_l[:, None]
                offs_n1 = BN + offs_n0
                tl.store(
                    w_base + offs_n0[None, :],
                    p0.to(AW.dtype.element_ty),
                    mask=m_ok & (offs_n0[None, :] < S),
                )
                tl.store(
                    w_base + offs_n1[None, :],
                    p1.to(AW.dtype.element_ty),
                    mask=m_ok & (offs_n1[None, :] < S),
                )
                acc = _pv_tile(p0, V, kv, 0, offs_dv, Dv, S, IEEE, BN, BDV)
                acc += _pv_tile(p1, V, kv, BN, offs_dv, Dv, S, IEEE, BN, BDV)
    else:
        if IS_CAUSAL:
            hi = (pid_m + 1) * BM + (S - L)
            hi = tl.minimum(hi, S)
            hi = tl.maximum(hi, 0)
        else:
            hi = S

        m_i = tl.full([BM], float("-inf"), tl.float32)
        l_i = tl.zeros([BM], tl.float32)

        for n0 in range(0, hi, BN):
            s = _qk_tile(
                q,
                K,
                kv,
                n0,
                offs_d,
                D,
                S,
                scale,
                Mask,
                Drop,
                pid_bh,
                offs_m,
                L,
                HAS_MASK,
                MASK_BOOL,
                HAS_DROP,
                IS_CAUSAL,
                IEEE,
                BN,
                BD,
            )
            m_new = tl.maximum(m_i, tl.max(s, 1))
            if HAS_MASK or HAS_DROP:
                safe = m_new != float("-inf")
                alpha = tl.where(safe, tl.exp(m_i - m_new), 0.0)
                p = tl.where(safe[:, None], tl.exp(s - m_new[:, None]), 0.0)
            else:
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new

        inv_l = 1.0 / tl.where(l_i == 0.0, 1.0, l_i)

        acc = tl.zeros([BM, BDV], tl.float32)
        for n0 in range(0, hi, BN):
            s = _qk_tile(
                q,
                K,
                kv,
                n0,
                offs_d,
                D,
                S,
                scale,
                Mask,
                Drop,
                pid_bh,
                offs_m,
                L,
                HAS_MASK,
                MASK_BOOL,
                HAS_DROP,
                IS_CAUSAL,
                IEEE,
                BN,
                BD,
            )
            p = tl.exp(s - m_i[:, None]) * inv_l[:, None]
            offs_n = n0 + tl.arange(0, BN)
            tl.store(
                w_base + offs_n[None, :],
                p.to(AW.dtype.element_ty),
                mask=m_ok & (offs_n[None, :] < S),
            )
            acc += _pv_tile(p, V, kv, n0, offs_dv, Dv, S, IEEE, BN, BDV)

        if IS_CAUSAL:
            zero = tl.zeros([BM, BN], AW.dtype.element_ty)
            for n0 in range(hi, S, BN):
                offs_n = n0 + tl.arange(0, BN)
                tl.store(
                    w_base + offs_n[None, :],
                    zero,
                    mask=m_ok & (offs_n[None, :] < S),
                )

    o_ptr = Out + pid_bh * L * Dv + offs_m[:, None] * Dv + offs_dv[None, :]
    tl.store(
        o_ptr,
        acc.to(Out.dtype.element_ty),
        mask=(offs_m[:, None] < L) & (offs_dv[None, :] < Dv),
    )


def _dense_mask(mask, shape):
    """Zero-copy broadcast when possible, dense copy otherwise."""
    if mask is None:
        return None
    try:
        m4 = torch.broadcast_to(mask, shape)
    except RuntimeError:
        m4 = mask.reshape(shape)
    if not m4.is_contiguous():
        m4 = m4.contiguous()
    return m4


def _scaled_dot_product_attention_math_for_mps(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    dropout_mask=None,
    *,
    scale=None,
    enable_gqa=False,
):
    assert dropout_p == 0.0, "Only dropout_p=0.0 is supported"

    q = query.contiguous()
    k = key.contiguous()
    v = value.contiguous()

    L = q.shape[-2]
    D = q.shape[-1]
    S = k.shape[-2]
    Dv = v.shape[-1]
    Hq = q.shape[-3] if q.dim() >= 3 else 1
    Hk = k.shape[-3] if k.dim() >= 3 else 1

    scale = 1.0 / (D**0.5) if scale is None else float(scale)

    batch_shape = q.shape[:-2]
    attn = torch.empty((*batch_shape, L, S), dtype=q.dtype, device=q.device)
    out = torch.empty((*batch_shape, L, Dv), dtype=q.dtype, device=q.device)

    BH = q.numel() // (L * D)
    if BH == 0 or L == 0 or S == 0:
        return out, attn

    gqa = bool(enable_gqa) and Hk != Hq
    if gqa:
        assert Hq % Hk == 0, "Hq must be divisible by Hk when enable_gqa"

    m4 = d4 = None
    if attn_mask is not None:
        m4 = _dense_mask(attn_mask, (BH // Hq, Hq, L, S))
    if dropout_mask is not None:
        d4 = _dense_mask(dropout_mask, (BH // Hq, Hq, L, S))

    has_mask = m4 is not None
    mask_bool = has_mask and m4.dtype == torch.bool
    has_drop = d4 is not None

    BD = max(triton.next_power_of_2(D), 16)
    BDV = max(triton.next_power_of_2(Dv), 16)
    es = q.element_size()
    BM = 64
    BN = 64
    stages = 2
    nchunk = 0

    # Register-resident variants: the key row block is held in NCHUNK tiles so
    # every QK product is evaluated once. Every key tile stays staged while its
    # dot runs, so the whole key range plus one value tile must fit the
    # ~48 KiB shared-memory regime that pipelines well, and all live score
    # tiles plus the accumulator and query tile must fit the register file.
    BNS = triton.next_power_of_2(S)
    regs = BM * BNS * 4 + BM * BDV * 4 + BM * BD * es
    if regs <= 100 * 1024:
        for chunks in (1, 2):
            bn = BNS // chunks
            if chunks > 1 and bn < 32:
                continue
            if (BNS * BD + bn * BDV) * es <= 49152:
                nchunk = chunks
                BN = bn
                stages = 1
                break

    if nchunk == 0:
        if es > 2:
            # fp32 dots run with input_precision="ieee", which stages extra
            # shared memory on this 64 KiB-per-block target.
            BN = 32
            stages = 1
        elif BD >= 128:
            # A 64x128 16-bit key tile staged at BN=64 fills the whole 64 KiB
            # shared budget, leaving no room to overlap key/value loads with
            # the dots. Halving the key tile keeps the efficient M=64 dot shape
            # but brings the footprint back to the ~40 KiB regime.
            BN = 32

    # Dummy pointer when a mask is absent (never dereferenced).
    if not has_mask:
        m4 = q
    if not has_drop:
        d4 = q

    grid = (triton.cdiv(L, BM), BH)
    _sdpa_math_fwd[grid](
        q,
        k,
        v,
        out,
        attn,
        m4,
        d4,
        scale,
        L,
        S,
        D,
        Dv,
        Hq,
        Hk,
        GQA=gqa,
        HAS_MASK=has_mask,
        MASK_BOOL=mask_bool,
        HAS_DROP=has_drop,
        IS_CAUSAL=bool(is_causal),
        IEEE=es > 2,
        NCHUNK=nchunk,
        BM=BM,
        BN=BN,
        BD=BD,
        BDV=BDV,
        num_warps=4,
        num_stages=stages,
    )
    return out, attn
