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

"""Math-path scaled dot-product attention with materialized attention weights.

Fused Triton implementation of
aten::_scaled_dot_product_attention_math (a.k.a. the "_for_mps" math path):

    scores = (Q @ K^T) * scale
    scores = scores + attn_mask           (bool mask: True -> -inf)
    scores = scores.masked_fill(causal)   (j > i -> -inf)
    P      = softmax(scores, dim=-1)
    O      = P @ V

Both O (batch, heads, L, Ev) and P (batch, heads, L, S) are materialized with
the query dtype.  One program owns a (BLOCK_M x S) slice of P and O for one
(batch, head): pass 1 streams K to build the online softmax statistics, pass 2
recomputes the scores, writes the normalized probabilities and accumulates
P @ V.  All softmax statistics, masking and normalization run in fp32.
"""

import logging
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_LOG2E = 1.4426950408889634


@triton.jit
def _sdpa_math_kernel(
    Q,
    K,
    V,
    MASK,
    Out,
    W,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_mn,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_wb,
    stride_wh,
    stride_wm,
    stride_wn,
    Q_LEN,
    KV_LEN,
    GROUP,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
    MASK_BOOL: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
    DOT_FP16: tl.constexpr,
):
    LOG2E: tl.constexpr = 1.4426950408889634

    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    kv_h = pid_h // GROUP

    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + kv_h * stride_kh
    v_base = V + pid_b * stride_vb + kv_h * stride_vh
    o_base = Out + pid_b * stride_ob + pid_h * stride_oh
    w_base = W + pid_b * stride_wb + pid_h * stride_wh
    m_base = MASK + pid_b * stride_mb + pid_h * stride_mh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n0 = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)

    q_mask = offs_m < Q_LEN
    d_mask = offs_d < HEAD_DIM
    dv_mask = offs_dv < HEAD_DIM_V

    # Keep Q/K/V in their native dtype so tl.dot uses the matrix cores; only
    # the softmax statistics and accumulators run in fp32.
    query = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    # fp32 inputs have no matrix-core fp32 dot; run the dots on fp16 operands
    # (exact fp16*fp16 products with fp32 accumulation) while keeping the
    # scores, softmax statistics and outputs in fp32.
    if DOT_FP16:
        query_dot = query.to(tl.float16)
    else:
        query_dot = query

    if SINGLE_PASS:
        # The whole key sequence fits in one tile: compute the scores once,
        # normalize with the exact max/sum softmax and consume them for both
        # the returned weights and the P@V product.
        offs_n = offs_n0
        kv_mask = offs_n < KV_LEN
        key = tl.load(
            k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        if DOT_FP16:
            qk = tl.dot(query_dot, tl.trans(key.to(tl.float16)))
        else:
            qk = tl.dot(query_dot, tl.trans(key))
        qk = (
            (qk.to(Q.dtype.element_ty) * sm_scale).to(Q.dtype.element_ty).to(tl.float32)
        )

        if HAS_MASK:
            m_ptrs = m_base + offs_m[:, None] * stride_mm + offs_n[None, :] * stride_mn
            mk = tl.load(m_ptrs, mask=q_mask[:, None] & kv_mask[None, :], other=0)
            if MASK_BOOL:
                qk = tl.where(mk, -float("inf"), qk)
            else:
                qk += mk.to(tl.float32)

        qk = tl.where(kv_mask[None, :], qk, -float("inf"))
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))

        row_max = tl.max(qk, 1)
        row_max = tl.where(row_max == -float("inf"), 0.0, row_max)
        p = tl.exp2((qk - row_max[:, None]) * LOG2E)
        row_sum = tl.sum(p, 1)
        p = p * (1.0 / tl.where(row_sum == 0.0, 1.0, row_sum))[:, None]

        w_ptrs = w_base + offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn
        tl.store(
            w_ptrs,
            p.to(W.dtype.element_ty),
            mask=q_mask[:, None] & kv_mask[None, :],
        )

        value = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd,
            mask=kv_mask[:, None] & dv_mask[None, :],
            other=0.0,
        )
        if DOT_FP16:
            acc = tl.dot(p.to(tl.float16), value.to(tl.float16))
        else:
            acc = tl.dot(p.to(value.dtype), value)

        o_ptrs = o_base + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od
        tl.store(
            o_ptrs,
            acc.to(Out.dtype.element_ty),
            mask=q_mask[:, None] & dv_mask[None, :],
        )
        return

    if IS_CAUSAL:
        # Top-left causal masking (j <= i), matching tril(diagonal=0).
        hi = tl.minimum((pid_m + 1) * BLOCK_M, KV_LEN)
    else:
        hi = KV_LEN

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Pass 1: online softmax statistics (no V traffic, no P writes).
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + offs_n0
        kv_mask = offs_n < KV_LEN

        key = tl.load(
            k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        # Match the reference dtype flow: the QK product is materialized in the
        # input dtype and then scaled, so both roundings are reproduced here.
        if DOT_FP16:
            qk = tl.dot(query_dot, tl.trans(key.to(tl.float16)))
        else:
            qk = tl.dot(query_dot, tl.trans(key))
        qk = (
            (qk.to(Q.dtype.element_ty) * sm_scale).to(Q.dtype.element_ty).to(tl.float32)
        )

        if HAS_MASK:
            m_ptrs = m_base + offs_m[:, None] * stride_mm + offs_n[None, :] * stride_mn
            load_mask = q_mask[:, None] & kv_mask[None, :]
            mk = tl.load(m_ptrs, mask=load_mask, other=0)
            if MASK_BOOL:
                qk = tl.where(mk, -float("inf"), qk)
            else:
                qk += mk.to(tl.float32)

        qk = tl.where(kv_mask[None, :], qk, -float("inf"))
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp2((qk - m_safe[:, None]) * LOG2E)
        alpha = tl.exp2((m_i - m_safe) * LOG2E)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_safe

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    inv_l = 1.0 / l_safe

    # Pass 2: recompute scores, materialize normalized weights, accumulate PV.
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + offs_n0
        kv_mask = offs_n < KV_LEN

        key = tl.load(
            k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        if DOT_FP16:
            qk = tl.dot(query_dot, tl.trans(key.to(tl.float16)))
        else:
            qk = tl.dot(query_dot, tl.trans(key))
        qk = (
            (qk.to(Q.dtype.element_ty) * sm_scale).to(Q.dtype.element_ty).to(tl.float32)
        )

        if HAS_MASK:
            m_ptrs = m_base + offs_m[:, None] * stride_mm + offs_n[None, :] * stride_mn
            load_mask = q_mask[:, None] & kv_mask[None, :]
            mk = tl.load(m_ptrs, mask=load_mask, other=0)
            if MASK_BOOL:
                qk = tl.where(mk, -float("inf"), qk)
            else:
                qk += mk.to(tl.float32)

        qk = tl.where(kv_mask[None, :], qk, -float("inf"))
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))

        p = tl.exp2((qk - m_i[:, None]) * LOG2E) * inv_l[:, None]

        w_ptrs = w_base + offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn
        tl.store(
            w_ptrs,
            p.to(W.dtype.element_ty),
            mask=q_mask[:, None] & kv_mask[None, :],
        )

        value = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd,
            mask=kv_mask[:, None] & dv_mask[None, :],
            other=0.0,
        )
        if DOT_FP16:
            acc = tl.dot(p.to(tl.float16), value.to(tl.float16), acc)
        else:
            acc = tl.dot(p.to(value.dtype), value, acc)

    if IS_CAUSAL:
        # Zero-fill the key blocks that are fully masked for every row here.
        zero = tl.zeros([BLOCK_M, BLOCK_N], dtype=W.dtype.element_ty)
        for start_n in range(hi, KV_LEN, BLOCK_N):
            offs_n = start_n + offs_n0
            kv_mask = offs_n < KV_LEN
            w_ptrs = w_base + offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn
            tl.store(
                w_ptrs,
                zero,
                mask=q_mask[:, None] & kv_mask[None, :],
            )

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od
    tl.store(
        o_ptrs,
        acc.to(Out.dtype.element_ty),
        mask=q_mask[:, None] & dv_mask[None, :],
    )


def _tile_config(q_len, kv_len, head_dim, head_dim_v, n_bh, elem_size):
    block_d = max(16, triton.next_power_of_2(head_dim))
    block_dv = max(16, triton.next_power_of_2(head_dim_v))
    wide = max(block_d, block_dv)

    # The per-CTA serial chain is ceil(kv_len/BLOCK_N) iterations per pass, so
    # make BLOCK_N as wide as the 64 KiB shared-memory budget allows.  The
    # (bn_cap, s2_max) pairs come from measured shared-memory usage: BLOCK_N up
    # to s2_max still fits with double buffering, wider tiles need one stage.
    if elem_size == 2:
        bn_cap, s2_max = (256, 128) if wide <= 64 else (128, 64)
    else:
        bn_cap, s2_max = (128, 64) if wide <= 64 else (64, 32)
    kv_pow2 = max(16, triton.next_power_of_2(kv_len))
    block_n = min(kv_pow2, bn_cap)
    num_stages = 2 if block_n <= s2_max else 1

    # One CTA per CU is all that fits (VGPR-limited), so the makespan is the
    # wave count times the per-CTA serial chain.  The chain length depends on
    # ceil(kv_len/BLOCK_N) iterations, not on BLOCK_M, so pick the smallest
    # BLOCK_M whose grid still fits a single CTA wave of num_sm programs.
    num_sm = 80
    max_block_m = max(16, min(128, 8192 // block_dv))
    block_m = min(64, max_block_m)
    for candidate in (16, 32, 64, 128):
        if candidate > max_block_m:
            break
        if n_bh * triton.cdiv(q_len, candidate) <= num_sm:
            block_m = candidate
            break
    if block_m > q_len:
        block_m = max(16, triton.next_power_of_2(q_len))
    if elem_size == 4 and wide > 64:
        block_m = min(block_m, 32)

    # Wide score tiles spread over 8 warps to keep the fp32 register tile small.
    num_warps = 8 if block_m * block_n >= 16384 else 4

    return block_m, block_n, block_d, block_dv, num_warps, num_stages


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
    logger.debug("GEMS_HYGON _SCALED_DOT_PRODUCT_ATTENTION_MATH_FOR_MPS")
    if dropout_p != 0.0:
        raise AssertionError("Only dropout_p=0.0 is supported")
    if dropout_mask is not None:
        raise AssertionError("dropout_mask is not supported")

    squeeze = query.dim() == 3
    if squeeze:
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)

    B, H, q_len, head_dim = query.shape
    kv_len = key.shape[-2]
    heads_kv = key.shape[1]
    head_dim_v = value.shape[-1]

    group = 1
    if enable_gqa and heads_kv != H:
        group = H // heads_kv

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    else:
        scale = float(scale)

    out = torch.empty((B, H, q_len, head_dim_v), dtype=query.dtype, device=query.device)
    weights = torch.empty((B, H, q_len, kv_len), dtype=query.dtype, device=query.device)

    has_mask = attn_mask is not None
    if has_mask:
        mask4 = attn_mask.expand(B, H, q_len, kv_len)
        mask_bool = attn_mask.dtype == torch.bool
        m_sb, m_sh, m_sm, m_sn = mask4.stride()
        mask_arg = mask4
    else:
        mask_bool = False
        m_sb = m_sh = m_sm = m_sn = 0
        mask_arg = query

    block_m, block_n, block_d, block_dv, num_warps, num_stages = _tile_config(
        q_len, kv_len, head_dim, head_dim_v, B * H, query.element_size()
    )

    grid = (triton.cdiv(q_len, block_m), H, B)
    _sdpa_math_kernel[grid](
        query,
        key,
        value,
        mask_arg,
        out,
        weights,
        scale,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        m_sb,
        m_sh,
        m_sm,
        m_sn,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        weights.stride(3),
        q_len,
        kv_len,
        group,
        HEAD_DIM=head_dim,
        HEAD_DIM_V=head_dim_v,
        BLOCK_D=block_d,
        BLOCK_DV=block_dv,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        IS_CAUSAL=bool(is_causal),
        HAS_MASK=has_mask,
        MASK_BOOL=mask_bool,
        SINGLE_PASS=kv_len <= block_n,
        DOT_FP16=query.dtype == torch.float32,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if squeeze:
        return out.squeeze(0), weights.squeeze(0)
    return out, weights
