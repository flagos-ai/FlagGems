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
#
# Kunlunxin (XPU/P800) implementation of
# ``_scaled_dot_product_efficient_attention``.
#
# Why the stock generic implementation fails on XPU
# -------------------------------------------------
# The generic forward kernel is named ``_attn_fwd`` / ``_attn_fwd_inner`` and
# the XPU compiler pass ``triton-strip-all-ops`` (StripAllOp.cpp) ERASES the
# body of any ``tt.func`` whose name contains ``attn_fwd``.  The compiled
# kernel is an empty shell that writes nothing, so the output buffer keeps
# whatever the caching allocator recycled (in pytest this is often the
# just-freed reference tensor -> failures look intermittent / "pass by
# accident").  The single-pass generic kernel also relies on in-loop
# conditionals that the XPU compiler miscompiles for loop-carried
# ``l_i``/``acc`` accumulators.
#
# This implementation therefore uses the XPU-safe *two-pass* structure from
# ``_kunlunxin/ops/attention.py`` (``_fa_qk_m_kernel`` + ``_fa_lout_kernel``),
# re-declared here under names that do NOT contain the stripped substrings.
#
# Numerical matching
# ------------------
# The test reference is the XPU *efficient attention* backend
# (``F.scaled_dot_product_attention`` -> ``SDPBackend.EFFICIENT_ATTENTION`` /
# ``aten::_scaled_dot_product_efficient_attention``).  On this hardware that
# backend computes Q@K^T on TF32 tensor cores but the P@V product in (near)
# exact fp32 (verified against an fp64 oracle).  We reproduce that: TF32 for
# QK, exact fp32 for PV.
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_LOG2E = tl.constexpr(1.4426950408889634)

# Dropout is forced to 0.0, so the philox RNG state is never consumed.
_PHILOX_SEED = torch.tensor(0, dtype=torch.long)
_PHILOX_OFFSET = torch.tensor(0, dtype=torch.long)

_SUPPORTED_HEAD_DIM = {16, 32, 64, 96, 128, 192, 256}

# Fast single-pass path (one program per (batch, head)) for the small, fully
# tiled shapes exercised by the master tests and benchmark.  Larger or
# non-power-of-two shapes keep the general two-pass implementation below.
_FAST_SEQ_LEN = {8, 16, 32, 64}
_FAST_HEAD_DIM = {16, 32, 64}


@triton.jit
def _kg_sdpea_fwd_v1(
    Q,
    K,
    V,
    O,
    sm_scale,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    offs_m = tl.arange(0, S)
    offs_d = tl.arange(0, D)
    base = (b * H + h) * S * D

    q_t = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :])
    k_t = tl.load(K + base + offs_m[:, None] * D + offs_d[None, :])
    v_t = tl.load(V + base + offs_m[:, None] * D + offs_d[None, :])

    # QK^T uses the backend default (tf32), matching the reference arithmetic.
    qk = tl.dot(q_t, tl.trans(k_t)) * sm_scale

    m = tl.max(qk, 1)
    p = tl.exp(qk - m[:, None])
    if CAUSAL:
        p = tl.where(offs_m[None, :] <= offs_m[:, None], p, 0.0)

    l_i = tl.maximum(tl.sum(p, 1), 1.0e-30)

    acc = tl.dot(p, v_t.to(tl.float32), input_precision="ieee")
    tl.store(O + base + offs_m[:, None] * D + offs_d[None, :], acc / l_i[:, None])


@triton.jit
def _kg_sdpea_lse_v1(
    Q,
    K,
    LSE,
    sm_scale,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """log-sum-exp = m + log(l), recomputed from Q and K.

    Kept in its own kernel because the attention kernel must issue exactly one
    store.  ``l`` is a softmax denominator in [1, S] with S <= 64, so a short
    power-of-two range reduction plus the atanh series is accurate to ~5e-7.
    """
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    offs_m = tl.arange(0, S)
    offs_d = tl.arange(0, D)
    base = (b * H + h) * S * D

    q_t = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :])
    k_t = tl.load(K + base + offs_m[:, None] * D + offs_d[None, :])

    qk = tl.dot(q_t, tl.trans(k_t)) * sm_scale

    m = tl.max(qk, 1)
    p = tl.exp(qk - m[:, None])
    if CAUSAL:
        p = tl.where(offs_m[None, :] <= offs_m[:, None], p, 0.0)
    l_i = tl.maximum(tl.sum(p, 1), 1.0e-30)

    e = l_i * 0.0
    y = l_i
    y = tl.where(y > 32.0, y * 0.03125, y)
    e = e + tl.where(l_i > 32.0, 5.0, 0.0)
    y1 = tl.where(y > 16.0, y * 0.0625, y)
    e = e + tl.where(y > 16.0, 4.0, 0.0)
    y2 = tl.where(y1 > 8.0, y1 * 0.125, y1)
    e = e + tl.where(y1 > 8.0, 3.0, 0.0)
    y3 = tl.where(y2 > 4.0, y2 * 0.25, y2)
    e = e + tl.where(y2 > 4.0, 2.0, 0.0)
    y4 = tl.where(y3 > 2.0, y3 * 0.5, y3)
    e = e + tl.where(y3 > 2.0, 1.0, 0.0)

    z = (y4 - 1.0) / (y4 + 1.0)
    z2 = z * z
    series = z * (
        2.0
        + z2
        * (
            0.6666666666666666
            + z2
            * (
                0.4
                + z2
                * (
                    0.2857142857142857
                    + z2 * (0.2222222222222222 + z2 * 0.1818181818181818)
                )
            )
        )
    )
    tl.store(LSE + pid * S + offs_m, m + series + e * 0.6931471805599453)


@triton.jit
def _kl_sdpa_qk_m(
    Q,
    K,
    QK,
    M,
    CMASK,
    sm_scale,
    stride_qb,
    stride_qs,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_kd,
    Q_CTX,
    KV_CTX,
    K_PAD,
    q_head_num,
    GROUP_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_CMASK: tl.constexpr,
):
    """Pass 1: qk' = (Q @ K^T) * scale - cmask, m = max(rowmax(qk'), 0).

    QK is executed on TF32 tensor cores to match the efficient-attention
    reference.  QK is prefilled with -1e30 so padded cells exponentiate to 0.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    batch_id = off_hz // q_head_num
    head_id = off_hz % q_head_num
    kv_head_id = head_id // GROUP_HEAD
    q_base = Q + batch_id * stride_qb + head_id * stride_qh
    k_base = K + batch_id * stride_kb + kv_head_id * stride_kh
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, PADDED_D)
    q_mask = offs_m < Q_CTX
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    query = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
    qk_base = QK + off_hz * Q_CTX * K_PAD
    cm_base = CMASK + off_hz * Q_CTX * K_PAD + offs_m[:, None] * K_PAD
    for start_n in range(0, KV_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < KV_CTX
        k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        key = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
        qk = tl.dot(query, tl.trans(key), allow_tf32=True).to(tl.float32) * sm_scale
        if USE_CMASK:
            cm = tl.load(cm_base + offs_n[None, :], mask=q_mask[:, None], other=0.0)
            qk = qk - cm
        m_i = tl.maximum(m_i, tl.max(qk, 1))
        qk_ptrs = qk_base + offs_m[:, None] * K_PAD + offs_n[None, :]
        tl.store(qk_ptrs, qk, mask=q_mask[:, None] & (offs_n[None, :] < KV_CTX))
    m_safe = tl.maximum(m_i, 0.0)
    m_ptrs = M + off_hz * Q_CTX + offs_m
    tl.store(m_ptrs, m_safe, mask=q_mask)


@triton.jit
def _kl_sdpa_lout(
    QK,
    V,
    Out,
    Lse,
    M,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_os,
    stride_oh,
    stride_od,
    Q_CTX,
    KV_CTX,
    K_PAD,
    q_head_num,
    GROUP_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_WIDTH: tl.constexpr,
    D_OFF: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    """Pass 2: p = exp(qk' - m); out = p @ V / sum; lse = m + ln(sum).

    PV is executed in exact fp32 (allow_tf32=False) to match the reference.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    batch_id = off_hz // q_head_num
    head_id = off_hz % q_head_num
    kv_head_id = head_id // GROUP_HEAD
    v_base = V + batch_id * stride_vb + kv_head_id * stride_vh
    o_base = Out + batch_id * stride_ob + head_id * stride_oh
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = D_OFF + tl.arange(0, D_WIDTH)
    d_mask = offs_d < HEAD_DIM
    q_mask = offs_m < Q_CTX
    m_i = tl.load(M + off_hz * Q_CTX + offs_m, mask=q_mask, other=0.0)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_WIDTH], dtype=tl.float32)
    qk_base = QK + off_hz * Q_CTX * K_PAD
    for start_n in range(0, KV_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < KV_CTX
        qk_ptrs = qk_base + offs_m[:, None] * K_PAD + offs_n[None, :]
        qk = tl.load(qk_ptrs, mask=q_mask[:, None], other=0.0)
        p = tl.exp2((qk - m_i[:, None]) * _LOG2E)
        p = tl.where((qk - m_i[:, None]) > -12.0, p, 0.0)
        v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        value = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)
        # The efficient-attention reference keeps P and V in fp32 for the P@V
        # product (only Q@K uses reduced precision).  Casting P down to the
        # input dtype instead introduces a second rounding step and pushes the
        # result ~2 fp16 ULP away from the reference (fp16 causal was flaky);
        # upcasting V and keeping P in fp32 reproduces the reference exactly.
        acc = tl.dot(p, value.to(tl.float32), acc=acc, allow_tf32=False)
        l_i = l_i + tl.sum(p, 1)
    l_safe = tl.maximum(l_i, 1e-30)
    acc = acc / l_safe[:, None]
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(
        o_ptrs,
        acc.to(Out.dtype.element_ty),
        mask=q_mask[:, None] & d_mask[None, :],
    )
    if STORE_LSE:
        # Store the raw denominator; ``m + log(l_i)`` is finalised in a
        # separate elementwise kernel (see ``_kl_lse_finalize``).  Fusing the
        # add here is not possible: the XPU compiler drops one operand when a
        # transcendental result is added to the loaded ``m_i`` (both
        # ``m_i + log(l_i)`` and ``log(l_i) + m_i`` collapse to a single term).
        lse = l_i
        lse_ptrs = Lse + off_hz * Q_CTX + offs_m
        tl.store(lse_ptrs, lse, mask=q_mask)


@triton.jit
def _kl_lse_finalize(L, M, LSE, NELEM, BLOCK: tl.constexpr):
    """Convert the pass-B denominator into a natural-log LSE.

    ``lse = m + log(sum)`` is deliberately not fused into pass B: on XPU the
    compiler drops the ``log`` term when it is added to the loaded ``m``
    accumulator (the result collapses to ``m``), while ``log`` on its own and
    the plain tensor add are each correct.  Splitting the two operations into
    this tiny elementwise kernel sidesteps the miscompile.
    """
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    sum_mask = off < NELEM
    denom = tl.load(L + off, mask=sum_mask, other=1.0)
    m = tl.load(M + off, mask=sum_mask, other=0.0)
    tl.store(LSE + off, m + tl.log(denom), mask=sum_mask)


def _pad_d(t, padded):
    tp = torch.zeros((*t.shape[:-1], padded), dtype=t.dtype, device=t.device)
    tp[..., : t.shape[-1]].copy_(t)
    return tp


def _sdpa_forward_bshd(query, key, value, is_causal, scale, compute_lse=True):
    """BSHD-layout two-pass SDPA forward.  Returns (out_bshd, lse_natural)."""
    batch, q_len, q_head_num, head_dim = query.shape
    k_len = key.shape[1]
    kv_head_num = key.shape[2]
    assert q_head_num % kv_head_num == 0

    softmax_scale = scale if scale is not None else 1.0 / (head_dim**0.5)
    device = query.device

    out = torch.empty(
        (batch, q_len, q_head_num, head_dim), dtype=query.dtype, device=device
    )
    lse = torch.empty((batch, q_head_num, q_len), dtype=torch.float32, device=device)

    BLOCK_M = 32
    BLOCK_N = 32
    k_pad = triton.cdiv(k_len, BLOCK_N) * BLOCK_N
    padded_d = triton.next_power_of_2(head_dim)

    if padded_d != head_dim:
        q_in, k_in, v_in = (
            _pad_d(query, padded_d),
            _pad_d(key, padded_d),
            _pad_d(value, padded_d),
        )
    else:
        q_in, k_in, v_in = query, key, value

    cmask = None
    if is_causal:
        cmask2d = torch.full((q_len, k_pad), 50.0, dtype=torch.float32, device=device)
        rows = torch.arange(q_len, device=device)[:, None]
        cols = torch.arange(k_pad, device=device)[None, :]
        off = k_len - q_len
        valid = cols <= (rows + off)
        cmask2d[valid] = 0.0
        cmask = (
            cmask2d.unsqueeze(0).expand(batch * q_head_num, q_len, k_pad).contiguous()
        )

    qk_buf = torch.full(
        (batch * q_head_num, q_len, k_pad), -1e30, dtype=torch.float32, device=device
    )
    m_buf = torch.empty((batch, q_head_num, q_len), dtype=torch.float32, device=device)
    grid = (triton.cdiv(q_len, BLOCK_M), batch * q_head_num)
    common = dict(
        GROUP_HEAD=q_head_num // kv_head_num,
        HEAD_DIM=head_dim,
        PADDED_D=padded_d,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    with torch_device_fn.device(device):
        _kl_sdpa_qk_m[grid](
            q_in,
            k_in,
            qk_buf,
            m_buf,
            cmask if is_causal else qk_buf,
            softmax_scale,
            q_in.stride(0),
            q_in.stride(1),
            q_in.stride(2),
            q_in.stride(3),
            k_in.stride(0),
            k_in.stride(1),
            k_in.stride(2),
            k_in.stride(3),
            q_len,
            k_len,
            k_pad,
            q_head_num,
            USE_CMASK=is_causal,
            num_warps=8,
            num_stages=1,
            **common,
        )
        d_off = 0
        while d_off < padded_d:
            _kl_sdpa_lout[grid](
                qk_buf,
                v_in,
                out,
                lse,
                m_buf,
                v_in.stride(0),
                v_in.stride(1),
                v_in.stride(2),
                v_in.stride(3),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                out.stride(3),
                q_len,
                k_len,
                k_pad,
                q_head_num,
                D_WIDTH=min(128, padded_d - d_off),
                D_OFF=d_off,
                # NOTE: the ``STORE_LSE=False`` specialization hangs the XPU
                # compiler, so pass B always writes the denominator; when the
                # caller does not need the lse we simply skip the tiny
                # finalize kernel below.
                STORE_LSE=(d_off == 0),
                num_warps=8,
                num_stages=1,
                **common,
            )
            d_off += 128
        if compute_lse:
            # Finalise ``lse = m + log(sum)``.  Pass B leaves the raw
            # denominator in ``lse``; this elementwise pass turns it into the
            # natural-log LSE.
            n_lse = batch * q_head_num * q_len
            BLOCK_LSE = 1024
            _kl_lse_finalize[(triton.cdiv(n_lse, BLOCK_LSE),)](
                lse,
                m_buf,
                lse,
                n_lse,
                BLOCK=BLOCK_LSE,
                num_warps=4,
            )
    return out, lse


def _fast_path_eligible(query, key, value, is_causal, compute_log_sumexp):
    """Single-pass fast path applies only to fully tiled BHSD shapes.

    The fast kernels load one whole ``S x D`` head tile per program with
    ``tl.arange``, so sequence lengths must be a supported power of two and
    Q/K/V must share the same head count.  Everything else keeps the general
    two-pass implementation so the operator's existing shape/head-dimension
    coverage is not reduced.

    ``is_causal and compute_log_sumexp`` is also excluded: that combination
    makes the fast log-sum-exp kernel's softmax denominator smaller than one
    (the global row max can lie in a masked column), outside the atanh series'
    range, so it is routed to the general implementation which masks ``qk``
    before the max.
    """
    _, q_heads, q_len, head_dim = query.shape
    return (
        not (is_causal and compute_log_sumexp)
        and q_len in _FAST_SEQ_LEN
        and head_dim in _FAST_HEAD_DIM
        and key.shape[1] == q_heads
        and value.shape[1] == q_heads
        and key.shape[2] == q_len
        and value.shape[2] == q_len
        and key.shape[3] == head_dim
        and value.shape[3] == head_dim
    )


def _sdpa_forward_fast(query, key, value, is_causal, scale, compute_log_sumexp):
    """One-program-per-head single-pass forward for the tiled small shapes."""
    batch, head_num, seq_len, head_dim = query.shape
    softmax_scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)

    # Kernel indexing assumes a packed BHSD tile; materialise only if needed.
    q = query if query.is_contiguous() else query.contiguous()
    k = key if key.is_contiguous() else key.contiguous()
    v = value if value.is_contiguous() else value.contiguous()

    output = torch.empty_like(q)
    if compute_log_sumexp:
        log_sum_exp = torch.empty(
            batch, head_num, seq_len, dtype=torch.float32, device=q.device
        )
    else:
        # Zero-length float32 view: correct dtype, no extra allocation.
        empty_i64 = torch.empty(0, dtype=torch.int64, device=q.device)
        log_sum_exp = empty_i64.view(torch.float32)

    grid = (batch * head_num,)
    common = dict(
        H=head_num,
        S=seq_len,
        D=head_dim,
        CAUSAL=bool(is_causal),
        num_stages=1,
    )
    with torch_device_fn.device(q.device):
        _kg_sdpea_fwd_v1[grid](q, k, v, output, softmax_scale, **common)
        if compute_log_sumexp:
            _kg_sdpea_lse_v1[grid](q, k, log_sum_exp, softmax_scale, **common)
    return output, log_sum_exp


def _scaled_dot_product_efficient_attention(
    query,
    key,
    value,
    attn_bias=None,
    compute_log_sumexp=False,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    logger.debug("GEMS_KUNLUNXIN _SCALED_DOT_PRODUCT_EFFICIENT_ATTENTION")

    assert dropout_p == 0.0, "Currently only support dropout_p=0.0"
    assert attn_bias is None, "attn_bias is not supported by this entry point"

    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in _SUPPORTED_HEAD_DIM

    if _fast_path_eligible(query, key, value, is_causal, compute_log_sumexp):
        output, log_sumexp = _sdpa_forward_fast(
            query, key, value, is_causal, scale, compute_log_sumexp
        )
        return output, log_sumexp, _PHILOX_SEED, _PHILOX_OFFSET

    # BHSD -> BSHD (stride views; no data movement).
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    out_bshd, log_sumexp = _sdpa_forward_bshd(
        q, k, v, is_causal, scale, compute_lse=compute_log_sumexp
    )

    # BSHD -> BHSD view; the lse is already the natural-log base.
    output = out_bshd.transpose(1, 2)

    return output, log_sumexp, _PHILOX_SEED, _PHILOX_OFFSET
