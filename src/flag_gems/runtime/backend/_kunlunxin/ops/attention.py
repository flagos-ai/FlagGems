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
import os

import torch

# import torch.nn.functional as F
import triton
import triton.language as tl

from flag_gems.config import use_c_extension
from flag_gems.runtime import torch_device_fn

from .flash_api import mha_varlan_fwd

logger = logging.getLogger(__name__)


# Modified from Triton tutorial: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    query,  #
    K_block_ptr,
    V_block_ptr,  #
    mask_block_ptr,  #
    stride_k_seqlen,
    stride_v_seqlen,
    stride_attn_mask_kv_seqlen,  #
    start_m,
    qk_scale,  #
    q_load_mask,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    KV_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    ACCUM_TYPE: tl.constexpr,
    GEMM_TYPE: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    # causal = False
    else:
        lo, hi = 0, KV_CTX

    K_block_ptr += lo * stride_k_seqlen
    V_block_ptr += lo * stride_v_seqlen
    if HAS_ATTN_MASK:
        mask_block_ptr += lo * stride_attn_mask_kv_seqlen

    LOG2E = 1.44269504  # log2(e) constant

    # loop over key, value and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        kv_load_mask = (start_n + offs_n) < KV_CTX
        # start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        key = tl.load(K_block_ptr, mask=kv_load_mask[None, :], other=0.0)
        if PRE_LOAD_V:
            value = tl.load(V_block_ptr, mask=kv_load_mask[:, None], other=0.0)

        qk = tl.dot(query, key, allow_tf32=(GEMM_TYPE == "tf32"))
        # incase not divisible.
        qk = tl.where(kv_load_mask[None, :], qk, -float("inf"))
        # qk = qk.to(tl.float32)

        if HAS_ATTN_MASK:
            attn_mask = tl.load(
                mask_block_ptr,
                mask=q_load_mask[:, None] & kv_load_mask[None, :],
                other=0.0,
            )

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])

            if HAS_ATTN_MASK:
                qk = qk * qk_scale + attn_mask
                qk *= LOG2E
                qk = qk + tl.where(mask, 0, -1.0e6)
            else:
                qk = qk * qk_scale * LOG2E + tl.where(mask, 0, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            qk *= qk_scale * LOG2E
            if HAS_ATTN_MASK:
                qk = qk + attn_mask
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        if not PRE_LOAD_V:
            value = tl.load(V_block_ptr, mask=kv_load_mask[:, None], other=0.0)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(query.dtype)
        p = p.to(value.dtype)
        acc = tl.dot(p, value, acc, allow_tf32=(GEMM_TYPE == "tf32"))
        # update m_i and l_i
        m_i = m_ij

        K_block_ptr += BLOCK_N * stride_k_seqlen
        V_block_ptr += BLOCK_N * stride_v_seqlen

        if HAS_ATTN_MASK:
            mask_block_ptr += BLOCK_N * stride_attn_mask_kv_seqlen

    return acc, l_i, m_i


# NOTE: we assert BLOCK_N <= HEAD_DIM in _attn_fwd, so for small head_dim,
# we need to generate more configs.
# [kunlunxin] single-config: on this backend the attention kernel body compiles
# to a launch-table stub, so multi-config autotuning would benchmark the stub on
# every new (KV_CTX, HEAD_DIM) key and fold ~100 ms into the first call. Only
# BLOCK_M is consumed here (it drives the grid).
configs = [
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 16, "PRE_LOAD_V": 0}, num_stages=2, num_warps=4
    )
]
SMALL_HEAD_DIM_CONFIGS = [
    triton.Config(
        {"BLOCK_M": BM, "BLOCK_N": BN, "PRE_LOAD_V": 0}, num_stages=s, num_warps=w
    )
    for BM in [64, 128]
    for BN in [16, 32]
    for s in [2, 3, 4]
    for w in [4, 8]
]
# configs += SMALL_HEAD_DIM_CONFIGS  # disabled: single-config mode (see above)


def _prune_sdpa_cfg(configs, nargs, **kwargs):
    hd = nargs.get("HEAD_DIM")
    if hd is None:
        return configs
    return [c for c in configs if c.kwargs["BLOCK_N"] <= hd]


@triton.autotune(
    configs=configs,  # single-config (see above)
    key=["KV_CTX", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": _prune_sdpa_cfg},
)
@triton.jit(
    do_not_specialize=[
        "stride_q_batch",
        "stride_q_head",
        "stride_q_seqlen",
        "stride_k_batch",
        "stride_k_head",
        "stride_k_seqlen",
        "stride_v_batch",
        "stride_v_head",
        "stride_v_seqlen",
        "stride_o_batch",
        "stride_o_head",
        "stride_o_seqlen",
        "Z",
        "q_head_num",
        "kv_head_num",
        "Q_CTX",
        "KV_CTX",
    ]
)
def _attn_fwd(
    Q,
    K,
    V,
    attn_mask,
    sm_scale,
    M,
    Out,  #
    stride_q_batch,
    stride_q_head,
    stride_q_seqlen,
    stride_q_headsize,
    stride_k_batch,
    stride_k_head,
    stride_k_seqlen,
    stride_k_headsize,
    stride_v_batch,
    stride_v_head,
    stride_v_seqlen,
    stride_v_headsize,
    stride_attn_mask_batch,
    stride_attn_mask_head,
    stride_attn_mask_q_seqlen,
    stride_attn_mask_kv_seqlen,
    stride_o_batch,
    stride_o_head,
    stride_o_seqlen,
    stride_o_headsize,
    Z,
    q_head_num,
    kv_head_num,
    GROUP_HEAD: tl.constexpr,
    Q_CTX,
    KV_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    ACCUM_TYPE: tl.constexpr,
    GEMM_TYPE: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    batch_id = off_hz // q_head_num
    head_id = off_hz % q_head_num
    kv_head_id = head_id // GROUP_HEAD

    q_offset = (
        batch_id.to(tl.int64) * stride_q_batch + head_id.to(tl.int64) * stride_q_head
    )
    o_offset = (
        batch_id.to(tl.int64) * stride_o_batch + head_id.to(tl.int64) * stride_o_head
    )
    kv_offset = (
        batch_id.to(tl.int64) * stride_k_batch + kv_head_id.to(tl.int64) * stride_k_head
    )

    offs_headsize = tl.arange(0, HEAD_DIM)

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_load_mask = offs_m < Q_CTX
    offs_n = tl.arange(0, BLOCK_N)

    Q_block_ptr = (
        Q
        + q_offset
        + offs_m[:, None] * stride_q_seqlen
        + offs_headsize[None, :] * stride_q_headsize
    )
    K_block_ptr = (
        K
        + kv_offset
        + offs_n[None, :] * stride_k_seqlen
        + offs_headsize[:, None] * stride_k_headsize
    )
    V_block_ptr = (
        V
        + kv_offset
        + offs_n[:, None] * stride_v_seqlen
        + offs_headsize[None, :] * stride_v_headsize
    )

    if HAS_ATTN_MASK:
        attn_mask_offset = (
            batch_id.to(tl.int64) * stride_attn_mask_batch
            + head_id.to(tl.int64) * stride_attn_mask_head
        )
        mask_block_ptr = (
            attn_mask
            + attn_mask_offset
            + offs_m[:, None] * stride_attn_mask_q_seqlen
            + offs_n[None, :] * stride_attn_mask_kv_seqlen
        )
    else:
        mask_block_ptr = None

    O_block_ptr = (
        Out
        + o_offset
        + offs_m[:, None] * stride_o_seqlen
        + offs_headsize[None, :] * stride_o_headsize
    )

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros(
        [BLOCK_M, HEAD_DIM], dtype=tl.float32 if ACCUM_TYPE == "float" else tl.float16
    )
    # load scales
    qk_scale = sm_scale
    # qk_scale *= 1.44269504  # 1/log(2)
    # load query: it will stay in SRAM throughout
    query = tl.load(Q_block_ptr, mask=q_load_mask[:, None], other=0.0)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            query,
            K_block_ptr,
            V_block_ptr,
            mask_block_ptr,
            stride_k_seqlen,
            stride_v_seqlen,
            stride_attn_mask_kv_seqlen,
            start_m,
            qk_scale,
            q_load_mask,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            4 - STAGE,
            offs_m,
            offs_n,
            KV_CTX,
            V.dtype.element_ty == tl.float8e5,
            HAS_ATTN_MASK,
            PRE_LOAD_V,
            ACCUM_TYPE,
            GEMM_TYPE,
        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            query,
            K_block_ptr,
            V_block_ptr,
            mask_block_ptr,
            stride_k_seqlen,
            stride_v_seqlen,
            stride_attn_mask_kv_seqlen,
            start_m,
            qk_scale,
            q_load_mask,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            2,
            offs_m,
            offs_n,
            KV_CTX,
            V.dtype.element_ty == tl.float8e5,
            HAS_ATTN_MASK,
            PRE_LOAD_V,
            ACCUM_TYPE,
            GEMM_TYPE,
        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * Q_CTX + offs_m
    tl.store(m_ptrs, m_i, mask=q_load_mask)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=q_load_mask[:, None])


# XPU backward is staged to keep multiple dot accumulators out of a single loop.
_STAGED_RED = 64
_STAGED_BLOCK_N = 8
_STAGED_BLOCK_D = 4
_STAGED_GRAD_R = 8


@triton.jit
def prob_dp_partial_kernel(
    Q,
    K,
    V,
    DO,
    SCORE_PARTIAL,
    DP_PARTIAL,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    D: tl.constexpr,
    D_CHUNKS: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N_: tl.constexpr,
    RED_: tl.constexpr,
):
    n_offs = tl.program_id(0) * BLOCK_N_ + tl.arange(0, BLOCK_N_)
    q_idx = tl.program_id(1)
    packed = tl.program_id(2)
    d_chunk = packed % D_CHUNKS
    query_bh = packed // D_CHUNKS
    batch_idx = query_bh // QUERY_HEADS
    query_head = query_bh % QUERY_HEADS
    kv_bh = batch_idx * KV_HEADS + query_head // GROUP_SIZE
    n_mask = n_offs < KV_LEN
    n_safe = tl.where(n_mask, n_offs, 0)
    score = tl.zeros((BLOCK_N_,), dtype=tl.float32)
    dp = tl.zeros((BLOCK_N_,), dtype=tl.float32)
    for d_offset in tl.static_range(RED_):
        d_idx = d_chunk * RED_ + d_offset
        if d_idx < D:
            q_value = tl.load(Q + (query_bh * Q_LEN + q_idx) * D + d_idx)
            do_value = tl.load(DO + (query_bh * Q_LEN + q_idx) * D + d_idx).to(
                tl.float32
            )
            k_value = tl.load(K + (kv_bh * KV_LEN + n_safe) * D + d_idx)
            v_value = tl.load(V + (kv_bh * KV_LEN + n_safe) * D + d_idx).to(tl.float32)
            # Gate after the multiply so the products keep the exact dtypes the
            # original code used (bit-identical results on the in-matrix shapes
            # where n_mask is all-true).  `n_safe` guarantees the loaded lanes
            # are real in-range elements, so the products are always finite.
            score += tl.where(n_mask, q_value * k_value, 0.0)
            dp += tl.where(n_mask, do_value * v_value, 0.0)
    partial_offs = ((query_bh * Q_LEN + q_idx) * KV_LEN + n_offs) * D_CHUNKS + d_chunk
    tl.store(SCORE_PARTIAL + partial_offs, score, mask=n_mask)
    tl.store(DP_PARTIAL + partial_offs, dp, mask=n_mask)


@triton.jit
def delta_partial_kernel(
    O,
    DO,
    DELTA_PARTIAL,
    Q_LEN: tl.constexpr,
    D: tl.constexpr,
    D_CHUNKS: tl.constexpr,
    RED_: tl.constexpr,
):
    q_idx = tl.program_id(0)
    packed = tl.program_id(1)
    d_chunk = packed % D_CHUNKS
    bh = packed // D_CHUNKS
    d_offs = d_chunk * RED_ + tl.arange(0, RED_)
    mask = d_offs < D
    o = tl.load(O + (bh * Q_LEN + q_idx) * D + d_offs, mask=mask, other=0.0).to(
        tl.float32
    )
    do = tl.load(DO + (bh * Q_LEN + q_idx) * D + d_offs, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.store(
        DELTA_PARTIAL + (bh * Q_LEN + q_idx) * D_CHUNKS + d_chunk,
        tl.sum(o * do, axis=0),
    )


@triton.jit
def prob_ds_finalize_kernel(
    SCORE_PARTIAL,
    DP_PARTIAL,
    DELTA_PARTIAL,
    LSE,
    P,
    P_DV,
    DS,
    SCALE,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    D_CHUNKS: tl.constexpr,
    LSE_STRIDE_BH: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_N_: tl.constexpr,
):
    n_offs = tl.program_id(0) * BLOCK_N_ + tl.arange(0, BLOCK_N_)
    q_idx = tl.program_id(1)
    bh = tl.program_id(2)
    n_mask = n_offs < KV_LEN
    partial_base = ((bh * Q_LEN + q_idx) * KV_LEN + n_offs) * D_CHUNKS
    score = tl.zeros((BLOCK_N_,), dtype=tl.float32)
    dp = tl.zeros((BLOCK_N_,), dtype=tl.float32)
    delta = 0.0
    delta_base = (bh * Q_LEN + q_idx) * D_CHUNKS
    for chunk in tl.static_range(D_CHUNKS):
        score += tl.load(SCORE_PARTIAL + partial_base + chunk, mask=n_mask, other=0.0)
        dp += tl.load(DP_PARTIAL + partial_base + chunk, mask=n_mask, other=0.0)
        delta += tl.load(DELTA_PARTIAL + delta_base + chunk)
    lse = tl.load(LSE + bh * LSE_STRIDE_BH + q_idx).to(tl.float32)
    valid = n_mask
    if IS_CAUSAL:
        valid = valid & (n_offs <= q_idx)
    p = tl.exp(score * SCALE)
    p = tl.where(valid, p, 0.0)
    p_dv = tl.exp2(score * SCALE - lse)
    p_dv = tl.where(valid, p_dv, 0.0)
    ds = p * (dp - delta) * SCALE
    out_offs = (bh * Q_LEN + q_idx) * KV_LEN + n_offs
    tl.store(P + out_offs, p, mask=n_mask)
    tl.store(P_DV + out_offs, p_dv, mask=n_mask)
    tl.store(DS + out_offs, ds, mask=n_mask)


@triton.jit
def normalize_prob_ds_kernel(
    P,
    DS,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    q_idx = tl.program_id(0)
    bh = tl.program_id(1)
    n_offs = tl.arange(0, BLOCK_KV)
    mask = n_offs < KV_LEN
    offs = (bh * Q_LEN + q_idx) * KV_LEN + n_offs
    p = tl.load(P + offs, mask=mask, other=0.0)
    norm = tl.sum(p, axis=0)
    p /= norm
    ds = tl.load(DS + offs, mask=mask, other=0.0) / norm
    tl.store(P + offs, p, mask=mask)
    tl.store(DS + offs, ds, mask=mask)


@triton.jit
def dq_partial_kernel(
    DS,
    K,
    PARTIAL,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    D: tl.constexpr,
    R_CHUNKS: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_R_: tl.constexpr,
    BLOCK_D_: tl.constexpr,
):
    q_idx = tl.program_id(0)
    d_base = tl.program_id(1) * BLOCK_D_
    packed = tl.program_id(2)
    r_chunk = packed % R_CHUNKS
    query_bh = packed // R_CHUNKS
    batch_idx = query_bh // QUERY_HEADS
    query_head = query_bh % QUERY_HEADS
    kv_bh = batch_idx * KV_HEADS + query_head // GROUP_SIZE
    r_offs = (r_chunk * BLOCK_R_ + tl.arange(0, BLOCK_R_)).to(tl.int32)
    r_mask = r_offs < KV_LEN
    # `other=0.0` is not honoured on this BLOCK_R_-lane load, and the value is
    # consumed by a cross-lane `tl.sum`, so tail garbage lands straight in dQ
    # whenever KV_LEN % BLOCK_R_ != 0 (e.g. kv_len = 65 or 100).  Clamp the
    # address into range and zero the tail explicitly in registers.
    r_safe = tl.where(r_mask, r_offs, 0)
    ds = tl.load(DS + (query_bh * Q_LEN + q_idx) * KV_LEN + r_safe).to(tl.float32)
    ds = tl.where(r_mask, ds, 0.0)
    partial_base = ((query_bh * Q_LEN + q_idx) * R_CHUNKS + r_chunk) * D + d_base
    for d_offset in tl.static_range(BLOCK_D_):
        d_idx = d_base + d_offset
        x = tl.load(
            K + (kv_bh * KV_LEN + r_safe) * D + d_idx,
            mask=r_mask & (d_idx < D),
            other=0.0,
        ).to(tl.float32)
        tl.store(
            PARTIAL + partial_base + d_offset, tl.sum(ds * x, axis=0), mask=d_idx < D
        )


@triton.jit
def dk_partial_kernel(
    DS,
    Q,
    PARTIAL,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    D: tl.constexpr,
    R_CHUNKS: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_R_: tl.constexpr,
    BLOCK_D_: tl.constexpr,
):
    n_idx = tl.program_id(0)
    d_base = tl.program_id(1) * BLOCK_D_
    packed = tl.program_id(2)
    r_chunk = packed % R_CHUNKS
    kv_bh = packed // R_CHUNKS
    r_offs = (r_chunk * BLOCK_R_ + tl.arange(0, BLOCK_R_)).to(tl.int32)
    r_mask = r_offs < GROUP_SIZE * Q_LEN
    # Same TritonXPU narrow-load hazard as in dq_partial_kernel: derive every
    # address from the clamped index so nothing is read out of range, and zero
    # the tail with `tl.where` instead of trusting `other=`.
    r_safe = tl.where(r_mask, r_offs, 0)
    group_head = r_safe // Q_LEN
    q_idx = r_safe % Q_LEN
    batch_idx = kv_bh // KV_HEADS
    kv_head = kv_bh % KV_HEADS
    query_head = kv_head * GROUP_SIZE + group_head
    query_bh = batch_idx * QUERY_HEADS + query_head
    ds = tl.load(DS + (query_bh * Q_LEN + q_idx) * KV_LEN + n_idx).to(tl.float32)
    ds = tl.where(r_mask, ds, 0.0)
    partial_base = ((kv_bh * KV_LEN + n_idx) * R_CHUNKS + r_chunk) * D + d_base
    for d_offset in tl.static_range(BLOCK_D_):
        d_idx = d_base + d_offset
        x = tl.load(
            Q + (query_bh * Q_LEN + q_idx) * D + d_idx,
            mask=r_mask & (d_idx < D),
            other=0.0,
        ).to(tl.float32)
        tl.store(
            PARTIAL + partial_base + d_offset, tl.sum(ds * x, axis=0), mask=d_idx < D
        )


@triton.jit
def dv_dot_kernel(
    P,
    DO,
    PARTIAL,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    D: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N_: tl.constexpr,
    BLOCK_Q_: tl.constexpr,
    BLOCK_D_: tl.constexpr,
):
    n_offs = tl.program_id(0) * BLOCK_N_ + tl.arange(0, BLOCK_N_)
    d_offs = tl.program_id(1) * BLOCK_D_ + tl.arange(0, BLOCK_D_)
    kv_bh = tl.program_id(2)
    batch_idx = kv_bh // KV_HEADS
    kv_head = kv_bh % KV_HEADS
    n_mask = n_offs < KV_LEN
    d_mask = d_offs < D
    acc = tl.zeros((BLOCK_N_, BLOCK_D_), dtype=tl.float32)
    for group_head in tl.static_range(GROUP_SIZE):
        query_head = kv_head * GROUP_SIZE + group_head
        query_bh = batch_idx * QUERY_HEADS + query_head
        for q_base in tl.static_range(0, Q_LEN, BLOCK_Q_):
            q_offs = q_base + tl.arange(0, BLOCK_Q_)
            q_mask = q_offs < Q_LEN
            p = tl.load(
                P + (query_bh * Q_LEN + q_offs[:, None]) * KV_LEN + n_offs[None, :],
                mask=q_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            grad_out = tl.load(
                DO + (query_bh * Q_LEN + q_offs[:, None]) * D + d_offs[None, :],
                mask=q_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(tl.trans(p), grad_out.to(tl.float32))
    tl.store(
        PARTIAL + (kv_bh * KV_LEN + n_offs[:, None]) * D + d_offs[None, :],
        acc,
        mask=n_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def grad_finalize_kernel(
    PARTIAL,
    OUT,
    ROWS: tl.constexpr,
    D: tl.constexpr,
    R_CHUNKS: tl.constexpr,
    BLOCK_D_: tl.constexpr,
):
    row = tl.program_id(0)
    d_base = tl.program_id(1) * BLOCK_D_
    bh = tl.program_id(2)
    out_base = (bh * ROWS + row) * D + d_base
    for d_offset in tl.static_range(BLOCK_D_):
        d_idx = d_base + d_offset
        acc = 0.0
        compensation = 0.0
        for chunk in tl.static_range(R_CHUNKS):
            value = tl.load(
                PARTIAL + ((bh * ROWS + row) * R_CHUNKS + chunk) * D + d_idx,
                mask=d_idx < D,
                other=0.0,
            )
            corrected = value - compensation
            updated = acc + corrected
            compensation = (updated - acc) - corrected
            acc = updated
        tl.store(OUT + out_base + d_offset, acc, mask=d_idx < D)


def scaled_dot_product_attention_forward(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    if dropout_p != 0.0:
        raise NotImplementedError(
            "Kunlunxin scaled_dot_product_attention_forward does not support dropout"
        )
    if attn_mask is not None and is_causal:
        raise RuntimeError("Explicit attn_mask should not be set when is_causal=True")

    batch, q_head_num, q_len, head_dim = query.shape
    kv_head_num = key.shape[1]
    kv_len = key.shape[2]
    value_dim = value.shape[3]
    assert key.shape[3] == head_dim, "key head dim must match query head dim"
    assert (
        q_head_num % kv_head_num == 0
    ), f"q_head_num {q_head_num} must be divisible by kv_head_num {kv_head_num}"
    device = query.device
    softmax_scale = scale if scale is not None else 1.0 / (head_dim**0.5)

    out = torch.empty(
        (batch, q_head_num, q_len, value_dim), dtype=value.dtype, device=device
    )
    lse = torch.empty((batch, q_head_num, q_len), dtype=torch.float32, device=device)

    if q_len % 128 == 0 and kv_len % 128 == 0:
        BLOCK_M = 128
        BLOCK_N = 128
    else:
        BLOCK_M = 32
        BLOCK_N = 32
    k_pad = triton.cdiv(kv_len, BLOCK_N) * BLOCK_N
    padded_dq = triton.next_power_of_2(head_dim)
    padded_dv = triton.next_power_of_2(value_dim)
    q_in = _fa_pad_d(query, padded_dq) if padded_dq != head_dim else query
    v_in = _fa_pad_d(value, padded_dv) if padded_dv != value_dim else value

    # Build the (B, H, Q, K_PAD) cmask: cm = -bias for allowed cells, +1e30
    # for masked cells.  n >= K cells are never stored by kernel A (they stay
    # -1e30 in qk_buf -- prefill only when there is padding -- and are zeroed
    # by kernel B's (qk' - m) < -12 test).
    cm_vals = None
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            cm_vals = torch.where(attn_mask, 0.0, 1e30)
        else:
            bias = attn_mask.to(torch.float32)
            bias = torch.where(
                bias == float("-inf"), torch.full_like(bias, -1e30), bias
            )
            cm_vals = -bias
        if cm_vals.dim() == 2:  # (Q, K)
            cm_vals = cm_vals[None, None]
        elif cm_vals.dim() == 3:  # (B, Q, K)
            cm_vals = cm_vals[:, None]
        elif cm_vals.dim() != 4:  # (B, H, Q, K)
            raise ValueError(f"attn_mask must be 2D/3D/4D, got {cm_vals.dim()}D")
        if cm_vals.shape[-2] != q_len or cm_vals.shape[-1] != kv_len:
            raise ValueError("attn_mask shape must be (Q, K) along the last dims")
        cm_vals = cm_vals.expand(batch, q_head_num, q_len, kv_len)

    use_cmask = cm_vals is not None or is_causal
    if use_cmask:
        cmask = torch.full(
            (batch, q_head_num, q_len, k_pad), 1e30, dtype=torch.float32, device=device
        )
        if cm_vals is not None:
            cmask[..., :kv_len] = cm_vals
        else:
            cmask[..., :kv_len] = 0.0
        if is_causal:
            # Bottom-right alignment: n <= m + (K - Q) (matches torch SDPA).
            # Build a (Q, K) pattern instead of a 4D broadcast where: the 4D
            # form materializes a B*H-sized f32 temp just to pick entries.
            rows = torch.arange(q_len, device=device)[:, None]
            cols = torch.arange(kv_len, device=device)[None, :]
            allowed = cols <= (rows + kv_len - q_len)  # (Q, K)
            cmask[..., :kv_len] = torch.where(allowed[None, None], 0.0, 1e30).expand(
                batch, q_head_num, q_len, kv_len
            )
        cmask = cmask.reshape(batch * q_head_num, q_len, k_pad)

    if kv_len % BLOCK_N == 0:
        # Every column is written by kernel A (no n >= KV_CTX padding), so the
        # -1e30 prefill is dead cost: torch.empty avoids a full-buffer write.
        qk_buf = torch.empty(
            (batch * q_head_num, q_len, k_pad), dtype=torch.float32, device=device
        )
    else:
        qk_buf = torch.full(
            (batch * q_head_num, q_len, k_pad),
            -1e30,
            dtype=torch.float32,
            device=device,
        )
    m_buf = torch.empty((batch, q_head_num, q_len), dtype=torch.float32, device=device)
    grid = (triton.cdiv(q_len, BLOCK_M), batch * q_head_num)
    common_q = dict(
        GROUP_HEAD=q_head_num // kv_head_num,
        HEAD_DIM=head_dim,
        PADDED_D=padded_dq,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    common_v = dict(
        GROUP_HEAD=q_head_num // kv_head_num,
        HEAD_DIM=value_dim,
        PADDED_D=padded_dv,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    with torch_device_fn.device(device):
        _fa_qk_m_kernel[grid](
            q_in,
            key,
            qk_buf,
            m_buf,
            cmask if use_cmask else qk_buf,
            softmax_scale,
            q_in.stride(0),
            q_in.stride(2),
            q_in.stride(1),
            q_in.stride(3),
            key.stride(0),
            key.stride(2),
            key.stride(1),
            key.stride(3),
            q_len,
            kv_len,
            k_pad,
            q_head_num,
            USE_CMASK=use_cmask,
            num_warps=8,
            num_stages=1,
            **common_q,
        )
        d_off = 0
        while d_off < padded_dv:
            _fa_lout_kernel[grid](
                qk_buf,
                v_in,
                out,
                lse,
                m_buf,
                v_in.stride(0),
                v_in.stride(2),
                v_in.stride(1),
                v_in.stride(3),
                out.stride(0),
                out.stride(2),
                out.stride(1),
                out.stride(3),
                q_len,
                kv_len,
                k_pad,
                q_head_num,
                D_WIDTH=min(128, padded_dv - d_off),
                D_OFF=d_off,
                STORE_LSE=False,
                num_warps=8,
                num_stages=1,
                **common_v,
            )
            d_off += 128

    return out


def _staged_attention_backward(do, query, key, value, o, lse, sm_scale, is_causal):
    batch, query_heads, query_length, head_dim = query.shape
    key_batch, kv_heads, kv_length, key_head_dim = key.shape
    assert key.shape == value.shape
    assert batch == key_batch and head_dim == key_head_dim
    assert head_dim <= 256 and query_length <= 1024 and kv_length <= 1024
    assert query_heads % kv_heads == 0
    assert do.is_contiguous()
    assert query.is_contiguous() and key.is_contiguous()
    assert value.is_contiguous() and o.is_contiguous()

    query_batch_heads = batch * query_heads
    kv_batch_heads = batch * kv_heads
    group_size = query_heads // kv_heads
    dim_chunks = triton.cdiv(head_dim, _STAGED_RED)
    score_partial = torch.empty(
        (query_batch_heads, query_length, kv_length, dim_chunks),
        device=query.device,
        dtype=torch.float32,
    )
    dp_partial = torch.empty_like(score_partial)
    delta_partial = torch.empty(
        (query_batch_heads, query_length, dim_chunks),
        device=query.device,
        dtype=torch.float32,
    )
    probability = torch.empty(
        (query_batch_heads, query_length, kv_length),
        device=query.device,
        dtype=torch.float32,
    )
    probability_dv = torch.empty_like(probability)
    ds = torch.empty_like(probability)
    launch_options = {
        "num_warps": 1,
        "num_stages": 1,
        "isCloseVectorization": True,
        "buffer_size_limit": 2048,
    }

    prob_dp_partial_kernel[
        (
            triton.cdiv(kv_length, _STAGED_BLOCK_N),
            query_length,
            query_batch_heads * dim_chunks,
        )
    ](
        query,
        key,
        value,
        do,
        score_partial,
        dp_partial,
        Q_LEN=query_length,
        KV_LEN=kv_length,
        D=head_dim,
        D_CHUNKS=dim_chunks,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_N_=_STAGED_BLOCK_N,
        RED_=_STAGED_RED,
        **launch_options,
    )
    delta_partial_kernel[(query_length, query_batch_heads * dim_chunks)](
        o,
        do,
        delta_partial,
        Q_LEN=query_length,
        D=head_dim,
        D_CHUNKS=dim_chunks,
        RED_=_STAGED_RED,
        **launch_options,
    )
    prob_ds_finalize_kernel[
        (triton.cdiv(kv_length, _STAGED_BLOCK_N), query_length, query_batch_heads)
    ](
        score_partial,
        dp_partial,
        delta_partial,
        lse,
        probability,
        probability_dv,
        ds,
        sm_scale,
        Q_LEN=query_length,
        KV_LEN=kv_length,
        D_CHUNKS=dim_chunks,
        LSE_STRIDE_BH=lse.stride(1),
        IS_CAUSAL=is_causal,
        BLOCK_N_=_STAGED_BLOCK_N,
        **launch_options,
    )
    normalize_prob_ds_kernel[(query_length, query_batch_heads)](
        probability,
        ds,
        Q_LEN=query_length,
        KV_LEN=kv_length,
        BLOCK_KV=triton.next_power_of_2(kv_length),
        **launch_options,
    )

    query_chunks = triton.cdiv(group_size * query_length, _STAGED_GRAD_R)
    kv_chunks = triton.cdiv(kv_length, _STAGED_GRAD_R)
    dq_partial = torch.empty(
        (query_batch_heads, query_length, kv_chunks, head_dim),
        device=query.device,
        dtype=torch.float32,
    )
    dk_partial = torch.empty(
        (kv_batch_heads, kv_length, query_chunks, head_dim),
        device=key.device,
        dtype=torch.float32,
    )
    dq = torch.empty_like(query).contiguous()
    dk = torch.empty(
        (batch, kv_heads, kv_length, head_dim),
        device=key.device,
        dtype=key.dtype,
    )
    dv = torch.empty(
        (batch, kv_heads, kv_length, head_dim),
        device=value.device,
        dtype=value.dtype,
    )

    dq_partial_kernel[
        (
            query_length,
            triton.cdiv(head_dim, _STAGED_BLOCK_D),
            query_batch_heads * kv_chunks,
        )
    ](
        ds,
        key,
        dq_partial,
        Q_LEN=query_length,
        KV_LEN=kv_length,
        D=head_dim,
        R_CHUNKS=kv_chunks,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_R_=_STAGED_GRAD_R,
        BLOCK_D_=_STAGED_BLOCK_D,
        **launch_options,
    )
    dk_partial_kernel[
        (
            kv_length,
            triton.cdiv(head_dim, _STAGED_BLOCK_D),
            kv_batch_heads * query_chunks,
        )
    ](
        ds,
        query,
        dk_partial,
        Q_LEN=query_length,
        KV_LEN=kv_length,
        D=head_dim,
        R_CHUNKS=query_chunks,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_R_=_STAGED_GRAD_R,
        BLOCK_D_=_STAGED_BLOCK_D,
        **launch_options,
    )
    grad_finalize_kernel[
        (query_length, triton.cdiv(head_dim, _STAGED_BLOCK_D), query_batch_heads)
    ](
        dq_partial,
        dq,
        ROWS=query_length,
        D=head_dim,
        R_CHUNKS=kv_chunks,
        BLOCK_D_=_STAGED_BLOCK_D,
        **launch_options,
    )
    grad_finalize_kernel[
        (kv_length, triton.cdiv(head_dim, _STAGED_BLOCK_D), kv_batch_heads)
    ](
        dk_partial,
        dk,
        ROWS=kv_length,
        D=head_dim,
        R_CHUNKS=query_chunks,
        BLOCK_D_=_STAGED_BLOCK_D,
        **launch_options,
    )
    dv_dot_kernel[
        (
            triton.cdiv(kv_length, _STAGED_BLOCK_N),
            triton.cdiv(head_dim, _STAGED_BLOCK_D),
            kv_batch_heads,
        )
    ](
        probability,
        do,
        dv,
        Q_LEN=query_length,
        KV_LEN=kv_length,
        D=head_dim,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_N_=_STAGED_BLOCK_N,
        BLOCK_Q_=64,
        BLOCK_D_=_STAGED_BLOCK_D,
        **launch_options,
    )
    return dq, dk, dv


def scaled_dot_product_attention_backward(
    do,
    query,
    key,
    value,
    o,
    M,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    logger.debug("GEMS_KUNLUNXIN SCALED_DOT_PRODUCT_ATTENTION_BACKWARD")
    head_dim = query.shape[-1]
    assert (
        attn_mask is None
    ), "staged attention backward does not support attention bias"
    assert dropout_p == 0.0, "Currently only support dropout_p=0.0"
    sm_scale = 1.0 / math.sqrt(head_dim) if scale is None else scale
    return _staged_attention_backward(do, query, key, value, o, M, sm_scale, is_causal)


def efficient_attention_backward(
    grad_out_,
    query,
    key,
    value,
    bias,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    logsumexp,
    dropout_p,
    philox_seed,
    philox_offset,
    custom_mask_type,
    bias_requires_grad,
    *,
    scale=None,
    num_splits_key=None,
    window_size=None,
    shared_storage_dqdkdv=False,
):
    """Kunlunxin implementation of ATen's dense efficient-attention backward."""
    assert bias is None and not bias_requires_grad, "attention bias is unsupported"
    assert dropout_p == 0.0, "dropout is unsupported"
    assert (
        cu_seqlens_q is None and cu_seqlens_k is None
    ), "varlen attention is unsupported"
    assert num_splits_key is None, "split-key attention is unsupported"
    assert window_size is None, "windowed attention is unsupported"
    assert not shared_storage_dqdkdv, "shared gradient storage is unsupported"

    if custom_mask_type == 0:
        is_causal = False
    elif custom_mask_type == 1:
        is_causal = True
    else:
        raise ValueError(f"unsupported custom_mask_type: {custom_mask_type}")

    q_len = query.shape[1]
    sm_scale = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
    d_query, d_key, d_value = _staged_attention_backward(
        grad_out_.permute(0, 2, 1, 3).contiguous(),
        query.permute(0, 2, 1, 3).contiguous(),
        key.permute(0, 2, 1, 3).contiguous(),
        value.permute(0, 2, 1, 3).contiguous(),
        out.permute(0, 2, 1, 3).contiguous(),
        logsumexp[:, :, :q_len].contiguous(),
        sm_scale,
        is_causal,
    )
    return (
        d_query.permute(0, 2, 1, 3).contiguous(),
        d_key.permute(0, 2, 1, 3).contiguous(),
        d_value.permute(0, 2, 1, 3).contiguous(),
        None,
    )


class ScaleDotProductAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        logger.debug("GEMS_KUNLUNXIN SCALED_DOT_PRODUCT_ATTENTION")
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = value.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        assert dropout_p == 0.0, "Currenty only support dropout_p=0.0"

        o = torch.empty_like(query, dtype=value.dtype)

        stage = 3 if is_causal else 1
        # Precision knobs kept for parity with the vendor implementation; the
        # defaults (fp32 accumulation, non-tf32 GEMM) preserve the existing math.
        fa_accum_type = os.environ.get("FLAG_GEMS_FA_ACCUM_TYPE", "float")
        fa_gemm_type = os.environ.get("FLAG_GEMS_FA_GEMM_TYPE", "float")

        if scale is None:
            sm_scale = 1.0 / (HEAD_DIM_K**0.5)
        else:
            sm_scale = scale

        q_head_num = query.shape[1]
        kv_head_num = key.shape[1]
        assert enable_gqa or q_head_num == kv_head_num, (
            f"q_head_num {q_head_num} != kv_head_num {kv_head_num}, "
            "enable_gqa must be True to support different head numbers."
        )

        grid = lambda args: (
            triton.cdiv(query.shape[2], args["BLOCK_M"]),
            query.shape[0] * query.shape[1],
            1,
        )

        if attn_mask is not None:
            HAS_ATTN_MASK = True
            if attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.to(query.dtype) * -1.0e6
            stride_attn_mask_batch = attn_mask.stride(0)
            stride_attn_mask_head = attn_mask.stride(1)
            stride_attn_mask_q_seqlen = attn_mask.stride(2)
            stride_attn_mask_kv_seqlen = attn_mask.stride(3)
        else:
            HAS_ATTN_MASK = False
            stride_attn_mask_batch = 1
            stride_attn_mask_head = 1
            stride_attn_mask_q_seqlen = 1
            stride_attn_mask_kv_seqlen = 1

        M = torch.empty(
            (query.shape[0], query.shape[1], query.shape[2]),
            device=query.device,
            dtype=torch.float32,
        )

        with torch_device_fn.device(query.device):
            _attn_fwd[grid](
                query,
                key,
                value,
                attn_mask,
                sm_scale,
                M,
                o,  #
                query.stride(0),
                query.stride(1),
                query.stride(2),
                query.stride(3),  #
                key.stride(0),
                key.stride(1),
                key.stride(2),
                key.stride(3),  #
                value.stride(0),
                value.stride(1),
                value.stride(2),
                value.stride(3),  #
                stride_attn_mask_batch,
                stride_attn_mask_head,
                stride_attn_mask_q_seqlen,
                stride_attn_mask_kv_seqlen,  #
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),  #
                query.shape[0],
                q_head_num,
                kv_head_num,  #
                q_head_num // kv_head_num,  # group_head
                query.shape[2],  #
                key.shape[2],  #
                HEAD_DIM_K,  #
                STAGE=stage,  #
                HAS_ATTN_MASK=HAS_ATTN_MASK,  #
                ACCUM_TYPE=fa_accum_type,
                GEMM_TYPE=fa_gemm_type,
            )

        ctx.save_for_backward(query, key, value, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = HEAD_DIM_K
        ctx.causal = is_causal
        ctx.enable_gqa = enable_gqa
        return o

    @staticmethod
    def backward(ctx, do):
        query, key, value, o, M = ctx.saved_tensors
        is_causal = ctx.causal
        enable_gqa = ctx.enable_gqa
        sm_scale = ctx.sm_scale
        dq, dk, dv = scaled_dot_product_attention_backward(
            do,
            query,
            key,
            value,
            o,
            M,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=sm_scale,
            enable_gqa=enable_gqa,
        )
        return dq, dk, dv, None, None, None, None, None


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    return scaled_dot_product_attention_forward(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        enable_gqa,
    )


def scaled_dot_product_efficient_attention_backward(
    grad_out,
    query,
    key,
    value,
    attn_bias,
    out,
    logsumexp,
    philox_seed,
    philox_offset,
    dropout_p,
    grad_input_mask,
    is_causal=False,
    *,
    scale=None,
):
    need_dq, need_dk, need_dv, need_dbias = grad_input_mask
    if dropout_p != 0.0:
        raise NotImplementedError(
            "Kunlunxin efficient attention backward does not support dropout"
        )
    if attn_bias is not None or need_dbias:
        raise NotImplementedError(
            "Kunlunxin efficient attention backward does not support attention bias"
        )

    # The native efficient-attention LSE is already in the primitive's BHS layout.
    logsumexp_base2 = logsumexp.contiguous()
    dq, dk, dv = scaled_dot_product_attention_backward(
        grad_out.contiguous(),
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        out.contiguous(),
        logsumexp_base2,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
    dbias = None
    if not need_dq:
        dq = torch.zeros_like(query)
    if not need_dk:
        dk = torch.zeros_like(key)
    if not need_dv:
        dv = torch.zeros_like(value)
    if not need_dbias:
        dbias = None
    return dq, dk, dv, dbias


_LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _fa_qk_m_kernel(
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
    """Pass 1: ``qk' = (Q @ K^T) * scale - cmask`` and ``m = max(rowmax(qk'), 0)``.

    When ``USE_CMASK``, the mask is folded into the stored qk here
    (``qk' = qk - cm``) from a dense ``(B*H, Q_CTX, K_PAD)`` f32 ``CMASK``:
    0.0/50.0 for the FA3 causal/window masks, ``-attn_mask`` / ``+1e30`` for
    SDPA.  Masking must NOT happen in kernel B: the XPU compiler corrupts
    that kernel's loop-carried ``l_i``/``acc`` when the loop body holds an
    in-loop load/branch.  ``QK`` is prefilled with -1e30 so cells with
    ``n >= KV_CTX`` stay at a value whose ``exp2`` underflows to exactly 0.0.

    ``m`` is computed over the MASK-APPLIED qk' (NOT the raw qk) so that
    ``m >= max(qk')``: the XPU ``exp2`` saturates at 1.0 for positive
    arguments, and kernel B's ``(qk' - m)`` must stay ``<= 0`` to avoid the
    saturation (this matters for positive additive masks, e.g.
    ``attn_mask = +3``, which raise qk' above rowmax of the raw qk).
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
        qk = tl.dot(query, tl.trans(key)).to(tl.float32) * sm_scale
        if USE_CMASK:
            cm = tl.load(cm_base + offs_n[None, :], mask=q_mask[:, None], other=0.0)
            qk = qk - cm
        # m MUST be taken over the mask-applied qk' (NOT the raw qk): the XPU
        # exp2 saturates at 1.0 for positive arguments (it implements
        # min(2^v, 1.0)), so kernel B relies on the invariant (qk' - m) <= 0.
        # A positive additive mask (attn_mask > 0) makes qk' > rowmax(raw qk),
        # which would leave those p cells clamped at 1.0 (wrong softmax).
        m_i = tl.maximum(m_i, tl.max(qk, 1))
        qk_ptrs = qk_base + offs_m[:, None] * K_PAD + offs_n[None, :]
        # separable (1D-broadcast) store mask; n >= KV_CTX stays -1e30.
        tl.store(qk_ptrs, qk, mask=q_mask[:, None] & (offs_n[None, :] < KV_CTX))
    m_safe = tl.maximum(m_i, 0.0)
    m_ptrs = M + off_hz * Q_CTX + offs_m
    tl.store(m_ptrs, m_safe, mask=q_mask)


@triton.jit
def _fa_lout_kernel(
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
    """Pass 2: ``p = exp2((qk' - m) * LOG2E) = e^(qk' - m)``, ``out = p @ V / sum``.

    The mask was already folded into ``qk`` by kernel A (``qk' = qk - cmask``)
    with ``m >= max(qk')`` (so ``(qk' - m) <= 0`` and the XPU ``exp2`` -- which
    saturates at 1.0 for positive arguments -- never leaves its exact range),
    and the ``n >= KV_CTX`` padding stays at ``-1e30``.  There is NO
    branch/load inside the loop: the XPU compiler corrupts the loop-carried
    ``l_i``/``acc`` otherwise.  The ``tl.where`` zeroes cells with
    ``(qk' - m) < -12`` explicitly because the XPU ``exp2``/underflow is
    inexact for inputs below ~-11.09: such cells would otherwise return
    ``2^-16`` instead of their true value (``1023 * 2^-16 = 0.016`` would
    leak into ``l_i`` of short rows).  The threshold is on ``(qk' - m)`` (not
    ``qk'``) so cells with very negative additive mask biases (e.g.
    ``attn_mask = -20``) are also zeroed exactly.
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
        acc = tl.dot(p.to(value.dtype), value, acc=acc)
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
        # Natural log: lse = m + ln(sum(exp2((qk - m) * LOG2E))).
        lse = m_i + tl.log(l_i)
        lse_ptrs = Lse + off_hz * Q_CTX + offs_m
        tl.store(lse_ptrs, lse, mask=q_mask)


def _fa_pad_d(t, padded):
    """Zero-pad the last dim of ``t`` up to ``padded`` (for non-pow2 D)."""
    tp = torch.zeros((*t.shape[:-1], padded), dtype=t.dtype, device=t.device)
    tp[..., : t.shape[-1]].copy_(t)
    return tp


def flash_attention_forward(
    query,
    key,
    value,
    cumulative_sequence_length_q,
    cumulative_sequence_length_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    return_debug_mask,
    *,
    scale=None,
    softcap=0.0,
    window_size_left=None,
    window_size_right=None,
    seqused_k=None,
    alibi_slopes=None,
    disable_splitkv=False,
):
    logger.debug("GEMS_KUNLUNXIN FLASH_ATTENTION_FORWARD")
    assert (
        cumulative_sequence_length_q is None and cumulative_sequence_length_k is None
    ), "varlen is not supported yet."
    if dropout_p != 0.0:
        raise NotImplementedError(
            "Kunlunxin flash_attention_forward does not support dropout"
        )
    if return_debug_mask:
        raise NotImplementedError(
            "Kunlunxin flash_attention_forward does not support debug mask"
        )
    if softcap > 0.0:
        # exp2-tanh softcap chain is miscompiled by the XPU backend in every
        # placement (see the two-pass notes above).
        raise NotImplementedError(
            "Kunlunxin flash_attention_forward does not support softcap"
        )
    if alibi_slopes is not None:
        # 0-d loaded scalar x 2D alibi is miscompiled by the XPU backend.
        raise NotImplementedError(
            "Kunlunxin flash_attention_forward does not support alibi_slopes"
        )

    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 96, 128, 192, 256}

    batch, q_len, q_head_num, head_dim = query.shape
    k_len = key.shape[1]
    kv_head_num = key.shape[2]
    assert (
        q_head_num % kv_head_num == 0
    ), f"q_head_num {q_head_num} must be divisible by kv_head_num {kv_head_num}"

    softmax_scale = scale if scale is not None else 1.0 / (head_dim**0.5)
    device = query.device

    out = torch.empty(
        (batch, q_len, q_head_num, head_dim), dtype=query.dtype, device=device
    )
    lse = torch.empty((batch, q_head_num, q_len), dtype=torch.float32, device=device)

    wl = -1 if window_size_left is None else window_size_left
    wr = -1 if window_size_right is None else window_size_right
    has_window = wl >= 0 or wr >= 0

    BLOCK_M = 32
    BLOCK_N = 32
    k_pad = triton.cdiv(k_len, BLOCK_N) * BLOCK_N
    padded_d = triton.next_power_of_2(head_dim)

    # The kernels process the full PADDED_D width with no d-mask on the load
    # side; pad the last dim with zeros when head_dim is not a power of 2
    # (stores are guarded by offs_d < HEAD_DIM).
    if padded_d != head_dim:
        q_in, k_in, v_in = (
            _fa_pad_d(query, padded_d),
            _fa_pad_d(key, padded_d),
            _fa_pad_d(value, padded_d),
        )
    else:
        q_in, k_in, v_in = query, key, value

    # cmask (B*H, Q, K_PAD): 0.0 = valid, 50.0 = masked (causal bound
    # n <= row+off, window bounds, and the n >= k_len padding).  Kernel A
    # folds it into the stored qk (qk' = qk - 50); kernel B zeroes cells with
    # (qk' - m) < -12 explicitly.
    cmask = None
    if is_causal or has_window:
        cmask2d = torch.full((q_len, k_pad), 50.0, dtype=torch.float32, device=device)
        rows = torch.arange(q_len, device=device)[:, None]
        cols = torch.arange(k_pad, device=device)[None, :]
        off = k_len - q_len
        if is_causal:
            valid = cols <= (rows + off)
        else:
            valid = torch.ones_like(cols, dtype=torch.bool)
        if wl >= 0:
            valid = valid & (cols >= (rows + off - wl))
        if wr >= 0:
            valid = valid & (cols <= (rows + off + wr))
        cmask2d[valid] = 0.0
        # Materialize the dense (B*H, Q, K_PAD) layout the kernel indexes
        # (off_hz * Q_CTX * K_PAD + m * K_PAD + n); head-independent.
        cmask = (
            cmask2d.unsqueeze(0).expand(batch * q_head_num, q_len, k_pad).contiguous()
        )

    # qk scratch prefilled with -1e30: exp2(-1e30) underflows to exactly 0.0
    # (the backend miscompiles exp2(-inf) to nan/inf).
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
        _fa_qk_m_kernel[grid](
            q_in,
            k_in,
            qk_buf,
            m_buf,
            cmask if (is_causal or has_window) else qk_buf,
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
            USE_CMASK=(is_causal or has_window),
            num_warps=8,
            num_stages=1,
            **common,
        )
        d_off = 0
        while d_off < padded_d:
            _fa_lout_kernel[grid](
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
                STORE_LSE=(d_off == 0),
                num_warps=8,
                num_stages=1,
                **common,
            )
            d_off += 128

    return out, lse, None, None, None


# Adapted from https://github.com/vllm-project/flash-attention/blob/main/vllm_flash_attn/flash_attn_interface.py
def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,  # only used for non-paged prefill
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    # Dummy FA3 arguments
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_splits: int = 0,
    fa_version: int = 2,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    if use_c_extension:
        logger.debug("GEMS_KUNLUNXIN FLASH_ATTN_VARLEN_FUNC")
        with torch_device_fn.device(q.device):
            out_cpp, softmax_lse = torch.ops.flag_gems.flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q,
                cu_seqlens_q,
                max_seqlen_k,
                cu_seqlens_k,
                seqused_k,
                q_v,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                softcap,
                alibi_slopes,
                deterministic,
                return_attn_probs,
                block_table,
                return_softmax_lse,
                out,
                scheduler_metadata,
                q_descale,
                k_descale,
                v_descale,
                fa_version,
            )
        return (out_cpp, softmax_lse) if return_softmax_lse else out_cpp
    else:
        logger.debug("GEMS_KUNLUNXIN FLASH_ATTN_VARLEN_FUNC")
        assert (
            cu_seqlens_k is not None or seqused_k is not None
        ), "cu_seqlens_k or seqused_k must be provided"
        assert (
            cu_seqlens_k is None or seqused_k is None
        ), "cu_seqlens_k and seqused_k cannot be provided at the same time"
        assert (
            block_table is None or seqused_k is not None
        ), "seqused_k must be provided if block_table is provided"
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        # custom op does not support non-tuple input
        if window_size is None:
            real_window_size = (-1, -1)
        else:
            assert len(window_size) == 2
            real_window_size = (window_size[0], window_size[1])
        q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
        dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)
        if fa_version != 2:
            raise RuntimeError("Only FA2 is implemented.")
        if num_splits > 0:
            raise RuntimeError("num_splits > 0 is not implemented in GEMS.")
        max_seqlen_q = (
            max_seqlen_q.item() if hasattr(max_seqlen_q, "item") else max_seqlen_q
        )
        max_seqlen_k = (
            max_seqlen_k.item() if hasattr(max_seqlen_k, "item") else max_seqlen_k
        )
        out, q, k, v, softmax_lse, *_ = mha_varlan_fwd(
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            # cu_seqlens_k not used since we use seqused_k, but flash_api.cpp
            # still wants it so we pass all zeros
            dummy_cu_seqlens_k if cu_seqlens_k is None else cu_seqlens_k,
            seqused_k,
            None,
            block_table,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            causal,
            real_window_size[0],
            real_window_size[1],
            softcap,
            return_softmax_lse and dropout_p > 0,
            None,
        )

    return (out, softmax_lse) if return_softmax_lse else out
