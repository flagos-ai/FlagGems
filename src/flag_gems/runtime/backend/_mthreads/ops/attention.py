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
from functools import partial

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.attention import scaled_dot_product_attention_backward
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


# MTHREADS SDPA forward. Light TCE schedule, no double-buffer.
# 1) Load current V right after QK (not the next tile) so softmax can overlap
#    PV operand setup, instead of reloading V after softmax.
# 2) Fuse acc*alpha into tl.dot C to drop a separate convert_layout rescale.
# No tl.range, no next-tile prefetch, no acc*0 extra FMA.
# Backward reuses the shared implementation from flag_gems.ops.attention.


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
    KV_CTX,
    fp8_v: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    HEAD_DIM_ACTUAL: tl.constexpr,
    IS_EVEN_MN: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    # causal = False
    else:
        lo, hi = 0, KV_CTX
    # Causal tutorial ranges assume K >= Q. When Q > K (e.g. 2048x256),
    # stage-1 hi = start_m*BLOCK_M walks past KV_CTX; the even path has
    # no kv mask and the OOB K/V loads turn later queries into NaN.
    hi = tl.minimum(hi, KV_CTX)
    lo = tl.minimum(lo, hi)

    K_block_ptr += lo * stride_k_seqlen
    V_block_ptr += lo * stride_v_seqlen
    if HAS_ATTN_MASK:
        mask_block_ptr += lo * stride_attn_mask_kv_seqlen

    # When IS_EVEN_MN is set, KV_CTX % BLOCK_N == 0 AND
    # HEAD_DIM_ACTUAL == HEAD_DIM, so neither kv_load_mask nor hd_mask
    # is needed; K/V loads skip the mask arithmetic entirely and Triton
    # is free to lower to b128 vectorized .cg loads.
    if IS_EVEN_MN:
        hd_mask_k = None
        hd_mask_v = None
    else:
        hd_mask = tl.arange(0, HEAD_DIM) < HEAD_DIM_ACTUAL
        hd_mask_k = hd_mask[:, None]
        hd_mask_v = hd_mask[None, :]

    # In Triton 3.2, plain range avoids speculative prefetch out-of-bounds on non-causal paths.
    for start_n in range(0, hi - lo, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if IS_EVEN_MN:
            kv_load_mask = None
        else:
            kv_load_mask = (lo + start_n + offs_n) < KV_CTX
        # -- compute qk ----
        # K shape: [HEAD_DIM, BLOCK_N]
        if IS_EVEN_MN:
            key = tl.load(K_block_ptr, cache_modifier=".cg")
            if PRE_LOAD_V:
                value = tl.load(V_block_ptr, cache_modifier=".cg")
        else:
            key = tl.load(
                K_block_ptr,
                mask=hd_mask_k & kv_load_mask[None, :],
                other=0.0,
                cache_modifier=".cg",
            )
            if PRE_LOAD_V:
                # V shape: [BLOCK_N, HEAD_DIM]
                value = tl.load(
                    V_block_ptr,
                    mask=kv_load_mask[:, None] & hd_mask_v,
                    other=0.0,
                    cache_modifier=".cg",
                    eviction_policy="evict_last",
                )

        qk = tl.dot(query, key, allow_tf32=False)
        if not IS_EVEN_MN:
            # Mask out columns past KV_CTX. No-op when IS_EVEN_MN.
            qk = tl.where(kv_load_mask[None, :], qk, -float("inf"))

        # Current V after QK: softmax ALU can overlap this load (PV data prep).
        # Reloading V after softmax made PRE_LOAD_V pay a second load. Load at
        # most once.
        if not PRE_LOAD_V:
            if IS_EVEN_MN:
                value = tl.load(V_block_ptr, cache_modifier=".cg")
            else:
                value = tl.load(
                    V_block_ptr,
                    mask=kv_load_mask[:, None] & hd_mask_v,
                    other=0.0,
                    cache_modifier=".cg",
                    eviction_policy="evict_last",
                )

        if HAS_ATTN_MASK:
            if IS_EVEN_MN:
                attn_mask = tl.load(mask_block_ptr, cache_modifier=".cg")
            else:
                attn_mask = tl.load(
                    mask_block_ptr,
                    mask=q_load_mask[:, None] & kv_load_mask[None, :],
                    other=0.0,
                )

        # Tutorial softmax: qk already in log2 space via qk_scale = sm_scale * log2(e).
        # Do not multiply qk_scale again on (m_old - m_new) or on m_ij.
        if STAGE == 2:
            mask = offs_m[:, None] >= (lo + start_n + offs_n[None, :])
            if HAS_ATTN_MASK:
                qk = qk * qk_scale + attn_mask
                qk = qk + tl.where(mask, 0, -1.0e6)
            else:
                qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
        else:
            # STAGE in {1, 3}: off-band or non-causal full pass
            qk = qk * qk_scale
            if HAS_ATTN_MASK:
                qk = qk + attn_mask

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij

        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(value.dtype)
        # Fuse rescale into MMA C: one convert as the accumulator operand,
        # no extra acc*0 FMA and no separate acc store.
        acc = tl.dot(p, value, acc * alpha[:, None], allow_tf32=False)
        # update m_i
        m_i = m_ij

        K_block_ptr += BLOCK_N * stride_k_seqlen
        V_block_ptr += BLOCK_N * stride_v_seqlen

        if HAS_ATTN_MASK:
            mask_block_ptr += BLOCK_N * stride_attn_mask_kv_seqlen

    return acc, l_i, m_i


# NOTE: we assert BLOCK_N <= HEAD_DIM in _attn_fwd, so for small head_dim,
# we need to generate more configs.
configs = runtime.get_tuned_config("attention")
SMALL_HEAD_DIM_CONFIGS = [
    triton.Config(
        {"BLOCK_M": BM, "BLOCK_N": BN, "PRE_LOAD_V": plv},
        num_stages=s,
        num_warps=w,
    )
    for BM in [64, 128]
    for BN in [32, 64]
    for s in [1, 2, 3, 4]
    for w in [4, 8]
    for plv in [0, 1]
]
configs += SMALL_HEAD_DIM_CONFIGS


def _qctx_bucket_strategy(q: int) -> int:
    if q <= 1024:
        return 1024
    if q <= 2048:
        return 2048
    if q <= 8192:
        return 8192
    return 16384


def _attn_keep(cfg, must_keep=None):
    BM = cfg.kwargs["BLOCK_M"]
    BN = cfg.kwargs["BLOCK_N"]
    s = cfg.num_stages
    # Hard rule: BN must be >= 32 to dodge the slice-layout assertion abort on MUSA
    if BN < 32:
        return False
    HD = 128
    sram_bytes = (BM * HD) * 2 + (HD * BN + BN * HD) * 2 * s + BM * HD * 4 + BM * BN * 4
    if sram_bytes > int(192 * 1024 * 0.8):
        return False
    return True


def _prune_attn_fwd_configs(configs, nargs, **kwargs):
    head_dim = kwargs.get("HEAD_DIM")
    if head_dim is None and isinstance(nargs, dict):
        head_dim = nargs.get("HEAD_DIM")
    if head_dim is None:
        return configs
    head_dim = int(head_dim)
    budget = 192 * 1024 - 4096
    pruned = []
    for cfg in configs:
        BM = cfg.kwargs["BLOCK_M"]
        BN = cfg.kwargs["BLOCK_N"]
        s = cfg.num_stages
        if BN < 32 or BN > head_dim:
            continue
        sram = (
            (BM * head_dim) * 2
            + (head_dim * BN + BN * head_dim) * 2 * s
            + BM * head_dim * 4
            + BM * BN * 4
        )
        if sram > budget:
            continue
        pruned.append(cfg)
    return pruned if pruned else configs


@libentry()
@libtuner(
    configs=list(
        filter(partial(_attn_keep, must_keep=SMALL_HEAD_DIM_CONFIGS), configs)
    ),
    strategy=[_qctx_bucket_strategy, "default"],
    key=["Q_CTX", "HEAD_DIM_ACTUAL"],
    prune_configs_by={"early_config_prune": _prune_attn_fwd_configs},
)
@triton.jit
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
    HEAD_DIM_ACTUAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
    IS_EVEN_MN: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)

    # IS_EVEN_MN is a host-side tl.constexpr. Deriving it here from runtime
    # Q_CTX/KV_CTX makes Triton 3.2/MUSA see None and fail:
    #   if IS_EVEN_MN: AttributeError('NoneType' ... 'type')

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
    # K and V may carry different batch/head strides (e.g. one is a BSHD
    # view); address each with its own strides instead of a shared offset.
    k_offset = (
        batch_id.to(tl.int64) * stride_k_batch + kv_head_id.to(tl.int64) * stride_k_head
    )
    v_offset = (
        batch_id.to(tl.int64) * stride_v_batch + kv_head_id.to(tl.int64) * stride_v_head
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
        + k_offset
        + offs_n[None, :] * stride_k_seqlen
        + offs_headsize[:, None] * stride_k_headsize
    )
    V_block_ptr = (
        V
        + v_offset
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
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    LOG2E = 1.44269504
    qk_scale = sm_scale * LOG2E
    hd_mask = offs_headsize < HEAD_DIM_ACTUAL

    if IS_EVEN_MN:
        query = tl.load(Q_block_ptr, cache_modifier=".cg")
    else:
        query = tl.load(
            Q_block_ptr,
            mask=q_load_mask[:, None] & hd_mask[None, :],
            other=0.0,
            cache_modifier=".cg",
        )

    # stage 1: off-band
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
            HEAD_DIM_ACTUAL,
            IS_EVEN_MN,
            2,
        )
    # stage 2: on-band
    if STAGE & 2:
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
            HEAD_DIM_ACTUAL,
            IS_EVEN_MN,
            2,
        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * Q_CTX + offs_m
    tl.store(m_ptrs, m_i, mask=q_load_mask)
    if IS_EVEN_MN:
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))
    else:
        tl.store(
            O_block_ptr,
            acc.to(Out.type.element_ty),
            mask=q_load_mask[:, None] & hd_mask[None, :],
        )


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
    logger.debug("GEMS SCALED DOT PRODUCT ATTENTION FORWARD")
    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert dropout_p == 0.0, "Currenty only support dropout_p=0.0"

    o = torch.empty_like(query, dtype=value.dtype)

    stage = 3 if is_causal else 1

    if scale is None:
        sm_scale = 1.0 / (HEAD_DIM_K**0.5)
    else:
        sm_scale = scale

    HEAD_DIM_ACTUAL = HEAD_DIM_K
    HEAD_DIM_K = triton.next_power_of_2(HEAD_DIM_K)
    # Configs use BLOCK_N >= 32, kernel asserts BLOCK_N <= HEAD_DIM.
    # head_dim=16 would otherwise fail to compile every BN=32/64 config.
    if HEAD_DIM_K < 64:
        HEAD_DIM_K = 64

    # Conservative even-path: Q divisible by all BM in {64,128}, K by all BN
    # in {32,64}. Otherwise keep masks. Must be a Python bool constexpr.
    IS_EVEN_MN = (
        query.shape[2] % 128 == 0
        and key.shape[2] % 64 == 0
        and HEAD_DIM_K == HEAD_DIM_ACTUAL
    )

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
        # The kernel runs the online softmax in base-2 space (qk_scale
        # carries log2(e)), while the PyTorch contract defines the additive
        # mask in natural-log space, so the mask must be converted too.
        LOG2E = 1.44269504
        if attn_mask.dtype == torch.bool:
            # PyTorch contract: True = take part in attention, False = masked
            # out. exp2(-1e6 * log2e) still underflows to exactly 0, so
            # fully-masked blocks keep contributing zero weight.
            attn_mask = torch.where(attn_mask, 0.0, -1.0e6) * LOG2E
        else:
            attn_mask = attn_mask * LOG2E
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
            HEAD_DIM_ACTUAL=HEAD_DIM_ACTUAL,  #
            STAGE=stage,  #
            HAS_ATTN_MASK=HAS_ATTN_MASK,  #
            IS_EVEN_MN=IS_EVEN_MN,  #
        )
    return o, M


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
        sm_scale = scale if scale is not None else 1.0 / (key.shape[-1] ** 0.5)
        o, M = scaled_dot_product_attention_forward(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            sm_scale,
            enable_gqa,
        )

        ctx.save_for_backward(query, key, value, o, M)
        ctx.sm_scale = sm_scale
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
    return ScaleDotProductAttention.apply(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        enable_gqa,
    )
