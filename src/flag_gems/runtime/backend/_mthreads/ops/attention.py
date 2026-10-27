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
from functools import partial

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.config import use_c_extension
from flag_gems.ops.flash_api import mha_fwd, mha_varlan_fwd, mha_varlan_fwd_opt
from flag_gems.ops.flash_kernel import keep  # noqa: F401  (kept for legacy refs)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


# MTHREADS SDPA forward. Light TCE schedule, no double-buffer.
# 1) Load current V right after QK (not the next tile) so softmax can overlap
#    PV operand setup, instead of reloading V after softmax.
# 2) Fuse acc*alpha into tl.dot C to drop a separate convert_layout rescale.
# No tl.range, no next-tile prefetch, no acc*0 extra FMA.
# Backward is implemented locally in this module (preprocess / dK-dV / dQ
# kernels plus host wrapper); nothing is imported from the shared
# flag_gems.ops.attention implementation.


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
                qk = qk + tl.where(mask, 0, -float("inf"))
            else:
                qk = qk * qk_scale + tl.where(mask, 0, -float("inf"))
        else:
            # STAGE in {1, 3}: off-band or non-causal full pass
            qk = qk * qk_scale
            if HAS_ATTN_MASK:
                qk = qk + attn_mask

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        # -inf-safe online softmax: when every position seen so far is masked
        # (-inf), m_ij is -inf and qk - m_ij would be NaN. Pivot on 0 instead:
        # exp2(-inf - 0) == 0 keeps p and alpha exact, l_i reaches 0, and the
        # epilogue yields 0/0 == NaN for fully masked rows, matching PyTorch.
        m_ij_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        p = tl.math.exp2(qk - m_ij_safe[:, None])
        alpha = tl.math.exp2(m_i - m_ij_safe)
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


@triton.jit
def _attn_bwd_preprocess(
    O,
    DO,
    Delta,
    Z,
    H,
    Q_CTX,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
    D_HEAD_ACTUAL: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = off_m < Q_CTX

    off_hz = tl.program_id(1)
    off_n = tl.arange(0, D_HEAD)
    hd_mask = off_n < D_HEAD_ACTUAL
    o = tl.load(
        O
        + off_hz * D_HEAD_ACTUAL * Q_CTX
        + off_m[:, None] * D_HEAD_ACTUAL
        + off_n[None, :],
        mask=mask[:, None] & hd_mask[None, :],
        other=0.0,
    )
    do = tl.load(
        DO
        + off_hz * D_HEAD_ACTUAL * Q_CTX
        + off_m[:, None] * D_HEAD_ACTUAL
        + off_n[None, :],
        mask=mask[:, None] & hd_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * Q_CTX + off_m, delta, mask=mask)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(
    dk,
    dv,  #
    Q,
    key,
    value,
    sm_scale,  #
    DO,  #
    M,
    D,  #
    stride_tok,
    stride_d,  #
    H,
    Q_CTX,
    KV_CTX,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_DMODEL: tl.constexpr,  #
    start_n,
    start_m,
    num_steps,  #
    MASK: tl.constexpr,
    BLOCK_DMODEL_ACTUAL: tl.constexpr,
):
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_n_mask = offs_n < KV_CTX

    offs_k = tl.arange(0, BLOCK_DMODEL)
    hd_mask = offs_k < BLOCK_DMODEL_ACTUAL

    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        offs_m_mask = offs_m < Q_CTX

        qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
        do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d

        qT = tl.load(qT_ptrs, mask=hd_mask[:, None] & offs_m_mask[None, :], other=0.0)

        m = tl.load(M + offs_m, mask=offs_m_mask, other=float("inf"))

        qkT = tl.dot(key, qT)
        m = tl.broadcast_to(m[None, :], (BLOCK_N1, BLOCK_M1))
        m = tl.where(offs_n_mask[:, None], m, float("inf"))
        pT = tl.math.exp2(qkT - m)

        mask = (offs_m < Q_CTX)[None, :] & (offs_n < KV_CTX)[None, :]
        if MASK:
            mask &= offs_m[None, :] >= offs_n[:, None]
        pT = tl.where(mask, pT, 0.0)

        do = tl.load(do_ptrs, mask=offs_m_mask[:, None] & hd_mask[None, :], other=0.0)

        dv += tl.dot(pT, do.to(tl.float32))
        Di = tl.load(D + offs_m, mask=offs_m_mask, other=0.0)

        dpT = tl.dot(value, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(qT.dtype)
        qT = tl.where(offs_m_mask[None, :], qT, 0.0)
        dsT = tl.where(offs_m_mask[None, :] & offs_n_mask[:, None], dsT, 0.0)
        dk += tl.dot(dsT, tl.trans(qT))
        curr_m += step_m
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
    dq,
    query,
    K,
    V,  #
    do,
    m,
    D,
    stride_tok,
    stride_d,  #
    H,
    Q_CTX,  #
    KV_CTX,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLOCK_DMODEL: tl.constexpr,
    start_m,
    start_n,
    num_steps,  #
    MASK: tl.constexpr,
    BLOCK_DMODEL_ACTUAL: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_m_mask = offs_m < Q_CTX

    offs_k = tl.arange(0, BLOCK_DMODEL)
    hd_mask = offs_k < BLOCK_DMODEL_ACTUAL
    Di = tl.load(D + offs_m, mask=offs_m_mask, other=0.0)
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        offs_n = curr_n + tl.arange(0, BLOCK_N2)
        offs_n_mask = offs_n < KV_CTX

        kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
        vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d

        kT = tl.load(kT_ptrs, mask=hd_mask[:, None] & offs_n_mask[None, :], other=0.0)
        vT = tl.load(vT_ptrs, mask=hd_mask[:, None] & offs_n_mask[None, :], other=0.0)
        qk = tl.dot(query, kT)
        p = tl.math.exp2(qk - m)
        mask = (offs_m < Q_CTX)[:, None] & (offs_n < KV_CTX)[None, :]
        if MASK:
            mask &= offs_m[:, None] >= offs_n[None, :]
        p = tl.where(mask, p, 0.0)
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = tl.where(mask, ds, 0.0).to(kT.dtype)
        dq += tl.dot(ds, tl.trans(kT))
        curr_n += step_n
    return dq


config_backward = runtime.get_tuned_config("attention_bwd")


def _prune_attn_bwd_configs(configs, nargs, **kwargs):
    pruned = []
    for config in configs:
        if kwargs.get("BLOCK_DMODEL", 64) <= 32:
            num_warps = min(config.num_warps, 2)
        else:
            num_warps = max(config.num_warps, 8)
        if num_warps != config.num_warps:
            config = triton.Config(
                config.kwargs,
                num_warps=num_warps,
                num_stages=config.num_stages,
                num_ctas=config.num_ctas,
                maxnreg=config.maxnreg,
                pre_hook=config.pre_hook,
            )
        pruned.append(config)
    return pruned


@libentry()
@libtuner(
    configs=config_backward,
    key=["KV_CTX", "BLOCK_DMODEL"],
    prune_configs_by={"early_config_prune": _prune_attn_bwd_configs},
)
@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    sm_scale,  #
    DO,  #
    DQ,
    DK,
    DV,  #
    M,
    D,
    stride_z,
    stride_h,
    stride_tok,
    stride_d,  #
    kv_stride_z,
    kv_stride_h,  #
    dk_stride_z,
    dk_stride_h,
    dk_stride_tok,  #
    H,  # query head num
    Q_CTX,  #
    KV_CTX,  #
    kv_head_num,  #
    GROUP_HEAD: tl.constexpr,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLK_SLICE_FACTOR: tl.constexpr,  #
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_ACTUAL: tl.constexpr,
    IS_CAUSAL: tl.constexpr = True,
):
    LN2: tl.constexpr = 0.6931471824645996

    bhid = tl.program_id(2)
    off_chz = (bhid * Q_CTX).to(tl.int64)
    batch_id = bhid // H
    q_head_id = bhid % H
    kv_head_id = q_head_id // GROUP_HEAD
    adj = (stride_h * q_head_id + stride_z * batch_id).to(tl.int64)
    kv_adj = (kv_stride_h * kv_head_id + kv_stride_z * batch_id).to(tl.int64)
    dk_adj = (dk_stride_h * q_head_id + dk_stride_z * batch_id).to(tl.int64)

    pid = tl.program_id(0)

    Q += adj
    K += kv_adj
    V += kv_adj
    DO += adj
    DQ += adj
    DK += dk_adj
    DV += dk_adj
    M += off_chz
    D += off_chz

    offs_k = tl.arange(0, BLOCK_DMODEL)
    hd_mask = offs_k < BLOCK_DMODEL_ACTUAL

    start_n = pid * BLOCK_N1
    if start_n < KV_CTX:
        dv = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)

        offs_n = start_n + tl.arange(0, BLOCK_N1)
        offs_n_mask = offs_n < KV_CTX
        key = tl.load(
            K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d,
            mask=offs_n_mask[:, None] & hd_mask[None, :],
            other=0.0,
        )
        value = tl.load(
            V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d,
            mask=offs_n_mask[:, None] & hd_mask[None, :],
            other=0.0,
        )

        MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR

        if IS_CAUSAL:
            start_m = start_n
            if start_m < Q_CTX:
                end_m = min(start_m + BLOCK_N1, Q_CTX)
                num_steps = (end_m - start_m + MASK_BLOCK_M1 - 1) // MASK_BLOCK_M1
                dk, dv = _attn_bwd_dkdv(
                    dk,
                    dv,  #
                    Q,
                    key,
                    value,
                    sm_scale,  #
                    DO,  #
                    M,
                    D,  #
                    stride_tok,
                    stride_d,  #
                    H,
                    Q_CTX,  #
                    KV_CTX,  #
                    MASK_BLOCK_M1,
                    BLOCK_N1,
                    BLOCK_DMODEL,  #
                    start_n,
                    start_m,
                    num_steps,  #
                    MASK=True,  #
                    BLOCK_DMODEL_ACTUAL=BLOCK_DMODEL_ACTUAL,
                )
                start_m += num_steps * MASK_BLOCK_M1
        else:
            start_m = 0

        remaining_m = Q_CTX - start_m
        num_steps = (remaining_m + BLOCK_M1 - 1) // BLOCK_M1
        if num_steps > 0:
            dk, dv = _attn_bwd_dkdv(
                dk,
                dv,  #
                Q,
                key,
                value,
                sm_scale,  #
                DO,  #
                M,
                D,  #
                stride_tok,
                stride_d,  #
                H,
                Q_CTX,  #
                KV_CTX,  #
                BLOCK_M1,
                BLOCK_N1,
                BLOCK_DMODEL,  #
                start_n,
                start_m,
                num_steps,  #
                MASK=False,  #
                BLOCK_DMODEL_ACTUAL=BLOCK_DMODEL_ACTUAL,
            )

        dv_ptrs = DV + offs_n[:, None] * dk_stride_tok + offs_k[None, :] * stride_d
        tl.store(dv_ptrs, dv, mask=offs_n_mask[:, None] & hd_mask[None, :])

        dk *= sm_scale
        dk_ptrs = DK + offs_n[:, None] * dk_stride_tok + offs_k[None, :] * stride_d
        tl.store(dk_ptrs, dk, mask=offs_n_mask[:, None] & hd_mask[None, :])

    start_m = pid * BLOCK_M2
    if start_m < Q_CTX:
        offs_m = start_m + tl.arange(0, BLOCK_M2)
        offs_m_mask = offs_m < Q_CTX
        query = tl.load(
            Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d,
            mask=offs_m_mask[:, None] & hd_mask[None, :],
            other=0.0,
        )
        dq = tl.zeros([BLOCK_M2, BLOCK_DMODEL], dtype=tl.float32)
        do = tl.load(
            DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d,
            mask=offs_m_mask[:, None] & hd_mask[None, :],
            other=0.0,
        )
        m = tl.load(M + offs_m, mask=offs_m_mask, other=float("inf"))
        m = m[:, None]

        MASK_BLOCK_N2: tl.constexpr = min(BLOCK_N2 // BLK_SLICE_FACTOR, BLOCK_M2)
        stage2_block_n: tl.constexpr = (
            BLOCK_N2 if BLOCK_M2 % BLOCK_N2 == 0 else MASK_BLOCK_N2
        )

        if IS_CAUSAL:
            diag_n = min(start_m, KV_CTX)
            end_n = min(start_m + BLOCK_M2, KV_CTX)
            num_steps = (end_n - diag_n + MASK_BLOCK_N2 - 1) // MASK_BLOCK_N2

            if num_steps > 0:
                dq = _attn_bwd_dq(
                    dq,
                    query,
                    K,
                    V,  #
                    do,
                    m,
                    D,  #
                    stride_tok,
                    stride_d,  #
                    H,
                    Q_CTX,  #
                    KV_CTX,  #
                    BLOCK_M2,
                    MASK_BLOCK_N2,
                    BLOCK_DMODEL,  #
                    start_m,
                    diag_n,
                    num_steps,  #
                    MASK=True,  #
                    BLOCK_DMODEL_ACTUAL=BLOCK_DMODEL_ACTUAL,
                )

            stage2_num_steps = (diag_n + stage2_block_n - 1) // stage2_block_n
        else:
            stage2_num_steps = (KV_CTX + stage2_block_n - 1) // stage2_block_n

        if stage2_num_steps > 0:
            dq = _attn_bwd_dq(
                dq,
                query,
                K,
                V,  #
                do,
                m,
                D,  #
                stride_tok,
                stride_d,  #
                H,
                Q_CTX,  #
                KV_CTX,  #
                BLOCK_M2,
                stage2_block_n,
                BLOCK_DMODEL,  #
                start_m,
                0,
                stage2_num_steps,  #
                MASK=False,  #
                BLOCK_DMODEL_ACTUAL=BLOCK_DMODEL_ACTUAL,
            )

        dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
        dq *= LN2
        tl.store(dq_ptrs, dq, mask=offs_m_mask[:, None] & hd_mask[None, :])


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
    assert value.shape[1] == kv_head_num, (
        f"key has {kv_head_num} heads but value has {value.shape[1]}; "
        "key and value must have the same number of heads."
    )
    if q_head_num != kv_head_num:
        assert q_head_num % kv_head_num == 0, (
            f"q_head_num {q_head_num} is not an integer multiple of "
            f"kv_head_num {kv_head_num}; GQA/MQA requires every KV head "
            "to serve an equal group of query heads."
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
            # out. Use -inf (not a large finite negative) so that fully
            # masked rows end as 0/0 == NaN exactly like torch, instead of
            # softmaxing uniformly over masked positions; the kernel's
            # online softmax pivots on 0 when m is -inf, so -inf never
            # produces NaN in partially masked rows.
            attn_mask = torch.where(attn_mask, 0.0, float("-inf"))
        elif attn_mask.dtype != torch.float32:
            attn_mask = attn_mask.to(torch.float32)
        attn_mask = attn_mask * LOG2E
        # Broadcast the mask right-aligned to [B, H, Q, KV] per the PyTorch
        # contract ([Q,KV] / [1,H,Q,KV] / [B,1,Q,KV] etc). expand() only
        # rewrites strides (zero stride on broadcast dims), so no [B,H,Q,KV]
        # copy is materialized and the kernel addresses broadcasts safely.
        mask_shape = (
            query.shape[0],
            query.shape[1],
            query.shape[2],
            key.shape[2],
        )
        while attn_mask.dim() < 4:
            attn_mask = attn_mask.unsqueeze(0)
        if tuple(attn_mask.shape) != mask_shape:
            if any(m != s and m != 1 for m, s in zip(attn_mask.shape, mask_shape)):
                raise RuntimeError(
                    f"attn_mask shape {tuple(attn_mask.shape)} is not "
                    f"broadcastable to {mask_shape}"
                )
            attn_mask = attn_mask.expand(mask_shape)
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
    logger.debug("GEMS SCALED DOT PRODUCT ATTENTION BACKWARD")
    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert dropout_p == 0.0, "Currenty only support dropout_p=0.0"

    if scale is None:
        sm_scale = 1.0 / (HEAD_DIM_K**0.5)
    else:
        sm_scale = scale

    assert do.is_contiguous()
    assert (
        query.is_contiguous()
        and key.is_contiguous()
        and value.is_contiguous()
        and o.is_contiguous()
    )
    assert query.stride() == o.stride() == do.stride()
    assert key.stride() == value.stride()

    BLOCK_DMODEL_ACTUAL = HEAD_DIM_K
    BLOCK_DMODEL = triton.next_power_of_2(HEAD_DIM_K)
    BATCH, Q_HEAD, Q_CTX = query.shape[:3]
    _, KV_HEAD, KV_CTX = key.shape[:3]
    group_head = Q_HEAD // KV_HEAD

    BLK_SLICE_FACTOR = 2
    RCP_LN2 = 1.0 / math.log(2)

    arg_k = key * (sm_scale * RCP_LN2)
    PRE_BLOCK = 256
    pre_grid = (triton.cdiv(Q_CTX, PRE_BLOCK), BATCH * Q_HEAD)

    delta = torch.empty_like(M)

    dq = torch.empty_like(query).contiguous()
    dk = torch.empty(
        (BATCH, Q_HEAD, KV_CTX, BLOCK_DMODEL_ACTUAL),
        device=key.device,
        dtype=key.dtype,
        memory_format=torch.contiguous_format,
    )
    dv = torch.empty(
        (BATCH, Q_HEAD, KV_CTX, BLOCK_DMODEL_ACTUAL),
        device=value.device,
        dtype=value.dtype,
        memory_format=torch.contiguous_format,
    )

    _attn_bwd_preprocess[pre_grid](
        o,
        do,  #
        delta,  #
        BATCH,
        Q_HEAD,
        Q_CTX,  #
        BLOCK_M=PRE_BLOCK,
        D_HEAD=BLOCK_DMODEL,  #
        D_HEAD_ACTUAL=BLOCK_DMODEL_ACTUAL,  #
    )

    grid = lambda meta: (
        max(
            triton.cdiv(KV_CTX, meta["BLOCK_N1"]),
            triton.cdiv(Q_CTX, meta["BLOCK_M2"]),
        ),
        1,
        BATCH * Q_HEAD,
    )

    _attn_bwd[grid](
        query,
        arg_k,
        value,
        sm_scale,
        do,
        dq,
        dk,
        dv,  #
        M,
        delta,  #
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),  #
        key.stride(0),
        key.stride(1),  #
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),  #
        Q_HEAD,
        Q_CTX,  #
        KV_CTX,  #
        KV_HEAD,  #
        GROUP_HEAD=group_head,  #
        BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
        BLOCK_DMODEL=BLOCK_DMODEL,  #
        BLOCK_DMODEL_ACTUAL=BLOCK_DMODEL_ACTUAL,  #
        IS_CAUSAL=is_causal,  #
    )

    if group_head > 1:
        dk = dk.reshape(
            BATCH, Q_HEAD // group_head, group_head, KV_CTX, BLOCK_DMODEL_ACTUAL
        )
        dv = dv.reshape(
            BATCH, Q_HEAD // group_head, group_head, KV_CTX, BLOCK_DMODEL_ACTUAL
        )
        dk = dk.sum(dim=2)
        dv = dv.sum(dim=2)

    return dq, dk, dv


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
    # The custom backward does not apply attn_mask and returns no gradient
    # for it; silently dropping either would produce wrong results. When a
    # mask is set and any differentiable input requires grad (Q, K, V, or a
    # float attn_mask), route through torch native instead: bool masks go to
    # the flash backend (exact, fast); additive masks go to the math
    # decomposition with the mask cast to the query dtype (MuDNN's math path
    # NaNs on an fp32 mask over fp16 Q/K/V). Masked inference stays on our
    # kernel.
    if (
        attn_mask is not None
        and torch.is_grad_enabled()
        and (
            query.requires_grad
            or key.requires_grad
            or value.requires_grad
            or attn_mask.requires_grad
        )
    ):
        if enable_gqa and query.shape[1] != key.shape[1]:
            # torch's math decomposition expands K/V heads via matmul, which
            # routes into the globally registered FlagGems aten::mm override
            # and fails on the expanded layouts. Reject explicitly instead
            # of emitting wrong gradients.
            raise RuntimeError(
                "flag_gems SDPA does not support attn_mask together with "
                "GQA (q_heads != kv_heads) when gradients are required; "
                "run the masked forward under no_grad, or call "
                "torch.nn.functional.scaled_dot_product_attention outside "
                "flag_gems for this case."
            )
        if attn_mask.dtype == torch.bool:
            return torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.MATH]):
            return torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask.to(query.dtype),
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
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
    logger.debug("GEMS FLASH_ATTENTION_FORWARD")
    assert (
        cumulative_sequence_length_q is None and cumulative_sequence_length_k is None
    ), "varlen is not supported yet."

    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    original_head_dim = HEAD_DIM_K
    supported_head_dims = (16, 32, 64, 96, 128, 192, 256)
    if HEAD_DIM_K not in supported_head_dims:
        padded_head_dim = None
        for d in supported_head_dims:
            if d >= HEAD_DIM_K:
                padded_head_dim = d
                break
        assert (
            padded_head_dim is not None
        ), f"Unsupported head dim {HEAD_DIM_K}, max supported is {supported_head_dims[-1]}"
        pad = padded_head_dim - HEAD_DIM_K
        query = F.pad(query, (0, pad))
        key = F.pad(key, (0, pad))
        value = F.pad(value, (0, pad))
        HEAD_DIM_K = padded_head_dim

    softmax_scale = scale or 1.0 / (original_head_dim**0.5)
    if window_size_left is not None:
        non_null_window_left = window_size_left
    else:
        non_null_window_left = -1
    if window_size_right is not None:
        non_null_window_right = window_size_right
    else:
        non_null_window_right = -1

    out = torch.empty_like(query)
    if cumulative_sequence_length_q is not None:
        out, q, k, v, lse, philox_seed, philox_offset, p = mha_varlan_fwd(
            query,
            key,
            value,
            out,
            cumulative_sequence_length_q,
            cumulative_sequence_length_k,
            seqused_k,
            None,  # leftpad_k
            None,  # block_table
            alibi_slopes,
            max_q,
            max_k,
            dropout_p,
            scale,
            False,
            is_causal,
            non_null_window_left,
            non_null_window_right,
            softcap,
            return_debug_mask and dropout_p > 0,
            None,
        )
    else:
        out, q, k, v, lse, philox_seed, philox_offset, p = mha_fwd(
            query,
            key,
            value,
            out,
            alibi_slopes,
            dropout_p,
            softmax_scale,
            is_causal,
            non_null_window_left,
            non_null_window_right,
            softcap,
            return_debug_mask,
            disable_splitkv=disable_splitkv,
        )

    if HEAD_DIM_K != original_head_dim:
        out = out[..., :original_head_dim]
    return (out, lse, philox_seed, philox_offset, p)


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    s_aux=None,
    num_splits: int = 0,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k=None,
    fa_version: int = 2,
):
    if fa_version != 2:
        raise RuntimeError("Only FA2 is implemented.")
    if num_splits > 0:
        raise RuntimeError("num_splits > 0 is not implemented in GEMS.")
    if use_c_extension:
        logger.debug("GEMS FLASH_ATTN_VARLEN_FUNC(C EXTENSION)")
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
                s_aux,
                num_splits,
                cp_world_size,
                cp_rank,
                cp_tot_seqused_k,
                fa_version,
            )
        return (out_cpp, softmax_lse) if return_softmax_lse else out_cpp
    else:
        logger.debug("GEMS FLASH_ATTN_VARLEN_FUNC")
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
        if window_size is None:
            real_window_size = (-1, -1)
        else:
            assert len(window_size) == 2
            real_window_size = (window_size[0], window_size[1])
        q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
        dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)
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


def flash_attn_varlen_opt_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    lse=None,
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    s_aux=None,
    num_splits: int = 0,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k=None,
    fa_version: int = 2,
):
    if fa_version != 2:
        raise RuntimeError("Only FA2 is implemented.")
    if num_splits > 0:
        raise RuntimeError("num_splits > 0 is not implemented in GEMS.")
    if use_c_extension:
        logger.debug("GEMS FLASH_ATTN_VARLEN_FUNC(C EXTENSION)")
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
                s_aux,
                num_splits,
                cp_world_size,
                cp_rank,
                cp_tot_seqused_k,
                fa_version,
            )
        return (out_cpp, softmax_lse) if return_softmax_lse else out_cpp
    else:
        logger.debug("GEMS FLASH_ATTN_VARLEN_OPT_FUNC")
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
        if window_size is None:
            real_window_size = (-1, -1)
        else:
            assert len(window_size) == 2
            real_window_size = (window_size[0], window_size[1])
        max_seqlen_q = (
            max_seqlen_q.item() if hasattr(max_seqlen_q, "item") else max_seqlen_q
        )
        max_seqlen_k = (
            max_seqlen_k.item() if hasattr(max_seqlen_k, "item") else max_seqlen_k
        )
        out, q, k, v, softmax_lse, *_ = mha_varlan_fwd_opt(
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k,
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
