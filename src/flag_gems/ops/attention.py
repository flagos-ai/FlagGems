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
from flag_gems.ops.flash_kernel import keep
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

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
    HEAD_DIM_ACTUAL: tl.constexpr,
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
    # mask for non-power-of-2 head dims: shape [HEAD_DIM]
    hd_mask = tl.arange(0, HEAD_DIM) < HEAD_DIM_ACTUAL

    # loop over key, value and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        kv_load_mask = (start_n + offs_n) < KV_CTX
        # start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        # K shape: [HEAD_DIM, BLOCK_N] — mask both head and kv-seq axes
        key = tl.load(
            K_block_ptr, mask=hd_mask[:, None] & kv_load_mask[None, :], other=0.0
        )
        if PRE_LOAD_V:
            # V shape: [BLOCK_N, HEAD_DIM] — mask both axes
            value = tl.load(
                V_block_ptr, mask=kv_load_mask[:, None] & hd_mask[None, :], other=0.0
            )

        qk = tl.dot(query, key, allow_tf32=False)
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
            value = tl.load(
                V_block_ptr, mask=kv_load_mask[:, None] & hd_mask[None, :], other=0.0
            )
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(query.dtype)
        p = p.to(value.dtype)
        acc = tl.dot(p, value, acc, allow_tf32=False)
        # update m_i and l_i
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
        {"BLOCK_M": BM, "BLOCK_N": BN, "PRE_LOAD_V": 0}, num_stages=s, num_warps=w
    )
    for BM in [64, 128]
    for BN in [16, 32]
    for s in [2, 3, 4]
    for w in [4, 8]
]
configs += SMALL_HEAD_DIM_CONFIGS


@libentry()
@libtuner(
    configs=list(filter(partial(keep, must_keep=SMALL_HEAD_DIM_CONFIGS), configs)),
    key=["KV_CTX", "HEAD_DIM"],
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
    PRE_LOAD_V: tl.constexpr,
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
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    # qk_scale *= 1.44269504  # 1/log(2)
    # mask for non-power-of-2 head dims
    hd_mask = offs_headsize < HEAD_DIM_ACTUAL
    # load query: it will stay in SRAM throughout
    query = tl.load(
        Q_block_ptr, mask=q_load_mask[:, None] & hd_mask[None, :], other=0.0
    )
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
            HEAD_DIM_ACTUAL,
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
            HEAD_DIM_ACTUAL,
        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * Q_CTX + offs_m
    tl.store(m_ptrs, m_i, mask=q_load_mask)
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
    # D_HEAD_ACTUAL is the real (unpadded) head size; use it for pointer
    # arithmetic so we stride correctly through contiguous tensors.
    # D_HEAD is the next power-of-2 block size used only for tl.arange.
    hd_mask = off_n < D_HEAD_ACTUAL
    # load
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
    # write-back
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
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    H,
    Q_CTX,
    KV_CTX,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_DMODEL: tl.constexpr,  #
    # Filled in by the wrapper.
    start_n,
    start_m,
    num_steps,  #
    MASK: tl.constexpr,
    BLOCK_DMODEL_ACTUAL: tl.constexpr,
):
    # BLOCK_M1: 32
    # BLOCK_N1: 128
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_n_mask = offs_n < KV_CTX  # (BLOCK_N1, )

    offs_k = tl.arange(0, BLOCK_DMODEL)  # (BLOCK_DMODEL, )
    hd_mask = offs_k < BLOCK_DMODEL_ACTUAL  # head dim mask

    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        offs_m = curr_m + tl.arange(0, BLOCK_M1)  # (BLOCK_M1, )
        offs_m_mask = offs_m < Q_CTX  # (BLOCK_M1, )

        qT_ptrs = (
            Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
        )  # (BLOCK_DMODEL, BLOCK_M1)
        do_ptrs = (
            DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
        )  # (BLOCK_M1, BLOCK_DMODEL)

        qT = tl.load(
            qT_ptrs, mask=hd_mask[:, None] & offs_m_mask[None, :], other=0.0
        )  # (BLOCK_DMODEL, BLOCK_M1)

        # Load m before computing qk to reduce pipeline stall.
        m = tl.load(M + offs_m, mask=offs_m_mask, other=float("inf"))  # (BLOCK_M1, )

        # key: (BLOCK_N1, BLOCK_DMODEL)
        qkT = tl.dot(key, qT)  # (BLOCK_N1, BLOCK_M1)
        m = tl.broadcast_to(m[None, :], (BLOCK_N1, BLOCK_M1))  # (BLOCK_N1, BLOCK_M1)
        m = tl.where(offs_n_mask[:, None], m, float("inf"))  # (BLOCK_N1, BLOCK_M1)
        pT = tl.math.exp2(qkT - m)
        # pT = tl.math.exp2(qkT - m[None, :])

        mask = (offs_m < Q_CTX)[None, :] & (offs_n < KV_CTX)[
            :, None
        ]  # (BLOCK_N1, BLOCK_M1)
        # Autoregressive masking.
        if MASK:
            mask &= offs_m[None, :] >= offs_n[:, None]
        pT = tl.where(mask, pT, 0.0)  # (BLOCK_N1, BLOCK_M1)

        do = tl.load(
            do_ptrs, mask=offs_m_mask[:, None] & hd_mask[None, :], other=0.0
        )  # (BLOCK_M1, BLOCK_DMODEL)

        # Compute dV.
        dv += tl.dot(pT, do.to(tl.float32))  # (BLOCK_N1, BLOCK_DMODEL)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m, mask=offs_m_mask, other=0.0)  # (BLOCK_M1, )

        # Compute dP and dS.
        dpT = tl.dot(value, tl.trans(do)).to(
            tl.float32
        )  # (BLOCK_N1, BLOCK_DMODEL) @ (BLOCK_M1, BLOCK_DMODEL).T -> (BLOCK_N1, BLOCK_M1)
        dsT = pT * (dpT - Di[None, :])  # (BLOCK_N1, BLOCK_M1)
        dsT = dsT.to(qT.dtype)
        qT = tl.where(offs_m_mask[None, :], qT, 0.0)  # (BLOCK_DMODEL, BLOCK_M1)
        dsT = tl.where(
            offs_m_mask[None, :] & offs_n_mask[:, None], dsT, 0.0
        )  # (BLOCK_N1, BLOCK_M1)
        dk += tl.dot(
            dsT, tl.trans(qT)
        )  # (BLOCK_N1, BLOCK_M1) @ (BLOCK_DMODEL, BLOCK_M1).T -> (BLOCK_N1, BLOCK_DMODEL)
        # Increment pointers.
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
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    H,
    Q_CTX,  #
    KV_CTX,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLOCK_DMODEL: tl.constexpr,
    # Filled in by the wrapper.
    start_m,
    start_n,
    num_steps,  #
    MASK: tl.constexpr,
    BLOCK_DMODEL_ACTUAL: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_m_mask = offs_m < Q_CTX

    offs_k = tl.arange(0, BLOCK_DMODEL)
    hd_mask = offs_k < BLOCK_DMODEL_ACTUAL  # head dim mask
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m, mask=offs_m_mask, other=0.0)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
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
        # Autoregressive masking.
        if MASK:
            # mask = (offs_m[:, None] >= offs_n[None, :])
            # mask = (offs_m[:, None] >= offs_n[None, :]) & (offs_m < N_CTX)[:, None] & (offs_n < N_CTX)[None, :]
            mask &= offs_m[:, None] >= offs_n[None, :]
        p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = tl.where(mask, ds, 0.0).to(kT.dtype)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
    return dq


config_backward = runtime.get_tuned_config("attention_bwd")


@libentry()
@libtuner(
    configs=config_backward,
    key=["KV_CTX", "BLOCK_DMODEL"],
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
    # shared by Q/K/V/DO.
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
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * Q_CTX).to(tl.int64)
    batch_id = bhid // H
    q_head_id = bhid % H
    kv_head_id = q_head_id // GROUP_HEAD
    adj = (stride_h * q_head_id + stride_z * batch_id).to(tl.int64)
    kv_adj = (kv_stride_h * kv_head_id + kv_stride_z * batch_id).to(tl.int64)
    dk_adj = (dk_stride_h * q_head_id + dk_stride_z * batch_id).to(tl.int64)

    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += kv_adj
    V += kv_adj
    DO += adj
    DQ += adj
    DK += dk_adj
    DV += dk_adj
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, BLOCK_DMODEL)
    hd_mask = offs_k < BLOCK_DMODEL_ACTUAL  # head dim mask

    # dK/dV: only execute when this pid covers a valid KV block
    start_n = pid * BLOCK_N1
    if start_n < KV_CTX:
        dv = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)

        # load K and V: they stay in SRAM throughout the inner loop.
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

        # Causal: masked diagonal phase, then unmasked above-diagonal phase.
        # Non-causal: skip masked phase, single unmasked pass over all Q rows.
        if IS_CAUSAL:
            # The causal mask is q_idx >= kv_idx, so for KV block starting at
            # start_n, the first Q row that can attend is start_n itself.
            start_m = start_n
            # Clamp to valid Q range
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
            # else: start_n >= Q_CTX, no Q rows can attend to this KV block
        else:
            start_m = 0

        # Unmasked phase (shared): traverse remaining Q rows.
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

        # Write back dK.
        dk *= sm_scale
        dk_ptrs = DK + offs_n[:, None] * dk_stride_tok + offs_k[None, :] * stride_d
        tl.store(dk_ptrs, dk, mask=offs_n_mask[:, None] & hd_mask[None, :])

    # dQ: only execute when this pid covers a valid Q block
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

        MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR

        if IS_CAUSAL:
            # Masked diagonal phase: KV columns [diag_n, end_n)
            # diag_n is the KV position where the causal boundary starts for
            # this Q block. Only needed when diag_n < KV_CTX.
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

            # Unmasked phase: KV columns [0, diag_n), all fully visible.
            stage2_num_steps = (diag_n + BLOCK_N2 - 1) // BLOCK_N2
        else:
            # Non-causal: single unmasked pass over all KV columns.
            stage2_num_steps = (KV_CTX + BLOCK_N2 - 1) // BLOCK_N2

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
                BLOCK_N2,
                BLOCK_DMODEL,  #
                start_m,
                0,
                stage2_num_steps,  #
                MASK=False,  #
                BLOCK_DMODEL_ACTUAL=BLOCK_DMODEL_ACTUAL,
            )

        # Write back dQ.
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
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert dropout_p == 0.0, "Currenty only support dropout_p=0.0"

    o = torch.empty_like(query, dtype=value.dtype)

    stage = 3 if is_causal else 1

    if scale is None:
        sm_scale = 1.0 / (HEAD_DIM_K**0.5)
    else:
        sm_scale = scale

    # Triton kernels require HEAD_DIM (constexpr block size) to be a power of 2.
    # We pass HEAD_DIM = next_power_of_2(HEAD_DIM_K) as the SRAM block size and
    # HEAD_DIM_ACTUAL = HEAD_DIM_K as the real dimension, then mask inside the kernel.
    # No tensor padding needed.
    HEAD_DIM_ACTUAL = HEAD_DIM_K
    HEAD_DIM_K = triton.next_power_of_2(HEAD_DIM_K)

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
            HEAD_DIM_ACTUAL=HEAD_DIM_ACTUAL,  #
            STAGE=stage,  #
            HAS_ATTN_MASK=HAS_ATTN_MASK,  #
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
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert dropout_p == 0.0, "Currenty only support dropout_p=0.0"

    # sm_scale is based on the original head dim
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

    # Triton kernels require BLOCK_DMODEL to be a power of 2.
    # Pass BLOCK_DMODEL = next_power_of_2(HEAD_DIM_K) as the SRAM block size and
    # BLOCK_DMODEL_ACTUAL = HEAD_DIM_K as the real dimension, masking inside kernels.
    BLOCK_DMODEL_ACTUAL = HEAD_DIM_K
    BLOCK_DMODEL = triton.next_power_of_2(HEAD_DIM_K)
    BATCH, Q_HEAD, Q_CTX = query.shape[:3]
    _, KV_HEAD, KV_CTX = key.shape[:3]
    group_head = Q_HEAD // KV_HEAD

    # NUM_WARPS, NUM_STAGES = 4, 1
    # BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
    BLK_SLICE_FACTOR = 2
    # RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)

    RCP_LN2 = 1.0 / math.log(2)

    arg_k = key * (sm_scale * RCP_LN2)
    # PRE_BLOCK = 128
    PRE_BLOCK = 256

    # PRE_BLOCK = 32
    # assert N_CTX % PRE_BLOCK == 0
    # pre_grid = (N_CTX // PRE_BLOCK, BATCH * Q_HEAD)
    pre_grid = (triton.cdiv(Q_CTX, PRE_BLOCK), BATCH * Q_HEAD)

    delta = torch.empty_like(M)

    # NOTE that dk & dv always have the same number of heads as q
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
            triton.cdiv(
                KV_CTX, meta["BLOCK_N1"]
            ),  # _attn_bwd_dq traverse the key-value sequence
            triton.cdiv(
                Q_CTX, meta["BLOCK_M2"]
            ),  # _attn_bwd_dkdv traverse the query sequence
        ),
        1,
        BATCH * Q_HEAD,
    )
    # logger.info(f"{triton.cdiv(Q_CTX, BLOCK_N1)=}")
    # logger.info(f"{M.shape=}")

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
        # BLOCK_M1=BLOCK_M1,
        # BLOCK_N1=BLOCK_N1,  #
        # BLOCK_M2=BLOCK_M2,
        # BLOCK_N2=BLOCK_N2,  #
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
            None,
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
    s_aux=None,
    num_splits: int = 0,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k=None,
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
        # custom op does not support non-tuple input
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


def flash_attn_varlen_opt_func(
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
    lse=None,
    # Dummy FA3 arguments
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
        # custom op does not support non-tuple input
        if window_size is None:
            real_window_size = (-1, -1)
        else:
            assert len(window_size) == 2
            real_window_size = (window_size[0], window_size[1])
        # q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
        # dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)
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
            # cu_seqlens_k not used since we use seqused_k, but flash_api.cpp
            # still wants it so we pass all zeros
            # dummy_cu_seqlens_k if cu_seqlens_k is None else cu_seqlens_k,
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
