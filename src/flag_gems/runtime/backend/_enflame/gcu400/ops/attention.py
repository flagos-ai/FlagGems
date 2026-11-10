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

import contextlib
import logging
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from flag_gems import runtime
from flag_gems.config import use_c_extension
from flag_gems.ops.flash_api import mha_varlan_fwd
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

from .flash_api import mha_fwd

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
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, min(start_m * BLOCK_M, KV_CTX)
    elif STAGE == 2:
        # Q_CTX > KV_CTX leaves on-band tiles entirely past the keys.  Visiting
        # one makes every logit -inf (fully masked), so m_ij == m_i == -inf and
        # exp2(-inf - -inf) is NaN.  Clamp the causal bands to KV_CTX; rows
        # beyond it are then covered by the off-band stage alone.
        lo, hi = min(start_m * BLOCK_M, KV_CTX), min((start_m + 1) * BLOCK_M, KV_CTX)
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
            value = tl.load(V_block_ptr, mask=kv_load_mask[:, None], other=0.0)
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


configs = [
    triton.Config(
        {"BLOCK_M": BM, "BLOCK_N": BN, "PRE_LOAD_V": PLV},
        num_stages=s,
        num_warps=w,
    )
    for BM in [64, 128]
    # BLOCK_N must be <= HEAD_DIM (static_assert in _attn_fwd); the prune
    # below keeps the largest width that fits, so list all supported widths.
    for BN in [16, 32, 64, 128]
    for PLV in [True, False]
    for s in [3]
    for w in [2, 4]
] + [
    # gcu400: HEAD_DIM=64 needs BLOCK_N<=64 (the kernel asserts it at compile
    # time).  Add only two candidates so the autotune space does not double
    # and exhaust host memory.
    triton.Config(
        {"BLOCK_M": BM, "BLOCK_N": 64, "PRE_LOAD_V": False},
        num_stages=3,
        num_warps=4,
    )
    for BM in [64, 128]
]


def _attn_fwd_prune_configs(configs, named_args, **kwargs):
    head_dim = named_args.get("HEAD_DIM", 128)
    block_n = max((bn for bn in (16, 32, 64, 128) if bn <= head_dim), default=16)
    return [c for c in configs if c.kwargs.get("BLOCK_N", 128) == block_n]


@triton.jit
def _attn_nomax_step(
    acc,
    p_sum,
    l_i,
    query,
    k_base,
    v_base,
    offs_m,
    offs_d,
    start_n,
    stride_k_seqlen,
    stride_k_headsize,
    stride_v_seqlen,
    stride_v_headsize,
    BLOCK_N: tl.constexpr,
    MASKED: tl.constexpr,
    DEFER_SUM: tl.constexpr,
):
    offs_n = tl.arange(0, BLOCK_N)
    pos = start_n + offs_n
    key = tl.load(
        k_base + offs_d[:, None] * stride_k_headsize + pos[None, :] * stride_k_seqlen
    )
    qk = tl.dot(query, key, allow_tf32=False)
    if MASKED:
        # exp2(-inf) == 0, so the causal mask needs no later fixup and the
        # deferred row sum stays exact.
        qk = tl.where(offs_m[:, None] >= pos[None, :], qk, -float("inf"))
    p = tl.math.exp2(qk)
    if DEFER_SUM:
        p_sum = p_sum + p
    else:
        l_i += tl.sum(p, 1)
    value = tl.load(
        v_base + pos[:, None] * stride_v_seqlen + offs_d[None, :] * stride_v_headsize
    )
    acc = tl.dot(p.to(value.dtype), value, acc, allow_tf32=False)
    return acc, p_sum, l_i


@triton.jit
def _attn_nomax_rowblock(
    start_m,
    Q,
    K,
    V,
    M,
    Out,
    sm_scale,
    q_offset,
    o_offset,
    kv_offset,
    stride_q_seqlen,
    stride_q_headsize,
    stride_k_seqlen,
    stride_k_headsize,
    stride_v_seqlen,
    stride_v_headsize,
    stride_o_seqlen,
    stride_o_headsize,
    off_hz,
    Q_CTX,
    KV_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DEFER_SUM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Every load here is unmasked; the host dispatches only when Q_CTX and
    # KV_CTX are multiples of the block sizes.
    query = tl.load(
        Q
        + q_offset
        + offs_m[:, None] * stride_q_seqlen
        + offs_d[None, :] * stride_q_headsize
    )
    # Fold the softmax scale (and log2(e)) into Q once instead of rescaling
    # the [BLOCK_M, BLOCK_N] score tile on every KV step.
    query = (query.to(tl.float32) * (sm_scale * 1.44269504)).to(query.dtype)

    k_base = K + kv_offset
    v_base = V + kv_offset

    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    p_sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    if CAUSAL:
        # Off-band: everything strictly above the diagonal block is unmasked.
        # Requires BLOCK_M % BLOCK_N == 0 so the split points are block aligned.
        # The `start_m > 0` guard matters: for row block 0 this range is empty,
        # and a 0-trip version of the loop miscompiles on GCU400 (sporadic
        # NaN/garbage in the first block's accumulator, ~1 element per row).
        if start_m > 0:
            for start_n in tl.range(0, start_m * BLOCK_M, BLOCK_N):
                acc, p_sum, l_i = _attn_nomax_step(
                    acc,
                    p_sum,
                    l_i,
                    query,
                    k_base,
                    v_base,
                    offs_m,
                    offs_d,
                    start_n,
                    stride_k_seqlen,
                    stride_k_headsize,
                    stride_v_seqlen,
                    stride_v_headsize,
                    BLOCK_N,
                    MASKED=False,
                    DEFER_SUM=DEFER_SUM,
                )
        # On-band: only the BLOCK_M/BLOCK_N blocks touching the diagonal.
        for start_n in tl.range(start_m * BLOCK_M, (start_m + 1) * BLOCK_M, BLOCK_N):
            acc, p_sum, l_i = _attn_nomax_step(
                acc,
                p_sum,
                l_i,
                query,
                k_base,
                v_base,
                offs_m,
                offs_d,
                start_n,
                stride_k_seqlen,
                stride_k_headsize,
                stride_v_seqlen,
                stride_v_headsize,
                BLOCK_N,
                MASKED=True,
                DEFER_SUM=DEFER_SUM,
            )
    else:
        for start_n in tl.range(0, KV_CTX, BLOCK_N):
            acc, p_sum, l_i = _attn_nomax_step(
                acc,
                p_sum,
                l_i,
                query,
                k_base,
                v_base,
                offs_m,
                offs_d,
                start_n,
                stride_k_seqlen,
                stride_k_headsize,
                stride_v_seqlen,
                stride_v_headsize,
                BLOCK_N,
                MASKED=False,
                DEFER_SUM=DEFER_SUM,
            )

    if DEFER_SUM:
        l_i = tl.sum(p_sum, 1)
    acc = acc / l_i[:, None]
    tl.store(
        Out
        + o_offset
        + offs_m[:, None] * stride_o_seqlen
        + offs_d[None, :] * stride_o_headsize,
        acc.to(Out.dtype.element_ty),
    )
    tl.store(M + off_hz * Q_CTX + offs_m, tl.math.log2(l_i))


@triton.jit
def _attn_fwd_nomax(
    Q,
    K,
    V,
    M,
    Out,
    sm_scale,
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
    stride_o_batch,
    stride_o_head,
    stride_o_seqlen,
    stride_o_headsize,
    q_head_num,
    Q_CTX,
    KV_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DEFER_SUM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    # Fast path for the mask-free, evenly divisible case (causal or not).
    #
    # Difference from _attn_fwd: the softmax keeps no running maximum. The row
    # shift in exp2 cancels in p / sum(p), so dropping it is exact as long as
    # the scaled scores stay inside exp2's exponent range -- which holds for
    # the usual 1/sqrt(d) scaling of unit-variance inputs. What it buys on
    # GCU400 is that `acc` is never read-modify-written per KV step: the
    # online form rescales the whole [BLOCK_M, HEAD_DIM] accumulator every
    # iteration, which forces the accumulator out of OACC and costs ~35%.
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    batch_id = off_hz // q_head_num
    head_id = off_hz % q_head_num

    q_offset = (
        batch_id.to(tl.int64) * stride_q_batch + head_id.to(tl.int64) * stride_q_head
    )
    o_offset = (
        batch_id.to(tl.int64) * stride_o_batch + head_id.to(tl.int64) * stride_o_head
    )
    kv_offset = (
        batch_id.to(tl.int64) * stride_k_batch + head_id.to(tl.int64) * stride_k_head
    )

    _attn_nomax_rowblock(
        start_m,
        Q,
        K,
        V,
        M,
        Out,
        sm_scale,
        q_offset,
        o_offset,
        kv_offset,
        stride_q_seqlen,
        stride_q_headsize,
        stride_k_seqlen,
        stride_k_headsize,
        stride_v_seqlen,
        stride_v_headsize,
        stride_o_seqlen,
        stride_o_headsize,
        off_hz,
        Q_CTX,
        KV_CTX,
        HEAD_DIM,
        BLOCK_M,
        BLOCK_N,
        DEFER_SUM=DEFER_SUM,
        CAUSAL=CAUSAL,
    )


def _nomax_launch_config(q_ctx, kv_ctx):
    """Tile choice for _attn_fwd_nomax, measured on GCU400.

    A 128x128 tile with 2 warps wins on every shape swept; it keeps the
    [BLOCK_M, HEAD_DIM] fp32 accumulator inside OACC while still giving each
    SIP enough programs to hide the K/V loads. Narrower tiles keep short
    sequences (and HEAD_DIM=64) on the same path.
    """
    block_n = next(bn for bn in (128, 64, 32) if kv_ctx % bn == 0)
    block_m = next(bm for bm in (128, 64, 32) if q_ctx % bm == 0)
    return block_m, block_n, 2


@libentry()
@libtuner(
    configs=configs,
    key=["KV_CTX", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": _attn_fwd_prune_configs},
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
        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * Q_CTX + offs_m
    tl.store(m_ptrs, m_i, mask=q_load_mask)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=q_load_mask[:, None])


@triton.jit
def _attn_bwd_preprocess(
    O, DO, Delta, Z, H, Q_CTX, BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = off_m < Q_CTX

    off_hz = tl.program_id(1)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(
        O + off_hz * D_HEAD * Q_CTX + off_m[:, None] * D_HEAD + off_n[None, :],
        mask=mask[:, None],
        other=0.0,
    )
    do = tl.load(
        DO + off_hz * D_HEAD * Q_CTX + off_m[:, None] * D_HEAD + off_n[None, :],
        mask=mask[:, None],
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
):
    # BLOCK_M1: 32
    # BLOCK_N1: 128
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_n_mask = offs_n < KV_CTX  # (BLOCK_N1, )

    offs_k = tl.arange(0, BLOCK_DMODEL)  # (BLOCK_DMODEL, )

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
            qT_ptrs, mask=offs_m_mask[None, :], other=0.0
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

        do = tl.load(do_ptrs, mask=offs_m_mask[:, None], other=0.0)
        # do = tl.load(do_ptrs, mask=offs_m_mask[:, None], other=0.0) # (BLOCK_M1, BLOCK_DMODEL)

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
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_m_mask = offs_m < Q_CTX

    offs_k = tl.arange(0, BLOCK_DMODEL)
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

        kT = tl.load(kT_ptrs, mask=offs_n_mask[None, :], other=0.0)
        vT = tl.load(vT_ptrs, mask=offs_n_mask[None, :], other=0.0)
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


def _attn_bwd_prune_configs(configs, named_args, **kwargs):
    # dq is computed at start_m = pid * BLOCK_M2, while the launch grid is
    # cdiv(Q_CTX, BLOCK_N1).  If BLOCK_M2 < BLOCK_N1 the tail of dq is never
    # written (it keeps uninitialized garbage), so drop those configs.
    block_n1 = max((c.kwargs["BLOCK_N1"] for c in configs), default=128)
    # GCU caps num_warps at 4 (GCUOptions); configs above that abort at launch.
    return [c for c in configs if c.kwargs["BLOCK_M2"] == block_n1 and c.num_warps <= 4]


@libentry()
@libtuner(
    configs=config_backward,
    key=["KV_CTX", "BLOCK_DMODEL"],
    prune_configs_by={"early_config_prune": _attn_bwd_prune_configs},
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
    dkv_stride_z,
    dkv_stride_h,  #
    H,  # query head num
    Q_CTX,  #
    KV_CTX,  #
    kv_head_num,  #
    GROUP_HEAD: tl.constexpr,  #
    IS_CAUSAL: tl.constexpr,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLK_SLICE_FACTOR: tl.constexpr,  #
    BLOCK_DMODEL: tl.constexpr,
):
    tl.device_assert(Q_CTX % BLOCK_M1 == 0, "Q_CTX must be a multiple of BLOCK_M1.")

    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * Q_CTX).to(tl.int64)
    batch_id = bhid // H
    q_head_id = bhid % H
    kv_head_id = q_head_id // GROUP_HEAD
    adj = (stride_h * q_head_id + stride_z * batch_id).to(tl.int64)
    kv_adj = (kv_stride_h * kv_head_id + kv_stride_z * batch_id).to(tl.int64)
    # dK/dV are allocated as (B, Q_HEAD, KV_CTX, D), so their head strides are
    # KV_CTX-based, not Q_CTX-based.  For non-square attention (Q_CTX !=
    # KV_CTX) reusing `adj` walks far past the dK/dV buffers and faults the
    # device (Sip exception).
    dkv_adj = (dkv_stride_h * q_head_id + dkv_stride_z * batch_id).to(tl.int64)

    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += kv_adj
    V += kv_adj
    DO += adj
    DQ += adj
    DK += dkv_adj
    DV += dkv_adj
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, BLOCK_DMODEL)

    start_n = pid * BLOCK_N1
    start_m = start_n

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_n_mask = offs_n < KV_CTX

    # A program whose KV tile is entirely past KV_CTX must not issue
    # the (masked) tile loads: with deep pipelining the GCU lowering
    # still DMAs the out-of-range tile and raises a Sip exception.
    if start_n < KV_CTX:
        dv = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)

        # load K and V: they stay in SRAM throughout the inner loop.
        key = tl.load(
            K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d,
            mask=offs_n_mask[:, None],
            other=0.0,
        )
        value = tl.load(
            V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d,
            mask=offs_n_mask[:, None],
            other=0.0,
        )

        if IS_CAUSAL:
            num_steps = BLOCK_N1 // MASK_BLOCK_M1

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
            )

            # Compute dK and dV for non-masked blocks.
            start_m += num_steps * MASK_BLOCK_M1
            remaining_m = Q_CTX - start_m
            num_steps = (remaining_m + BLOCK_M1 - 1) // BLOCK_M1

            if (num_steps > 0) & (start_m < Q_CTX):
                dk, dv = _attn_bwd_dkdv(  #
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
                )
        else:
            # Non-causal: every KV block receives gradients from all Q blocks.
            start_m = 0
            num_steps = (Q_CTX + BLOCK_M1 - 1) // BLOCK_M1
            if num_steps > 0:
                dk, dv = _attn_bwd_dkdv(  #
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
                )
        # tl.device_print("dv: ", dv)

        dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
        tl.store(dv_ptrs, dv, mask=offs_n_mask[:, None])

        # Write back dK.
        dk *= sm_scale
        dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
        tl.store(dk_ptrs, dk, mask=offs_n_mask[:, None])

    # THIS BLOCK DOES DQ:
    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    start_m = pid * BLOCK_M2

    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_m_mask = offs_m < Q_CTX

    query = tl.load(
        Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d,
        mask=offs_m_mask[:, None],
        other=0.0,
    )
    dq = tl.zeros([BLOCK_M2, BLOCK_DMODEL], dtype=tl.float32)
    do = tl.load(
        DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d,
        mask=offs_m_mask[:, None],
        other=0.0,
    )

    m = tl.load(M + offs_m, mask=offs_m_mask, other=float("inf"))
    m = m[:, None]

    if IS_CAUSAL:
        end_n = min(start_m + BLOCK_M2, KV_CTX)  # Ensure end_n does not exceed N_CTX
        num_steps = (end_n - start_n + MASK_BLOCK_N2 - 1) // MASK_BLOCK_N2

        # Stage 1 - Compute dQ for masked (diagonal) blocks.
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
                start_n,
                num_steps,  #
                MASK=True,  #
            )

        # Stage 2 - non-masked blocks.  Walk from 0: deriving the start as
        # stage2_end_n - num_steps*BLOCK_N2 goes negative when KV_CTX is not a
        # multiple of BLOCK_N2, and a negative offset is not caught by the
        # `offs_n < KV_CTX` mask, so it reads out of bounds.
        stage2_end_n = min(start_n, KV_CTX)
        stage2_num_steps = (stage2_end_n + BLOCK_N2 - 1) // BLOCK_N2

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
            )
    else:
        # Non-causal: every Q block attends to all KV blocks.
        num_steps = (KV_CTX + BLOCK_N2 - 1) // BLOCK_N2
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
                BLOCK_N2,
                BLOCK_DMODEL,  #
                start_m,
                0,
                num_steps,  #
                MASK=False,  #
            )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    # tl.store(dq_ptrs, dq)

    tl.store(dq_ptrs, dq, mask=offs_m_mask[:, None])


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
    logger.debug("GEMS_ENFLAME SCALED_DOT_PRODUCT_ATTENTION_FORWARD")
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    assert dropout_p == 0.0, "Currenty only support dropout_p=0.0"

    o = torch.empty_like(query, dtype=value.dtype)

    stage = 3 if is_causal else 1

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

    # Fast path: no attention mask, evenly divisible sequence lengths, causal
    # or not. Measured on GCU400 to be the cheapest formulation; see
    # _attn_fwd_nomax for why the running max is dropped.
    q_ctx, kv_ctx = query.shape[2], key.shape[2]
    causal_fast = bool(stage == 3)
    if (
        not HAS_ATTN_MASK
        and stage in (1, 3)
        and q_ctx == kv_ctx
        and dropout_p == 0.0
        and HEAD_DIM_K in (64, 128)
        # _attn_fwd_nomax indexes K/V by the query head directly; grouped-query
        # shapes need the q_head_num // kv_head_num mapping of the generic path.
        and q_head_num == kv_head_num
    ):
        block_m, block_n, num_warps = _nomax_launch_config(q_ctx, kv_ctx)
        # The causal split runs [0, start_m*BLOCK_M) then the diagonal blocks,
        # so the split point has to be block aligned.
        if (
            q_ctx % block_m == 0
            and kv_ctx % block_n == 0
            and (not causal_fast or block_m % block_n == 0)
        ):
            # Deferring the row sum always wins on GCU400: the reduction is a
            # shuffle tree, and paying it per KV step costs more than carrying
            # an extra [BLOCK_M, BLOCK_N] fp32 accumulator (measured on all
            # three core shapes, 4-16 KV steps).
            defer_sum = True
            M = torch.empty(
                (query.shape[0], query.shape[1], query.shape[2]),
                device=query.device,
                dtype=torch.float32,
            )
            grid_m = q_ctx // block_m
            _attn_fwd_nomax[(grid_m, query.shape[0] * q_head_num)](
                query,
                key,
                value,
                M,
                o,
                sm_scale,
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
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),
                q_head_num,
                q_ctx,
                kv_ctx,
                HEAD_DIM=HEAD_DIM_K,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                DEFER_SUM=defer_sum,
                CAUSAL=causal_fast,
                num_warps=num_warps,
                num_stages=2,
            )
            return o, M

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
    logger.debug("GEMS_ENFLAME SCALED_DOT_PRODUCT_ATTENTION_BACKWARD")
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
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

    BLOCK_DMODEL = HEAD_DIM_K
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
        (BATCH, Q_HEAD, KV_CTX, HEAD_DIM_K),
        device=key.device,
        dtype=key.dtype,
        memory_format=torch.contiguous_format,
    )
    dv = torch.empty(
        (BATCH, Q_HEAD, KV_CTX, HEAD_DIM_V),
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
    )

    max_block_n1 = (
        max([cfg.kwargs["BLOCK_N1"] for cfg in config_backward])
        if config_backward
        else 128
    )
    max_block_m2 = (
        max([cfg.kwargs["BLOCK_M2"] for cfg in config_backward])
        if config_backward
        else 128
    )
    # One grid serves both halves of the kernel: dK/dV tiles walk KV_CTX, dQ
    # tiles walk Q_CTX.  For non-square attention these need different counts,
    # so launch the larger and let each half mask/guard its own excess.  Using
    # only cdiv(Q_CTX, ...) left dK/dV tiles past Q_CTX/BN1 unwritten (garbage).
    grid = (
        max(
            triton.cdiv(Q_CTX, max_block_m2),
            triton.cdiv(KV_CTX, max_block_n1),
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
        dk.stride(1),  #
        Q_HEAD,
        Q_CTX,  #
        KV_CTX,  #
        KV_HEAD,  #
        GROUP_HEAD=group_head,  #
        IS_CAUSAL=is_causal,  #
        # BLOCK_M1=BLOCK_M1,
        # BLOCK_N1=BLOCK_N1,  #
        # BLOCK_M2=BLOCK_M2,
        # BLOCK_N2=BLOCK_N2,  #
        BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
        BLOCK_DMODEL=BLOCK_DMODEL,  #
        # num_warps=NUM_WARPS,  #
        # num_stages=NUM_STAGES,  #
    )

    if group_head > 1:
        dk = dk.reshape(BATCH, Q_HEAD // group_head, group_head, KV_CTX, HEAD_DIM_K)
        dv = dv.reshape(BATCH, Q_HEAD // group_head, group_head, KV_CTX, HEAD_DIM_V)
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
    logger.debug("GEMS_ENFLAME FLASH_ATTENTION_FORWARD")
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


# GCU400 allots 896 KiB of VDMem (DSM) per CTA. The paged varlen attention
# kernel keeps its K/V tiles and dot temporaries there; with HEAD_DIM=128 a
# BLOCK_N of 128 needs 1.17~1.26 MiB and the compiler aborts with
# "local buffer allocated in VDMem exceeds!". This is a hardware limit
# (triton_gcu GCUOptions.max_dsm), so widening the software cap does not help.
# Clamp BLOCK_N to a budget-safe tile before dispatching instead of letting
# the JIT compilation fail.
def _dsm_safe_block_n(head_size: int) -> int:
    """Largest BLOCK_N that keeps the paged varlen kernel inside VDMem."""
    # Measured on GCU400 with HEAD_DIM=128: BLOCK_N=64 tops out at 892928 B,
    # BLOCK_N=128 needs 1175552~1261568 B and fails to compile. BLOCK_K scales
    # with HEAD_DIM, so the safe tile is inversely proportional to it.
    budget = max(32, min(128, 8192 // max(1, head_size)))
    # tl.arange requires a power-of-two tile (HEAD_DIM=192 would give 42).
    return 1 << (budget.bit_length() - 1)


@contextlib.contextmanager
def _dsm_budget_guard(head_size: int):
    """Temporarily clamp the MHA heuristic BLOCK_N for one dispatch.

    The patch is scoped to the enclosing call and restored in ``finally``.
    FlagGems dispatches operators from a single Python thread, so no lock is
    needed here.
    """
    original = runtime.get_heuristic_config
    safe_bn = _dsm_safe_block_n(head_size)

    def guarded(op_name):
        cfg = original(op_name)
        if not op_name.startswith("mha_block_"):
            return cfg
        block_n = cfg.get("BLOCK_N")
        if block_n is None:
            return cfg
        return {
            **cfg,
            "BLOCK_N": lambda args, fn=block_n, limit=safe_bn: min(fn(args), limit),
        }

    runtime.get_heuristic_config = guarded
    try:
        yield
    finally:
        runtime.get_heuristic_config = original


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
        logger.debug("GEMS_ENFLAME FLASH_ATTN_VARLEN_FUNC")
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
        logger.debug("GEMS_ENFLAME FLASH_ATTN_VARLEN_FUNC")
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
        with _dsm_budget_guard(q.shape[-1]):
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
