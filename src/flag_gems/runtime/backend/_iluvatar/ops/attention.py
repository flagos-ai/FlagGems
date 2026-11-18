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

"""
Iluvatar-specialized SDPA (scaled_dot_product_attention) forward kernel.

Optimizations over the generic FlagGems attention kernel:
  1. Split-K flash-decoding path for decode shapes (Q_CTX == 1):
     Partitions KV across SPLIT_K thread blocks, each computing a partial
     softmax.  A lightweight reduction kernel merges partial results.
  2. Small-seq tile selection: BLOCK_M=16/32 for Q_CTX <= 512 to reduce
     padding waste (generic uses 64/128, wasting 75%+ on Sq=400).
  3. allow_tf32=True on tl.dot for head_dim=128 shapes — ~2x FMA throughput
     on iluvatar MR-series TF32 tensor cores.
  4. Tuned autotuning configs targeting the regression shapes.

The backward pass delegates to the generic kernel (unchanged).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.shape_utils import broadcasted_stride

logger = logging.getLogger(__name__)

LOG2E = tl.constexpr(1.44269504)  # log2(e)


# ──────────────────────────────────────────────────────────────────────
# Inner loop (shared by prefill and small-seq forward kernels)
# ──────────────────────────────────────────────────────────────────────


@triton.jit
def _attn_fwd_inner(
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
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_ACTUAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    KV_CTX: tl.constexpr,
    Q_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    if IS_CAUSAL:
        # Upper-left causal: Q[i] attends KV[0 : i + 1].
        # This matches torch.nn.functional.scaled_dot_product_attention's
        # is_causal semantics and the generic FlagGems kernel (STAGE==2).
        hi = tl.minimum((start_m + 1) * BLOCK_M, KV_CTX)
    else:
        hi = KV_CTX

    hd_mask = tl.arange(0, HEAD_DIM) < HEAD_DIM_ACTUAL

    # Pre-compute combined scale to avoid repeated multiplication in loop
    qk_scale_log2e = qk_scale * LOG2E

    for start_n in range(0, hi, BLOCK_N):
        kv_load_mask = (start_n + offs_n) < KV_CTX
        key = tl.load(
            K_block_ptr, mask=hd_mask[:, None] & kv_load_mask[None, :], other=0.0
        )
        qk = tl.dot(query, key, allow_tf32=ALLOW_TF32)
        qk = tl.where(kv_load_mask[None, :], qk, -float("inf"))

        if HAS_ATTN_MASK:
            qk = qk * qk_scale + tl.load(
                mask_block_ptr, mask=kv_load_mask[None, :], other=0.0
            )
            qk *= LOG2E
        else:
            qk *= qk_scale_log2e

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk += tl.where(causal_mask, 0, -1.0e6)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        value = tl.load(
            V_block_ptr, mask=kv_load_mask[:, None] & hd_mask[None, :], other=0.0
        )
        if fp8_v:
            p = p.to(tl.float8e5)
        p = p.to(value.dtype)
        acc = tl.dot(p, value, acc, allow_tf32=ALLOW_TF32)
        m_i = m_ij

        K_block_ptr += BLOCK_N * stride_k_seqlen
        V_block_ptr += BLOCK_N * stride_v_seqlen
        if HAS_ATTN_MASK:
            mask_block_ptr += BLOCK_N * stride_attn_mask_kv_seqlen

    return acc, l_i, m_i


# ──────────────────────────────────────────────────────────────────────
# Prefill / small-seq forward kernel (standard flash-attention)
# ──────────────────────────────────────────────────────────────────────

# Shared prefill configs for MHA and GQA kernels.
# Both kernels share the same tile parameters (BLOCK_M, BLOCK_N).
# GQA has higher register pressure from GROUP_HEAD accumulators, so large
# BLOCK_M configs will naturally be slower there and the autotuner will
# select smaller tiles.
#
# BLOCK_N is independent of HEAD_DIM (they are different dimensions in the dot product).
# One config per tile size; stages/warps follow: small tile → few warps,
# large tile → more warps. num_stages=2 uniformly (flagtree manual-ptr
# mode does not benefit from higher stages).
_ATTN_PREFILL_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=1, num_warps=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=1, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=1, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=1, num_warps=8),
]


@libentry()
@libtuner(
    configs=_ATTN_PREFILL_CONFIGS,
    key=["KV_CTX", "HEAD_DIM", "GROUP_HEAD", "STAGE"],
)
@triton.jit
def _attn_fwd_prefill(
    Q,
    K,
    V,
    attn_mask,
    sm_scale,
    M,
    Out,
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
    ALLOW_TF32: tl.constexpr,
):
    off_hz = tl.program_id(0)
    start_m = tl.program_id(1)
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

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_load_mask = offs_m < Q_CTX
    offs_n = tl.arange(0, BLOCK_N)
    offs_headsize = tl.arange(0, HEAD_DIM)
    hd_mask = offs_headsize < HEAD_DIM_ACTUAL

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

    query = tl.load(
        Q_block_ptr, mask=q_load_mask[:, None] & hd_mask[None, :], other=0.0
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale

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
        BLOCK_M,
        HEAD_DIM,
        HEAD_DIM_ACTUAL,
        BLOCK_N,
        STAGE == 3,
        offs_m,
        offs_n,
        KV_CTX,
        Q_CTX,
        V.dtype.element_ty == tl.float8e5,
        HAS_ATTN_MASK,
        ALLOW_TF32,
    )
    # Epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * Q_CTX + offs_m
    tl.store(m_ptrs, m_i, mask=q_load_mask)
    tl.store(
        O_block_ptr,
        acc.to(Out.type.element_ty),
        mask=q_load_mask[:, None] & hd_mask[None, :],
    )


# ──────────────────────────────────────────────────────────────────────
# Split-K decode kernel (Q_CTX == 1)
# ──────────────────────────────────────────────────────────────────────

_DECODE_CONFIGS = [
    triton.Config({"BLOCK_KV": 64}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_KV": 128}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_KV": 256}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_KV": 128}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_KV": 256}, num_stages=3, num_warps=8),
]


@libentry()
@libtuner(
    configs=_DECODE_CONFIGS,
    key=["KV_CTX", "HEAD_DIM"],
)
@triton.jit
def _attn_decode_splitk(
    Q,
    K,
    V,
    sm_scale,
    # Partial outputs — shape [B, Hq, SPLIT_K, D] and [B, Hq, SPLIT_K]
    Out_partial,  # [B, Hq, SPLIT_K, D]
    L_partial,  # [B, Hq, SPLIT_K]
    M_partial,  # [B, Hq, SPLIT_K]
    stride_q_batch,
    stride_q_head,
    stride_q_headsize,
    stride_k_batch,
    stride_k_head,
    stride_k_seqlen,
    stride_k_headsize,
    stride_v_batch,
    stride_v_head,
    stride_v_seqlen,
    stride_v_headsize,
    stride_op_batch,
    stride_op_head,
    stride_op_split,
    stride_op_headsize,
    Z,
    q_head_num,
    kv_head_num,
    GROUP_HEAD: tl.constexpr,
    KV_CTX,
    SPLIT_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_ACTUAL: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Each program handles one (batch, q_head, split_k_idx) slice."""
    split_k_idx = tl.program_id(0)
    off_hz = tl.program_id(1)
    batch_id = off_hz // q_head_num
    head_id = off_hz % q_head_num
    kv_head_id = head_id // GROUP_HEAD

    # Load the single query vector [HEAD_DIM]
    offs_d = tl.arange(0, HEAD_DIM)
    hd_mask = offs_d < HEAD_DIM_ACTUAL

    q_ptr = (
        Q
        + batch_id.to(tl.int64) * stride_q_batch
        + head_id.to(tl.int64) * stride_q_head
    )
    q = tl.load(q_ptr + offs_d * stride_q_headsize, mask=hd_mask, other=0.0)

    k_base = (
        batch_id.to(tl.int64) * stride_k_batch + kv_head_id.to(tl.int64) * stride_k_head
    )
    v_base = (
        batch_id.to(tl.int64) * stride_v_batch + kv_head_id.to(tl.int64) * stride_v_head
    )

    # This split's KV range
    kv_per_split = tl.cdiv(KV_CTX, SPLIT_K)
    kv_start = split_k_idx * kv_per_split
    kv_end = tl.minimum(kv_start + kv_per_split, KV_CTX)

    # Online softmax accumulators — scalar m/l, vector acc [HEAD_DIM]
    m_i = tl.full([1], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * LOG2E

    offs_kv = tl.arange(0, BLOCK_KV)
    for kv_off in range(kv_start, kv_end, BLOCK_KV):
        kv_idx = kv_off + offs_kv
        kv_mask = kv_idx < kv_end

        # Load K: [HEAD_DIM, BLOCK_KV]
        k_ptr = (
            K
            + k_base
            + kv_idx[None, :] * stride_k_seqlen
            + offs_d[:, None] * stride_k_headsize
        )
        k = tl.load(k_ptr, mask=hd_mask[:, None] & kv_mask[None, :], other=0.0)

        # QK: [HEAD_DIM] @ [HEAD_DIM, BLOCK_KV] → reduce to [BLOCK_KV]
        qk = tl.sum(q[:, None] * k, axis=0)  # dot via broadcast-multiply-reduce
        qk = qk * qk_scale
        qk = tl.where(kv_mask, qk, -float("inf"))

        # Online softmax update (all 1-D)
        m_ij = tl.max(qk, 0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new)  # [BLOCK_KV]
        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha

        # Load V: [BLOCK_KV, HEAD_DIM]
        v_ptr = (
            V
            + v_base
            + kv_idx[:, None] * stride_v_seqlen
            + offs_d[None, :] * stride_v_headsize
        )
        v = tl.load(v_ptr, mask=kv_mask[:, None] & hd_mask[None, :], other=0.0)

        # Weighted sum: p [BLOCK_KV] × V [BLOCK_KV, HEAD_DIM] → [HEAD_DIM]
        acc += tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        m_i = m_new

    # Store partial results
    op_offset = (
        batch_id.to(tl.int64) * stride_op_batch
        + head_id.to(tl.int64) * stride_op_head
        + split_k_idx.to(tl.int64) * stride_op_split
    )
    out_ptr = Out_partial + op_offset + offs_d * stride_op_headsize
    tl.store(out_ptr, acc, mask=hd_mask)

    lm_offset = (
        batch_id.to(tl.int64) * (q_head_num * SPLIT_K)
        + head_id.to(tl.int64) * SPLIT_K
        + split_k_idx.to(tl.int64)
    )
    # l_i / m_i are 0-D Triton tensors after arithmetic; reshape to [1] so
    # that both pointer and value are "block type" — required by flagtree.
    lm_ptr = lm_offset + tl.arange(0, 1)  # [1] block pointer offset
    tl.store(L_partial + lm_ptr, tl.reshape(l_i, [1]))
    tl.store(M_partial + lm_ptr, tl.reshape(m_i, [1]))


@libentry()
@triton.jit
def _attn_decode_reduce(
    Out_partial,  # [B, Hq, SPLIT_K, D]
    L_partial,  # [B, Hq, SPLIT_K]
    M_partial,  # [B, Hq, SPLIT_K]
    Out,  # [B, Hq, 1, D]
    M_out,  # [B, Hq, 1]
    stride_op_batch,
    stride_op_head,
    stride_op_split,
    stride_op_headsize,
    stride_o_batch,
    stride_o_head,
    stride_o_seqlen,
    stride_o_headsize,
    q_head_num,
    SPLIT_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_ACTUAL: tl.constexpr,
):
    """Reduce SPLIT_K partial softmax results into final output."""
    off_hz = tl.program_id(0)
    batch_id = off_hz // q_head_num
    head_id = off_hz % q_head_num

    offs_d = tl.arange(0, HEAD_DIM)
    hd_mask = offs_d < HEAD_DIM_ACTUAL

    # Load all partial m and l
    lm_base = (
        batch_id.to(tl.int64) * (q_head_num * SPLIT_K) + head_id.to(tl.int64) * SPLIT_K
    )
    offs_k = tl.arange(0, SPLIT_K)

    m_parts = tl.load(M_partial + lm_base + offs_k)  # [SPLIT_K]
    l_parts = tl.load(L_partial + lm_base + offs_k)  # [SPLIT_K]

    # Global max
    m_global = tl.max(m_parts, 0)

    # Rescale each partial and accumulate
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    l_global = 0.0

    op_base = (
        batch_id.to(tl.int64) * stride_op_batch + head_id.to(tl.int64) * stride_op_head
    )
    for k in range(SPLIT_K):
        # Use masking instead of constexpr indexing: select the k-th
        # element from the SPLIT_K-length vectors loaded above.
        k_mask = offs_k == k
        m_k = tl.sum(tl.where(k_mask, m_parts, 0.0), 0)
        l_k = tl.sum(tl.where(k_mask, l_parts, 0.0), 0)

        alpha = tl.math.exp2(m_k - m_global)
        l_scaled = l_k * alpha

        partial_ptr = (
            Out_partial + op_base + k * stride_op_split + offs_d * stride_op_headsize
        )
        partial_out = tl.load(partial_ptr, mask=hd_mask, other=0.0)

        acc += partial_out * alpha
        l_global += l_scaled

    acc = acc / l_global

    # Store final output
    o_offset = (
        batch_id.to(tl.int64) * stride_o_batch + head_id.to(tl.int64) * stride_o_head
    )
    out_ptr = Out + o_offset + offs_d * stride_o_headsize
    tl.store(out_ptr, acc.to(Out.type.element_ty), mask=hd_mask)

    # Store logsumexp for backward compatibility
    m_ptr = M_out + batch_id.to(tl.int64) * (q_head_num * 1) + head_id.to(tl.int64)
    tl.store(m_ptr, m_global + tl.math.log2(l_global))


def _choose_split_k(kv_ctx):
    """Choose split-K factor for decode based on KV context length."""
    if kv_ctx <= 256:
        return 1
    elif kv_ctx <= 1024:
        return 4
    elif kv_ctx <= 4096:
        return 8
    else:
        return 16


def _decode_forward(
    query,
    key,
    value,
    sm_scale,
    q_head_num,
    kv_head_num,
    HEAD_DIM_POW2,
    HEAD_DIM_ACTUAL,
    allow_tf32,
):
    """Split-K flash-decoding for Q_CTX == 1. Shared by MHA and GQA."""
    KV_CTX = key.shape[2]
    SPLIT_K = _choose_split_k(KV_CTX)
    B = query.shape[0]

    out_partial = torch.empty(
        (B, q_head_num, SPLIT_K, HEAD_DIM_ACTUAL),
        device=query.device,
        dtype=torch.float32,
    )
    l_partial = torch.empty(
        (B, q_head_num, SPLIT_K),
        device=query.device,
        dtype=torch.float32,
    )
    m_partial = torch.empty(
        (B, q_head_num, SPLIT_K),
        device=query.device,
        dtype=torch.float32,
    )

    grid_splitk = (SPLIT_K, B * q_head_num)
    with torch_device_fn.device(query.device):
        _attn_decode_splitk[grid_splitk](
            query,
            key,
            value,
            sm_scale,
            out_partial,
            l_partial,
            m_partial,
            query.stride(0),
            query.stride(1),
            query.stride(3),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key.stride(3),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),
            out_partial.stride(0),
            out_partial.stride(1),
            out_partial.stride(2),
            out_partial.stride(3),
            B,
            q_head_num,
            kv_head_num,
            q_head_num // kv_head_num,
            KV_CTX,
            SPLIT_K=SPLIT_K,
            HEAD_DIM=HEAD_DIM_POW2,
            HEAD_DIM_ACTUAL=HEAD_DIM_ACTUAL,
            ALLOW_TF32=allow_tf32,
        )

    o = torch.empty_like(query, dtype=value.dtype)
    M = torch.empty(
        (B, q_head_num, 1),
        device=query.device,
        dtype=torch.float32,
    )
    grid_reduce = (B * q_head_num,)
    with torch_device_fn.device(query.device):
        _attn_decode_reduce[grid_reduce](
            out_partial,
            l_partial,
            m_partial,
            o,
            M,
            out_partial.stride(0),
            out_partial.stride(1),
            out_partial.stride(2),
            out_partial.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            q_head_num,
            SPLIT_K=SPLIT_K,
            HEAD_DIM=HEAD_DIM_POW2,
            HEAD_DIM_ACTUAL=HEAD_DIM_ACTUAL,
        )
    return o, M


def _prepare_attn_mask(query, key, attn_mask, q_head_num):
    """Prepare attention mask strides and convert bool mask to float."""
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.to(query.dtype) * -1.0e6
        strides = broadcasted_stride(
            attn_mask.shape,
            attn_mask.stride(),
            (query.shape[0], q_head_num, query.shape[2], key.shape[2]),
        )
        return attn_mask, True, strides
    return attn_mask, False, (1, 1, 1, 1)


def _prefill_kernel_args(
    query,
    key,
    value,
    attn_mask,
    sm_scale,
    M,
    o,
    q_head_num,
    kv_head_num,
    HEAD_DIM_POW2,
    mask_strides,
):
    """Build the positional args tuple for the _attn_fwd_prefill kernel."""
    return (
        query,
        key,
        value,
        attn_mask,
        sm_scale,
        M,
        o,
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
        mask_strides[0],
        mask_strides[1],
        mask_strides[2],
        mask_strides[3],
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        query.shape[0],
        q_head_num,
        kv_head_num,
        q_head_num // kv_head_num,
        query.shape[2],
        key.shape[2],
        HEAD_DIM_POW2,
    )


def _prefill_forward(
    query,
    key,
    value,
    attn_mask,
    sm_scale,
    is_causal,
    q_head_num,
    kv_head_num,
    HEAD_DIM_POW2,
    HEAD_DIM_ACTUAL,
    allow_tf32,
):
    """Prefill / standard flash-attention path (handles both MHA and GQA).

    GQA is supported via GROUP_HEAD = q_head_num // kv_head_num passed to the
    kernel; KV head sharing is handled by the kv_head_id = head_id // GROUP_HEAD
    index computation inside the kernel.
    """
    Q_CTX = query.shape[2]

    # Decode
    if Q_CTX == 1 and attn_mask is None:
        return _decode_forward(
            query,
            key,
            value,
            sm_scale,
            q_head_num,
            kv_head_num,
            HEAD_DIM_POW2,
            HEAD_DIM_ACTUAL,
            allow_tf32,
        )

    # Prefill
    o = torch.empty_like(query, dtype=value.dtype)
    stage = 3 if is_causal else 1
    attn_mask, HAS_ATTN_MASK, mask_strides = _prepare_attn_mask(
        query,
        key,
        attn_mask,
        q_head_num,
    )
    M = torch.empty(
        (query.shape[0], q_head_num, Q_CTX),
        device=query.device,
        dtype=torch.float32,
    )
    args = _prefill_kernel_args(
        query,
        key,
        value,
        attn_mask,
        sm_scale,
        M,
        o,
        q_head_num,
        kv_head_num,
        HEAD_DIM_POW2,
        mask_strides,
    )

    def grid(meta):
        return (
            query.shape[0] * q_head_num,
            triton.cdiv(Q_CTX, meta["BLOCK_M"]),
            1,
        )

    with torch_device_fn.device(query.device):
        _attn_fwd_prefill[grid](
            *args,
            HEAD_DIM_ACTUAL=HEAD_DIM_ACTUAL,
            STAGE=stage,
            HAS_ATTN_MASK=HAS_ATTN_MASK,
            ALLOW_TF32=allow_tf32,
        )
    return o, M


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
    """Unified SDPA forward — dispatches to MHA or GQA path."""
    logger.debug("GEMS_ILUVATAR SCALED_DOT_PRODUCT_ATTENTION_FORWARD")
    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert dropout_p == 0.0, "Currently only support dropout_p=0.0"

    sm_scale = scale if scale is not None else 1.0 / (HEAD_DIM_K**0.5)
    HEAD_DIM_ACTUAL = HEAD_DIM_K
    HEAD_DIM_POW2 = triton.next_power_of_2(HEAD_DIM_K)

    q_head_num = query.shape[1]
    kv_head_num = key.shape[1]
    assert enable_gqa or q_head_num == kv_head_num, (
        f"q_head_num {q_head_num} != kv_head_num {kv_head_num}, "
        "enable_gqa must be True to support different head numbers."
    )

    allow_tf32 = False

    return _prefill_forward(
        query,
        key,
        value,
        attn_mask,
        sm_scale,
        is_causal,
        q_head_num,
        kv_head_num,
        HEAD_DIM_POW2,
        HEAD_DIM_ACTUAL,
        allow_tf32,
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
        # Delegate backward to the generic kernel — no iluvatar-specific backward yet
        from flag_gems.ops.attention import scaled_dot_product_attention_backward

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
