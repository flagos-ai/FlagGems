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

"""MTHREADS-specific scaled_dot_product_attention forward kernel.

Four optimizations versus the shared Triton flash-attention kernel, all
preserving the shared kernel's math contract (identical online-softmax
formulation in the log2 domain, same ``M`` statistics, fp32 accumulators,
same output semantics):

1. K is loaded as a row-major ``[BLOCK_N, HEAD_DIM]`` tile (sequence rows)
   and transposed in-register before the QK dot. The shared kernel's
   ``[HEAD_DIM, BLOCK_N]`` column tile strides along the KV sequence axis
   between rows, which defeats the MUSA global-to-shared copy pipeline.
2. The KV loop is split into unmasked full tiles plus a single masked tail
   tile, removing the per-tile tail predicates from the hot loop (~99%
   fewer conditional instructions in the KV loop, per MCU profiling).
3. ``num_stages=1`` for the KV loop: the MUSA pipeliner's staging never
   overlaps the online-softmax dependency chain, so deeper staging only
   costs shared memory (96 KB at stages=3 vs 56 KB at stages=1).
4. ``Q_CTX``/``KV_CTX`` are ``tl.constexpr``: the masked tail tile compiles
   to straight-line code instead of an ``scf.if`` region, and loop bounds
   are compile-time constants.

Changes 3 and 4 act jointly on CTA residency: with 8 warps on MTT S5000
either change alone still leaves the kernel bound at 2 resident CTAs per
MP (by shared memory or by the register file, respectively); together
they reach 3 CTAs/MP, hiding the serial per-tile chain. A 2x2 ablation
and resource numbers are recorded in the PR description.

The fast path only accepts a conservatively validated subset: 4D fp16/bf16
tensors with identical dtypes, batch, head counts, head dims and KV lengths,
head dim a power of two and >= 64, non-causal, no attn_mask, no dropout, no
GQA. All other inputs fall back to the shared implementation unchanged, so
unsupported and invalid inputs keep the shared path's behavior and error
semantics. The backward pass always reuses the shared Triton kernel.

Measured on MTT S5000 (bf16, B=1, heads=32, Q=4096, KV=4122, BSHD views):
12.8 ms (shared) -> 6.45 ms.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_LOG2E = 1.4426950408889634


@triton.jit
def _attn_tile_update(
    acc,
    l_i,
    m_i,
    query,
    K_block_ptr,
    V_block_ptr,
    qk_scale,
    offs_m,
    offs_n,
    start_n,
    MASKED: tl.constexpr,
    kv_ctx,
):
    """One KV tile of the online softmax (log2-domain statistics).

    ``MASKED`` selects the guarded form used only for the final partial tile;
    full tiles take the predicate-free path.
    """
    if MASKED:
        kv_load_mask = (start_n + offs_n) < kv_ctx
        kt = tl.load(K_block_ptr, mask=kv_load_mask[:, None], other=0.0)
        value = tl.load(V_block_ptr, mask=kv_load_mask[:, None], other=0.0)
        qk = tl.dot(query, tl.trans(kt), allow_tf32=False)
        qk = tl.where(kv_load_mask[None, :], qk, -float("inf"))
    else:
        kt = tl.load(K_block_ptr)
        value = tl.load(V_block_ptr)
        qk = tl.dot(query, tl.trans(kt), allow_tf32=False)

    qk *= qk_scale
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]

    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    acc = tl.dot(p.to(query.dtype), value, acc, allow_tf32=False)
    m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_inner_mthreads(
    acc,
    l_i,
    m_i,
    query,
    K_block_ptr,
    V_block_ptr,
    stride_k_seqlen,
    stride_v_seqlen,
    qk_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    offs_m,
    offs_n,
    KV_CTX: tl.constexpr,
):
    # non-causal full range only: causal inputs are routed to the shared
    # kernel by the launcher (see the guard there)
    lo, hi = 0, KV_CTX
    K_block_ptr += lo * stride_k_seqlen
    V_block_ptr += lo * stride_v_seqlen

    # hot loop over full tiles: no load masks, no predicates
    n_full = (hi // BLOCK_N) * BLOCK_N
    for start_n in range(lo, n_full, BLOCK_N):
        acc, l_i, m_i = _attn_tile_update(
            acc,
            l_i,
            m_i,
            query,
            K_block_ptr,
            V_block_ptr,
            qk_scale,
            offs_m,
            offs_n,
            start_n,
            MASKED=False,
            kv_ctx=hi,
        )
        K_block_ptr += BLOCK_N * stride_k_seqlen
        V_block_ptr += BLOCK_N * stride_v_seqlen
    # single masked tail tile when the range is not tile-aligned; the
    # pointers already sit at n_full after the full-tile loop
    if n_full < hi:
        acc, l_i, m_i = _attn_tile_update(
            acc,
            l_i,
            m_i,
            query,
            K_block_ptr,
            V_block_ptr,
            qk_scale,
            offs_m,
            offs_n,
            n_full,
            MASKED=True,
            kv_ctx=hi,
        )

    return acc, l_i, m_i


@libentry()
@triton.jit
def _attn_fwd_mthreads(
    Q,
    K,
    V,
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
    stride_o_batch,
    stride_o_head,
    stride_o_seqlen,
    stride_o_headsize,
    q_head_num,
    Q_CTX: tl.constexpr,
    KV_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    batch_id = off_hz // q_head_num
    head_id = off_hz % q_head_num

    # the fast path guarantees q_heads == kv_heads (guarded at launch), and
    # K/V use their own batch/head strides — their layouts may differ
    q_offset = (
        batch_id.to(tl.int64) * stride_q_batch + head_id.to(tl.int64) * stride_q_head
    )
    o_offset = (
        batch_id.to(tl.int64) * stride_o_batch + head_id.to(tl.int64) * stride_o_head
    )
    k_offset = (
        batch_id.to(tl.int64) * stride_k_batch + head_id.to(tl.int64) * stride_k_head
    )
    v_offset = (
        batch_id.to(tl.int64) * stride_v_batch + head_id.to(tl.int64) * stride_v_head
    )

    offs_headsize = tl.arange(0, HEAD_DIM)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_load_mask = offs_m < Q_CTX
    offs_n = tl.arange(0, BLOCK_N)

    Q_block_ptr = (
        Q
        + q_offset
        + offs_m[:, None] * stride_q_seqlen
        + offs_headsize[None, :] * stride_q_headsize
    )
    # K/V as [BLOCK_N, HEAD_DIM] row-major tiles (sequence rows)
    K_block_ptr = (
        K
        + k_offset
        + offs_n[:, None] * stride_k_seqlen
        + offs_headsize[None, :] * stride_k_headsize
    )
    V_block_ptr = (
        V
        + v_offset
        + offs_n[:, None] * stride_v_seqlen
        + offs_headsize[None, :] * stride_v_headsize
    )
    O_block_ptr = (
        Out
        + o_offset
        + offs_m[:, None] * stride_o_seqlen
        + offs_headsize[None, :] * stride_o_headsize
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    query = tl.load(Q_block_ptr, mask=q_load_mask[:, None], other=0.0)

    acc, l_i, m_i = _attn_fwd_inner_mthreads(
        acc,
        l_i,
        m_i,
        query,
        K_block_ptr,
        V_block_ptr,
        stride_k_seqlen,
        stride_v_seqlen,
        sm_scale,
        BLOCK_M,
        BLOCK_N,
        offs_m,
        offs_n,
        KV_CTX,
    )

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * Q_CTX + offs_m
    tl.store(m_ptrs, m_i, mask=q_load_mask)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=q_load_mask[:, None])


class _MthreadsSdpaForward(torch.autograd.Function):
    """Autograd wrapper: optimized forward, shared Triton backward."""

    @staticmethod
    def forward(ctx, query, key, value, o, M, sm_scale, is_causal):
        ctx.save_for_backward(query, key, value, o, M)
        ctx.sm_scale = sm_scale
        ctx.causal = is_causal
        return o

    @staticmethod
    def backward(ctx, do):
        from flag_gems.ops.attention import (
            scaled_dot_product_attention_backward as shared_backward,
        )

        query, key, value, o, M = ctx.saved_tensors
        dq, dk, dv = shared_backward(
            do,
            query,
            key,
            value,
            o,
            M,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=ctx.causal,
            scale=ctx.sm_scale,
            enable_gqa=False,
        )
        return dq, dk, dv, None, None, None, None


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
    """MTHREADS SDPA entry: optimized forward, shared backward.

    Fast path: 4D fp16/bf16, matching dtypes/batch/head counts/head dims/KV
    lengths across Q/K/V, head dim a power of two >= 64, non-causal, no
    attn_mask, no dropout, no GQA. Everything else falls back to the shared
    implementation (which also keeps the shared kernel's assertions and error
    semantics for invalid inputs, e.g. mismatched head counts without
    enable_gqa).
    """
    logger.debug("GEMS_MTHREADS SCALED_DOT_PRODUCT_ATTENTION FORWARD")
    from flag_gems.ops.attention import scaled_dot_product_attention as shared_sdpa

    if (
        attn_mask is not None
        or dropout_p != 0.0
        or enable_gqa
        or is_causal
        # This optimization targets non-causal workloads only; causal
        # inputs keep the shared kernel. (The causal variant of this
        # kernel measured slower under the previous stages=3 config;
        # it has not been re-evaluated under the current stages=1 +
        # constexpr configuration, so no claim is made either way.)
        # Route causal to the shared implementation.
        or query.dim() != 4
        or query.dtype not in (torch.float16, torch.bfloat16)
        or key.dtype != query.dtype
        or value.dtype != query.dtype
        or query.shape[0] != key.shape[0]
        or query.shape[0] != value.shape[0]
        or query.shape[1] != key.shape[1]
        or key.shape[1] != value.shape[1]
        or query.shape[-1] != key.shape[-1]
        or key.shape[-1] != value.shape[-1]
        or key.shape[2] != value.shape[2]
        or query.shape[2] == 0
        or key.shape[2] == 0
        # head dim must be a power of two: unmasked tile loads read the full
        # padded HEAD_DIM, so non-power-of-2 dims (e.g. 96) need the shared
        # kernel's column masking
        or query.shape[-1] < 64
        or (query.shape[-1] & (query.shape[-1] - 1)) != 0
    ):
        return shared_sdpa(
            query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
        )

    sm_scale = scale if scale is not None else 1.0 / (query.shape[-1] ** 0.5)
    o = torch.empty_like(query, dtype=value.dtype)
    M = torch.empty(
        (query.shape[0], query.shape[1], query.shape[2]),
        device=query.device,
        dtype=torch.float32,
    )
    grid = (triton.cdiv(query.shape[2], 64), query.shape[0] * query.shape[1], 1)
    with torch_device_fn.device(query.device):
        _attn_fwd_mthreads[grid](
            query,
            key,
            value,
            sm_scale * _LOG2E,  # fold LOG2E: statistics stay in the log2 domain
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
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            query.shape[1],
            Q_CTX=query.shape[2],
            KV_CTX=key.shape[2],
            HEAD_DIM=query.shape[-1],
            BLOCK_M=64,
            BLOCK_N=32,
            num_warps=8,
            # Residency needs BOTH of this launch's changes together (2x2
            # ablation, MTT S5000): stages=1 drops staging shared memory
            # 96 -> 56 KB, and the constexpr Q_CTX/KV_CTX above drop
            # registers 165 -> 158 (no dynamic loop bounds / scf.if tail).
            # Either change alone still leaves the kernel at 2 CTA/MP
            # (bound by shared memory or registers, respectively); together
            # they reach 3 CTA/MP, which hides the serial online-softmax
            # chain: 10.05 -> 6.45 ms on B1 H32 Q4096 KV4122 D128 bf16
            # (the pipeliner's staging never overlapped that chain anyway:
            # MCU shows issue slots stalled ~103% of resident cycles at
            # every depth).
            num_stages=1,
        )
    if torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value)):
        o = _MthreadsSdpaForward.apply(query, key, value, o, M, sm_scale, is_causal)
    return o
