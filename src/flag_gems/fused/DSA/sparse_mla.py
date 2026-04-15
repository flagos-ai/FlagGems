"""
Combined Sparse MLA kernel: v91 + v99 with s_q-based dispatch.

- v99 (explicit software pipelining): better for small s_q (decode)
- v91 (flash-style softmax + overlapped K loads): better for large s_q (prefill)

Threshold tuning:
    Run: python benchmark.py --s_q 1 32 64 128 256 512 1024 2048 4096
    then adjust SQ_THRESHOLD below.
"""
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# ============================================================
# Crossover threshold: v99 when s_q <= this, v91 otherwise
# ============================================================
SQ_THRESHOLD = 128


# ============================================================
# v91 kernel — Flash-style softmax + overlapped K loads
# ============================================================


@triton.autotune(
    configs=[
        triton.Config({"BK": 128, "BH": 16}, num_warps=8, num_stages=5),
        triton.Config({"BK": 128, "BH": 16}, num_warps=8, num_stages=6),
        triton.Config({"BK": 128, "BH": 16}, num_warps=4, num_stages=5),
        triton.Config({"BK": 128, "BH": 16}, num_warps=4, num_stages=6),
        triton.Config({"BK": 128, "BH": 16}, num_warps=2, num_stages=5),
        triton.Config({"BK": 128, "BH": 16}, num_warps=2, num_stages=7),
        triton.Config({"BK": 128, "BH": 16}, num_warps=2, num_stages=8),
        triton.Config({"BK": 128, "BH": 16}, num_warps=2, num_stages=10),
        triton.Config({"BK": 128, "BH": 32}, num_warps=8, num_stages=4),
        triton.Config({"BK": 128, "BH": 32}, num_warps=8, num_stages=5),
        triton.Config({"BK": 128, "BH": 32}, num_warps=4, num_stages=5),
        triton.Config({"BK": 64, "BH": 16}, num_warps=4, num_stages=5),
        triton.Config({"BK": 64, "BH": 16}, num_warps=4, num_stages=7),
        triton.Config({"BK": 64, "BH": 16}, num_warps=8, num_stages=5),
        triton.Config({"BK": 64, "BH": 16}, num_warps=2, num_stages=5),
        triton.Config({"BK": 64, "BH": 16}, num_warps=2, num_stages=8),
        triton.Config({"BK": 64, "BH": 32}, num_warps=8, num_stages=5),
        triton.Config({"BK": 64, "BH": 64}, num_warps=8, num_stages=4),
    ],
    key=["topk", "num_qo_heads", "D_CKV", "HAS_KPE"],
)
@triton.jit
def _kernel_v91(
    q_nope_ptr,
    q_pe_ptr,
    kv_cache_ptr,
    sparse_indices_ptr,
    output_ptr,
    lse_ptr,
    num_tokens,
    num_qo_heads,
    topk,
    stride_qn_b,
    stride_qn_h,
    stride_qn_d,
    stride_qp_b,
    stride_qp_h,
    stride_qp_d,
    stride_kv_row,
    stride_kv_d,
    stride_sparse_b,
    stride_sparse_t,
    stride_out_b,
    stride_out_h,
    stride_out_d,
    stride_lse_b,
    stride_lse_h,
    sm_scale: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    D_CKV: tl.constexpr,
    D_KPE: tl.constexpr,
    HAS_KPE: tl.constexpr,
):
    num_head_blocks: tl.constexpr = (num_qo_heads + BH - 1) // BH
    pid = tl.program_id(0)
    token_idx = pid // num_head_blocks
    head_block_idx = pid % num_head_blocks

    offs_bh = tl.arange(0, BH)
    offs_d_nope = tl.arange(0, D_CKV)
    offs_bk = tl.arange(0, BK)
    head_mask = head_block_idx * BH + offs_bh < num_qo_heads
    log_scale: tl.constexpr = sm_scale * 1.44269504

    q_base = q_nope_ptr + token_idx * stride_qn_b + head_block_idx * BH * stride_qn_h
    q_nope = tl.load(
        q_base + offs_bh[:, None] * stride_qn_h + offs_d_nope[None, :] * stride_qn_d,
        mask=head_mask[:, None],
        other=0.0,
    )
    if HAS_KPE:
        offs_d_pe = tl.arange(0, D_KPE)
        qp_base = q_pe_ptr + token_idx * stride_qp_b + head_block_idx * BH * stride_qp_h
        q_pe = tl.load(
            qp_base + offs_bh[:, None] * stride_qp_h + offs_d_pe[None, :] * stride_qp_d,
            mask=head_mask[:, None],
            other=0.0,
        )

    m_i = tl.full([BH], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BH], dtype=tl.float32)
    acc = tl.zeros([BH, D_CKV], dtype=tl.float32)
    idx_base = sparse_indices_ptr + token_idx * stride_sparse_b

    for offset in range(0, topk, BK):
        kv_mask = offset + offs_bk < topk
        flat_kv_idx = tl.load(
            idx_base + (offset + offs_bk) * stride_sparse_t,
            mask=kv_mask,
            other=0,
            eviction_policy="evict_first",
        )
        row_offsets = flat_kv_idx * stride_kv_row
        k_nope = tl.load(
            kv_cache_ptr + row_offsets[:, None] + offs_d_nope[None, :] * stride_kv_d,
            mask=kv_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        if HAS_KPE:
            k_pe = tl.load(
                kv_cache_ptr
                + row_offsets[:, None]
                + (D_CKV + offs_d_pe)[None, :] * stride_kv_d,
                mask=kv_mask[:, None],
                other=0.0,
                eviction_policy="evict_last",
            )

        qk = tl.dot(q_nope, tl.trans(k_nope))
        if HAS_KPE:
            qk += tl.dot(q_pe, tl.trans(k_pe))
        qk = qk * log_scale
        qk = tl.where(kv_mask[None, :], qk, float("-inf"))

        qk_max = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, qk_max)
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), k_nope, acc)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    acc = acc / l_i[:, None]
    lse_val = m_i * 0.6931471805599453 + tl.log(l_i)

    o_base = output_ptr + token_idx * stride_out_b + head_block_idx * BH * stride_out_h
    tl.store(
        o_base + offs_bh[:, None] * stride_out_h + offs_d_nope[None, :] * stride_out_d,
        acc.to(tl.bfloat16),
        mask=head_mask[:, None],
    )
    l_base = lse_ptr + token_idx * stride_lse_b + head_block_idx * BH * stride_lse_h
    tl.store(l_base + offs_bh * stride_lse_h, lse_val, mask=head_mask)


# ============================================================
# v99 kernel — Explicit software pipelining
# ============================================================


@triton.autotune(
    configs=[
        triton.Config({"BK": 128, "BH": 16}, num_warps=4, num_stages=4),
        triton.Config({"BK": 128, "BH": 16}, num_warps=4, num_stages=3),
        triton.Config({"BK": 128, "BH": 16}, num_warps=8, num_stages=4),
        triton.Config({"BK": 128, "BH": 16}, num_warps=2, num_stages=4),
        triton.Config({"BK": 128, "BH": 32}, num_warps=4, num_stages=4),
        triton.Config({"BK": 64, "BH": 16}, num_warps=4, num_stages=4),
        triton.Config({"BK": 64, "BH": 16}, num_warps=4, num_stages=3),
        triton.Config({"BK": 64, "BH": 16}, num_warps=8, num_stages=4),
        triton.Config({"BK": 64, "BH": 32}, num_warps=4, num_stages=4),
        triton.Config({"BK": 64, "BH": 64}, num_warps=4, num_stages=3),
    ],
    key=["topk", "num_qo_heads", "D_CKV", "HAS_KPE"],
)
@triton.jit
def _kernel_v99(
    q_nope_ptr,
    q_pe_ptr,
    kv_cache_ptr,
    sparse_indices_ptr,
    output_ptr,
    lse_ptr,
    num_tokens,
    num_qo_heads,
    topk,
    stride_qn_b,
    stride_qn_h,
    stride_qn_d,
    stride_qp_b,
    stride_qp_h,
    stride_qp_d,
    stride_kv_row,
    stride_kv_d,
    stride_sparse_b,
    stride_sparse_t,
    stride_out_b,
    stride_out_h,
    stride_out_d,
    stride_lse_b,
    stride_lse_h,
    sm_scale: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    D_CKV: tl.constexpr,
    D_KPE: tl.constexpr,
    HAS_KPE: tl.constexpr,
):
    num_head_blocks: tl.constexpr = (num_qo_heads + BH - 1) // BH
    pid = tl.program_id(0)
    token_idx = pid // num_head_blocks
    head_block_idx = pid % num_head_blocks

    offs_bh = tl.arange(0, BH)
    offs_d_nope = tl.arange(0, D_CKV)
    offs_bk = tl.arange(0, BK)
    head_mask = head_block_idx * BH + offs_bh < num_qo_heads
    log_scale: tl.constexpr = sm_scale * 1.44269504

    q_base = q_nope_ptr + token_idx * stride_qn_b + head_block_idx * BH * stride_qn_h
    q_nope = tl.load(
        q_base + offs_bh[:, None] * stride_qn_h + offs_d_nope[None, :] * stride_qn_d,
        mask=head_mask[:, None],
        other=0.0,
    )
    if HAS_KPE:
        offs_d_pe = tl.arange(0, D_KPE)
        qp_base = q_pe_ptr + token_idx * stride_qp_b + head_block_idx * BH * stride_qp_h
        q_pe = tl.load(
            qp_base + offs_bh[:, None] * stride_qp_h + offs_d_pe[None, :] * stride_qp_d,
            mask=head_mask[:, None],
            other=0.0,
        )

    m_i = tl.full([BH], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BH], dtype=tl.float32)
    acc = tl.zeros([BH, D_CKV], dtype=tl.float32)
    idx_base = sparse_indices_ptr + token_idx * stride_sparse_b
    num_iters = (topk + BK - 1) // BK

    for iter_idx in range(num_iters):
        offset = iter_idx * BK
        kv_mask = offset + offs_bk < topk
        flat_kv_idx = tl.load(
            idx_base + (offset + offs_bk) * stride_sparse_t,
            mask=kv_mask,
            other=0,
        )
        row_offsets = flat_kv_idx * stride_kv_row
        k_nope = tl.load(
            kv_cache_ptr + row_offsets[:, None] + offs_d_nope[None, :] * stride_kv_d,
            mask=kv_mask[:, None],
            other=0.0,
        )

        if HAS_KPE:
            k_pe = tl.load(
                kv_cache_ptr
                + row_offsets[:, None]
                + (D_CKV + offs_d_pe)[None, :] * stride_kv_d,
                mask=kv_mask[:, None],
                other=0.0,
            )
            qk = tl.dot(q_nope, tl.trans(k_nope), out_dtype=tl.float32)
            qk = tl.dot(q_pe, tl.trans(k_pe), qk, out_dtype=tl.float32)
        else:
            qk = tl.dot(q_nope, tl.trans(k_nope), out_dtype=tl.float32)

        qk = qk * log_scale
        qk = tl.where(kv_mask[None, :], qk, float("-inf"))

        qk_max = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, qk_max)
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), k_nope, acc)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    acc = acc / l_i[:, None]
    lse_val = m_i * 0.6931471805599453 + tl.log(l_i)

    o_base = output_ptr + token_idx * stride_out_b + head_block_idx * BH * stride_out_h
    tl.store(
        o_base + offs_bh[:, None] * stride_out_h + offs_d_nope[None, :] * stride_out_d,
        acc.to(tl.bfloat16),
        mask=head_mask[:, None],
    )
    l_base = lse_ptr + token_idx * stride_lse_b + head_block_idx * BH * stride_lse_h
    tl.store(l_base + offs_bh * stride_lse_h, lse_val, mask=head_mask)


# ============================================================
# Shared launcher
# ============================================================


def _run_kernel(
    kernel_fn,
    q_nope,
    q_pe,
    kv_cache,
    sparse_indices,
    sm_scale,
    output,
    lse,
    d_ckv,
    d_kpe,
    has_kpe=True,
):
    num_tokens, num_qo_heads, _ = q_nope.shape
    topk = sparse_indices.shape[-1]
    D_CKV = triton.next_power_of_2(d_ckv)
    D_KPE = triton.next_power_of_2(d_kpe) if d_kpe > 0 else 16

    def grid(META):
        return (triton.cdiv(num_qo_heads, META["BH"]) * num_tokens,)

    kernel_fn[grid](
        q_nope,
        q_pe,
        kv_cache,
        sparse_indices,
        output,
        lse,
        num_tokens,
        num_qo_heads,
        topk,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        q_pe.stride(0),
        q_pe.stride(1),
        q_pe.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        sparse_indices.stride(0),
        sparse_indices.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        sm_scale=sm_scale,
        D_CKV=D_CKV,
        D_KPE=D_KPE,
        HAS_KPE=has_kpe,
    )
    return output, lse


# ============================================================
# Unified entry point with s_q dispatch
# ============================================================


def sparse_prefill_fwd(q, kv, indices, sm_scale, d_v, attn_sink=None, topk_length=None):
    """Sparse MLA forward with automatic kernel dispatch.

    Uses v99 for s_q <= SQ_THRESHOLD (decode), v91 otherwise (prefill).

    Args:
        q:       [s_q, h_q, d_qk]       bf16
        kv:      [s_kv, h_kv, d_qk]     bf16
        indices: [s_q, h_kv, topk]       int32
    Returns:
        (output, max_logits, lse)
    """
    logger.debug("GEMS SPARSE MLA")
    s_q, h_q, d_qk = q.shape
    d_kpe = d_qk - d_v
    MIN_KPE = 16

    q_nope = q[:, :, :d_v].contiguous()
    if d_kpe > 0:
        q_pe = q[:, :, d_v:].contiguous()
    else:
        q_pe = torch.zeros(s_q, h_q, MIN_KPE, dtype=q.dtype, device=q.device)

    kv_cache = kv.squeeze(1)
    sparse_indices = indices.squeeze(1).contiguous()
    output = torch.zeros(s_q, h_q, d_v, dtype=q.dtype, device=q.device)
    lse = torch.zeros(s_q, h_q, dtype=torch.float32, device=q.device)
    has_kpe = d_kpe > 0

    # Dispatch
    kernel_fn = _kernel_v99 if s_q <= SQ_THRESHOLD else _kernel_v91

    _run_kernel(
        kernel_fn,
        q_nope,
        q_pe,
        kv_cache,
        sparse_indices,
        sm_scale,
        output,
        lse,
        d_ckv=d_v,
        d_kpe=d_kpe if d_kpe > 0 else MIN_KPE,
        has_kpe=has_kpe,
    )

    max_logits = torch.zeros(s_q, h_q, dtype=torch.float32, device=q.device)
    return output, max_logits, lse
