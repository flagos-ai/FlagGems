"""Triton flash_mla_with_kvcache for DeepSeek V4 sparse MLA decode attention.

Replaces vLLM's FlashMLA CUDA kernel for the decode path with a pure Triton
implementation using token-parallel split-reduce.

Background:
    In DeepSeek V4 decode, each step computes sparse attention over selected
    KV cache tokens via indices (SWA + optional extra compressed cache).
    The KV cache uses FP8 (E4M3) for NoPE dims with UE8M0 per-64-dim block
    scales and BF16 for RoPE dims, stored in a Structure-of-Arrays (SoA)
    layout per page block.

Strategy:
    Token-parallel split-reduce approach:
    1. Split topk tokens across SPLIT_TOKENS programs for parallelism.
    2. Each program computes partial (m, l, acc) independently via online
       softmax — no loop-carried dependencies across splits.
    3. A lightweight reduction kernel combines partial results with
       numerically stable rescaling.

Performance (DeepSeek V4 config, H20 GPU):
    - B=1,  topk=128:     0.21x vs vLLM CUDA
    - B=4,  topk=128:     0.37x vs vLLM CUDA
    - B=32, topk=128:     0.57x vs vLLM CUDA
    - B=1,  topk=128+256: 0.25x vs vLLM CUDA
    - B=32, topk=128+256: 0.51x vs vLLM CUDA
"""

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# DeepSeek V4 MLA cache layout constants
_HEAD_DIM = 512
_NOPE_DIM = 448
_ROPE_DIM = 64
# FP8 NoPE (448 bytes) + BF16 RoPE (64 * 2 = 128 bytes)
_TOKEN_DATA_BYTES = 576
# 7 UE8M0 scales (one per 64-dim NoPE block) + 1 padding byte
_SCALE_BYTES = 8
_HEAD_BYTES = _TOKEN_DATA_BYTES + _SCALE_BYTES  # 584

# 1 / ln(2) for exp2-based softmax
_LOG2E = 1.4426950408889634
# ln(2) for converting log2 back to natural log
_LN2 = 0.6931471805599453


def _get_split_tokens(max_topk):
    """Select token-parallel split count based on topk size.

    More splits = more SM parallelism, but each split processes fewer tokens.
    Returns 1/2/4 depending on topk to balance parallelism vs. overhead.
    """
    if max_topk <= 32:
        return 1
    elif max_topk <= 64:
        return 2
    return 4


_DECODE_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=2),
]


@triton.autotune(
    configs=_DECODE_CONFIGS,
    key=["MAX_TOPK", "MAX_EXTRA_TOPK", "HAS_EXTRA"],
)
@triton.jit
def _flash_mla_sparse_decode_kernel(
    Q,
    K_CACHE,
    INDICES,
    TOPK_LEN,
    ATTN_SINK,
    EXTRA_K_CACHE,
    EXTRA_INDICES,
    EXTRA_TOPK_LEN,
    PARTIAL_M,
    PARTIAL_L,
    PARTIAL_ACC,
    stride_qb,
    stride_qs,
    stride_qh,
    stride_qd,
    k_block_stride,
    extra_k_block_stride,
    stride_idx_b,
    stride_idx_s,
    stride_idx_t,
    stride_eidx_b,
    stride_eidx_s,
    stride_eidx_t,
    softmax_scale,
    page_block_size,
    extra_page_block_size,
    batch_size,
    num_heads,
    TOKEN_DATA_BYTES_C: tl.constexpr,
    SCALE_BYTES_C: tl.constexpr,
    NOPE_DIM_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_TOPK: tl.constexpr,
    MAX_EXTRA_TOPK: tl.constexpr,
    SPLIT_TOKENS: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    HAS_ATTN_SINK: tl.constexpr,
):
    """Compute partial attention for a subset of topk tokens.

    Each program handles tokens [t_start, t_end) and writes partial
    (m, l, acc) to global memory for later reduction.
    """
    pid = tl.program_id(0)
    split_idx = pid % SPLIT_TOKENS
    bh_idx = pid // SPLIT_TOKENS
    b_idx = bh_idx // num_heads
    h_idx = bh_idx % num_heads

    if b_idx >= batch_size:
        return

    q_base = b_idx * stride_qb + h_idx * stride_qh
    d_range = tl.arange(0, BLOCK_D)

    # Load Q into registers — reused across all tokens in this split
    q0 = tl.load(Q + q_base + 0 * BLOCK_D + d_range).to(tl.float32)
    q1 = tl.load(Q + q_base + 1 * BLOCK_D + d_range).to(tl.float32)
    q2 = tl.load(Q + q_base + 2 * BLOCK_D + d_range).to(tl.float32)
    q3 = tl.load(Q + q_base + 3 * BLOCK_D + d_range).to(tl.float32)

    m_val = -float("inf")
    l_val = 0.0

    acc0 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_D], dtype=tl.float32)

    topk = tl.load(TOPK_LEN + b_idx).to(tl.int32)

    # Token range for this split
    tokens_per_split = (MAX_TOPK + SPLIT_TOKENS - 1) // SPLIT_TOKENS
    t_start = split_idx * tokens_per_split
    t_end = tl.minimum((split_idx + 1) * tokens_per_split, MAX_TOPK)

    # Masks for splitting 128-dim blocks into two 64-dim halves.
    # NoPE uses 7 FP8 blocks of 64 dims each; the 8th 64-dim slice is BF16 RoPE.
    mask_lo = d_range < 64
    mask_hi = d_range >= 64

    for t in range(t_start, t_end):
        if t < topk:
            token_idx = tl.load(
                INDICES + b_idx * stride_idx_b + t * stride_idx_t
            ).to(tl.int32)
            page = token_idx // page_block_size
            offset = token_idx % page_block_size

            data_base = page * k_block_stride + offset * TOKEN_DATA_BYTES_C
            scale_base = (
                page * k_block_stride
                + page_block_size * TOKEN_DATA_BYTES_C
                + offset * SCALE_BYTES_C
            )

            fp8_ptr = (K_CACHE + data_base).to(tl.pointer_type(tl.float8e4nv))

            # Load 7 UE8M0 scales — one per 64-dim NoPE block
            s0 = tl.math.exp2(
                (tl.load(K_CACHE + scale_base + 0).to(tl.int32) - 127).to(tl.float32)
            )
            s1 = tl.math.exp2(
                (tl.load(K_CACHE + scale_base + 1).to(tl.int32) - 127).to(tl.float32)
            )
            s2 = tl.math.exp2(
                (tl.load(K_CACHE + scale_base + 2).to(tl.int32) - 127).to(tl.float32)
            )
            s3 = tl.math.exp2(
                (tl.load(K_CACHE + scale_base + 3).to(tl.int32) - 127).to(tl.float32)
            )
            s4 = tl.math.exp2(
                (tl.load(K_CACHE + scale_base + 4).to(tl.int32) - 127).to(tl.float32)
            )
            s5 = tl.math.exp2(
                (tl.load(K_CACHE + scale_base + 5).to(tl.int32) - 127).to(tl.float32)
            )
            s6 = tl.math.exp2(
                (tl.load(K_CACHE + scale_base + 6).to(tl.int32) - 127).to(tl.float32)
            )

            # Dequantize 4 x BLOCK_D chunks covering HEAD_DIM=512.
            # Each BLOCK_D=128 chunk spans two 64-dim FP8 blocks with distinct scales.
            kv0_lo = (
                tl.load(
                    fp8_ptr + tl.where(mask_lo, d_range, 0),
                    mask=mask_lo,
                    other=0.0,
                ).to(tl.float32)
                * s0
            )
            kv0_hi = (
                tl.load(
                    fp8_ptr + 64 + tl.where(mask_hi, d_range - 64, 0),
                    mask=mask_hi,
                    other=0.0,
                ).to(tl.float32)
                * s1
            )
            kv0 = tl.where(mask_lo, kv0_lo, kv0_hi)

            kv1_lo = (
                tl.load(
                    fp8_ptr + 128 + tl.where(mask_lo, d_range, 0),
                    mask=mask_lo,
                    other=0.0,
                ).to(tl.float32)
                * s2
            )
            kv1_hi = (
                tl.load(
                    fp8_ptr + 192 + tl.where(mask_hi, d_range - 64, 0),
                    mask=mask_hi,
                    other=0.0,
                ).to(tl.float32)
                * s3
            )
            kv1 = tl.where(mask_lo, kv1_lo, kv1_hi)

            kv2_lo = (
                tl.load(
                    fp8_ptr + 256 + tl.where(mask_lo, d_range, 0),
                    mask=mask_lo,
                    other=0.0,
                ).to(tl.float32)
                * s4
            )
            kv2_hi = (
                tl.load(
                    fp8_ptr + 320 + tl.where(mask_hi, d_range - 64, 0),
                    mask=mask_hi,
                    other=0.0,
                ).to(tl.float32)
                * s5
            )
            kv2 = tl.where(mask_lo, kv2_lo, kv2_hi)

            # Last chunk: 64 FP8 dims (NoPE block 6) + 64 BF16 dims (RoPE)
            kv3_fp8 = (
                tl.load(
                    fp8_ptr + 384 + tl.where(mask_lo, d_range, 0),
                    mask=mask_lo,
                    other=0.0,
                ).to(tl.float32)
                * s6
            )
            bf16_ptr = (K_CACHE + data_base + NOPE_DIM_C).to(
                tl.pointer_type(tl.bfloat16)
            )
            kv3_bf16 = tl.load(
                bf16_ptr + tl.where(mask_hi, d_range - 64, 0),
                mask=mask_hi,
                other=0.0,
            ).to(tl.float32)
            kv3 = tl.where(mask_lo, kv3_fp8, kv3_bf16)

            # Fused dot product across all 4 chunks
            score = tl.sum(q0 * kv0 + q1 * kv1 + q2 * kv2 + q3 * kv3)
            s = score * softmax_scale

            # Online softmax update
            m_new = tl.maximum(m_val, s)
            alpha = tl.math.exp2((m_val - m_new) * _LOG2E)
            p = tl.math.exp2((s - m_new) * _LOG2E)
            l_val = l_val * alpha + p

            acc0 = acc0 * alpha + p * kv0
            acc1 = acc1 * alpha + p * kv1
            acc2 = acc2 * alpha + p * kv2
            acc3 = acc3 * alpha + p * kv3

            m_val = m_new

    # Process extra (compressed) KV cache tokens if present
    if HAS_EXTRA:
        extra_topk = tl.load(EXTRA_TOPK_LEN + b_idx).to(tl.int32)

        extra_tokens_per_split = (MAX_EXTRA_TOPK + SPLIT_TOKENS - 1) // SPLIT_TOKENS
        et_start = split_idx * extra_tokens_per_split
        et_end = tl.minimum(
            (split_idx + 1) * extra_tokens_per_split, MAX_EXTRA_TOPK
        )

        for t in range(et_start, et_end):
            if t < extra_topk:
                token_idx = tl.load(
                    EXTRA_INDICES + b_idx * stride_eidx_b + t * stride_eidx_t
                ).to(tl.int32)
                page = token_idx // extra_page_block_size
                offset = token_idx % extra_page_block_size

                data_base = (
                    page * extra_k_block_stride + offset * TOKEN_DATA_BYTES_C
                )
                scale_base = (
                    page * extra_k_block_stride
                    + extra_page_block_size * TOKEN_DATA_BYTES_C
                    + offset * SCALE_BYTES_C
                )

                fp8_ptr = (EXTRA_K_CACHE + data_base).to(
                    tl.pointer_type(tl.float8e4nv)
                )

                s0 = tl.math.exp2(
                    (tl.load(EXTRA_K_CACHE + scale_base + 0).to(tl.int32) - 127).to(
                        tl.float32
                    )
                )
                s1 = tl.math.exp2(
                    (tl.load(EXTRA_K_CACHE + scale_base + 1).to(tl.int32) - 127).to(
                        tl.float32
                    )
                )
                s2 = tl.math.exp2(
                    (tl.load(EXTRA_K_CACHE + scale_base + 2).to(tl.int32) - 127).to(
                        tl.float32
                    )
                )
                s3 = tl.math.exp2(
                    (tl.load(EXTRA_K_CACHE + scale_base + 3).to(tl.int32) - 127).to(
                        tl.float32
                    )
                )
                s4 = tl.math.exp2(
                    (tl.load(EXTRA_K_CACHE + scale_base + 4).to(tl.int32) - 127).to(
                        tl.float32
                    )
                )
                s5 = tl.math.exp2(
                    (tl.load(EXTRA_K_CACHE + scale_base + 5).to(tl.int32) - 127).to(
                        tl.float32
                    )
                )
                s6 = tl.math.exp2(
                    (tl.load(EXTRA_K_CACHE + scale_base + 6).to(tl.int32) - 127).to(
                        tl.float32
                    )
                )

                kv0_lo = (
                    tl.load(
                        fp8_ptr + tl.where(mask_lo, d_range, 0),
                        mask=mask_lo,
                        other=0.0,
                    ).to(tl.float32)
                    * s0
                )
                kv0_hi = (
                    tl.load(
                        fp8_ptr + 64 + tl.where(mask_hi, d_range - 64, 0),
                        mask=mask_hi,
                        other=0.0,
                    ).to(tl.float32)
                    * s1
                )
                kv0 = tl.where(mask_lo, kv0_lo, kv0_hi)

                kv1_lo = (
                    tl.load(
                        fp8_ptr + 128 + tl.where(mask_lo, d_range, 0),
                        mask=mask_lo,
                        other=0.0,
                    ).to(tl.float32)
                    * s2
                )
                kv1_hi = (
                    tl.load(
                        fp8_ptr + 192 + tl.where(mask_hi, d_range - 64, 0),
                        mask=mask_hi,
                        other=0.0,
                    ).to(tl.float32)
                    * s3
                )
                kv1 = tl.where(mask_lo, kv1_lo, kv1_hi)

                kv2_lo = (
                    tl.load(
                        fp8_ptr + 256 + tl.where(mask_lo, d_range, 0),
                        mask=mask_lo,
                        other=0.0,
                    ).to(tl.float32)
                    * s4
                )
                kv2_hi = (
                    tl.load(
                        fp8_ptr + 320 + tl.where(mask_hi, d_range - 64, 0),
                        mask=mask_hi,
                        other=0.0,
                    ).to(tl.float32)
                    * s5
                )
                kv2 = tl.where(mask_lo, kv2_lo, kv2_hi)

                kv3_fp8 = (
                    tl.load(
                        fp8_ptr + 384 + tl.where(mask_lo, d_range, 0),
                        mask=mask_lo,
                        other=0.0,
                    ).to(tl.float32)
                    * s6
                )
                bf16_ptr = (EXTRA_K_CACHE + data_base + NOPE_DIM_C).to(
                    tl.pointer_type(tl.bfloat16)
                )
                kv3_bf16 = tl.load(
                    bf16_ptr + tl.where(mask_hi, d_range - 64, 0),
                    mask=mask_hi,
                    other=0.0,
                ).to(tl.float32)
                kv3 = tl.where(mask_lo, kv3_fp8, kv3_bf16)

                score = tl.sum(q0 * kv0 + q1 * kv1 + q2 * kv2 + q3 * kv3)
                s = score * softmax_scale

                m_new = tl.maximum(m_val, s)
                alpha = tl.math.exp2((m_val - m_new) * _LOG2E)
                p = tl.math.exp2((s - m_new) * _LOG2E)
                l_val = l_val * alpha + p

                acc0 = acc0 * alpha + p * kv0
                acc1 = acc1 * alpha + p * kv1
                acc2 = acc2 * alpha + p * kv2
                acc3 = acc3 * alpha + p * kv3

                m_val = m_new

    # Store partial results for reduction
    partial_base = b_idx * num_heads * SPLIT_TOKENS + h_idx * SPLIT_TOKENS + split_idx
    tl.store(PARTIAL_M + partial_base, m_val)
    tl.store(PARTIAL_L + partial_base, l_val)

    acc_base = partial_base * _HEAD_DIM
    tl.store(PARTIAL_ACC + acc_base + 0 * BLOCK_D + d_range, acc0)
    tl.store(PARTIAL_ACC + acc_base + 1 * BLOCK_D + d_range, acc1)
    tl.store(PARTIAL_ACC + acc_base + 2 * BLOCK_D + d_range, acc2)
    tl.store(PARTIAL_ACC + acc_base + 3 * BLOCK_D + d_range, acc3)


@triton.jit
def _flash_mla_reduction_kernel(
    PARTIAL_M,
    PARTIAL_L,
    PARTIAL_ACC,
    ATTN_SINK,
    OUT,
    LSE,
    stride_ob,
    stride_os,
    stride_oh,
    stride_od,
    batch_size,
    num_heads,
    SPLIT_TOKENS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_ATTN_SINK: tl.constexpr,
):
    """Reduce partial (m, l, acc) from split-token programs into final output."""
    pid = tl.program_id(0)
    b_idx = pid // num_heads
    h_idx = pid % num_heads

    if b_idx >= batch_size:
        return

    d_range = tl.arange(0, BLOCK_D)
    partial_base = b_idx * num_heads * SPLIT_TOKENS + h_idx * SPLIT_TOKENS

    # Find global max across all splits
    m_global = -float("inf")
    for s in range(SPLIT_TOKENS):
        m_s = tl.load(PARTIAL_M + partial_base + s)
        m_global = tl.maximum(m_global, m_s)

    # Rescale and accumulate
    l_global = 0.0
    acc0 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_D], dtype=tl.float32)

    for s in range(SPLIT_TOKENS):
        m_s = tl.load(PARTIAL_M + partial_base + s)
        l_s = tl.load(PARTIAL_L + partial_base + s)

        alpha = tl.math.exp2((m_s - m_global) * _LOG2E)
        l_global = l_global + l_s * alpha

        acc_base = (partial_base + s) * _HEAD_DIM
        acc0 += tl.load(PARTIAL_ACC + acc_base + 0 * BLOCK_D + d_range) * alpha
        acc1 += tl.load(PARTIAL_ACC + acc_base + 1 * BLOCK_D + d_range) * alpha
        acc2 += tl.load(PARTIAL_ACC + acc_base + 2 * BLOCK_D + d_range) * alpha
        acc3 += tl.load(PARTIAL_ACC + acc_base + 3 * BLOCK_D + d_range) * alpha

    # Normalize by total attention weight
    if l_global > 0:
        inv_l = 1.0 / l_global
        acc0 *= inv_l
        acc1 *= inv_l
        acc2 *= inv_l
        acc3 *= inv_l

    lse_val = m_global + tl.math.log2(l_global) * _LN2

    # Attention sink: scale output by sigmoid(lse - sink) to dampen sink tokens
    if HAS_ATTN_SINK:
        sink_val = tl.load(ATTN_SINK + h_idx)
        sink_scale = 1.0 / (
            1.0 + tl.math.exp2((sink_val - lse_val) * _LOG2E)
        )
        acc0 *= sink_scale
        acc1 *= sink_scale
        acc2 *= sink_scale
        acc3 *= sink_scale

    # Store final output
    o_base = b_idx * stride_ob + h_idx * stride_oh
    tl.store(OUT + o_base + 0 * BLOCK_D + d_range, acc0.to(tl.bfloat16))
    tl.store(OUT + o_base + 1 * BLOCK_D + d_range, acc1.to(tl.bfloat16))
    tl.store(OUT + o_base + 2 * BLOCK_D + d_range, acc2.to(tl.bfloat16))
    tl.store(OUT + o_base + 3 * BLOCK_D + d_range, acc3.to(tl.bfloat16))
    tl.store(LSE + b_idx * num_heads + h_idx, lse_val)


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata,
    num_splits: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton sparse MLA decode attention with FP8 KV cache.

    Signature matches vLLM FlashMLA CUDA kernel for drop-in compatibility.
    Accepts both 2D (num_blocks, block_stride) and 4D
    (num_blocks, block_size, 1, HEAD_BYTES) KV cache layouts.

    Args:
        q: (B, 1, H_q, 512) bf16 query tensor
        k_cache: KV cache in uint8, either 2D or 4D layout
        block_table: Not used in sparse mode
        cache_seqlens: Not used in sparse mode
        head_dim_v: Must be 512
        tile_scheduler_metadata: Not used (compatibility placeholder)
        num_splits: Not used
        softmax_scale: Attention scale (default: head_dim_v ** -0.5)
        causal: Not used in sparse decode
        is_fp8_kvcache: Must be True
        indices: (B, 1, topk) int32 sparse token indices
        attn_sink: (H_q,) float32 attention sink bias (optional)
        extra_k_cache: Extra KV cache for compressed tokens (optional)
        extra_indices_in_kvcache: (B, 1, extra_topk) int32 (optional)
        topk_length: (B,) int32 valid index count per batch
        extra_topk_length: (B,) int32 extra valid index count (optional)
        out: Pre-allocated output tensor (optional)

    Returns:
        out: (B, 1, H_q, 512) bf16
        lse: (B, H_q, 1) float32
    """
    logger.debug("GEMS FLASH_MLA_WITH_KVCACHE")

    B, S_q, H_q, D = q.shape
    assert S_q == 1, "Only decode (seq_len_q=1) is supported"
    assert D == _HEAD_DIM == head_dim_v, f"head_dim_v must be {_HEAD_DIM}"
    assert is_fp8_kvcache, "is_fp8_kvcache must be True"
    assert indices is not None, "indices must be provided for sparse attention"
    assert topk_length is not None, "topk_length must be provided"

    if softmax_scale is None:
        softmax_scale = head_dim_v ** -0.5

    # Handle 4D cache layout (num_blocks, block_size, 1, HEAD_BYTES) -> 2D
    if k_cache.ndim == 4:
        k_cache = k_cache.reshape(k_cache.shape[0], -1)
    if extra_k_cache is not None and extra_k_cache.ndim == 4:
        extra_k_cache = extra_k_cache.reshape(extra_k_cache.shape[0], -1)

    if out is None:
        out = torch.empty(B, 1, H_q, D, dtype=q.dtype, device=q.device)
    lse = torch.empty(B, H_q, dtype=torch.float32, device=q.device)

    block_stride = k_cache.shape[1]
    block_size = block_stride // _HEAD_BYTES

    HAS_EXTRA = (
        extra_k_cache is not None and extra_indices_in_kvcache is not None
    )
    HAS_ATTN_SINK = attn_sink is not None and not torch.all(
        attn_sink == float("-inf")
    )

    MAX_TOPK = indices.shape[-1]
    MAX_EXTRA_TOPK = (
        extra_indices_in_kvcache.shape[-1] if HAS_EXTRA else 1
    )

    k_block_stride = k_cache.stride(0)
    extra_k_block_stride = extra_k_cache.stride(0) if HAS_EXTRA else 0

    extra_block_size = 1
    if HAS_EXTRA:
        extra_block_stride = extra_k_cache.shape[1]
        extra_block_size = extra_block_stride // _HEAD_BYTES

    SPLIT_TOKENS = _get_split_tokens(MAX_TOPK)
    BLOCK_D = 128

    # Partial buffers for split-token reduction
    partial_m = torch.empty(
        B * H_q * SPLIT_TOKENS, dtype=torch.float32, device=q.device
    )
    partial_l = torch.empty(
        B * H_q * SPLIT_TOKENS, dtype=torch.float32, device=q.device
    )
    partial_acc = torch.empty(
        B * H_q * SPLIT_TOKENS * _HEAD_DIM, dtype=torch.float32, device=q.device
    )

    # Placeholder tensors for optional arguments
    empty_u8 = torch.empty(0, device=q.device, dtype=torch.uint8)
    empty_i32 = torch.empty(0, device=q.device, dtype=torch.int32)
    empty_f32 = torch.empty(0, device=q.device, dtype=torch.float32)

    grid = (B * H_q * SPLIT_TOKENS,)

    _flash_mla_sparse_decode_kernel[grid](
        q,
        k_cache,
        indices,
        topk_length,
        attn_sink if attn_sink is not None else empty_f32,
        extra_k_cache if HAS_EXTRA else empty_u8,
        extra_indices_in_kvcache if HAS_EXTRA else empty_i32,
        extra_topk_length if HAS_EXTRA else empty_i32,
        partial_m,
        partial_l,
        partial_acc,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k_block_stride,
        extra_k_block_stride,
        indices.stride(0),
        indices.stride(1),
        indices.stride(2),
        extra_indices_in_kvcache.stride(0) if HAS_EXTRA else 0,
        extra_indices_in_kvcache.stride(1) if HAS_EXTRA else 0,
        extra_indices_in_kvcache.stride(2) if HAS_EXTRA else 0,
        softmax_scale,
        block_size,
        extra_block_size,
        B,
        H_q,
        TOKEN_DATA_BYTES_C=_TOKEN_DATA_BYTES,
        SCALE_BYTES_C=_SCALE_BYTES,
        NOPE_DIM_C=_NOPE_DIM,
        BLOCK_D=BLOCK_D,
        MAX_TOPK=MAX_TOPK,
        MAX_EXTRA_TOPK=MAX_EXTRA_TOPK,
        SPLIT_TOKENS=SPLIT_TOKENS,
        HAS_EXTRA=HAS_EXTRA,
        HAS_ATTN_SINK=False,
    )

    grid = (B * H_q,)

    _flash_mla_reduction_kernel[grid](
        partial_m,
        partial_l,
        partial_acc,
        attn_sink if attn_sink is not None else empty_f32,
        out,
        lse,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        B,
        H_q,
        SPLIT_TOKENS=SPLIT_TOKENS,
        BLOCK_D=BLOCK_D,
        HAS_ATTN_SINK=HAS_ATTN_SINK,
        num_warps=4,
        num_stages=1,
    )

    return out, lse.unsqueeze(-1)
