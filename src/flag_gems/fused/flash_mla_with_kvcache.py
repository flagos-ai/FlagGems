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
Triton implementation of flash_mla_with_kvcache for MLA attention.
Supports both sparse (FP8 KV cache + topk indices) and dense (paged attention) modes.
Only supports sm90 (Hopper) architecture.
"""

import dataclasses
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.utils.triton_version_utils import has_triton_tle

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE = True
    except ImportError:
        tle = None
        HAS_TLE = False
else:
    tle = None
    HAS_TLE = False


# TLE constants for decode
TLE_DECODE_BK = 64
TLE_DECODE_BH = 64
TLE_DECODE_PAIR_BLOCKS = 2
TLE_DECODE_WORKER_NUM_WARPS = 4


# ============================================================================
# Data structures (compatible with original CUDA interface)
# ============================================================================


@dataclasses.dataclass
class FlashMLASchedMeta:
    """Stores tile scheduler metadata for FlashMLA."""

    @dataclasses.dataclass
    class Config:
        b: int
        s_q: int
        h_q: int
        page_block_size: int
        h_k: int
        causal: bool
        is_fp8_kvcache: bool
        topk: Optional[int]
        extra_page_block_size: Optional[int]
        extra_topk: Optional[int]

    have_initialized: bool = False
    config: Optional[Config] = None
    tile_scheduler_metadata: Optional[torch.Tensor] = None
    num_splits: Optional[torch.Tensor] = None


def get_mla_metadata(*args, **kwargs) -> Tuple[FlashMLASchedMeta, None]:
    """Returns an empty FlashMLASchedMeta instance."""
    return FlashMLASchedMeta(), None


# ============================================================================
# Sparse decode kernel (FP8 KV cache + topk indices)
#
# KV cache layout per token (656 bytes total):
#   [0:512]   - NoPE part: 512 float8_e4m3 values
#   [512:528] - Scale factors: 4 float32 values (each for 128 FP8 values)
#   [528:656] - RoPE part: 64 bfloat16 values
#
# The NoPE part (after dequantization) serves as BOTH K and V for MLA.
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BK": 64, "BH": 64}, num_warps=8, num_stages=2),
        triton.Config({"BK": 64, "BH": 64}, num_warps=8, num_stages=4),
    ],
    key=["HQ", "DQK", "TOPK", "HAVE_ATTN_SINK", "HAVE_TOPK_LENGTH", "IS_FP8"],
)
@triton.jit
def _sparse_decode_kernel(
    q,
    kv,
    kv_scales,
    kv_rope,
    indices,
    attn_sink,
    topk_length,
    sm_scale: tl.constexpr,
    output,
    lse,
    stride_qb,
    stride_qsq,
    stride_qh,
    stride_kvn,
    stride_scales_n,
    stride_rope_n,
    stride_ib,
    stride_isq,
    stride_ob,
    stride_osq,
    stride_oh,
    stride_lseb,
    stride_lseh,
    SQ,
    HQ: tl.constexpr,
    DQK: tl.constexpr,
    SKV,
    TOPK: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    IS_FP8: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
):
    """
    Sparse decode kernel with online softmax.
    Grid: (batch_size * seq_q * ceil(HQ / BH),)
    Each program handles BH heads for one (batch, seq_q) position.

    For FP8 mode:
      - kv: [num_tokens, 512] float8_e4m3fn (NoPE part)
      - kv_scales: [num_tokens, 4] float32 (per-128-element scales)
      - kv_rope: [num_tokens, 64] bfloat16 (RoPE part)
    For BF16 mode:
      - kv: [num_tokens, DQK] bfloat16 (full KV)
      - kv_scales, kv_rope: unused
    """
    num_head_blocks: tl.constexpr = (HQ + BH - 1) // BH
    pid = tl.program_id(0)
    i_b = pid // (SQ * num_head_blocks)
    remainder = pid % (SQ * num_head_blocks)
    i_sq = remainder // num_head_blocks
    i_sq = i_sq.to(tl.int64)
    i_gbh = remainder % num_head_blocks
    gbh_base = i_gbh * BH

    DP: tl.constexpr = 512
    BDP: tl.constexpr = 256

    # Base pointers
    q_base = q + i_b * stride_qb + i_sq * stride_qsq + gbh_base * stride_qh
    kv_base = kv
    t_base = indices + i_b * stride_ib + i_sq * stride_isq
    attn_sink_ptr = attn_sink + gbh_base if HAVE_ATTN_SINK else 0
    topk_length_ptr = topk_length + i_b if HAVE_TOPK_LENGTH else 0
    o_base = output + i_b * stride_ob + i_sq * stride_osq + gbh_base * stride_oh
    l_base = lse + i_b * stride_lseb + gbh_base * stride_lseh + i_sq

    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, BDP)
    if DQK == 576:
        offs_td = tl.arange(0, 64)
    offs_t = tl.arange(0, BK)

    # Load Q in two halves [BH, 256] x 2
    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :]
    q_blk0 = tl.load(q_ptr, eviction_policy="evict_first")
    q_blk1 = tl.load(q_ptr + BDP, eviction_policy="evict_first")
    if DQK == 576:
        tq_ptr = q_base + DP + offs_h[:, None] * stride_qh + offs_td[None, :]
        tq_blk = tl.load(tq_ptr, eviction_policy="evict_first")

    # Online softmax accumulators
    max_log = tl.full([BH], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
    acc0 = tl.zeros([BH, BDP], dtype=tl.float32)
    acc1 = tl.zeros([BH, BDP], dtype=tl.float32)

    topk_len = tl.load(topk_length_ptr) if HAVE_TOPK_LENGTH else TOPK
    NK = tl.cdiv(topk_len, BK)
    for ck in range(NK):
        # Load indices
        t_ptr = BK * ck + offs_t
        t_msk = t_ptr < topk_len
        t_ptr += t_base
        kv_ids = tl.load(t_ptr, t_msk, other=-1)
        mask_ids = (kv_ids < SKV) & (kv_ids >= 0)
        kv_ids = tl.where(mask_ids, kv_ids, 0)

        if IS_FP8:
            # FP8 mode: load FP8 values and dequantize with per-128-element scales
            # Load NoPE FP8 data: [BDP, BK] for each half
            kv_ptr = kv_base + offs_d[:, None] + kv_ids[None, :] * stride_kvn
            kv_fp8_0 = tl.load(kv_ptr, cache_modifier=".cg")  # [256, BK] float8
            kv_fp8_1 = tl.load(kv_ptr + BDP, cache_modifier=".cg")  # [256, BK] float8

            # Load 4 scales per token separately
            # Scale layout: [num_tokens, 4] float32
            scale0 = tl.load(kv_scales + kv_ids * stride_scales_n + 0)  # [BK]
            scale1 = tl.load(kv_scales + kv_ids * stride_scales_n + 1)  # [BK]
            scale2 = tl.load(kv_scales + kv_ids * stride_scales_n + 2)  # [BK]
            scale3 = tl.load(kv_scales + kv_ids * stride_scales_n + 3)  # [BK]

            # Dequantize first half [256, BK]:
            #   elements [0:128] use scale0, elements [128:256] use scale1
            mask_lo = offs_d[:, None] < 128
            kv_blk0 = tl.where(
                mask_lo,
                kv_fp8_0.to(tl.float32) * scale0[None, :],
                kv_fp8_0.to(tl.float32) * scale1[None, :],
            ).to(tl.bfloat16)

            # Dequantize second half [256, BK]:
            #   elements [0:128] use scale2, elements [128:256] use scale3
            kv_blk1 = tl.where(
                mask_lo,
                kv_fp8_1.to(tl.float32) * scale2[None, :],
                kv_fp8_1.to(tl.float32) * scale3[None, :],
            ).to(tl.bfloat16)
        else:
            # BF16 mode: load directly
            kv_ptr = kv_base + offs_d[:, None] + kv_ids[None, :] * stride_kvn
            kv_blk0 = tl.load(kv_ptr, cache_modifier=".cg")  # [BDP, BK]
            kv_blk1 = tl.load(kv_ptr + BDP, cache_modifier=".cg")  # [BDP, BK]

        # Compute QK^T
        qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)
        qk = tl.dot(q_blk1, kv_blk1, qk, out_dtype=tl.float32)
        if DQK == 576:
            if IS_FP8:
                # RoPE part from separate tensor
                rope_ptr = kv_rope + offs_td[:, None] + kv_ids[None, :] * stride_rope_n
                tkv_blk = tl.load(rope_ptr, cache_modifier=".cg")
            else:
                tkv_ptr = kv_base + DP + offs_td[:, None] + kv_ids[None, :] * stride_kvn
                tkv_blk = tl.load(tkv_ptr, cache_modifier=".cg")
            qk = tl.dot(tq_blk, tkv_blk, qk, out_dtype=tl.float32)
        qk *= sm_scale

        # Mask invalid tokens
        qk = tl.where(mask_ids[None, :], qk, float("-inf"))

        # Online softmax
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        exp_qk = tl.math.exp(qk - new_max[:, None])
        sum_qk = tl.sum(exp_qk, axis=1)
        alpha = tl.math.exp(max_log - new_max)
        sum_exp = sum_exp * alpha + sum_qk

        # Accumulate P @ V (V = K NoPE for MLA)
        acc0 = tl.dot(
            exp_qk.to(tl.bfloat16),
            kv_blk0.trans(),
            acc0 * alpha[:, None],
            out_dtype=tl.float32,
        )
        acc1 = tl.dot(
            exp_qk.to(tl.bfloat16),
            kv_blk1.trans(),
            acc1 * alpha[:, None],
            out_dtype=tl.float32,
        )
        max_log = new_max

    # Finalize output
    valid_mask = max_log != float("-inf")
    max_log = tl.where(valid_mask, max_log, float("-inf"))

    orig_lse = max_log + tl.math.log(sum_exp)
    lse_out = tl.where(valid_mask, orig_lse, float("inf"))
    tl.store(l_base + offs_h * stride_lseh, lse_out)

    if HAVE_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + offs_h)
        sum_exp_new_lse = tl.math.exp(orig_lse) + tl.math.exp(sink)
        factor = tl.math.exp(max_log) / sum_exp_new_lse
    else:
        factor = 1.0 / sum_exp

    out_vals0 = tl.where(valid_mask[:, None], acc0 * factor[:, None], 0.0)
    out_vals1 = tl.where(valid_mask[:, None], acc1 * factor[:, None], 0.0)

    # Store output
    o_ptr = o_base + offs_h[:, None] * stride_oh + offs_d[None, :]
    tl.store(o_ptr, out_vals0.to(tl.bfloat16))
    tl.store(o_ptr + BDP, out_vals1.to(tl.bfloat16))


# ============================================================================
# Sparse decode kernel for FlashMLA MODEL1 layout
#
# MODEL1 is FlashMLA's internal name for the d_qk=512 / 584-byte layout.
# It is not a model name. Per page:
#   [0:page_block_size*576]      - token data
#     per token: 448 FP8 NoPE + 64 BF16 RoPE
#   [page_block_size*576:...]    - 8 uint8 E8M0 scales per token
#
# The 512-dim output uses both NoPE and RoPE values as V:
#   output[0:448]   = weighted NoPE
#   output[448:512] = weighted RoPE
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BK": 32, "BH": 64}, num_warps=4, num_stages=1),
        triton.Config({"BK": 32, "BH": 64}, num_warps=8, num_stages=1),
    ],
    key=[
        "HQ",
        "TOPK",
        "EXTRA_TOPK",
        "HAVE_ATTN_SINK",
        "HAVE_TOPK_LENGTH",
        "HAVE_EXTRA",
        "HAVE_EXTRA_TOPK_LENGTH",
    ],
)
@triton.jit
def _sparse_decode_model1_kernel(
    q,
    kv,
    indices,
    extra_kv,
    extra_indices,
    attn_sink,
    topk_length,
    extra_topk_length,
    sm_scale: tl.constexpr,
    output,
    lse,
    stride_qb,
    stride_qsq,
    stride_qh,
    stride_kv_block,
    stride_ib,
    stride_isq,
    stride_extra_kv_block,
    stride_eib,
    stride_eisq,
    stride_ob,
    stride_osq,
    stride_oh,
    stride_lseb,
    stride_lseh,
    SQ,
    HQ: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    EXTRA_PAGE_SIZE: tl.constexpr,
    NUM_BLOCKS,
    EXTRA_NUM_BLOCKS,
    TOPK: tl.constexpr,
    EXTRA_TOPK: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    HAVE_EXTRA: tl.constexpr,
    HAVE_EXTRA_TOPK_LENGTH: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
):
    num_head_blocks: tl.constexpr = (HQ + BH - 1) // BH
    pid = tl.program_id(0)
    i_b = pid // (SQ * num_head_blocks)
    remainder = pid % (SQ * num_head_blocks)
    i_sq = remainder // num_head_blocks
    i_sq = i_sq.to(tl.int64)
    i_gbh = remainder % num_head_blocks
    gbh_base = i_gbh * BH

    NOPE: tl.constexpr = 448
    ROPE: tl.constexpr = 64
    # D: tl.constexpr = 512
    BDP: tl.constexpr = 256
    TOKEN_DATA_BYTES: tl.constexpr = 576
    SCALE_BYTES: tl.constexpr = 8

    q_base = q + i_b * stride_qb + i_sq * stride_qsq + gbh_base * stride_qh
    t_base = indices + i_b * stride_ib + i_sq * stride_isq
    et_base = extra_indices + i_b * stride_eib + i_sq * stride_eisq
    attn_sink_ptr = attn_sink + gbh_base if HAVE_ATTN_SINK else 0
    topk_length_ptr = topk_length + i_b if HAVE_TOPK_LENGTH else 0
    extra_topk_length_ptr = extra_topk_length + i_b if HAVE_EXTRA_TOPK_LENGTH else 0
    o_base = output + i_b * stride_ob + i_sq * stride_osq + gbh_base * stride_oh
    l_base = lse + i_b * stride_lseb + gbh_base * stride_lseh + i_sq

    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, BDP)
    offs_t = tl.arange(0, BK)
    offs_rope = tl.arange(0, ROPE)

    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :]
    q_blk0 = tl.load(q_ptr, eviction_policy="evict_first")
    q_blk1_nope = tl.load(
        q_ptr + BDP,
        mask=offs_d[None, :] < (NOPE - BDP),
        other=0.0,
        eviction_policy="evict_first",
    )
    q_rope = tl.load(
        q_base + offs_h[:, None] * stride_qh + (NOPE + offs_rope[None, :]),
        eviction_policy="evict_first",
    )

    max_log = tl.full([BH], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
    acc0 = tl.zeros([BH, BDP], dtype=tl.float32)
    acc1 = tl.zeros([BH, BDP], dtype=tl.float32)

    topk_len = tl.load(topk_length_ptr) if HAVE_TOPK_LENGTH else TOPK
    NK = tl.cdiv(topk_len, BK)
    for ck in range(NK):
        t_offs = BK * ck + offs_t
        t_msk = t_offs < topk_len
        kv_ids = tl.load(t_base + t_offs, t_msk, other=-1)
        block_ids = kv_ids // PAGE_SIZE
        rel_ids = kv_ids - block_ids * PAGE_SIZE
        valid_ids = t_msk & (kv_ids >= 0) & (block_ids < NUM_BLOCKS)
        block_ids = tl.where(valid_ids, block_ids, 0)
        rel_ids = tl.where(valid_ids, rel_ids, 0)

        token_base = (
            kv + block_ids.to(tl.int64) * stride_kv_block + rel_ids * TOKEN_DATA_BYTES
        )
        scale_base = (
            kv
            + block_ids.to(tl.int64) * stride_kv_block
            + PAGE_SIZE * TOKEN_DATA_BYTES
            + rel_ids * SCALE_BYTES
        )

        kv_fp8_0_u8 = tl.load(
            token_base[None, :] + offs_d[:, None],
            mask=valid_ids[None, :],
            other=0,
            cache_modifier=".cg",
        )
        kv_fp8_1_u8 = tl.load(
            token_base[None, :] + (BDP + offs_d[:, None]),
            mask=valid_ids[None, :] & (offs_d[:, None] < (NOPE - BDP)),
            other=0,
            cache_modifier=".cg",
        )

        scale0_u8 = tl.load(scale_base + 0, mask=valid_ids, other=127)
        scale1_u8 = tl.load(scale_base + 1, mask=valid_ids, other=127)
        scale2_u8 = tl.load(scale_base + 2, mask=valid_ids, other=127)
        scale3_u8 = tl.load(scale_base + 3, mask=valid_ids, other=127)
        scale4_u8 = tl.load(scale_base + 4, mask=valid_ids, other=127)
        scale5_u8 = tl.load(scale_base + 5, mask=valid_ids, other=127)
        scale6_u8 = tl.load(scale_base + 6, mask=valid_ids, other=127)

        scale0 = (scale0_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale1 = (scale1_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale2 = (scale2_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale3 = (scale3_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale4 = (scale4_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale5 = (scale5_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale6 = (scale6_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)

        kv_fp8_0 = kv_fp8_0_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_0 = tl.where(
            offs_d[:, None] < 64,
            scale0[None, :],
            tl.where(
                offs_d[:, None] < 128,
                scale1[None, :],
                tl.where(offs_d[:, None] < 192, scale2[None, :], scale3[None, :]),
            ),
        )
        kv_blk0 = (kv_fp8_0 * scale_0).to(tl.bfloat16)

        kv_fp8_1 = kv_fp8_1_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_1 = tl.where(
            offs_d[:, None] < 64,
            scale4[None, :],
            tl.where(offs_d[:, None] < 128, scale5[None, :], scale6[None, :]),
        )
        nope_tail = (kv_fp8_1 * scale_1).to(tl.bfloat16)

        rope_ptr = (token_base + NOPE).to(tl.pointer_type(tl.bfloat16))
        rope_blk = tl.load(
            rope_ptr[None, :] + offs_rope[:, None],
            mask=valid_ids[None, :],
            other=0.0,
            cache_modifier=".cg",
        )

        kv_blk1 = tl.where(
            offs_d[:, None] < (NOPE - BDP),
            nope_tail,
            tl.load(
                rope_ptr[None, :] + (offs_d[:, None] - (NOPE - BDP)),
                mask=valid_ids[None, :] & (offs_d[:, None] >= (NOPE - BDP)),
                other=0.0,
                cache_modifier=".cg",
            ),
        )

        qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)
        qk = tl.dot(q_blk1_nope, nope_tail, qk, out_dtype=tl.float32)
        qk = tl.dot(q_rope, rope_blk, qk, out_dtype=tl.float32)
        qk *= sm_scale
        qk = tl.where(valid_ids[None, :], qk, float("-inf"))

        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        exp_qk = tl.math.exp(qk - new_max[:, None])
        sum_qk = tl.sum(exp_qk, axis=1)
        alpha = tl.math.exp(max_log - new_max)
        sum_exp = sum_exp * alpha + sum_qk
        acc0 = tl.dot(
            exp_qk.to(tl.bfloat16),
            kv_blk0.trans(),
            acc0 * alpha[:, None],
            out_dtype=tl.float32,
        )
        acc1 = tl.dot(
            exp_qk.to(tl.bfloat16),
            kv_blk1.trans(),
            acc1 * alpha[:, None],
            out_dtype=tl.float32,
        )
        max_log = new_max

    if HAVE_EXTRA:
        extra_topk_len = (
            tl.load(extra_topk_length_ptr) if HAVE_EXTRA_TOPK_LENGTH else EXTRA_TOPK
        )
        ENK = tl.cdiv(extra_topk_len, BK)
        for ck in range(ENK):
            t_offs = BK * ck + offs_t
            t_msk = t_offs < extra_topk_len
            kv_ids = tl.load(et_base + t_offs, t_msk, other=-1)
            block_ids = kv_ids // EXTRA_PAGE_SIZE
            rel_ids = kv_ids - block_ids * EXTRA_PAGE_SIZE
            valid_ids = t_msk & (kv_ids >= 0) & (block_ids < EXTRA_NUM_BLOCKS)
            block_ids = tl.where(valid_ids, block_ids, 0)
            rel_ids = tl.where(valid_ids, rel_ids, 0)

            token_base = (
                extra_kv
                + block_ids.to(tl.int64) * stride_extra_kv_block
                + rel_ids * TOKEN_DATA_BYTES
            )
            scale_base = (
                extra_kv
                + block_ids.to(tl.int64) * stride_extra_kv_block
                + EXTRA_PAGE_SIZE * TOKEN_DATA_BYTES
                + rel_ids * SCALE_BYTES
            )

            kv_fp8_0_u8 = tl.load(
                token_base[None, :] + offs_d[:, None],
                mask=valid_ids[None, :],
                other=0,
                cache_modifier=".cg",
            )
            kv_fp8_1_u8 = tl.load(
                token_base[None, :] + (BDP + offs_d[:, None]),
                mask=valid_ids[None, :] & (offs_d[:, None] < (NOPE - BDP)),
                other=0,
                cache_modifier=".cg",
            )

            scale0_u8 = tl.load(scale_base + 0, mask=valid_ids, other=127)
            scale1_u8 = tl.load(scale_base + 1, mask=valid_ids, other=127)
            scale2_u8 = tl.load(scale_base + 2, mask=valid_ids, other=127)
            scale3_u8 = tl.load(scale_base + 3, mask=valid_ids, other=127)
            scale4_u8 = tl.load(scale_base + 4, mask=valid_ids, other=127)
            scale5_u8 = tl.load(scale_base + 5, mask=valid_ids, other=127)
            scale6_u8 = tl.load(scale_base + 6, mask=valid_ids, other=127)

            scale0 = (scale0_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            scale1 = (scale1_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            scale2 = (scale2_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            scale3 = (scale3_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            scale4 = (scale4_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            scale5 = (scale5_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            scale6 = (scale6_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)

            kv_fp8_0 = kv_fp8_0_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scale_0 = tl.where(
                offs_d[:, None] < 64,
                scale0[None, :],
                tl.where(
                    offs_d[:, None] < 128,
                    scale1[None, :],
                    tl.where(offs_d[:, None] < 192, scale2[None, :], scale3[None, :]),
                ),
            )
            kv_blk0 = (kv_fp8_0 * scale_0).to(tl.bfloat16)

            kv_fp8_1 = kv_fp8_1_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scale_1 = tl.where(
                offs_d[:, None] < 64,
                scale4[None, :],
                tl.where(offs_d[:, None] < 128, scale5[None, :], scale6[None, :]),
            )
            nope_tail = (kv_fp8_1 * scale_1).to(tl.bfloat16)

            rope_ptr = (token_base + NOPE).to(tl.pointer_type(tl.bfloat16))
            rope_blk = tl.load(
                rope_ptr[None, :] + offs_rope[:, None],
                mask=valid_ids[None, :],
                other=0.0,
                cache_modifier=".cg",
            )
            kv_blk1 = tl.where(
                offs_d[:, None] < (NOPE - BDP),
                nope_tail,
                tl.load(
                    rope_ptr[None, :] + (offs_d[:, None] - (NOPE - BDP)),
                    mask=valid_ids[None, :] & (offs_d[:, None] >= (NOPE - BDP)),
                    other=0.0,
                    cache_modifier=".cg",
                ),
            )

            qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)
            qk = tl.dot(q_blk1_nope, nope_tail, qk, out_dtype=tl.float32)
            qk = tl.dot(q_rope, rope_blk, qk, out_dtype=tl.float32)
            qk *= sm_scale
            qk = tl.where(valid_ids[None, :], qk, float("-inf"))

            new_max = tl.maximum(max_log, tl.max(qk, axis=1))
            exp_qk = tl.math.exp(qk - new_max[:, None])
            sum_qk = tl.sum(exp_qk, axis=1)
            alpha = tl.math.exp(max_log - new_max)
            sum_exp = sum_exp * alpha + sum_qk
            acc0 = tl.dot(
                exp_qk.to(tl.bfloat16),
                kv_blk0.trans(),
                acc0 * alpha[:, None],
                out_dtype=tl.float32,
            )
            acc1 = tl.dot(
                exp_qk.to(tl.bfloat16),
                kv_blk1.trans(),
                acc1 * alpha[:, None],
                out_dtype=tl.float32,
            )
            max_log = new_max

    valid_mask = max_log != float("-inf")
    orig_lse = max_log + tl.math.log(sum_exp)
    lse_out = tl.where(valid_mask, orig_lse, float("inf"))
    tl.store(l_base + offs_h * stride_lseh, lse_out)

    if HAVE_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + offs_h)
        sum_exp_new_lse = tl.math.exp(orig_lse) + tl.math.exp(sink)
        factor = tl.math.exp(max_log) / sum_exp_new_lse
    else:
        factor = 1.0 / sum_exp

    out_vals0 = tl.where(valid_mask[:, None], acc0 * factor[:, None], 0.0)
    out_vals1 = tl.where(valid_mask[:, None], acc1 * factor[:, None], 0.0)
    o_ptr = o_base + offs_h[:, None] * stride_oh + offs_d[None, :]
    tl.store(o_ptr, out_vals0.to(tl.bfloat16))
    tl.store(o_ptr + BDP, out_vals1.to(tl.bfloat16))


# ============================================================================
# Dense decode kernel (paged attention with block_table)
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    ],
    key=["HQ", "DQK", "HAVE_CAUSAL"],
)
@triton.jit
def _dense_decode_kernel(
    Q_ptr,
    stride_q_b,
    stride_q_sq,
    stride_q_h,
    KV_cache,
    stride_kv_bs,
    Block_table,
    stride_bt_b,
    Seq_lens,
    Out,
    stride_o_b,
    stride_o_sq,
    stride_o_h,
    LSE,
    stride_lse_b,
    stride_lse_h,
    sm_scale,
    SQ,
    HQ: tl.constexpr,
    DQK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HAVE_CAUSAL: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Dense decode kernel with paged attention and online softmax.
    Grid: (ceil(HQ / BLOCK_H), batch_size * seq_q)
    """
    pid_h_block = tl.program_id(0)
    pid_b_sq = tl.program_id(1)
    i_b = pid_b_sq // SQ
    i_sq = pid_b_sq % SQ

    cur_head = pid_h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_head = cur_head < HQ

    # Load Q: NoPE part [BLOCK_H, HEAD_DIM_V] and RoPE part [BLOCK_H, DQK-HEAD_DIM_V]
    offs_d_nope = tl.arange(0, HEAD_DIM_V)
    offs_q_nope = (
        i_b * stride_q_b
        + i_sq * stride_q_sq
        + cur_head[:, None] * stride_q_h
        + offs_d_nope[None, :]
    )
    q_nope = tl.load(Q_ptr + offs_q_nope, mask=mask_head[:, None], other=0.0)

    offs_d_pe = tl.arange(HEAD_DIM_V, DQK)
    offs_q_pe = (
        i_b * stride_q_b
        + i_sq * stride_q_sq
        + cur_head[:, None] * stride_q_h
        + offs_d_pe[None, :]
    )
    q_pe = tl.load(Q_ptr + offs_q_pe, mask=mask_head[:, None], other=0.0)

    # Online softmax accumulators
    e_max = tl.full([BLOCK_H], value=float("-inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, HEAD_DIM_V], dtype=tl.float32)

    cur_batch_seq_len = tl.load(Seq_lens + i_b)
    Block_table += i_b * stride_bt_b

    offs_n = tl.arange(0, BLOCK_N)
    loop_time = cur_batch_seq_len // BLOCK_N
    remainder = cur_batch_seq_len % BLOCK_N

    for i in range(0, loop_time):
        kv_page_number = tl.load(Block_table + offs_n // PAGE_SIZE)
        kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE

        # Load V (NoPE part)
        offs_v_c = kv_loc[:, None] * stride_kv_bs + offs_d_nope[None, :]
        v_c = tl.load(KV_cache + offs_v_c)
        k_c = tl.trans(v_c)

        # QK = q_nope @ k_nope^T
        qk = tl.dot(q_nope, k_c)

        # Add RoPE contribution
        offs_k_pe = kv_loc[None, :] * stride_kv_bs + offs_d_pe[:, None]
        k_pe = tl.load(KV_cache + offs_k_pe)
        qk = tl.dot(q_pe, k_pe, acc=qk)
        qk *= sm_scale

        # Online softmax update
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc = tl.dot(p.to(v_c.dtype), v_c, acc=acc)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max
        offs_n += BLOCK_N

    if remainder:
        mask_kvsplit = offs_n < cur_batch_seq_len
        kv_page_number = tl.load(
            Block_table + offs_n // PAGE_SIZE, mask=mask_kvsplit, other=0
        )
        kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE

        offs_v_c = kv_loc[:, None] * stride_kv_bs + offs_d_nope[None, :]
        v_c = tl.load(KV_cache + offs_v_c, mask=mask_kvsplit[:, None], other=0.0)
        k_c = tl.trans(v_c)

        qk = tl.dot(q_nope, k_c)

        offs_k_pe = kv_loc[None, :] * stride_kv_bs + offs_d_pe[:, None]
        k_pe = tl.load(KV_cache + offs_k_pe, mask=mask_kvsplit[None, :], other=0.0)
        qk = tl.dot(q_pe, k_pe, acc=qk)
        qk *= sm_scale

        qk = tl.where(mask_kvsplit[None, :], qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc = tl.dot(p.to(v_c.dtype), v_c, acc=acc)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    # Store output
    offs_o = (
        i_b * stride_o_b
        + i_sq * stride_o_sq
        + cur_head[:, None] * stride_o_h
        + offs_d_nope[None, :]
    )
    tl.store(
        Out + offs_o,
        (acc / e_sum[:, None]).to(Out.dtype.element_ty),
        mask=mask_head[:, None],
    )

    # Store LSE
    lse_val = e_max + tl.math.log(e_sum)
    lse_offset = i_b * stride_lse_b + cur_head * stride_lse_h + i_sq
    tl.store(LSE + lse_offset, lse_val, mask=mask_head)


# ============================================================================
# Main dispatch function
# ============================================================================


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta,
    num_splits: None = None,
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
    """
    Triton implementation of flash_mla_with_kvcache.
    Functionally equivalent to the CUDA implementation.

    Returns:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v)
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32
    """
    sched_meta = tile_scheduler_metadata
    assert isinstance(sched_meta, FlashMLASchedMeta)
    assert num_splits is None
    assert q.ndim == 4
    assert k_cache.ndim == 4

    topk = indices.shape[-1] if indices is not None else None
    extra_k_page_block_size = (
        extra_k_cache.shape[1] if extra_k_cache is not None else None
    )
    extra_topk_val = (
        extra_indices_in_kvcache.shape[-1]
        if extra_indices_in_kvcache is not None
        else None
    )

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if not sched_meta.have_initialized:
        if indices is not None:
            assert not causal, "causal must be False when sparse attention is enabled"
        sched_meta.have_initialized = True
        sched_meta.config = FlashMLASchedMeta.Config(
            q.shape[0],
            q.shape[1],
            q.shape[2],
            k_cache.shape[1],
            k_cache.shape[2],
            causal,
            is_fp8_kvcache,
            topk,
            extra_k_page_block_size,
            extra_topk_val,
        )
    else:
        helper_msg = (
            " Your input arguments are inconsistent with sched_meta. Please make "
            "sure the input arguments are consistent across different invocations "
            "of flash_mla_with_kvcache on the same sched_meta."
        )
        assert sched_meta.config is not None
        assert sched_meta.config.b == q.shape[0], (
            "sched_meta.config.b must be equal to batch_size." + helper_msg
        )
        assert sched_meta.config.s_q == q.shape[1], (
            "sched_meta.config.s_q must be equal to seq_len_q." + helper_msg
        )
        assert sched_meta.config.h_q == q.shape[2], (
            "sched_meta.config.h_q must be equal to num_heads_q." + helper_msg
        )
        assert sched_meta.config.page_block_size == k_cache.shape[1], (
            "sched_meta.config.page_block_size must be equal to page_block_size."
            + helper_msg
        )
        assert sched_meta.config.h_k == k_cache.shape[2], (
            "sched_meta.config.h_k must be equal to num_heads_k." + helper_msg
        )
        assert sched_meta.config.causal == causal, (
            "sched_meta.config.causal must be equal to causal." + helper_msg
        )
        assert sched_meta.config.is_fp8_kvcache == is_fp8_kvcache, (
            "sched_meta.config.is_fp8_kvcache must be equal to is_fp8_kvcache."
            + helper_msg
        )
        assert sched_meta.config.topk == topk, (
            "sched_meta.config.topk must be equal to the last dim of indices."
            + helper_msg
        )
        assert sched_meta.config.extra_page_block_size == extra_k_page_block_size, (
            "sched_meta.config.extra_page_block_size must be equal to the "
            "page_block_size of extra_k_cache." + helper_msg
        )
        assert sched_meta.config.extra_topk == extra_topk_val, (
            "sched_meta.config.extra_topk must be equal to the last dim of "
            "extra_indices_in_kvcache." + helper_msg
        )

    batch_size, seq_q, num_heads_q, head_dim_k = q.shape
    num_heads_k = k_cache.shape[2]

    if out is None:
        out = torch.empty(
            (batch_size, seq_q, num_heads_q, head_dim_v),
            dtype=q.dtype,
            device=q.device,
        )
    else:
        assert out.shape == (batch_size, seq_q, num_heads_q, head_dim_v)
        assert out.dtype == q.dtype
        assert out.device == q.device
        assert out.stride(-1) == 1
    lse = torch.empty(
        (batch_size, num_heads_q, seq_q),
        dtype=torch.float32,
        device=q.device,
    )

    if indices is not None:
        assert not causal, "causal must be False when sparse attention is enabled"
        assert is_fp8_kvcache, "is_fp8_kvcache must be True for sparse attention"
        assert (
            num_heads_k == 1
        ), "Currently only MQA (h_kv == 1) is supported for sparse decoding"
        assert head_dim_v == 512, "Only head_size_v == 512 is supported"
        assert num_heads_q in (64, 128), "Only h_q == 64 or 128 is supported"
        assert head_dim_k in (
            512,
            576,
        ), "Only head_size_k == 512 or 576 is supported for sparse decoding"
        assert q.dtype == torch.bfloat16
        assert k_cache.dtype in (torch.float8_e4m3fn, torch.int8, torch.uint8)
        assert topk is not None and topk > 0
        assert topk % 64 == 0, "topk must be divisible by 64"
        assert indices.ndim == 3 and indices.shape[:2] == (batch_size, seq_q)
        assert indices.dtype == torch.int32
        assert indices.stride(-1) == 1
        if topk_length is not None:
            assert topk_length.shape == (batch_size,)
            assert topk_length.dtype == torch.int32
            assert topk_length.is_contiguous()
        if attn_sink is not None:
            assert attn_sink.shape == (num_heads_q,)
            assert attn_sink.dtype == torch.float32
        if extra_k_cache is not None:
            assert extra_indices_in_kvcache is not None, (
                "extra_indices_in_kvcache must be provided when extra_k_cache "
                "is provided"
            )
            assert extra_k_cache.dtype in (
                torch.float8_e4m3fn,
                torch.int8,
                torch.uint8,
            )
        else:
            assert extra_indices_in_kvcache is None, (
                "extra_indices_in_kvcache must not be provided when extra_k_cache "
                "is not provided"
            )
            assert extra_topk_length is None, (
                "extra_topk_length must not be provided when extra_k_cache is "
                "not provided"
            )
        if extra_indices_in_kvcache is not None:
            assert extra_indices_in_kvcache.ndim == 3
            assert extra_indices_in_kvcache.shape[:2] == (batch_size, seq_q)
            assert extra_indices_in_kvcache.dtype == torch.int32
            assert extra_indices_in_kvcache.stride(-1) == 1
            assert extra_indices_in_kvcache.shape[-1] % 64 == 0
        if extra_topk_length is not None:
            assert extra_topk_length.shape == (batch_size,)
            assert extra_topk_length.dtype == torch.int32
            assert extra_topk_length.is_contiguous()
        if head_dim_k == 576:
            assert (
                k_cache.shape[-1] == 656
            ), "V32 sparse FP8 cache must use 656 bytes per token"
            assert (
                k_cache.stride(1) == 656
            ), "The whole block must be contiguous for V32 KV cache"
            assert topk_length is None, "V3.2/V32 does not support dynamic topk length"
            assert extra_k_cache is None, "V3.2/V32 does not support extra KV cache"
            assert (
                extra_indices_in_kvcache is None
            ), "V3.2/V32 does not support extra indices"
            assert (
                extra_topk_length is None
            ), "V3.2/V32 does not support extra topk length"
        else:
            assert (
                k_cache.shape[-1] == 584
            ), "MODEL1 sparse FP8 cache must use 584 bytes per token"
            assert (
                k_cache.stride(1) == 584
            ), "The whole block must be contiguous for MODEL1 KV cache"
            if extra_k_cache is not None:
                assert extra_k_cache.ndim == 4
                assert extra_k_cache.shape[2] == 1
                assert extra_k_cache.shape[-1] == 584
                assert extra_k_cache.stride(1) == 584
        _sparse_decode_dispatch(
            q,
            k_cache,
            indices,
            out,
            lse,
            attn_sink,
            topk_length,
            extra_k_cache,
            extra_indices_in_kvcache,
            extra_topk_length,
            batch_size,
            seq_q,
            num_heads_q,
            head_dim_k,
            head_dim_v,
            topk,
            k_cache.shape[1],
            softmax_scale,
            is_fp8_kvcache,
        )
    else:
        assert (
            attn_sink is None
            and extra_k_cache is None
            and extra_indices_in_kvcache is None
            and topk_length is None
            and extra_topk_length is None
        ), (
            "indices, attn_sink, extra_k_cache, extra_indices_in_kvcache, "
            "topk_length and extra_topk_length must be None when dense "
            "attention is used."
        )
        assert block_table is not None and cache_seqlens is not None, (
            "block_table and cache_seqlens must be provided when dense attention "
            "is used."
        )
        assert num_heads_k == 1, "Only num_heads_k == 1 is supported for dense MLA"
        if seq_q > 1 and causal:
            raise NotImplementedError(
                "causal dense attention with seq_q > 1 is not implemented"
            )
        _dense_decode_dispatch(
            q,
            k_cache,
            block_table,
            cache_seqlens,
            out,
            lse,
            batch_size,
            seq_q,
            num_heads_q,
            head_dim_k,
            head_dim_v,
            k_cache.shape[1],
            softmax_scale,
            causal,
        )

    return out, lse


# ============================================================================
# Kernel launch helpers
# ============================================================================


def _sparse_decode_dispatch(
    q,
    kv,
    indices,
    out,
    lse,
    attn_sink,
    topk_length,
    extra_kv,
    extra_indices,
    extra_topk_length,
    batch_size,
    seq_q,
    num_heads_q,
    head_dim_k,
    head_dim_v,
    topk,
    page_block_size,
    softmax_scale,
    is_fp8_kvcache,
):
    """Launch sparse decode kernel."""
    BH = 64
    num_head_blocks = (num_heads_q + BH - 1) // BH
    grid = (batch_size * seq_q * num_head_blocks,)

    skv = kv.shape[0] * page_block_size

    if head_dim_k == 512:
        _sparse_decode_model1_kernel[grid](
            q,
            kv,
            indices,
            extra_kv if extra_kv is not None else kv,
            extra_indices if extra_indices is not None else indices,
            attn_sink if attn_sink is not None else None,
            topk_length if topk_length is not None else None,
            extra_topk_length if extra_topk_length is not None else None,
            softmax_scale,
            out,
            lse,
            # Q strides
            q.stride(0),
            q.stride(1),
            q.stride(2),
            # KV and indices strides
            kv.stride(0),
            indices.stride(0),
            indices.stride(1),
            extra_kv.stride(0) if extra_kv is not None else kv.stride(0),
            extra_indices.stride(0) if extra_indices is not None else indices.stride(0),
            extra_indices.stride(1) if extra_indices is not None else indices.stride(1),
            # Output strides
            out.stride(0),
            out.stride(1),
            out.stride(2),
            # LSE strides
            lse.stride(0),
            lse.stride(1),
            # Scalar args
            seq_q,
            num_heads_q,
            page_block_size,
            extra_kv.shape[1] if extra_kv is not None else 1,
            kv.shape[0],
            extra_kv.shape[0] if extra_kv is not None else 0,
            topk,
            extra_indices.shape[-1] if extra_indices is not None else 0,
            attn_sink is not None,
            topk_length is not None,
            extra_kv is not None,
            extra_topk_length is not None,
        )
        return

    if is_fp8_kvcache:
        # FP8 mode: kv has shape [num_blocks, page_block_size, 1, 656]
        # Layout per token (656 bytes):
        #   [0:512]   - 512 float8_e4m3fn values (NoPE)
        #   [512:528] - 4 float32 scales (16 bytes)
        #   [528:656] - 64 bfloat16 values (RoPE, 128 bytes)
        kv_bytes = kv.reshape(-1, 656).contiguous()  # [num_tokens, 656] uint8

        # NoPE FP8 part: first 512 bytes as float8_e4m3fn
        kv_nope = (
            kv_bytes[:, :512].contiguous().view(torch.float8_e4m3fn)
        )  # [num_tokens, 512]
        stride_kvn = kv_nope.stride(0)

        # Scales: bytes [512:528] as 4 float32 values
        kv_scales = (
            kv_bytes[:, 512:528].contiguous().view(torch.float32)
        )  # [num_tokens, 4]
        stride_scales_n = kv_scales.stride(0)

        # RoPE BF16 part: bytes [528:656] as 64 bfloat16 values
        kv_rope = (
            kv_bytes[:, 528:656].contiguous().view(torch.bfloat16)
        )  # [num_tokens, 64]
        stride_rope_n = kv_rope.stride(0)
    else:
        # BF16 mode: kv has shape [num_blocks, page_block_size, 1, head_dim_k]
        kv_nope = kv.reshape(-1, kv.shape[-1]).contiguous()
        stride_kvn = kv_nope.stride(0)
        kv_scales = kv_nope  # unused, pass same tensor
        stride_scales_n = 0
        kv_rope = kv_nope  # unused, pass same tensor
        stride_rope_n = 0

    # # TLE warp specialization path TODO
    # if _can_use_tle_sparse_decode(q, indices, head_dim_v, head_dim_k, is_fp8_kvcache):
    #     _tle_sparse_decode_launch(
    #         q, kv_nope, kv_scales, kv_rope, indices, out, lse,
    #         attn_sink, topk_length,
    #         batch_size, seq_q, num_heads_q,
    #         head_dim_k, head_dim_v, topk, skv,
    #         softmax_scale, is_fp8_kvcache,
    #         stride_kvn, stride_scales_n, stride_rope_n,
    #     )
    #     return

    _sparse_decode_kernel[grid](
        q,
        kv_nope,
        kv_scales,
        kv_rope,
        indices,
        attn_sink if attn_sink is not None else None,
        topk_length if topk_length is not None else None,
        softmax_scale,
        out,
        lse,
        # Q strides
        q.stride(0),
        q.stride(1),
        q.stride(2),
        # KV strides
        stride_kvn,
        stride_scales_n,
        stride_rope_n,
        # Indices strides
        indices.stride(0),
        indices.stride(1),
        # Output strides
        out.stride(0),
        out.stride(1),
        out.stride(2),
        # LSE strides
        lse.stride(0),
        lse.stride(1),
        # Scalar args
        seq_q,
        num_heads_q,
        head_dim_k,
        skv,
        topk,
        attn_sink is not None,
        topk_length is not None,
        is_fp8_kvcache,
    )


def _dense_decode_dispatch(
    q,
    kv_cache,
    block_table,
    cache_seqlens,
    out,
    lse,
    batch_size,
    seq_q,
    num_heads_q,
    head_dim_k,
    head_dim_v,
    page_block_size,
    softmax_scale,
    causal,
):
    """Launch dense decode kernel."""
    BLOCK_H = 64
    num_head_blocks = (num_heads_q + BLOCK_H - 1) // BLOCK_H

    # KV cache: [num_blocks, page_block_size, num_heads_k, head_dim_k]
    # Flatten to [num_tokens_total, head_dim_k] for paged access
    kv_flat = kv_cache.view(-1, head_dim_k).contiguous()
    block_table = block_table.contiguous()

    # TLE warp specialization path
    if _can_use_tle_dense_decode(q, kv_cache, block_table, head_dim_v, page_block_size):
        _tle_dense_decode_launch(
            q,
            kv_flat,
            block_table,
            cache_seqlens,
            out,
            lse,
            batch_size,
            seq_q,
            num_heads_q,
            head_dim_k,
            head_dim_v,
            page_block_size,
            softmax_scale,
            causal,
        )
        return

    grid = (num_head_blocks, batch_size * seq_q)

    _dense_decode_kernel[grid](
        q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_flat,
        kv_flat.stride(0),
        block_table,
        block_table.stride(0),
        cache_seqlens,
        out,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse,
        lse.stride(0),
        lse.stride(1),
        softmax_scale,
        seq_q,
        num_heads_q,
        head_dim_k,
        head_dim_v,
        page_block_size,
        causal,
    )


# ============================================================================
# TLE Warp Specialization path for sparse decode
# ============================================================================


def _tle_decode_enabled() -> bool:
    value = os.environ.get("FLAGGEMS_FLASHMLA_DECODE_TLE", "1").lower()
    return value not in {"0", "false", "off", "no"}


def _can_use_tle_sparse_decode(
    q: torch.Tensor,
    indices: torch.Tensor,
    head_dim_v: int,
    head_dim_k: int,
    is_fp8: bool,
) -> bool:
    if not (HAS_TLE and _tle_decode_enabled()):
        return False
    if q.device.type != "cuda":
        return False
    batch_size, seq_q, num_heads_q, d_qk = q.shape
    TOPK = indices.shape[-1]
    return (
        head_dim_v == 512
        and d_qk in (512, 576)
        and num_heads_q % TLE_DECODE_BH == 0
        and TOPK > 0
        and TOPK % (TLE_DECODE_BK * TLE_DECODE_PAIR_BLOCKS) == 0
    )


def _can_use_tle_dense_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    head_dim_v: int,
    page_block_size: int,
) -> bool:
    if not (HAS_TLE and _tle_decode_enabled()):
        return False
    if q.device.type != "cuda":
        return False
    batch_size, seq_q, num_heads_q, d_qk = q.shape
    return (
        head_dim_v == 512
        and d_qk in (512, 576)
        and num_heads_q % TLE_DECODE_BH == 0
        and page_block_size == TLE_DECODE_BK
    )


def _set_triton_descriptor_allocator(device: torch.device) -> None:
    def alloc_fn(size: int, align: int, stream):
        _ = align
        _ = stream
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)


def _tle_sparse_decode_launch(
    q,
    kv_nope,
    kv_scales,
    kv_rope,
    indices,
    out,
    lse,
    attn_sink,
    topk_length,
    batch_size,
    seq_q,
    num_heads_q,
    head_dim_k,
    head_dim_v,
    topk,
    skv,
    softmax_scale,
    is_fp8_kvcache,
    stride_kvn,
    stride_scales_n,
    stride_rope_n,
):
    """Launch TLE warp-specialized sparse decode kernel."""
    from triton.tools.tensor_descriptor import TensorDescriptor

    _set_triton_descriptor_allocator(q.device)

    BH = TLE_DECODE_BH
    BK = TLE_DECODE_BK
    D = head_dim_v  # 512
    TD = head_dim_k - D  # 64 for DQK=576, 0 for DQK=512
    DP = triton.next_power_of_2(D)
    DPH = DP // 2
    HAVE_TAIL = TD > 0
    TDP = triton.next_power_of_2(TD) if HAVE_TAIL else 1
    G = num_heads_q
    RH = G // BH

    # Reshape q for TensorDescriptor: [batch*seq_q*HQ, DQK]
    q_flat = q.reshape(batch_size * seq_q * num_heads_q, head_dim_k).contiguous()
    out_flat = out.reshape(batch_size * seq_q * num_heads_q, head_dim_v)

    q_desc = TensorDescriptor(
        q_flat,
        shape=[batch_size * seq_q * num_heads_q, head_dim_k],
        strides=[head_dim_k, 1],
        block_shape=[BH, DPH],
    )
    if HAVE_TAIL:
        tq_desc = TensorDescriptor(
            q_flat,
            shape=[batch_size * seq_q * num_heads_q, head_dim_k],
            strides=[head_dim_k, 1],
            block_shape=[BH, TDP],
        )
    else:
        tq_desc = q_desc
    output_desc = TensorDescriptor(
        out_flat,
        shape=[batch_size * seq_q * num_heads_q, D],
        strides=[D, 1],
        block_shape=[BH, DPH],
    )

    # Grid: one program per (batch*seq_q, head_block)
    grid = (batch_size * seq_q * RH,)

    # Indices stride: [batch, seq_q, topk] -> stride for batch*seq_q dim
    stride_isq = (
        indices.stride(0) * indices.stride(1) // indices.stride(1)
        if seq_q == 1
        else indices.stride(1)
    )
    # For shape [batch, seq_q, topk]: stride_isq = topk (contiguous)
    stride_isq = topk

    _tle_sparse_decode_fwd[grid](
        q_desc,
        tq_desc,
        output_desc,
        kv_nope,
        kv_scales,
        kv_rope,
        indices.reshape(batch_size * seq_q, topk).contiguous(),
        attn_sink,
        topk_length,
        softmax_scale,
        out_flat,
        lse.reshape(batch_size * seq_q, num_heads_q).contiguous(),
        batch_size * seq_q,
        num_heads_q,
        head_dim_k,
        skv,
        topk,
        attn_sink is not None,
        topk_length is not None,
        is_fp8_kvcache,
        D,
        TD,
        DP,
        TDP,
        G,
        RH,
        HAVE_TAIL,
        BK,
        BH,
        TLE_DECODE_PAIR_BLOCKS,
        stride_kvn,
        stride_scales_n,
        stride_rope_n,
        indices.stride(0),
        stride_isq,
        num_warps=TLE_DECODE_WORKER_NUM_WARPS,
        num_stages=1,
    )


if HAS_TLE:

    @triton.jit
    def _tle_sparse_decode_producer(
        k0_l_writer,
        k0_r_writer,
        k1_l_writer,
        k1_r_writer,
        valid_writer,
        kv_nope_base,
        kv_scales_base,
        kv_rope_base,
        t_base,
        topk_len_ptr,
        D: tl.constexpr,
        TD: tl.constexpr,
        DPH: tl.constexpr,
        TDP: tl.constexpr,
        SKV,
        TOPK: tl.constexpr,
        HAVE_TOPK_LENGTH: tl.constexpr,
        HAVE_TAIL: tl.constexpr,
        IS_FP8: tl.constexpr,
        BK: tl.constexpr,
        stride_kvn,
        stride_scales_n,
        stride_rope_n,
    ):
        """
        Producer warpgroup: loads KV data from global memory to shared memory.
        For FP8 mode: loads FP8 NoPE + scales + RoPE, dequantizes FP8 to BF16.
        For BF16 mode: loads BF16 KV directly.
        """
        topk_len = tl.load(topk_len_ptr) if HAVE_TOPK_LENGTH else TOPK
        max_col = SKV - 1
        NK = tl.cdiv(topk_len, BK)
        NPAIRS = tl.cdiv(NK, 2)
        offs_t = tl.arange(0, BK)
        offs_tile = tl.arange(0, 64)
        kv_tile_rows = tl.broadcast_to(offs_t[:, None], (BK, 64))

        for pair in tl.range(NPAIRS):
            ck0 = pair * 2
            ck1 = ck0 + 1

            # Load indices for both blocks
            t_offs0 = BK * ck0 + offs_t
            t_msk0 = t_offs0 < topk_len
            kv_ids0 = tl.load(t_base + t_offs0, t_msk0, other=-1)
            valid0 = t_msk0 & (kv_ids0 <= max_col) & (kv_ids0 >= 0)

            t_offs1 = BK * ck1 + offs_t
            t_msk1 = t_offs1 < topk_len
            kv_ids1 = tl.load(t_base + t_offs1, t_msk1, other=-1)
            valid1 = t_msk1 & (kv_ids1 <= max_col) & (kv_ids1 >= 0)

            # Process k0_l (left half of block 0)
            k0_l_slot = k0_l_writer.acquire(pair)
            for tile in tl.static_range(0, DPH, 64):
                k_cols = tile + offs_tile
                k_cols_b = tl.broadcast_to(k_cols[None, :], (BK, 64))

                if IS_FP8:
                    # Load FP8 data
                    kv_ptr = (
                        kv_nope_base + k_cols[None, :] + kv_ids0[:, None] * stride_kvn
                    )
                    k0_l_msk = valid0[:, None] & (k_cols < D)[None, :]
                    k0_l_fp8 = tl.load(
                        kv_ptr, mask=k0_l_msk, other=0.0, eviction_policy="evict_last"
                    )

                    # Load scales for dequantization
                    # Each 128 elements share one scale
                    scale_idx = k_cols // 128  # 0 or 1 for left half
                    scale0 = tl.load(
                        kv_scales_base + kv_ids0 * stride_scales_n + scale_idx,
                        mask=valid0,
                        other=1.0,
                    )

                    # Dequantize: FP8 * scale -> BF16
                    k0_l_blk = (k0_l_fp8.to(tl.float32) * scale0[:, None]).to(
                        tl.bfloat16
                    )
                else:
                    # BF16 mode: load directly
                    kv_ptr = (
                        kv_nope_base + k_cols[None, :] + kv_ids0[:, None] * stride_kvn
                    )
                    k0_l_msk = valid0[:, None] & (k_cols < D)[None, :]
                    k0_l_blk = tl.load(
                        kv_ptr, mask=k0_l_msk, other=0.0, eviction_policy="evict_last"
                    )

                tl.store(
                    tle.gpu.local_ptr(k0_l_slot.sK, (kv_tile_rows, k_cols_b)),
                    k0_l_blk,
                    mask=valid0[:, None] & (k_cols < D)[None, :],
                )
            k0_l_writer.commit(pair)

            # Process k1_r (right half of block 1)
            k1_r_slot = k1_r_writer.acquire(pair)
            for tile in tl.static_range(0, DPH, 64):
                k_cols = DPH + tile + offs_tile
                k_cols_b = tl.broadcast_to(k_cols[None, :], (BK, 64))

                if IS_FP8:
                    kv_ptr = (
                        kv_nope_base + k_cols[None, :] + kv_ids1[:, None] * stride_kvn
                    )
                    k1_r_msk = valid1[:, None] & (k_cols < D)[None, :]
                    k1_r_fp8 = tl.load(
                        kv_ptr, mask=k1_r_msk, other=0.0, eviction_policy="evict_last"
                    )

                    # Scale index: 2 or 3 for right half
                    scale_idx = 2 + (k_cols - DPH) // 128
                    scale1 = tl.load(
                        kv_scales_base + kv_ids1 * stride_scales_n + scale_idx,
                        mask=valid1,
                        other=1.0,
                    )

                    k1_r_blk = (k1_r_fp8.to(tl.float32) * scale1[:, None]).to(
                        tl.bfloat16
                    )
                else:
                    kv_ptr = (
                        kv_nope_base + k_cols[None, :] + kv_ids1[:, None] * stride_kvn
                    )
                    k1_r_msk = valid1[:, None] & (k_cols < D)[None, :]
                    k1_r_blk = tl.load(
                        kv_ptr, mask=k1_r_msk, other=0.0, eviction_policy="evict_last"
                    )

                tl.store(
                    tle.gpu.local_ptr(k1_r_slot.sK, (kv_tile_rows, k_cols_b)),
                    k1_r_blk,
                    mask=valid1[:, None] & (k_cols < D)[None, :],
                )

            # Load RoPE tail if needed
            if HAVE_TAIL:
                offs_td = tl.arange(0, TDP)
                if IS_FP8:
                    k1_r_tail_ptr = (
                        kv_rope_base
                        + offs_td[None, :]
                        + kv_ids1[:, None] * stride_rope_n
                    )
                else:
                    k1_r_tail_ptr = (
                        kv_nope_base
                        + D
                        + offs_td[None, :]
                        + kv_ids1[:, None] * stride_kvn
                    )
                k1_r_tail_msk = valid1[:, None] & (offs_td < TD)[None, :]
                k1_r_tail_blk = tl.load(
                    k1_r_tail_ptr,
                    mask=k1_r_tail_msk,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                tl.store(
                    tle.gpu.local_ptr(k1_r_slot.sK_tail),
                    k1_r_tail_blk,
                    mask=k1_r_tail_msk,
                )
            k1_r_writer.commit(pair)

            # Process k0_r (right half of block 0)
            k0_r_slot = k0_r_writer.acquire(pair)
            for tile in tl.static_range(0, DPH, 64):
                k_cols = DPH + tile + offs_tile
                k_cols_b = tl.broadcast_to(k_cols[None, :], (BK, 64))

                if IS_FP8:
                    kv_ptr = (
                        kv_nope_base + k_cols[None, :] + kv_ids0[:, None] * stride_kvn
                    )
                    k0_r_msk = valid0[:, None] & (k_cols < D)[None, :]
                    k0_r_fp8 = tl.load(
                        kv_ptr, mask=k0_r_msk, other=0.0, eviction_policy="evict_last"
                    )

                    scale_idx = 2 + (k_cols - DPH) // 128
                    scale0 = tl.load(
                        kv_scales_base + kv_ids0 * stride_scales_n + scale_idx,
                        mask=valid0,
                        other=1.0,
                    )

                    k0_r_blk = (k0_r_fp8.to(tl.float32) * scale0[:, None]).to(
                        tl.bfloat16
                    )
                else:
                    kv_ptr = (
                        kv_nope_base + k_cols[None, :] + kv_ids0[:, None] * stride_kvn
                    )
                    k0_r_msk = valid0[:, None] & (k_cols < D)[None, :]
                    k0_r_blk = tl.load(
                        kv_ptr, mask=k0_r_msk, other=0.0, eviction_policy="evict_last"
                    )

                tl.store(
                    tle.gpu.local_ptr(k0_r_slot.sK, (kv_tile_rows, k_cols_b)),
                    k0_r_blk,
                    mask=valid0[:, None] & (k_cols < D)[None, :],
                )

            if HAVE_TAIL:
                offs_td = tl.arange(0, TDP)
                if IS_FP8:
                    k0_r_tail_ptr = (
                        kv_rope_base
                        + offs_td[None, :]
                        + kv_ids0[:, None] * stride_rope_n
                    )
                else:
                    k0_r_tail_ptr = (
                        kv_nope_base
                        + D
                        + offs_td[None, :]
                        + kv_ids0[:, None] * stride_kvn
                    )
                k0_r_tail_msk = valid0[:, None] & (offs_td < TD)[None, :]
                k0_r_tail_blk = tl.load(
                    k0_r_tail_ptr,
                    mask=k0_r_tail_msk,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                tl.store(
                    tle.gpu.local_ptr(k0_r_slot.sK_tail),
                    k0_r_tail_blk,
                    mask=k0_r_tail_msk,
                )
            k0_r_writer.commit(pair)

            # Process k1_l (left half of block 1)
            k1_l_slot = k1_l_writer.acquire(pair)
            for tile in tl.static_range(0, DPH, 64):
                k_cols = tile + offs_tile
                k_cols_b = tl.broadcast_to(k_cols[None, :], (BK, 64))

                if IS_FP8:
                    kv_ptr = (
                        kv_nope_base + k_cols[None, :] + kv_ids1[:, None] * stride_kvn
                    )
                    k1_l_msk = valid1[:, None] & (k_cols < D)[None, :]
                    k1_l_fp8 = tl.load(
                        kv_ptr, mask=k1_l_msk, other=0.0, eviction_policy="evict_last"
                    )

                    scale_idx = k_cols // 128
                    scale1 = tl.load(
                        kv_scales_base + kv_ids1 * stride_scales_n + scale_idx,
                        mask=valid1,
                        other=1.0,
                    )

                    k1_l_blk = (k1_l_fp8.to(tl.float32) * scale1[:, None]).to(
                        tl.bfloat16
                    )
                else:
                    kv_ptr = (
                        kv_nope_base + k_cols[None, :] + kv_ids1[:, None] * stride_kvn
                    )
                    k1_l_msk = valid1[:, None] & (k_cols < D)[None, :]
                    k1_l_blk = tl.load(
                        kv_ptr, mask=k1_l_msk, other=0.0, eviction_policy="evict_last"
                    )

                tl.store(
                    tle.gpu.local_ptr(k1_l_slot.sK, (kv_tile_rows, k_cols_b)),
                    k1_l_blk,
                    mask=valid1[:, None] & (k_cols < D)[None, :],
                )
            k1_l_writer.commit(pair)

            # Store validity masks
            valid_slot = valid_writer.acquire(pair)
            valid_row0 = tl.full([BK], 0, dtype=tl.int32)
            valid_row1 = tl.full([BK], 1, dtype=tl.int32)
            valid_ptr0 = tle.gpu.local_ptr(valid_slot.is_kv_valid, (valid_row0, offs_t))
            valid_ptr1 = tle.gpu.local_ptr(valid_slot.is_kv_valid, (valid_row1, offs_t))
            tl.store(valid_ptr0, valid0.to(tl.int8))
            tl.store(valid_ptr1, valid1.to(tl.int8))
            valid_writer.commit(pair)

    @triton.jit
    def _tle_sparse_decode_consumer0(
        q_writer,
        q_reader,
        q_desc,
        tq_desc,
        k0_l_reader,
        k0_r_qk_reader,
        k1_l_remote_reader,
        valid_reader,
        sM_wg0_writer,
        sM_wg1_reader,
        sS0_writer,
        sS1_reader,
        sL_wg0_writer,
        sL_wg1_reader,
        output_desc,
        output_row,
        h_base,
        topk_len_ptr,
        attn_sink_base,
        log_scale: tl.constexpr,
        D: tl.constexpr,
        TD: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
        HAVE_ATTN_SINK: tl.constexpr,
        TOPK: tl.constexpr,
        HAVE_TOPK_LENGTH: tl.constexpr,
        HAVE_TAIL: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        DPH: tl.constexpr,
        TDP: tl.constexpr,
        G: tl.constexpr,
    ):
        """Consumer 0: computes QK^T + online softmax + P@V_left."""
        topk_len = tl.load(topk_len_ptr) if HAVE_TOPK_LENGTH else TOPK
        offs_h = tl.arange(0, BH)
        offs_dh = tl.arange(0, DPH)
        mask_h = h_base + offs_h < G
        mask_od_l = offs_dh < D
        kv_rows = tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, DPH))
        kv_cols_l = tl.broadcast_to(offs_dh[None, :], (BK, DPH))
        kv_cols_r = tl.broadcast_to((DPH + offs_dh)[None, :], (BK, DPH))

        # Load Q into shared memory (one-shot)
        q_write_slot = q_writer.acquire(0)
        tle.gpu.copy(q_desc, q_write_slot.sQ_l, [BH, DPH], [output_row, 0])
        tle.gpu.copy(q_desc, q_write_slot.sQ_r, [BH, DPH], [output_row, DPH])
        if HAVE_TAIL:
            tle.gpu.copy(tq_desc, q_write_slot.sQ_tail, [BH, TDP], [output_row, D])
        q_writer.commit(0)

        q_slot = q_reader.wait(0).slot
        q_l_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_l)
        q_r_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_r)

        max_prev = tl.full([BH], -1.0e30, dtype=tl.float32)
        sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
        acc_l = tl.zeros([BH, DPH], dtype=tl.float32)

        NK = tl.cdiv(topk_len, BK)
        NPAIRS = tl.cdiv(NK, 2)
        for pair in tl.range(NPAIRS):
            # Wait for k0_l data
            k0_l_wait = k0_l_reader.wait(pair)
            k0_l_slot = k0_l_wait.slot

            q_l_blk = tl.load(q_l_smem_ptr)
            q_r_blk = tl.load(q_r_smem_ptr)
            k0_l_blk = tl.load(tle.gpu.local_ptr(k0_l_slot.sK, (kv_rows, kv_cols_l)))

            # QK for block 0: q_l @ k0_l^T + q_r @ k0_r^T + q_tail @ k0_tail^T
            qk0 = tl.full([BH, BK], 0.0, dtype=tl.float32)
            qk0 = tl.dot(q_l_blk, tl.trans(k0_l_blk), qk0, out_dtype=tl.float32)

            # Wait for k0_r
            k0_r_wait = k0_r_qk_reader.wait(pair)
            k0_r_slot = k0_r_wait.slot
            k0_r_blk = tl.load(tle.gpu.local_ptr(k0_r_slot.sK, (kv_rows, kv_cols_r)))
            qk0 = tl.dot(q_r_blk, tl.trans(k0_r_blk), qk0, out_dtype=tl.float32)

            if HAVE_TAIL:
                q_tail_blk = tl.load(tle.gpu.local_ptr(q_slot.sQ_tail))
                k0_t_blk = tl.load(tle.gpu.local_ptr(k0_r_slot.sK_tail))
                qk0 = tl.dot(q_tail_blk, tl.trans(k0_t_blk), qk0, out_dtype=tl.float32)

            # Get validity mask for block 0
            valid_wait = valid_reader.wait(pair)
            row0 = tl.full([BK], 0, dtype=tl.int32)
            valid0 = (
                tl.load(
                    tle.gpu.local_ptr(
                        valid_wait.slot.is_kv_valid, (row0, tl.arange(0, BK))
                    )
                ).to(tl.int32)
                == 1
            )

            qk0 = tl.where(valid0[None, :], qk0, float("-inf"))

            # Compute local softmax for block 0 only
            local_max = tl.maximum(max_prev, tl.max(qk0, axis=1))
            alpha = tl.math.exp2((max_prev - local_max) * log_scale)
            prob0 = tl.math.exp2(qk0 * log_scale - local_max[:, None] * log_scale)
            sum_exp = sum_exp * alpha + tl.sum(prob0, axis=1)
            acc_l = acc_l * alpha[:, None]
            prob0_b = prob0.to(OUT_DTYPE)

            # Send local_max to consumer1
            sM_wg0_slot = sM_wg0_writer.acquire(pair)
            tl.store(tle.gpu.local_ptr(sM_wg0_slot.sM), local_max)
            sM_wg0_writer.commit(pair)

            # Accumulate P@V_left with prob0
            k0_l_blk = tl.load(tle.gpu.local_ptr(k0_l_slot.sK, (kv_rows, kv_cols_l)))
            acc_l = tl.dot(prob0_b, k0_l_blk, acc_l, out_dtype=tl.float32)
            k0_l_reader.release(pair)
            k0_r_qk_reader.release(pair)

            # Wait for max_next from consumer1 (merged max of block0 and block1)
            sM_wg1_wait = sM_wg1_reader.wait(pair)
            max_next = tl.load(tle.gpu.local_ptr(sM_wg1_wait.slot.sM))
            sM_wg1_reader.release(pair)

            # Rescale prob0 and acc_l using the global max
            final_scale = tl.math.exp2((local_max - max_next) * log_scale)
            sum_exp = sum_exp * final_scale
            acc_l = acc_l * final_scale[:, None]

            # Send rescaled prob0 to consumer1
            prob0_scaled = prob0 * final_scale[:, None]
            sS0_slot = sS0_writer.acquire(pair)
            tl.store(tle.gpu.local_ptr(sS0_slot.sS0), prob0_scaled.to(OUT_DTYPE))
            sS0_writer.commit(pair)

            # Receive prob1 from consumer1 and accumulate k1_l
            sS1_wait = sS1_reader.wait(pair)
            prob1 = tl.load(tle.gpu.local_ptr(sS1_wait.slot.sS1))
            k1_l_wait = k1_l_remote_reader.wait(pair)
            k1_l_blk = tl.load(
                tle.gpu.local_ptr(k1_l_wait.slot.sK, (kv_rows, kv_cols_l))
            )
            acc_l = tl.dot(prob1, k1_l_blk, acc_l, out_dtype=tl.float32)
            sS1_reader.release(pair)
            k1_l_remote_reader.release(pair)

            valid_reader.release(pair)

            max_prev = max_next

        # Exchange final sum_exp with consumer1
        sL_wg0_slot = sL_wg0_writer.acquire(0)
        tl.store(tle.gpu.local_ptr(sL_wg0_slot.sL), sum_exp)
        sL_wg0_writer.commit(0)
        sL_wg1_wait = sL_wg1_reader.wait(1)
        peer_sum = tl.load(tle.gpu.local_ptr(sL_wg1_wait.slot.sL))
        total_sum = sum_exp + peer_sum
        sL_wg1_reader.release(1)

        is_no_valid_tokens = total_sum == 0.0
        inv_total_sum = tl.fdiv(1.0, total_sum)
        out_l_vals = acc_l * inv_total_sum[:, None]
        if HAVE_ATTN_SINK:
            fin_log = (
                max_prev * log_scale + tl.math.log2(total_sum)
            ) * 0.6931471805599453
            sink = tl.load(attn_sink_base + h_base + offs_h, mask_h, other=0.0)
            sink_scale = tl.fdiv(1.0, 1.0 + tl.math.exp(sink - fin_log))
            out_l_vals = out_l_vals * sink_scale[:, None]
        out_l_vals = tl.where(is_no_valid_tokens[:, None], 0.0, out_l_vals)
        o_l_msk = mask_h[:, None] & mask_od_l[None, :]
        tl.store(q_l_smem_ptr, out_l_vals.to(OUT_DTYPE), o_l_msk)
        tle.gpu.copy(q_slot.sQ_l, output_desc, [BH, DPH], [output_row, 0])

    @triton.jit
    def _tle_sparse_decode_consumer1(
        q_reader,
        k1_r_reader,
        k1_l_qk_reader,
        k0_r_remote_reader,
        valid_reader,
        sM_wg1_writer,
        sM_wg0_reader,
        sS1_writer,
        sS0_reader,
        sL_wg1_writer,
        sL_wg0_reader,
        output_desc,
        output_row,
        lse_base,
        h_base,
        topk_len_ptr,
        attn_sink_base,
        log_scale: tl.constexpr,
        D: tl.constexpr,
        TD: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
        HAVE_ATTN_SINK: tl.constexpr,
        TOPK: tl.constexpr,
        HAVE_TOPK_LENGTH: tl.constexpr,
        HAVE_TAIL: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        DPH: tl.constexpr,
        TDP: tl.constexpr,
        G: tl.constexpr,
    ):
        """Consumer 1: computes P@V_right, exchanges softmax state with consumer0."""
        topk_len = tl.load(topk_len_ptr) if HAVE_TOPK_LENGTH else TOPK
        offs_h = tl.arange(0, BH)
        offs_dh = tl.arange(0, DPH)
        mask_h = h_base + offs_h < G
        mask_od_r = DPH + offs_dh < D
        kv_rows = tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, DPH))
        kv_cols_l = tl.broadcast_to(offs_dh[None, :], (BK, DPH))
        kv_cols_r = tl.broadcast_to((DPH + offs_dh)[None, :], (BK, DPH))

        q_slot = q_reader.wait(0).slot
        q_l_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_l)
        q_r_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_r)

        max_prev = tl.full([BH], -1.0e30, dtype=tl.float32)
        sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
        acc_r = tl.zeros([BH, DPH], dtype=tl.float32)

        NK = tl.cdiv(topk_len, BK)
        NPAIRS = tl.cdiv(NK, 2)
        for pair in tl.range(NPAIRS):
            # Wait for k1_r data
            k1_r_wait = k1_r_reader.wait(pair)
            k1_r_slot = k1_r_wait.slot

            q_l_blk = tl.load(q_l_smem_ptr)
            q_r_blk = tl.load(q_r_smem_ptr)
            k1_r_blk = tl.load(tle.gpu.local_ptr(k1_r_slot.sK, (kv_rows, kv_cols_r)))

            # QK for block 1
            qk1 = tl.full([BH, BK], 0.0, dtype=tl.float32)
            qk1 = tl.dot(q_r_blk, tl.trans(k1_r_blk), qk1, out_dtype=tl.float32)
            if HAVE_TAIL:
                q_tail_blk = tl.load(tle.gpu.local_ptr(q_slot.sQ_tail))
                k1_t_blk = tl.load(tle.gpu.local_ptr(k1_r_slot.sK_tail))
                qk1 = tl.dot(q_tail_blk, tl.trans(k1_t_blk), qk1, out_dtype=tl.float32)

            k1_l_wait = k1_l_qk_reader.wait(pair)
            k1_l_slot = k1_l_wait.slot
            k1_l_blk = tl.load(tle.gpu.local_ptr(k1_l_slot.sK, (kv_rows, kv_cols_l)))
            qk1 = tl.dot(q_l_blk, tl.trans(k1_l_blk), qk1, out_dtype=tl.float32)

            # Get validity mask for block 1
            valid_wait = valid_reader.wait(pair)
            row1 = tl.full([BK], 1, dtype=tl.int32)
            valid1 = (
                tl.load(
                    tle.gpu.local_ptr(
                        valid_wait.slot.is_kv_valid, (row1, tl.arange(0, BK))
                    )
                ).to(tl.int32)
                == 1
            )

            qk1 = tl.where(valid1[None, :], qk1, float("-inf"))
            valid_reader.release(pair)

            # Receive candidate0 (local_max) from consumer0
            sM_wg0_wait = sM_wg0_reader.wait(pair)
            candidate0 = tl.load(tle.gpu.local_ptr(sM_wg0_wait.slot.sM))
            sM_wg0_reader.release(pair)

            # Compute candidate1 and merge to get global max_next
            candidate1 = tl.maximum(max_prev, tl.max(qk1, axis=1))
            max_next = tl.maximum(candidate1, candidate0)

            # Send max_next back to consumer0
            sM_wg1_slot = sM_wg1_writer.acquire(pair)
            tl.store(tle.gpu.local_ptr(sM_wg1_slot.sM), max_next)
            sM_wg1_writer.commit(pair)

            # Compute prob1 using global max_next
            alpha = tl.math.exp2((max_prev - max_next) * log_scale)
            prob1 = tl.math.exp2(qk1 * log_scale - max_next[:, None] * log_scale)
            sum_exp = sum_exp * alpha + tl.sum(prob1, axis=1)
            acc_r = acc_r * alpha[:, None]
            prob1_b = prob1.to(OUT_DTYPE)

            k1_l_qk_reader.release(pair)

            # Accumulate P@V_right with prob1
            acc_r = tl.dot(prob1_b, k1_r_blk, acc_r, out_dtype=tl.float32)

            # Send prob1 to consumer0
            sS1_slot = sS1_writer.acquire(pair)
            tl.store(tle.gpu.local_ptr(sS1_slot.sS1), prob1_b)
            sS1_writer.commit(pair)

            # Receive rescaled prob0 from consumer0 and accumulate k0_r
            sS0_wait = sS0_reader.wait(pair)
            prob0 = tl.load(tle.gpu.local_ptr(sS0_wait.slot.sS0))
            k0_r_wait = k0_r_remote_reader.wait(pair)
            k0_r_blk = tl.load(
                tle.gpu.local_ptr(k0_r_wait.slot.sK, (kv_rows, kv_cols_r))
            )
            acc_r = tl.dot(prob0, k0_r_blk, acc_r, out_dtype=tl.float32)
            k1_r_reader.release(pair)
            sS0_reader.release(pair)
            k0_r_remote_reader.release(pair)

            max_prev = max_next

        # Exchange final sum_exp with consumer0
        sL_wg1_slot = sL_wg1_writer.acquire(1)
        tl.store(tle.gpu.local_ptr(sL_wg1_slot.sL), sum_exp)
        sL_wg1_writer.commit(1)
        sL_wg0_wait = sL_wg0_reader.wait(0)
        peer_sum = tl.load(tle.gpu.local_ptr(sL_wg0_wait.slot.sL))
        total_sum = sum_exp + peer_sum
        sL_wg0_reader.release(0)

        is_no_valid_tokens = total_sum == 0.0
        inv_total_sum = tl.fdiv(1.0, total_sum)
        out_r_vals = acc_r * inv_total_sum[:, None]
        if HAVE_ATTN_SINK:
            fin_log = (
                max_prev * log_scale + tl.math.log2(total_sum)
            ) * 0.6931471805599453
            sink = tl.load(attn_sink_base + h_base + offs_h, mask_h, other=0.0)
            sink_scale = tl.fdiv(1.0, 1.0 + tl.math.exp(sink - fin_log))
            out_r_vals = out_r_vals * sink_scale[:, None]
        out_r_vals = tl.where(is_no_valid_tokens[:, None], 0.0, out_r_vals)
        o_r_msk = mask_h[:, None] & mask_od_r[None, :]
        tl.store(q_r_smem_ptr, out_r_vals.to(OUT_DTYPE), o_r_msk)
        tle.gpu.copy(q_slot.sQ_r, output_desc, [BH, DPH], [output_row, DPH])

        # Store LSE
        lse_val = (max_prev * log_scale + tl.math.log2(total_sum)) * 0.6931471805599453
        lse_val = tl.where(is_no_valid_tokens, float("inf"), lse_val)
        tl.store(lse_base + offs_h, lse_val, mask=mask_h)

    @triton.jit
    def _tle_sparse_decode_fwd(
        q_desc,
        tq_desc,
        output_desc,
        kv_nope,
        kv_scales,
        kv_rope,
        indices,
        attn_sink,
        topk_length,
        sm_scale: tl.constexpr,
        output,
        lse,
        BATCH_SQ,
        HQ: tl.constexpr,
        DQK: tl.constexpr,
        SKV,
        TOPK: tl.constexpr,
        HAVE_ATTN_SINK: tl.constexpr,
        HAVE_TOPK_LENGTH: tl.constexpr,
        IS_FP8: tl.constexpr,
        D: tl.constexpr,
        TD: tl.constexpr,
        DP: tl.constexpr,
        TDP: tl.constexpr,
        G: tl.constexpr,
        RH: tl.constexpr,
        HAVE_TAIL: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        PAIR_BLOCKS: tl.constexpr,
        stride_kvn,
        stride_scales_n,
        stride_rope_n,
        stride_ib,
        stride_isq,
    ):
        DPH: tl.constexpr = DP // 2
        stride_lm = HQ

        pid = tl.program_id(0)
        programs_per_bsq: tl.constexpr = RH
        i_bsq = pid // programs_per_bsq
        i_rh = pid % programs_per_bsq
        h_base = i_rh * BH
        i_bsq64 = i_bsq.to(tl.int64)

        kv_nope_base = kv_nope
        kv_scales_base = kv_scales
        kv_rope_base = kv_rope
        t_base = indices + i_bsq64 * stride_isq
        topk_len_ptr = topk_length + i_bsq64 if HAVE_TOPK_LENGTH else indices
        attn_sink_base = attn_sink if HAVE_ATTN_SINK else lse
        l_base = lse + i_bsq64 * stride_lm + h_base
        q_row = i_bsq * HQ + h_base
        _ = output
        _ = BATCH_SQ
        _ = DQK

        sQ_l_smem = tle.gpu.alloc(
            [1, BH, DPH], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem
        )
        sQ_r_smem = tle.gpu.alloc(
            [1, BH, DPH], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem
        )
        if HAVE_TAIL:
            sQ_tail_smem = tle.gpu.alloc(
                [1, BH, TDP], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem
            )
            q_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="decode_sQ",
                readers=("wg0", "wg1"),
                one_shot=True,
                sQ_l=sQ_l_smem,
                sQ_r=sQ_r_smem,
                sQ_tail=sQ_tail_smem,
            )
        else:
            q_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="decode_sQ",
                readers=("wg0", "wg1"),
                one_shot=True,
                sQ_l=sQ_l_smem,
                sQ_r=sQ_r_smem,
            )

        sK0_smem = tle.gpu.alloc(
            [1, BK, DP], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem
        )
        sK1_smem = tle.gpu.alloc(
            [1, BK, DP], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem
        )
        if HAVE_TAIL:
            sK0_tail_smem = tle.gpu.alloc(
                [1, BK, TDP], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem
            )
            sK1_tail_smem = tle.gpu.alloc(
                [1, BK, TDP], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem
            )
            sS0_smem = sK0_tail_smem
            sS1_smem = sK1_tail_smem
        else:
            sS0_smem = tle.gpu.alloc(
                [1, BH, BK], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem
            )
            sS1_smem = tle.gpu.alloc(
                [1, BH, BK], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem
            )
        is_kv_valid_smem = tle.gpu.alloc(
            [1, 2, BK], dtype=tl.int8, layout=None, scope=tle.gpu.smem
        )
        sM_smem = tle.gpu.alloc(
            [1, BH], dtype=tl.float32, layout=None, scope=tle.gpu.smem
        )
        sL_smem = tle.gpu.alloc(
            [2, BH], dtype=tl.float32, layout=None, scope=tle.gpu.smem
        )

        # Pipe definitions
        if HAVE_TAIL:
            k0_l_pipe = tle.pipe(
                capacity=1, scope="cta", name="decode_k0_l", sK=sK0_smem
            )
            k0_r_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="decode_k0_r",
                readers=("qk", "remote"),
                sK=sK0_smem,
                sK_tail=sK0_tail_smem,
            )
            k1_l_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="decode_k1_l",
                readers=("qk", "remote"),
                sK=sK1_smem,
            )
            k1_r_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="decode_k1_r",
                sK=sK1_smem,
                sK_tail=sK1_tail_smem,
            )
        else:
            k0_l_pipe = tle.pipe(
                capacity=1, scope="cta", name="decode_k0_l", sK=sK0_smem
            )
            k0_r_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="decode_k0_r",
                readers=("qk", "remote"),
                sK=sK0_smem,
            )
            k1_l_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="decode_k1_l",
                readers=("qk", "remote"),
                sK=sK1_smem,
            )
            k1_r_pipe = tle.pipe(
                capacity=1, scope="cta", name="decode_k1_r", sK=sK1_smem
            )

        is_kv_valid_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="decode_valid",
            readers=("wg0", "wg1"),
            is_kv_valid=is_kv_valid_smem,
        )
        sM_wg0_pipe = tle.pipe(
            capacity=1, scope="cta", name="decode_wg0_max", sM=sM_smem
        )
        sM_wg1_pipe = tle.pipe(
            capacity=1, scope="cta", name="decode_wg1_max", sM=sM_smem
        )
        sS0_pipe = tle.pipe(capacity=1, scope="cta", name="decode_sS0", sS0=sS0_smem)
        sS1_pipe = tle.pipe(capacity=1, scope="cta", name="decode_sS1", sS1=sS1_smem)
        sL_wg0_pipe = tle.pipe(
            capacity=2, scope="cta", name="decode_sL_wg0", sL=sL_smem
        )
        sL_wg1_pipe = tle.pipe(
            capacity=2, scope="cta", name="decode_sL_wg1", sL=sL_smem
        )

        log_scale: tl.constexpr = sm_scale * 1.4426950408889634

        tle.gpu.warp_specialize(
            [
                (
                    _tle_sparse_decode_consumer0,
                    (
                        q_pipe.writer(),
                        q_pipe.reader("wg0"),
                        q_desc,
                        tq_desc,
                        k0_l_pipe.reader(),
                        k0_r_pipe.reader("qk"),
                        k1_l_pipe.reader("remote", fields=("sK",)),
                        is_kv_valid_pipe.reader("wg0"),
                        sM_wg0_pipe.writer(),
                        sM_wg1_pipe.reader(),
                        sS0_pipe.writer(),
                        sS1_pipe.reader(),
                        sL_wg0_pipe.writer(),
                        sL_wg1_pipe.reader(),
                        output_desc,
                        q_row,
                        h_base,
                        topk_len_ptr,
                        attn_sink_base,
                        log_scale,
                        D,
                        TD,
                        tl.bfloat16,
                        HAVE_ATTN_SINK,
                        TOPK,
                        HAVE_TOPK_LENGTH,
                        HAVE_TAIL,
                        BK,
                        BH,
                        DPH,
                        TDP,
                        G,
                    ),
                ),
                (
                    _tle_sparse_decode_consumer1,
                    (
                        q_pipe.reader("wg1"),
                        k1_r_pipe.reader(),
                        k1_l_pipe.reader("qk"),
                        k0_r_pipe.reader("remote", fields=("sK",)),
                        is_kv_valid_pipe.reader("wg1"),
                        sM_wg1_pipe.writer(),
                        sM_wg0_pipe.reader(),
                        sS1_pipe.writer(),
                        sS0_pipe.reader(),
                        sL_wg1_pipe.writer(),
                        sL_wg0_pipe.reader(),
                        output_desc,
                        q_row,
                        l_base,
                        h_base,
                        topk_len_ptr,
                        attn_sink_base,
                        log_scale,
                        D,
                        TD,
                        tl.bfloat16,
                        HAVE_ATTN_SINK,
                        TOPK,
                        HAVE_TOPK_LENGTH,
                        HAVE_TAIL,
                        BK,
                        BH,
                        DPH,
                        TDP,
                        G,
                    ),
                ),
                (
                    _tle_sparse_decode_producer,
                    (
                        k0_l_pipe.writer(),
                        k0_r_pipe.writer(),
                        k1_l_pipe.writer(),
                        k1_r_pipe.writer(),
                        is_kv_valid_pipe.writer(),
                        kv_nope_base,
                        kv_scales_base,
                        kv_rope_base,
                        t_base,
                        topk_len_ptr,
                        D,
                        TD,
                        DPH,
                        TDP,
                        SKV,
                        TOPK,
                        HAVE_TOPK_LENGTH,
                        HAVE_TAIL,
                        IS_FP8,
                        BK,
                        stride_kvn,
                        stride_scales_n,
                        stride_rope_n,
                    ),
                ),
            ],
            [4, 4],
            [216, 72],
        )


# ============================================================================
# TLE Warp Specialization path for dense decode
# ============================================================================


def _tle_dense_decode_launch(
    q,
    kv_flat,
    block_table,
    cache_seqlens,
    out,
    lse,
    batch_size,
    seq_q,
    num_heads_q,
    head_dim_k,
    head_dim_v,
    page_block_size,
    softmax_scale,
    causal,
):
    """Launch TLE warp-specialized dense decode kernel."""
    from triton.tools.tensor_descriptor import TensorDescriptor

    _set_triton_descriptor_allocator(q.device)

    BH = TLE_DECODE_BH
    BK = TLE_DECODE_BK
    D = head_dim_v  # 512
    TD = head_dim_k - D  # 64 for DQK=576, 0 for DQK=512
    DP = triton.next_power_of_2(D)
    DPH = DP // 2
    HAVE_TAIL = TD > 0
    TDP = triton.next_power_of_2(TD) if HAVE_TAIL else 1
    G = num_heads_q
    RH = G // BH

    # Reshape q for TensorDescriptor: [batch*seq_q*HQ, DQK]
    q_flat = q.reshape(batch_size * seq_q * num_heads_q, head_dim_k).contiguous()
    out_flat = out.reshape(batch_size * seq_q * num_heads_q, head_dim_v)

    q_desc = TensorDescriptor(
        q_flat,
        shape=[batch_size * seq_q * num_heads_q, head_dim_k],
        strides=[head_dim_k, 1],
        block_shape=[BH, DPH],
    )
    if HAVE_TAIL:
        tq_desc = TensorDescriptor(
            q_flat,
            shape=[batch_size * seq_q * num_heads_q, head_dim_k],
            strides=[head_dim_k, 1],
            block_shape=[BH, TDP],
        )
    else:
        tq_desc = q_desc
    output_desc = TensorDescriptor(
        out_flat,
        shape=[batch_size * seq_q * num_heads_q, D],
        strides=[D, 1],
        block_shape=[BH, DPH],
    )

    # Grid: one program per (batch*seq_q, head_block)
    grid = (batch_size * seq_q * RH,)

    # Reshape block_table and cache_seqlens for kernel
    block_table_flat = block_table.reshape(batch_size * seq_q, -1).contiguous()
    cache_seqlens_flat = cache_seqlens.reshape(batch_size * seq_q).contiguous()

    _tle_dense_decode_fwd[grid](
        q_desc,
        tq_desc,
        output_desc,
        kv_flat,
        block_table_flat,
        cache_seqlens_flat,
        softmax_scale,
        out_flat,
        lse.reshape(batch_size * seq_q, num_heads_q).contiguous(),
        batch_size * seq_q,
        num_heads_q,
        head_dim_k,
        page_block_size,
        causal,
        D,
        TD,
        DP,
        TDP,
        G,
        RH,
        HAVE_TAIL,
        BK,
        BH,
        TLE_DECODE_PAIR_BLOCKS,
        kv_flat.stride(0),
        block_table_flat.stride(0),
        num_warps=TLE_DECODE_WORKER_NUM_WARPS,
        num_stages=1,
    )


if HAS_TLE:

    @triton.jit
    def _tle_dense_decode_producer(
        k0_l_writer,
        k0_r_writer,
        k1_l_writer,
        k1_r_writer,
        is_kv_valid_writer,
        kv_base,
        block_table_ptr,
        cache_seqlen,
        D: tl.constexpr,
        TD: tl.constexpr,
        DPH: tl.constexpr,
        TDP: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        HAVE_TAIL: tl.constexpr,
        BK: tl.constexpr,
        stride_kvn: tl.constexpr,
        stride_bt: tl.constexpr,
    ):
        """
        Producer: Load KV pages from paged cache to shared memory.
        Key difference from sparse: pages are contiguous, enabling efficient loads.
        """
        num_pages = tl.cdiv(cache_seqlen, PAGE_SIZE)
        NPAIRS = tl.cdiv(num_pages, 2)

        offs_t = tl.arange(0, BK)
        offs_tile = tl.arange(0, 64)
        kv_tile_rows = tl.broadcast_to(offs_t[:, None], (BK, 64))

        for pair in tl.range(NPAIRS):
            page_idx0 = pair * 2
            page_idx1 = page_idx0 + 1

            # Load physical page numbers from block_table (stride within row is 1)
            phys_page0 = tl.load(
                block_table_ptr + page_idx0,
                mask=page_idx0 < tl.cdiv(cache_seqlen, PAGE_SIZE),
                other=0,
            )
            phys_page1 = tl.load(
                block_table_ptr + page_idx1,
                mask=page_idx1 < tl.cdiv(cache_seqlen, PAGE_SIZE),
                other=0,
            )

            # Compute base addresses for contiguous page data
            base0 = phys_page0.to(tl.int64) * PAGE_SIZE * stride_kvn
            base1 = phys_page1.to(tl.int64) * PAGE_SIZE * stride_kvn

            # Validity masks for partial last page
            t_offs0 = page_idx0 * PAGE_SIZE + offs_t
            t_offs1 = page_idx1 * PAGE_SIZE + offs_t
            valid0 = t_offs0 < cache_seqlen
            valid1 = t_offs1 < cache_seqlen

            # Store validity masks
            valid_slot = is_kv_valid_writer.acquire(pair)
            valid_row0 = tl.full([BK], 0, dtype=tl.int32)
            valid_row1 = tl.full([BK], 1, dtype=tl.int32)
            tl.store(
                tle.gpu.local_ptr(valid_slot.is_kv_valid, (valid_row0, offs_t)),
                valid0.to(tl.int8),
            )
            tl.store(
                tle.gpu.local_ptr(valid_slot.is_kv_valid, (valid_row1, offs_t)),
                valid1.to(tl.int8),
            )
            is_kv_valid_writer.commit(pair)

            # Load page0 left half (NoPE [:256])
            k0_l_slot = k0_l_writer.acquire(pair)
            for tile in tl.static_range(0, DPH, 64):
                k_cols = tile + offs_tile
                k_cols_b = tl.broadcast_to(k_cols[None, :], (BK, 64))
                k0_l_ptr = (
                    kv_base + base0 + offs_t[:, None] * stride_kvn + k_cols[None, :]
                )
                k0_l_msk = valid0[:, None] & (k_cols < D)[None, :]
                k0_l_blk = tl.load(
                    k0_l_ptr, mask=k0_l_msk, other=0.0, eviction_policy="evict_last"
                )
                tl.store(
                    tle.gpu.local_ptr(k0_l_slot.sK, (kv_tile_rows, k_cols_b)),
                    k0_l_blk,
                    mask=k0_l_msk,
                )
            k0_l_writer.commit(pair)

            # Load page1 right half (NoPE [256:512])
            k1_r_slot = k1_r_writer.acquire(pair)
            for tile in tl.static_range(0, DPH, 64):
                k_cols = DPH + tile + offs_tile
                k_cols_b = tl.broadcast_to(k_cols[None, :], (BK, 64))
                k1_r_ptr = (
                    kv_base + base1 + offs_t[:, None] * stride_kvn + k_cols[None, :]
                )
                k1_r_msk = valid1[:, None] & (k_cols < D)[None, :]
                k1_r_blk = tl.load(
                    k1_r_ptr, mask=k1_r_msk, other=0.0, eviction_policy="evict_last"
                )
                tl.store(
                    tle.gpu.local_ptr(k1_r_slot.sK, (kv_tile_rows, k_cols_b)),
                    k1_r_blk,
                    mask=k1_r_msk,
                )
            if HAVE_TAIL:
                offs_td = tl.arange(0, TDP)
                k1_r_tail_ptr = (
                    kv_base
                    + base1
                    + offs_t[:, None] * stride_kvn
                    + (D + offs_td)[None, :]
                )
                k1_r_tail_msk = valid1[:, None] & (offs_td < TD)[None, :]
                k1_r_tail_blk = tl.load(
                    k1_r_tail_ptr,
                    mask=k1_r_tail_msk,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                tl.store(
                    tle.gpu.local_ptr(k1_r_slot.sK_tail),
                    k1_r_tail_blk,
                    mask=k1_r_tail_msk,
                )
            k1_r_writer.commit(pair)

            # Load page0 right half (NoPE [256:512])
            k0_r_slot = k0_r_writer.acquire(pair)
            for tile in tl.static_range(0, DPH, 64):
                k_cols = DPH + tile + offs_tile
                k_cols_b = tl.broadcast_to(k_cols[None, :], (BK, 64))
                k0_r_ptr = (
                    kv_base + base0 + offs_t[:, None] * stride_kvn + k_cols[None, :]
                )
                k0_r_msk = valid0[:, None] & (k_cols < D)[None, :]
                k0_r_blk = tl.load(
                    k0_r_ptr, mask=k0_r_msk, other=0.0, eviction_policy="evict_last"
                )
                tl.store(
                    tle.gpu.local_ptr(k0_r_slot.sK, (kv_tile_rows, k_cols_b)),
                    k0_r_blk,
                    mask=k0_r_msk,
                )
            if HAVE_TAIL:
                offs_td = tl.arange(0, TDP)
                k0_r_tail_ptr = (
                    kv_base
                    + base0
                    + offs_t[:, None] * stride_kvn
                    + (D + offs_td)[None, :]
                )
                k0_r_tail_msk = valid0[:, None] & (offs_td < TD)[None, :]
                k0_r_tail_blk = tl.load(
                    k0_r_tail_ptr,
                    mask=k0_r_tail_msk,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                tl.store(
                    tle.gpu.local_ptr(k0_r_slot.sK_tail),
                    k0_r_tail_blk,
                    mask=k0_r_tail_msk,
                )
            k0_r_writer.commit(pair)

            # Load page1 left half (NoPE [:256])
            k1_l_slot = k1_l_writer.acquire(pair)
            for tile in tl.static_range(0, DPH, 64):
                k_cols = tile + offs_tile
                k_cols_b = tl.broadcast_to(k_cols[None, :], (BK, 64))
                k1_l_ptr = (
                    kv_base + base1 + offs_t[:, None] * stride_kvn + k_cols[None, :]
                )
                k1_l_msk = valid1[:, None] & (k_cols < D)[None, :]
                k1_l_blk = tl.load(
                    k1_l_ptr, mask=k1_l_msk, other=0.0, eviction_policy="evict_last"
                )
                tl.store(
                    tle.gpu.local_ptr(k1_l_slot.sK, (kv_tile_rows, k_cols_b)),
                    k1_l_blk,
                    mask=k1_l_msk,
                )
            k1_l_writer.commit(pair)

    @triton.jit
    def _tle_dense_decode_consumer0(
        q_writer,
        q_reader,
        q_desc,
        tq_desc,
        k0_l_reader,
        k0_r_qk_reader,
        k1_l_remote_reader,
        is_kv_valid_reader,
        sM_wg0_writer,
        sM_wg1_reader,
        sS0_writer,
        sS1_reader,
        sL_wg0_writer,
        sL_wg1_reader,
        output_desc,
        output_row,
        h_base,
        cache_seqlen,
        log_scale: tl.constexpr,
        D: tl.constexpr,
        TD: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
        HAVE_TAIL: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        DPH: tl.constexpr,
        TDP: tl.constexpr,
        G: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
    ):
        """Consumer 0: QK^T left half + softmax + P@V_left."""
        offs_h = tl.arange(0, BH)
        offs_dh = tl.arange(0, DPH)
        mask_h = h_base + offs_h < G
        mask_od_l = offs_dh < D
        kv_rows = tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, DPH))
        kv_cols_l = tl.broadcast_to(offs_dh[None, :], (BK, DPH))
        kv_cols_r = tl.broadcast_to((DPH + offs_dh)[None, :], (BK, DPH))

        # Load Q once
        q_write_slot = q_writer.acquire(0)
        tle.gpu.copy(q_desc, q_write_slot.sQ_l, [BH, DPH], [output_row, 0])
        tle.gpu.copy(q_desc, q_write_slot.sQ_r, [BH, DPH], [output_row, DPH])
        if HAVE_TAIL:
            tle.gpu.copy(tq_desc, q_write_slot.sQ_tail, [BH, TDP], [output_row, D])
        q_writer.commit(0)

        q_slot = q_reader.wait(0).slot
        q_l_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_l)
        q_r_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_r)
        max_prev = tl.full([BH], -1.0e30, dtype=tl.float32)
        sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
        acc_l = tl.zeros([BH, DPH], dtype=tl.float32)

        num_pages = tl.cdiv(cache_seqlen, PAGE_SIZE)
        NPAIRS = tl.cdiv(num_pages, 2)

        for pair in tl.range(NPAIRS):
            # Compute QK^T for page0
            k0_l_wait = k0_l_reader.wait(pair)
            k0_l_slot = k0_l_wait.slot

            q_l_blk = tl.load(q_l_smem_ptr)
            q_r_blk = tl.load(q_r_smem_ptr)
            k0_l_blk = tl.load(tle.gpu.local_ptr(k0_l_slot.sK, (kv_rows, kv_cols_l)))

            qk0 = tl.full([BH, BK], 0.0, dtype=tl.float32)
            qk0 = tl.dot(q_l_blk, tl.trans(k0_l_blk), qk0, out_dtype=tl.float32)

            k0_r_wait = k0_r_qk_reader.wait(pair)
            k0_r_slot = k0_r_wait.slot
            k0_r_blk = tl.load(tle.gpu.local_ptr(k0_r_slot.sK, (kv_rows, kv_cols_r)))
            qk0 = tl.dot(q_r_blk, tl.trans(k0_r_blk), qk0, out_dtype=tl.float32)
            if HAVE_TAIL:
                q_tail_blk = tl.load(tle.gpu.local_ptr(q_slot.sQ_tail))
                k0_t_blk = tl.load(tle.gpu.local_ptr(k0_r_slot.sK_tail))
                qk0 = tl.dot(q_tail_blk, tl.trans(k0_t_blk), qk0, out_dtype=tl.float32)

            # Apply validity mask
            valid_wait = is_kv_valid_reader.wait(pair)
            row0 = tl.full([BK], 0, dtype=tl.int32)
            valid0 = (
                tl.load(
                    tle.gpu.local_ptr(
                        valid_wait.slot.is_kv_valid, (row0, tl.arange(0, BK))
                    )
                )
                != 0
            )
            qk0 = tl.where(valid0[None, :], qk0, float("-inf"))
            is_kv_valid_reader.release(pair)

            # Online softmax
            local_max = tl.maximum(max_prev, tl.max(qk0, axis=1))
            alpha = tl.math.exp2((max_prev - local_max) * log_scale)
            prob0 = tl.math.exp2(qk0 * log_scale - local_max[:, None] * log_scale)
            sum_exp = sum_exp * alpha + tl.sum(prob0, axis=1)
            acc_l = acc_l * alpha[:, None]
            prob0_b = prob0.to(OUT_DTYPE)

            # Send local max to consumer1
            sM_wg0_slot = sM_wg0_writer.acquire(pair)
            tl.store(tle.gpu.local_ptr(sM_wg0_slot.sM), local_max)
            sM_wg0_writer.commit(pair)

            # Accumulate P@V_left
            k0_l_blk = tl.load(tle.gpu.local_ptr(k0_l_slot.sK, (kv_rows, kv_cols_l)))
            acc_l = tl.dot(prob0_b, k0_l_blk, acc_l, out_dtype=tl.float32)
            k0_l_reader.release(pair)
            k0_r_qk_reader.release(pair)

            # Receive final max from consumer1
            sM_wg1_wait = sM_wg1_reader.wait(pair)
            max_next = tl.load(tle.gpu.local_ptr(sM_wg1_wait.slot.sM))
            sM_wg1_reader.release(pair)

            # Rescale with final max
            final_scale = tl.math.exp2((local_max - max_next) * log_scale)
            sum_exp = sum_exp * final_scale
            acc_l = acc_l * final_scale[:, None]

            # Send rescaled prob0 to consumer1
            prob0_scaled = prob0 * final_scale[:, None]
            sS0_slot = sS0_writer.acquire(pair)
            tl.store(tle.gpu.local_ptr(sS0_slot.sS0), prob0_scaled.to(OUT_DTYPE))
            sS0_writer.commit(pair)

            # Receive prob1 and accumulate P@V_left from page1
            sS1_wait = sS1_reader.wait(pair)
            prob1 = tl.load(tle.gpu.local_ptr(sS1_wait.slot.sS1))
            k1_l_wait = k1_l_remote_reader.wait(pair)
            k1_l_blk = tl.load(
                tle.gpu.local_ptr(k1_l_wait.slot.sK, (kv_rows, kv_cols_l))
            )
            acc_l = tl.dot(prob1, k1_l_blk, acc_l, out_dtype=tl.float32)
            sS1_reader.release(pair)
            k1_l_remote_reader.release(pair)

            max_prev = max_next

        # Exchange sum_exp with consumer1
        sL_wg0_slot = sL_wg0_writer.acquire(0)
        tl.store(tle.gpu.local_ptr(sL_wg0_slot.sL), sum_exp)
        sL_wg0_writer.commit(0)
        sL_wg1_wait = sL_wg1_reader.wait(1)
        peer_sum = tl.load(tle.gpu.local_ptr(sL_wg1_wait.slot.sL))
        total_sum = sum_exp + peer_sum
        sL_wg1_reader.release(1)

        # Normalize and write output left half
        is_no_valid_tokens = total_sum == 0.0
        inv_total_sum = tl.fdiv(1.0, total_sum)
        out_l_vals = acc_l * inv_total_sum[:, None]
        out_l_vals = tl.where(is_no_valid_tokens[:, None], 0.0, out_l_vals)
        o_l_msk = mask_h[:, None] & mask_od_l[None, :]
        tl.store(q_l_smem_ptr, out_l_vals.to(OUT_DTYPE), o_l_msk)
        tle.gpu.copy(q_slot.sQ_l, output_desc, [BH, DPH], [output_row, 0])

    @triton.jit
    def _tle_dense_decode_consumer1(
        q_reader,
        k1_r_reader,
        k1_l_qk_reader,
        k0_r_remote_reader,
        is_kv_valid_reader,
        sM_wg1_writer,
        sM_wg0_reader,
        sS1_writer,
        sS0_reader,
        sL_wg1_writer,
        sL_wg0_reader,
        final_lse_smem,
        output_desc,
        output_row,
        l_base,
        h_base,
        cache_seqlen,
        log_scale: tl.constexpr,
        D: tl.constexpr,
        TD: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
        HAVE_TAIL: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        DPH: tl.constexpr,
        TDP: tl.constexpr,
        G: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
    ):
        """Consumer 1: QK^T right half + P@V_right."""
        offs_h = tl.arange(0, BH)
        offs_dh = tl.arange(0, DPH)
        mask_h = h_base + offs_h < G
        mask_od_r = DPH + offs_dh < D
        kv_rows = tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, DPH))
        kv_cols_l = tl.broadcast_to(offs_dh[None, :], (BK, DPH))
        kv_cols_r = tl.broadcast_to((DPH + offs_dh)[None, :], (BK, DPH))

        q_slot = q_reader.wait(0).slot
        q_l_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_l)
        q_r_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_r)
        max_prev = tl.full([BH], -1.0e30, dtype=tl.float32)
        sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
        acc_r = tl.zeros([BH, DPH], dtype=tl.float32)

        num_pages = tl.cdiv(cache_seqlen, PAGE_SIZE)
        NPAIRS = tl.cdiv(num_pages, 2)

        for pair in tl.range(NPAIRS):
            # Compute QK^T for page1
            k1_r_wait = k1_r_reader.wait(pair)
            k1_r_slot = k1_r_wait.slot

            q_l_blk = tl.load(q_l_smem_ptr)
            q_r_blk = tl.load(q_r_smem_ptr)
            k1_r_blk = tl.load(tle.gpu.local_ptr(k1_r_slot.sK, (kv_rows, kv_cols_r)))

            qk1 = tl.full([BH, BK], 0.0, dtype=tl.float32)
            qk1 = tl.dot(q_r_blk, tl.trans(k1_r_blk), qk1, out_dtype=tl.float32)
            if HAVE_TAIL:
                q_tail_blk = tl.load(tle.gpu.local_ptr(q_slot.sQ_tail))
                k1_t_blk = tl.load(tle.gpu.local_ptr(k1_r_slot.sK_tail))
                qk1 = tl.dot(q_tail_blk, tl.trans(k1_t_blk), qk1, out_dtype=tl.float32)

            k1_l_wait = k1_l_qk_reader.wait(pair)
            k1_l_slot = k1_l_wait.slot
            k1_l_blk = tl.load(tle.gpu.local_ptr(k1_l_slot.sK, (kv_rows, kv_cols_l)))
            qk1 = tl.dot(q_l_blk, tl.trans(k1_l_blk), qk1, out_dtype=tl.float32)

            # Apply validity mask
            valid_wait = is_kv_valid_reader.wait(pair)
            row1 = tl.full([BK], 1, dtype=tl.int32)
            valid1 = (
                tl.load(
                    tle.gpu.local_ptr(
                        valid_wait.slot.is_kv_valid, (row1, tl.arange(0, BK))
                    )
                )
                != 0
            )
            qk1 = tl.where(valid1[None, :], qk1, float("-inf"))
            is_kv_valid_reader.release(pair)

            # Receive candidate0 from consumer0
            sM_wg0_wait = sM_wg0_reader.wait(pair)
            candidate0 = tl.load(tle.gpu.local_ptr(sM_wg0_wait.slot.sM))
            sM_wg0_reader.release(pair)

            # Compute final max
            candidate1 = tl.maximum(max_prev, tl.max(qk1, axis=1))
            max_next = tl.maximum(candidate1, candidate0)
            sM_wg1_slot = sM_wg1_writer.acquire(pair)
            tl.store(tle.gpu.local_ptr(sM_wg1_slot.sM), max_next)
            sM_wg1_writer.commit(pair)

            # Online softmax
            alpha = tl.math.exp2((max_prev - max_next) * log_scale)
            prob1 = tl.math.exp2(qk1 * log_scale - max_next[:, None] * log_scale)
            sum_exp = sum_exp * alpha + tl.sum(prob1, axis=1)
            acc_r = acc_r * alpha[:, None]
            prob1_b = prob1.to(OUT_DTYPE)

            k1_l_qk_reader.release(pair)

            # Accumulate P@V_right from page1
            acc_r = tl.dot(prob1_b, k1_r_blk, acc_r, out_dtype=tl.float32)

            # Send prob1 to consumer0
            sS1_slot = sS1_writer.acquire(pair)
            tl.store(tle.gpu.local_ptr(sS1_slot.sS1), prob1_b)
            sS1_writer.commit(pair)

            # Receive prob0 and accumulate P@V_right from page0
            sS0_wait = sS0_reader.wait(pair)
            prob0 = tl.load(tle.gpu.local_ptr(sS0_wait.slot.sS0))
            k0_r_wait = k0_r_remote_reader.wait(pair)
            k0_r_blk = tl.load(
                tle.gpu.local_ptr(k0_r_wait.slot.sK, (kv_rows, kv_cols_r))
            )
            acc_r = tl.dot(prob0, k0_r_blk, acc_r, out_dtype=tl.float32)
            k1_r_reader.release(pair)
            sS0_reader.release(pair)
            k0_r_remote_reader.release(pair)
            max_prev = max_next

        # Exchange sum_exp with consumer0
        sL_wg1_slot = sL_wg1_writer.acquire(1)
        tl.store(tle.gpu.local_ptr(sL_wg1_slot.sL), sum_exp)
        sL_wg1_writer.commit(1)
        sL_wg0_wait = sL_wg0_reader.wait(0)
        peer_sum = tl.load(tle.gpu.local_ptr(sL_wg0_wait.slot.sL))
        total_sum = sum_exp + peer_sum
        sL_wg0_reader.release(0)

        # Normalize and write output right half
        is_no_valid_tokens = total_sum == 0.0
        inv_total_sum = tl.fdiv(1.0, total_sum)
        out_r_vals = acc_r * inv_total_sum[:, None]
        final_max_logits_log2 = max_prev * log_scale
        fin_log = (final_max_logits_log2 + tl.math.log2(total_sum)) * 0.6931471805599453
        out_r_vals = tl.where(is_no_valid_tokens[:, None], 0.0, out_r_vals)
        o_r_msk = mask_h[:, None] & mask_od_r[None, :]
        tl.store(q_r_smem_ptr, out_r_vals.to(OUT_DTYPE), o_r_msk)
        tle.gpu.copy(q_slot.sQ_r, output_desc, [BH, DPH], [output_row, DPH])

        # Write LSE
        fin_log = tl.where(is_no_valid_tokens, float("inf"), fin_log)
        tl.store(tle.gpu.local_ptr(final_lse_smem), fin_log, mask_h)
        fin_log = tl.load(tle.gpu.local_ptr(final_lse_smem), mask_h, other=float("inf"))
        tl.store(l_base + offs_h, fin_log, mask_h)

    @triton.jit
    def _tle_dense_decode_fwd(
        q_desc,
        tq_desc,
        output_desc,
        kv_cache,
        block_table,
        cache_seqlens,
        sm_scale: tl.constexpr,
        output,
        lse,
        BS,
        G: tl.constexpr,
        DQK: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        CAUSAL: tl.constexpr,
        D: tl.constexpr,
        TD: tl.constexpr,
        DP: tl.constexpr,
        TDP: tl.constexpr,
        H: tl.constexpr,
        RH: tl.constexpr,
        HAVE_TAIL: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        PAIR_BLOCKS: tl.constexpr,
        stride_kvn: tl.constexpr,
        stride_bt: tl.constexpr,
    ):
        DPH: tl.constexpr = DP // 2
        stride_lm = G

        pid = tl.program_id(0)
        i_sq = pid // RH
        i_rh = pid % RH
        h_base = i_rh * BH
        output_row = i_sq * G + h_base
        i_sq64 = i_sq.to(tl.int64)

        cache_seqlen = tl.load(cache_seqlens + i_sq64)
        block_table_ptr = block_table + i_sq64 * stride_bt
        kv_base = kv_cache
        l_base = lse + i_sq64 * stride_lm + h_base
        _ = output
        _ = BS
        _ = DQK
        _ = CAUSAL

        sQ_l_smem = tle.gpu.alloc(
            [1, BH, DPH],
            dtype=kv_cache.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        sQ_r_smem = tle.gpu.alloc(
            [1, BH, DPH],
            dtype=kv_cache.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        if HAVE_TAIL:
            sQ_tail_smem = tle.gpu.alloc(
                [1, BH, TDP],
                dtype=kv_cache.dtype.element_ty,
                layout=None,
                scope=tle.gpu.smem,
            )
            q_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="dense_sQ",
                readers=("wg0", "wg1"),
                one_shot=True,
                sQ_l=sQ_l_smem,
                sQ_r=sQ_r_smem,
                sQ_tail=sQ_tail_smem,
            )
        else:
            q_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="dense_sQ",
                readers=("wg0", "wg1"),
                one_shot=True,
                sQ_l=sQ_l_smem,
                sQ_r=sQ_r_smem,
            )

        sK0_smem = tle.gpu.alloc(
            [1, BK, DP],
            dtype=kv_cache.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        sK1_smem = tle.gpu.alloc(
            [1, BK, DP],
            dtype=kv_cache.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        if HAVE_TAIL:
            sK0_tail_smem = tle.gpu.alloc(
                [1, BK, TDP],
                dtype=kv_cache.dtype.element_ty,
                layout=None,
                scope=tle.gpu.smem,
            )
            sK1_tail_smem = tle.gpu.alloc(
                [1, BK, TDP],
                dtype=kv_cache.dtype.element_ty,
                layout=None,
                scope=tle.gpu.smem,
            )
            sS0_smem = sK0_tail_smem
        else:
            sS0_smem = tle.gpu.alloc(
                [1, BH, BK],
                dtype=kv_cache.dtype.element_ty,
                layout=None,
                scope=tle.gpu.smem,
            )

        is_kv_valid_smem = tle.gpu.alloc(
            [1, PAIR_BLOCKS, BK],
            dtype=tl.int8,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )

        k0_l_pipe = tle.pipe(capacity=1, scope="cta", name="dense_sK0_l", sK=sK0_smem)
        if HAVE_TAIL:
            k0_r_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="dense_sK0_r",
                readers=("qk", "remote"),
                sK=sK0_smem,
                sK_tail=sK0_tail_smem,
            )
        else:
            k0_r_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="dense_sK0_r",
                readers=("qk", "remote"),
                sK=sK0_smem,
            )
        k1_l_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="dense_sK1_l",
            readers=("qk", "remote"),
            sK=sK1_smem,
        )
        if HAVE_TAIL:
            k1_r_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="dense_sK1_r",
                sK=sK1_smem,
                sK_tail=sK1_tail_smem,
            )
        else:
            k1_r_pipe = tle.pipe(
                capacity=1, scope="cta", name="dense_sK1_r", sK=sK1_smem
            )

        is_kv_valid_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="dense_is_kv_valid",
            readers=("wg0", "wg1"),
            is_kv_valid=is_kv_valid_smem,
        )

        sM_smem = tle.gpu.alloc(
            [1, BH],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        sS1_smem = tle.gpu.alloc(
            [1, BH, BK],
            dtype=kv_cache.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        sL_smem = tle.gpu.alloc(
            [2, BH],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        final_lse_smem = tle.gpu.alloc(
            [BH],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )

        sM_wg0_pipe = tle.pipe(
            capacity=1, scope="cta", name="dense_wg0_max", sM=sM_smem
        )
        sM_wg1_pipe = tle.pipe(
            capacity=1, scope="cta", name="dense_wg1_max", sM=sM_smem
        )
        sS0_pipe = tle.pipe(capacity=1, scope="cta", name="dense_sS0", sS0=sS0_smem)
        sS1_pipe = tle.pipe(capacity=1, scope="cta", name="dense_sS1", sS1=sS1_smem)
        sL_wg0_pipe = tle.pipe(capacity=2, scope="cta", name="dense_sL_wg0", sL=sL_smem)
        sL_wg1_pipe = tle.pipe(capacity=2, scope="cta", name="dense_sL_wg1", sL=sL_smem)

        log_scale: tl.constexpr = sm_scale * 1.4426950408889634

        tle.gpu.warp_specialize(
            [
                (
                    _tle_dense_decode_consumer0,
                    (
                        q_pipe.writer(),
                        q_pipe.reader("wg0"),
                        q_desc,
                        tq_desc,
                        k0_l_pipe.reader(),
                        k0_r_pipe.reader("qk"),
                        k1_l_pipe.reader("remote", fields=("sK",)),
                        is_kv_valid_pipe.reader("wg0"),
                        sM_wg0_pipe.writer(),
                        sM_wg1_pipe.reader(),
                        sS0_pipe.writer(),
                        sS1_pipe.reader(),
                        sL_wg0_pipe.writer(),
                        sL_wg1_pipe.reader(),
                        output_desc,
                        output_row,
                        h_base,
                        cache_seqlen,
                        log_scale,
                        D,
                        TD,
                        kv_cache.dtype.element_ty,
                        HAVE_TAIL,
                        BK,
                        BH,
                        DPH,
                        TDP,
                        G,
                        PAGE_SIZE,
                    ),
                ),
                (
                    _tle_dense_decode_consumer1,
                    (
                        q_pipe.reader("wg1"),
                        k1_r_pipe.reader(),
                        k1_l_pipe.reader("qk"),
                        k0_r_pipe.reader("remote", fields=("sK",)),
                        is_kv_valid_pipe.reader("wg1"),
                        sM_wg1_pipe.writer(),
                        sM_wg0_pipe.reader(),
                        sS1_pipe.writer(),
                        sS0_pipe.reader(),
                        sL_wg1_pipe.writer(),
                        sL_wg0_pipe.reader(),
                        final_lse_smem,
                        output_desc,
                        output_row,
                        l_base,
                        h_base,
                        cache_seqlen,
                        log_scale,
                        D,
                        TD,
                        kv_cache.dtype.element_ty,
                        HAVE_TAIL,
                        BK,
                        BH,
                        DPH,
                        TDP,
                        G,
                        PAGE_SIZE,
                    ),
                ),
                (
                    _tle_dense_decode_producer,
                    (
                        k0_l_pipe.writer(),
                        k0_r_pipe.writer(),
                        k1_l_pipe.writer(),
                        k1_r_pipe.writer(),
                        is_kv_valid_pipe.writer(),
                        kv_base,
                        block_table_ptr,
                        cache_seqlen,
                        D,
                        TD,
                        DPH,
                        TDP,
                        PAGE_SIZE,
                        HAVE_TAIL,
                        BK,
                        stride_kvn,
                        stride_bt,
                    ),
                ),
            ],
            [4, 4],
            [216, 72],
        )
