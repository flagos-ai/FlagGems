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
import os
import weakref

import torch
import triton
import triton.language as tl

from flag_gems.fused.mhc.mhc_pre import mhc_pre as _general_mhc_pre

logger = logging.getLogger(__name__)

_SUPPORTED_HC = (2, 4)
_PART_BLOCK = {2: 512, 4: 1024}
_ROW_BLOCK_MAX = 8192
_MIX_BLOCK = 64
_SQR_BLOCK_MAX = 4096

_FN_BF16_CACHE: weakref.WeakKeyDictionary[torch.Tensor, tuple[int, torch.Tensor]] = (
    weakref.WeakKeyDictionary()
)


def _get_fn_bf16_cached(fn: torch.Tensor) -> torch.Tensor:
    if fn.requires_grad or torch.is_grad_enabled():
        return fn.to(dtype=torch.bfloat16)
    version = fn._version
    cached = _FN_BF16_CACHE.get(fn)
    if cached is not None:
        cached_version, cached_bf16 = cached
        if cached_version == version:
            return cached_bf16
    fn_bf16 = fn.to(dtype=torch.bfloat16)
    _FN_BF16_CACHE[fn] = (version, fn_bf16)
    return fn_bf16


# ─────────────────────────────── kernels ───────────────────────────────
@triton.jit
def _sqrsum_kernel(
    residual_ptr,  # (N, HC, H) bf16, contiguous
    sqrsum_ptr,  # (N,) f32
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per-token sum of squares over (HC, H), one program per token.

    The internal loop over BLOCK-wide tiles keeps each program reading a long
    contiguous run (high HBM throughput), and BLOCK always divides TOTAL so no
    masking is needed.
    """
    pid = tl.program_id(0)
    base = pid * TOTAL
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in tl.static_range(0, TOTAL, BLOCK):
        v = tl.load(residual_ptr + base + off + tl.arange(0, BLOCK)).to(tl.float32)
        acc += v * v
    tl.store(sqrsum_ptr + pid, tl.sum(acc))


@triton.jit
def _mix_kernel_hc4(
    sqrsum_ptr,  # (N,) f32
    gemm_out_ptr,  # (24, N) f32
    hc_scale_ptr,  # (3,) f32
    hc_base_ptr,  # (24,) f32
    post_mix_ptr,  # (4, N) f32
    comb_mix_ptr,  # (16, N) f32
    pre_mix_ptr,  # (4, N) f32
    N,
    H: tl.constexpr,
    BN: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
):
    """Per-token mixes + Sinkhorn (HC=4), vectorised over BN tokens.

    All buffers are transposed so the token dimension N is contiguous, which
    keeps every load/store coalesced along N (the strided-row layout of
    (N, C) would otherwise turn each [BN] access into a stride-C gather).
    """
    pid = tl.program_id(0)
    offs = pid * BN + tl.arange(0, BN)
    mask = offs < N

    sq = tl.load(sqrsum_ptr + offs, mask=mask, other=0.0)
    rms_inv = tl.rsqrt(sq / (4 * H) + rms_eps)

    scale_0 = tl.load(hc_scale_ptr + 0)
    scale_1 = tl.load(hc_scale_ptr + 1)
    scale_2 = tl.load(hc_scale_ptr + 2)

    pre_0 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + 0 * N + offs, mask=mask, other=0.0)
            * rms_inv
            * scale_0
            + tl.load(hc_base_ptr + 0)
        )
        + hc_pre_eps
    )
    pre_1 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + 1 * N + offs, mask=mask, other=0.0)
            * rms_inv
            * scale_0
            + tl.load(hc_base_ptr + 1)
        )
        + hc_pre_eps
    )
    pre_2 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + 2 * N + offs, mask=mask, other=0.0)
            * rms_inv
            * scale_0
            + tl.load(hc_base_ptr + 2)
        )
        + hc_pre_eps
    )
    pre_3 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + 3 * N + offs, mask=mask, other=0.0)
            * rms_inv
            * scale_0
            + tl.load(hc_base_ptr + 3)
        )
        + hc_pre_eps
    )
    tl.store(pre_mix_ptr + 0 * N + offs, pre_0, mask=mask)
    tl.store(pre_mix_ptr + 1 * N + offs, pre_1, mask=mask)
    tl.store(pre_mix_ptr + 2 * N + offs, pre_2, mask=mask)
    tl.store(pre_mix_ptr + 3 * N + offs, pre_3, mask=mask)

    post_0 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + 4 * N + offs, mask=mask, other=0.0)
            * rms_inv
            * scale_1
            + tl.load(hc_base_ptr + 4)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + 0 * N + offs, post_0, mask=mask)
    post_1 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + 5 * N + offs, mask=mask, other=0.0)
            * rms_inv
            * scale_1
            + tl.load(hc_base_ptr + 5)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + 1 * N + offs, post_1, mask=mask)
    post_2 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + 6 * N + offs, mask=mask, other=0.0)
            * rms_inv
            * scale_1
            + tl.load(hc_base_ptr + 6)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + 2 * N + offs, post_2, mask=mask)
    post_3 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + 7 * N + offs, mask=mask, other=0.0)
            * rms_inv
            * scale_1
            + tl.load(hc_base_ptr + 7)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + 3 * N + offs, post_3, mask=mask)

    cm_00 = tl.load(
        gemm_out_ptr + 8 * N + offs, mask=mask, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + 8)
    cm_01 = tl.load(
        gemm_out_ptr + 9 * N + offs, mask=mask, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + 9)
    cm_02 = tl.load(
        gemm_out_ptr + 10 * N + offs, mask=mask, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + 10)
    cm_03 = tl.load(
        gemm_out_ptr + 11 * N + offs, mask=mask, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + 11)
    cm_10 = tl.load(
        gemm_out_ptr + 12 * N + offs, mask=mask, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + 12)
    cm_11 = tl.load(
        gemm_out_ptr + 13 * N + offs, mask=mask, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + 13)
    cm_12 = tl.load(
        gemm_out_ptr + 14 * N + offs, mask=mask, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + 14)
    cm_13 = tl.load(
        gemm_out_ptr + 15 * N + offs, mask=mask, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + 15)
    cm_20 = tl.load(
        gemm_out_ptr + 16 * N + offs, mask=mask, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + 16)
    cm_21 = tl.load(
        gemm_out_ptr + 17 * N + offs, mask=mask, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + 17)
    cm_22 = tl.load(
        gemm_out_ptr + 18 * N + offs, mask=mask, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + 18)
    cm_23 = tl.load(
        gemm_out_ptr + 19 * N + offs, mask=mask, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + 19)
    cm_30 = tl.load(
        gemm_out_ptr + 20 * N + offs, mask=mask, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + 20)
    cm_31 = tl.load(
        gemm_out_ptr + 21 * N + offs, mask=mask, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + 21)
    cm_32 = tl.load(
        gemm_out_ptr + 22 * N + offs, mask=mask, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + 22)
    cm_33 = tl.load(
        gemm_out_ptr + 23 * N + offs, mask=mask, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + 23)

    # row softmax + eps
    rm = tl.maximum(tl.maximum(cm_00, cm_01), tl.maximum(cm_02, cm_03))
    e_00 = tl.exp(cm_00 - rm)
    e_01 = tl.exp(cm_01 - rm)
    e_02 = tl.exp(cm_02 - rm)
    e_03 = tl.exp(cm_03 - rm)
    rs = e_00 + e_01 + e_02 + e_03
    inv_rs = 1.0 / rs
    cm_00 = e_00 * inv_rs + hc_sinkhorn_eps
    cm_01 = e_01 * inv_rs + hc_sinkhorn_eps
    cm_02 = e_02 * inv_rs + hc_sinkhorn_eps
    cm_03 = e_03 * inv_rs + hc_sinkhorn_eps

    rm = tl.maximum(tl.maximum(cm_10, cm_11), tl.maximum(cm_12, cm_13))
    e_10 = tl.exp(cm_10 - rm)
    e_11 = tl.exp(cm_11 - rm)
    e_12 = tl.exp(cm_12 - rm)
    e_13 = tl.exp(cm_13 - rm)
    rs = e_10 + e_11 + e_12 + e_13
    inv_rs = 1.0 / rs
    cm_10 = e_10 * inv_rs + hc_sinkhorn_eps
    cm_11 = e_11 * inv_rs + hc_sinkhorn_eps
    cm_12 = e_12 * inv_rs + hc_sinkhorn_eps
    cm_13 = e_13 * inv_rs + hc_sinkhorn_eps

    rm = tl.maximum(tl.maximum(cm_20, cm_21), tl.maximum(cm_22, cm_23))
    e_20 = tl.exp(cm_20 - rm)
    e_21 = tl.exp(cm_21 - rm)
    e_22 = tl.exp(cm_22 - rm)
    e_23 = tl.exp(cm_23 - rm)
    rs = e_20 + e_21 + e_22 + e_23
    inv_rs = 1.0 / rs
    cm_20 = e_20 * inv_rs + hc_sinkhorn_eps
    cm_21 = e_21 * inv_rs + hc_sinkhorn_eps
    cm_22 = e_22 * inv_rs + hc_sinkhorn_eps
    cm_23 = e_23 * inv_rs + hc_sinkhorn_eps

    rm = tl.maximum(tl.maximum(cm_30, cm_31), tl.maximum(cm_32, cm_33))
    e_30 = tl.exp(cm_30 - rm)
    e_31 = tl.exp(cm_31 - rm)
    e_32 = tl.exp(cm_32 - rm)
    e_33 = tl.exp(cm_33 - rm)
    rs = e_30 + e_31 + e_32 + e_33
    inv_rs = 1.0 / rs
    cm_30 = e_30 * inv_rs + hc_sinkhorn_eps
    cm_31 = e_31 * inv_rs + hc_sinkhorn_eps
    cm_32 = e_32 * inv_rs + hc_sinkhorn_eps
    cm_33 = e_33 * inv_rs + hc_sinkhorn_eps

    # column normalize (+ eps in denominator)
    cs0 = cm_00 + cm_10 + cm_20 + cm_30
    cs1 = cm_01 + cm_11 + cm_21 + cm_31
    cs2 = cm_02 + cm_12 + cm_22 + cm_32
    cs3 = cm_03 + cm_13 + cm_23 + cm_33
    inv_cs0 = 1.0 / (cs0 + hc_sinkhorn_eps)
    inv_cs1 = 1.0 / (cs1 + hc_sinkhorn_eps)
    inv_cs2 = 1.0 / (cs2 + hc_sinkhorn_eps)
    inv_cs3 = 1.0 / (cs3 + hc_sinkhorn_eps)
    cm_00 *= inv_cs0
    cm_10 *= inv_cs0
    cm_20 *= inv_cs0
    cm_30 *= inv_cs0
    cm_01 *= inv_cs1
    cm_11 *= inv_cs1
    cm_21 *= inv_cs1
    cm_31 *= inv_cs1
    cm_02 *= inv_cs2
    cm_12 *= inv_cs2
    cm_22 *= inv_cs2
    cm_32 *= inv_cs2
    cm_03 *= inv_cs3
    cm_13 *= inv_cs3
    cm_23 *= inv_cs3
    cm_33 *= inv_cs3

    # Sinkhorn iterations (runtime loop keeps XPU compile fast)
    for _ in range(sinkhorn_repeat - 1):
        rs0 = cm_00 + cm_01 + cm_02 + cm_03
        rs1 = cm_10 + cm_11 + cm_12 + cm_13
        rs2 = cm_20 + cm_21 + cm_22 + cm_23
        rs3 = cm_30 + cm_31 + cm_32 + cm_33
        inv_rs0 = 1.0 / (rs0 + hc_sinkhorn_eps)
        inv_rs1 = 1.0 / (rs1 + hc_sinkhorn_eps)
        inv_rs2 = 1.0 / (rs2 + hc_sinkhorn_eps)
        inv_rs3 = 1.0 / (rs3 + hc_sinkhorn_eps)
        cm_00 *= inv_rs0
        cm_01 *= inv_rs0
        cm_02 *= inv_rs0
        cm_03 *= inv_rs0
        cm_10 *= inv_rs1
        cm_11 *= inv_rs1
        cm_12 *= inv_rs1
        cm_13 *= inv_rs1
        cm_20 *= inv_rs2
        cm_21 *= inv_rs2
        cm_22 *= inv_rs2
        cm_23 *= inv_rs2
        cm_30 *= inv_rs3
        cm_31 *= inv_rs3
        cm_32 *= inv_rs3
        cm_33 *= inv_rs3
        cs0 = cm_00 + cm_10 + cm_20 + cm_30
        cs1 = cm_01 + cm_11 + cm_21 + cm_31
        cs2 = cm_02 + cm_12 + cm_22 + cm_32
        cs3 = cm_03 + cm_13 + cm_23 + cm_33
        inv_cs0 = 1.0 / (cs0 + hc_sinkhorn_eps)
        inv_cs1 = 1.0 / (cs1 + hc_sinkhorn_eps)
        inv_cs2 = 1.0 / (cs2 + hc_sinkhorn_eps)
        inv_cs3 = 1.0 / (cs3 + hc_sinkhorn_eps)
        cm_00 *= inv_cs0
        cm_01 *= inv_cs1
        cm_02 *= inv_cs2
        cm_03 *= inv_cs3
        cm_10 *= inv_cs0
        cm_11 *= inv_cs1
        cm_12 *= inv_cs2
        cm_13 *= inv_cs3
        cm_20 *= inv_cs0
        cm_21 *= inv_cs1
        cm_22 *= inv_cs2
        cm_23 *= inv_cs3
        cm_30 *= inv_cs0
        cm_31 *= inv_cs1
        cm_32 *= inv_cs2
        cm_33 *= inv_cs3

    tl.store(comb_mix_ptr + 0 * N + offs, cm_00, mask=mask)
    tl.store(comb_mix_ptr + 1 * N + offs, cm_01, mask=mask)
    tl.store(comb_mix_ptr + 2 * N + offs, cm_02, mask=mask)
    tl.store(comb_mix_ptr + 3 * N + offs, cm_03, mask=mask)
    tl.store(comb_mix_ptr + 4 * N + offs, cm_10, mask=mask)
    tl.store(comb_mix_ptr + 5 * N + offs, cm_11, mask=mask)
    tl.store(comb_mix_ptr + 6 * N + offs, cm_12, mask=mask)
    tl.store(comb_mix_ptr + 7 * N + offs, cm_13, mask=mask)
    tl.store(comb_mix_ptr + 8 * N + offs, cm_20, mask=mask)
    tl.store(comb_mix_ptr + 9 * N + offs, cm_21, mask=mask)
    tl.store(comb_mix_ptr + 10 * N + offs, cm_22, mask=mask)
    tl.store(comb_mix_ptr + 11 * N + offs, cm_23, mask=mask)
    tl.store(comb_mix_ptr + 12 * N + offs, cm_30, mask=mask)
    tl.store(comb_mix_ptr + 13 * N + offs, cm_31, mask=mask)
    tl.store(comb_mix_ptr + 14 * N + offs, cm_32, mask=mask)
    tl.store(comb_mix_ptr + 15 * N + offs, cm_33, mask=mask)


@triton.jit
def _mix_kernel_hc2(
    sqrsum_ptr,  # (N,) f32
    gemm_out_ptr,  # (8, N) f32
    hc_scale_ptr,
    hc_base_ptr,
    post_mix_ptr,  # (2, N) f32
    comb_mix_ptr,  # (4, N) f32
    pre_mix_ptr,  # (2, N) f32
    N,
    H: tl.constexpr,
    BN: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
):
    """Per-token mixes + 2x2 Sinkhorn, vectorised over BN tokens."""
    pid = tl.program_id(0)
    offs = pid * BN + tl.arange(0, BN)
    mask = offs < N

    sq = tl.load(sqrsum_ptr + offs, mask=mask, other=0.0)
    rms_inv = tl.rsqrt(sq / (2 * H) + rms_eps)

    scale_0 = tl.load(hc_scale_ptr + 0)
    scale_1 = tl.load(hc_scale_ptr + 1)
    scale_2 = tl.load(hc_scale_ptr + 2)

    pre_0 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + 0 * N + offs, mask=mask, other=0.0)
            * rms_inv
            * scale_0
            + tl.load(hc_base_ptr + 0)
        )
        + hc_pre_eps
    )
    pre_1 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + 1 * N + offs, mask=mask, other=0.0)
            * rms_inv
            * scale_0
            + tl.load(hc_base_ptr + 1)
        )
        + hc_pre_eps
    )
    tl.store(pre_mix_ptr + 0 * N + offs, pre_0, mask=mask)
    tl.store(pre_mix_ptr + 1 * N + offs, pre_1, mask=mask)

    post_0 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + 2 * N + offs, mask=mask, other=0.0)
            * rms_inv
            * scale_1
            + tl.load(hc_base_ptr + 2)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + 0 * N + offs, post_0, mask=mask)
    post_1 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + 3 * N + offs, mask=mask, other=0.0)
            * rms_inv
            * scale_1
            + tl.load(hc_base_ptr + 3)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + 1 * N + offs, post_1, mask=mask)

    cm_00 = tl.load(
        gemm_out_ptr + 4 * N + offs, mask=mask, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + 4)
    cm_01 = tl.load(
        gemm_out_ptr + 5 * N + offs, mask=mask, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + 5)
    cm_10 = tl.load(
        gemm_out_ptr + 6 * N + offs, mask=mask, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + 6)
    cm_11 = tl.load(
        gemm_out_ptr + 7 * N + offs, mask=mask, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + 7)

    rm = tl.maximum(cm_00, cm_01)
    e_00 = tl.exp(cm_00 - rm)
    e_01 = tl.exp(cm_01 - rm)
    rs = e_00 + e_01
    cm_00 = e_00 / rs + hc_sinkhorn_eps
    cm_01 = e_01 / rs + hc_sinkhorn_eps
    rm = tl.maximum(cm_10, cm_11)
    e_10 = tl.exp(cm_10 - rm)
    e_11 = tl.exp(cm_11 - rm)
    rs = e_10 + e_11
    cm_10 = e_10 / rs + hc_sinkhorn_eps
    cm_11 = e_11 / rs + hc_sinkhorn_eps

    cs0 = cm_00 + cm_10
    cs1 = cm_01 + cm_11
    inv_cs0 = 1.0 / (cs0 + hc_sinkhorn_eps)
    inv_cs1 = 1.0 / (cs1 + hc_sinkhorn_eps)
    cm_00 *= inv_cs0
    cm_10 *= inv_cs0
    cm_01 *= inv_cs1
    cm_11 *= inv_cs1

    for _ in range(sinkhorn_repeat - 1):
        rs0 = cm_00 + cm_01
        rs1 = cm_10 + cm_11
        inv_rs0 = 1.0 / (rs0 + hc_sinkhorn_eps)
        inv_rs1 = 1.0 / (rs1 + hc_sinkhorn_eps)
        cm_00 *= inv_rs0
        cm_01 *= inv_rs0
        cm_10 *= inv_rs1
        cm_11 *= inv_rs1
        cs0 = cm_00 + cm_10
        cs1 = cm_01 + cm_11
        inv_cs0 = 1.0 / (cs0 + hc_sinkhorn_eps)
        inv_cs1 = 1.0 / (cs1 + hc_sinkhorn_eps)
        cm_00 *= inv_cs0
        cm_10 *= inv_cs0
        cm_01 *= inv_cs1
        cm_11 *= inv_cs1

    tl.store(comb_mix_ptr + 0 * N + offs, cm_00, mask=mask)
    tl.store(comb_mix_ptr + 1 * N + offs, cm_01, mask=mask)
    tl.store(comb_mix_ptr + 2 * N + offs, cm_10, mask=mask)
    tl.store(comb_mix_ptr + 3 * N + offs, cm_11, mask=mask)


@triton.jit
def _weighted_row_kernel(
    residual_ptr,  # (N, HC, H) bf16
    pre_mix_ptr,  # (N, HC) f32
    layer_input_ptr,  # (N, H) bf16
    H: tl.constexpr,
    HC: tl.constexpr,
    B: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Whole-row elementwise weighted sum (grid (N,)).

    EVEN selects the no-mask fast path when B == H (power-of-two hidden size),
    which reads at ~3x the bandwidth of the masked path. For non-power-of-two
    H the masked path is unavoidable: a smaller unmasked block (the largest
    power-of-two divisor of H) collapses to <30 GB/s on this backend, so the
    masked B=next_pow2(H) form wins despite the masked-memory overhead.
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, B)
    base = pid * HC * H
    acc = tl.zeros([B], dtype=tl.float32)
    for k in tl.static_range(HC):
        pk = tl.load(pre_mix_ptr + pid * HC + k)
        if EVEN:
            r = tl.load(residual_ptr + base + k * H + offs).to(tl.float32)
        else:
            m = offs < H
            r = tl.load(residual_ptr + base + k * H + offs, mask=m, other=0.0).to(
                tl.float32
            )
        acc += pk * r
    if EVEN:
        tl.store(layer_input_ptr + pid * H + offs, acc.to(tl.bfloat16))
    else:
        m = offs < H
        tl.store(layer_input_ptr + pid * H + offs, acc.to(tl.bfloat16), mask=m)


# ─────────────────────────────── dispatch ───────────────────────────────


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    mHC pre block (kunlunxin / XPU specialized).

    Same interface and semantics as `flag_gems.fused.mhc.mhc_pre.mhc_pre`.
    Uses the XPU kernels above for hc_mult in {2, 4} with hidden_size % 256 ==
    0 (the full mhc test / benchmark matrix); other shapes go to the general
    implementation.
    """
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    use_xpu = hc_mult in _SUPPORTED_HC and hidden_size % 256 == 0
    if not use_xpu:
        return _general_mhc_pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits=n_splits,
        )

    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32

    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape == (hc_mult3, hc_hidden_size)

    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, hc_mult, hidden_size).contiguous()
    num_tokens = residual_flat.shape[0]
    device = residual.device

    # ── Step 1: GEMM (gems triton matmul, f32 accumulator -> bf16 out) ──
    x_flat = residual_flat.reshape(num_tokens, hc_hidden_size)
    fn_bf16 = _get_fn_bf16_cached(fn)
    from flag_gems.runtime.backend._kunlunxin.ops.mm import mm as _gems_mm

    gemm_out = _gems_mm(x_flat, fn_bf16.t()).float()  # (N, hc_mult3)
    # Transpose to (hc_mult3, N) so the mix kernel reads each of the hc_mult3
    # columns as a contiguous [BN] slice along N (coalesced). Small buffer.
    gemm_out_t = gemm_out.t().contiguous()

    # ── Step 2: fused sqrsum + norm + mix + sinkhorn + weighted sum ──
    total = hc_mult * hidden_size
    sqrsum = torch.empty(num_tokens, dtype=torch.float32, device=device)
    pre_t = torch.empty(hc_mult, num_tokens, dtype=torch.float32, device=device)
    post_t = torch.empty(hc_mult, num_tokens, dtype=torch.float32, device=device)
    comb_t = torch.empty(
        hc_mult * hc_mult, num_tokens, dtype=torch.float32, device=device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )

    sqr_block = min(_SQR_BLOCK_MAX, total & -total)
    _sqrsum_kernel[(num_tokens,)](
        residual_flat,
        sqrsum,
        TOTAL=total,
        BLOCK=sqr_block,
        num_warps=4,
        num_stages=1,
    )
    mix_block = _MIX_BLOCK
    mix_grid = (triton.cdiv(num_tokens, mix_block),)
    mix_kwargs = dict(
        sqrsum_ptr=sqrsum,
        N=num_tokens,
        H=hidden_size,
        BN=mix_block,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        num_warps=2,
        num_stages=1,
    )
    mix_common = dict(
        gemm_out_ptr=gemm_out_t,
        hc_scale_ptr=hc_scale,
        hc_base_ptr=hc_base,
        post_mix_ptr=post_t,
        comb_mix_ptr=comb_t,
        pre_mix_ptr=pre_t,
    )
    if hc_mult == 4:
        _mix_kernel_hc4[mix_grid](**mix_common, **mix_kwargs)
    else:
        _mix_kernel_hc2[mix_grid](**mix_common, **mix_kwargs)

    # Transpose the small (hc_mult, N) / (hc_mult^2, N) outputs back to the
    # (N, ...) layout expected downstream (memory-only copies).
    pre_mix = pre_t.t().contiguous()
    post_mix = post_t.t().contiguous()
    comb_mix = comb_t.t().contiguous()

    row_block = min(_ROW_BLOCK_MAX, triton.next_power_of_2(hidden_size))
    row_even = hidden_size == row_block
    _weighted_row_kernel[(num_tokens,)](
        residual_flat,
        pre_mix,
        layer_input,
        H=hidden_size,
        HC=hc_mult,
        B=row_block,
        EVEN=row_even,
        num_warps=8,
        num_stages=1,
    )

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)
    return post_mix, comb_mix, layer_input


# ────────────────────────────── wiring ──────────────────────────────


def _use_general_for_ab():
    """A/B escape hatch: set FLAGGEMS_XPU_MHC_PRE_GENERAL=1 to force the
    general implementation (used only for baseline measurement / ablation)."""
    return os.environ.get("FLAGGEMS_XPU_MHC_PRE_GENERAL", "0") == "1"


def _install():
    """Wire the XPU implementation into the direct-import entrypoint.

    The mhc fused family is called via direct module import
    (`from flag_gems.fused.mhc.mhc_pre import mhc_pre`) in both
    tests/test_mhc_ops.py and benchmark/test_mhc.py, so the normal
    SpecOpRegistrar namespace swap can not reach it. Replace the attribute on
    the already-imported module (loaded during `import flag_gems`).
    """
    if _use_general_for_ab():
        return
    import sys

    mod = sys.modules.get("flag_gems.fused.mhc.mhc_pre")
    if mod is not None:
        cur = getattr(mod, "mhc_pre", None)
        if cur is _general_mhc_pre:
            mod.mhc_pre = mhc_pre


_install()
