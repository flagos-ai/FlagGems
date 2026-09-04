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
Triton implementation of the mHC Pre operator (kunlunxin / XPU specialized).

Why a specialized file (XPU, measured 2026-08-20):
- The general implementation in `flag_gems/fused/mhc/mhc_pre.py` uses
  ``@triton.autotune`` (10 configs). On XPU triton, autotune recompiles ALL
  configs for every new key -> a single hc_mult=4 case takes >15min just
  compiling, blowing the 900s per-case gate (measured: one (512,1280,4)
  benchmark case still compiling after 20+ min).
- The generic (HC != 4) path of the general kernel implements the Sinkhorn
  iterations by storing to / loading back from global memory inside one
  program. On XPU, scalar stores followed by loads to the same address within
  a single program do NOT see the updated value (measured: readback probe
  returns 5.0 vs expected 3.0), so comb_mix is wrong (~88% mismatched
  elements for hc_mult=2, case n512_h1280).
- Masked tails are unreliable on XPU reductions: masked lanes contribute
  adjacent-memory garbage even with `other=0.0` and further zeroing (measured
  up to ~58% error, H=1280/BLOCK=1024). This file therefore:
  * computes the rms sqrsum through an exact-tile PARTIAL kernel
    (BLOCK=1024 for hc_mult=4, 512 for hc_mult=2; hidden sizes
    {1280,2560,4096} are all divisible by 256 so every tile is exact and
    unmasked);
  * keeps the Sinkhorn and all mix math in registers (no global read-back);
  * masks only in the final elementwise weighted pass (no reduction there,
    masked loads/stores are safe in that pattern).

Kernel decomposition (3 kernels, all single-shot, no internal loops):
  1. sqrsum partials   grid (N, cdiv(HC*H, B))  -> (N, T) partials
  2. mixes + sinkhorn  grid (N,)                -> post / comb / pre buffers
  3. weighted row      grid (N,)                -> layer_input (masked, safe)

Key points:
- NO @triton.autotune, single config, num_stages=1.
- H, HC are tl.constexpr; Sinkhorn repeat loop is a runtime loop
  (`range(sinkhorn_repeat - 1)`); a `tl.static_range` unroll of the Sinkhorn
  iterations is pathologically slow / hangs the XPU compiler (measured).
- For hc_mult in (2, 4) and hidden_size % 256 == 0 the XPU kernels below are
  used; any other shape is routed to the general implementation to preserve
  upstream behavior.
"""

import logging
import os
import weakref

import torch
import triton
import triton.language as tl

from flag_gems.fused.mhc.mhc_pre import mhc_pre as _general_mhc_pre

logger = logging.getLogger(__name__)

_SUPPORTED_HC = (2, 4)
# exact-tile sizes: hidden sizes {1280,2560,4096} all satisfy
# 4*H % 1024 == 0 and 2*H % 512 == 0.
_PART_BLOCK = {2: 512, 4: 1024}
_ROW_BLOCK_MAX = 8192

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
def _sqrsum_partials_kernel(
    residual_ptr,  # (N, HC, H) bf16, contiguous
    part_ptr,  # (N, T) f32, T = HC*H // B
    H: tl.constexpr,
    HC: tl.constexpr,
    B: tl.constexpr,
):
    """Exact unmasked per-token tile squares (grid (N, cdiv(HC*H, B)))."""
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    offs = pid_t * B + tl.arange(0, B)
    v = tl.load(residual_ptr + pid_n * HC * H + offs).to(tl.float32)
    tl.store(part_ptr + pid_n * (HC * H // B) + pid_t, tl.sum(v * v))


@triton.jit
def _mix_kernel_hc4(
    part_ptr,  # (N, 16) f32
    gemm_out_ptr,  # (N, 24) f32
    hc_scale_ptr,  # (3,) f32
    hc_base_ptr,  # (24,) f32
    post_mix_ptr,  # (N, 4) f32
    comb_mix_ptr,  # (N, 16) f32
    pre_mix_ptr,  # (N, 4) f32
    T: tl.constexpr,
    H: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
):
    """Per-token mixes + Sinkhorn (HC=4), all in registers."""
    pid = tl.program_id(0)
    sq = 0.0
    for i in tl.static_range(T):
        sq += tl.load(part_ptr + pid * T + i)
    rms_inv = tl.rsqrt(sq / (4 * H) + rms_eps)

    scale_0 = tl.load(hc_scale_ptr + 0)
    scale_1 = tl.load(hc_scale_ptr + 1)
    scale_2 = tl.load(hc_scale_ptr + 2)
    go = pid * 24

    pre_0 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go + 0) * rms_inv * scale_0
            + tl.load(hc_base_ptr + 0)
        )
        + hc_pre_eps
    )
    pre_1 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go + 1) * rms_inv * scale_0
            + tl.load(hc_base_ptr + 1)
        )
        + hc_pre_eps
    )
    pre_2 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go + 2) * rms_inv * scale_0
            + tl.load(hc_base_ptr + 2)
        )
        + hc_pre_eps
    )
    pre_3 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go + 3) * rms_inv * scale_0
            + tl.load(hc_base_ptr + 3)
        )
        + hc_pre_eps
    )
    tl.store(pre_mix_ptr + pid * 4 + 0, pre_0)
    tl.store(pre_mix_ptr + pid * 4 + 1, pre_1)
    tl.store(pre_mix_ptr + pid * 4 + 2, pre_2)
    tl.store(pre_mix_ptr + pid * 4 + 3, pre_3)

    post_0 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go + 4) * rms_inv * scale_1
            + tl.load(hc_base_ptr + 4)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + pid * 4 + 0, post_0)
    post_1 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go + 5) * rms_inv * scale_1
            + tl.load(hc_base_ptr + 5)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + pid * 4 + 1, post_1)
    post_2 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go + 6) * rms_inv * scale_1
            + tl.load(hc_base_ptr + 6)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + pid * 4 + 2, post_2)
    post_3 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go + 7) * rms_inv * scale_1
            + tl.load(hc_base_ptr + 7)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + pid * 4 + 3, post_3)

    cb = 8
    cm_00 = tl.load(gemm_out_ptr + go + cb + 0) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 0
    )
    cm_01 = tl.load(gemm_out_ptr + go + cb + 1) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 1
    )
    cm_02 = tl.load(gemm_out_ptr + go + cb + 2) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 2
    )
    cm_03 = tl.load(gemm_out_ptr + go + cb + 3) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 3
    )
    cm_10 = tl.load(gemm_out_ptr + go + cb + 4) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 4
    )
    cm_11 = tl.load(gemm_out_ptr + go + cb + 5) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 5
    )
    cm_12 = tl.load(gemm_out_ptr + go + cb + 6) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 6
    )
    cm_13 = tl.load(gemm_out_ptr + go + cb + 7) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 7
    )
    cm_20 = tl.load(gemm_out_ptr + go + cb + 8) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 8
    )
    cm_21 = tl.load(gemm_out_ptr + go + cb + 9) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 9
    )
    cm_22 = tl.load(gemm_out_ptr + go + cb + 10) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 10
    )
    cm_23 = tl.load(gemm_out_ptr + go + cb + 11) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 11
    )
    cm_30 = tl.load(gemm_out_ptr + go + cb + 12) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 12
    )
    cm_31 = tl.load(gemm_out_ptr + go + cb + 13) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 13
    )
    cm_32 = tl.load(gemm_out_ptr + go + cb + 14) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 14
    )
    cm_33 = tl.load(gemm_out_ptr + go + cb + 15) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 15
    )

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

    cmb = pid * 16
    tl.store(comb_mix_ptr + cmb + 0, cm_00)
    tl.store(comb_mix_ptr + cmb + 1, cm_01)
    tl.store(comb_mix_ptr + cmb + 2, cm_02)
    tl.store(comb_mix_ptr + cmb + 3, cm_03)
    tl.store(comb_mix_ptr + cmb + 4, cm_10)
    tl.store(comb_mix_ptr + cmb + 5, cm_11)
    tl.store(comb_mix_ptr + cmb + 6, cm_12)
    tl.store(comb_mix_ptr + cmb + 7, cm_13)
    tl.store(comb_mix_ptr + cmb + 8, cm_20)
    tl.store(comb_mix_ptr + cmb + 9, cm_21)
    tl.store(comb_mix_ptr + cmb + 10, cm_22)
    tl.store(comb_mix_ptr + cmb + 11, cm_23)
    tl.store(comb_mix_ptr + cmb + 12, cm_30)
    tl.store(comb_mix_ptr + cmb + 13, cm_31)
    tl.store(comb_mix_ptr + cmb + 14, cm_32)
    tl.store(comb_mix_ptr + cmb + 15, cm_33)


@triton.jit
def _mix_kernel_hc2(
    part_ptr,  # (N, T) f32
    gemm_out_ptr,  # (N, 8) f32
    hc_scale_ptr,
    hc_base_ptr,
    post_mix_ptr,
    comb_mix_ptr,
    pre_mix_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
):
    """Per-token mixes + 2x2 Sinkhorn (register-resident)."""
    pid = tl.program_id(0)
    sq = 0.0
    for i in tl.static_range(T):
        sq += tl.load(part_ptr + pid * T + i)
    rms_inv = tl.rsqrt(sq / (2 * H) + rms_eps)

    scale_0 = tl.load(hc_scale_ptr + 0)
    scale_1 = tl.load(hc_scale_ptr + 1)
    scale_2 = tl.load(hc_scale_ptr + 2)

    go = pid * 8

    pre_0 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go + 0) * rms_inv * scale_0
            + tl.load(hc_base_ptr + 0)
        )
        + hc_pre_eps
    )
    pre_1 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go + 1) * rms_inv * scale_0
            + tl.load(hc_base_ptr + 1)
        )
        + hc_pre_eps
    )
    tl.store(pre_mix_ptr + pid * 2 + 0, pre_0)
    tl.store(pre_mix_ptr + pid * 2 + 1, pre_1)

    post_0 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go + 2) * rms_inv * scale_1
            + tl.load(hc_base_ptr + 2)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + pid * 2 + 0, post_0)
    post_1 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go + 3) * rms_inv * scale_1
            + tl.load(hc_base_ptr + 3)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + pid * 2 + 1, post_1)

    cm_00 = tl.load(gemm_out_ptr + go + 4) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + 4
    )
    cm_01 = tl.load(gemm_out_ptr + go + 5) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + 5
    )
    cm_10 = tl.load(gemm_out_ptr + go + 6) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + 6
    )
    cm_11 = tl.load(gemm_out_ptr + go + 7) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + 7
    )

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

    cmb = pid * 4
    tl.store(comb_mix_ptr + cmb + 0, cm_00)
    tl.store(comb_mix_ptr + cmb + 1, cm_01)
    tl.store(comb_mix_ptr + cmb + 2, cm_10)
    tl.store(comb_mix_ptr + cmb + 3, cm_11)


@triton.jit
def _weighted_row_kernel(
    residual_ptr,  # (N, HC, H) bf16
    pre_mix_ptr,  # (N, HC) f32
    layer_input_ptr,  # (N, H) bf16
    H: tl.constexpr,
    HC: tl.constexpr,
    B: tl.constexpr,
):
    """Whole-row elementwise weighted sum (grid (N,)); masked lanes safe here
    (elementwise only, no reductions) - mask only guards the tail store."""
    pid = tl.program_id(0)
    offs = tl.arange(0, B)
    m = offs < H
    base = pid * HC * H
    acc = tl.zeros([B], dtype=tl.float32)
    for k in tl.static_range(HC):
        pk = tl.load(pre_mix_ptr + pid * HC + k)
        r = tl.load(residual_ptr + base + k * H + offs, mask=m, other=0.0).to(
            tl.float32
        )
        acc += pk * r
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

    # ── Step 1: GEMM (vendor bf16 matmul) ──
    x_flat = residual_flat.reshape(num_tokens, hc_hidden_size)
    fn_bf16 = _get_fn_bf16_cached(fn)
    gemm_out = torch.mm(x_flat, fn_bf16.t()).float()

    # ── Step 2: fused sqrsum + norm + mix + sinkhorn + weighted sum ──
    T = (hc_mult * hidden_size) // _PART_BLOCK[hc_mult]
    part = torch.empty(num_tokens, T, dtype=torch.float32, device=device)
    pre_mix = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=device)
    post_mix = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=device)
    comb_mix = torch.empty(
        num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )

    grid_tiles = (num_tokens, T)
    _sqrsum_partials_kernel[grid_tiles](
        residual_flat,
        part,
        H=hidden_size,
        HC=hc_mult,
        B=_PART_BLOCK[hc_mult],
        num_warps=4,
        num_stages=1,
    )
    mix_kwargs = dict(
        part_ptr=part,
        T=T,
        H=hidden_size,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        num_warps=4,
        num_stages=1,
    )
    mix_common = dict(
        gemm_out_ptr=gemm_out,
        hc_scale_ptr=hc_scale,
        hc_base_ptr=hc_base,
        post_mix_ptr=post_mix,
        comb_mix_ptr=comb_mix,
        pre_mix_ptr=pre_mix,
    )
    if hc_mult == 4:
        _mix_kernel_hc4[(num_tokens,)](**mix_common, **mix_kwargs)
    else:
        _mix_kernel_hc2[(num_tokens,)](**mix_common, **mix_kwargs)
    row_block = min(triton.next_power_of_2(hidden_size), 8192)
    _weighted_row_kernel[(num_tokens,)](
        residual_flat,
        pre_mix,
        layer_input,
        H=hidden_size,
        HC=hc_mult,
        B=row_block,
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
