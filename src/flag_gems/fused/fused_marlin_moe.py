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

# SPDX-License-Identifier: Apache-2.0
"""
Fused Marlin MoE for FlagGems.

Aligns the interface of vLLM v0.20.0:
    vllm/model_executor/layers/fused_moe/fused_marlin_moe.py :: fused_marlin_moe

PHASE 2 (this file): bypass `fused_experts_impl`'s dequant-then-FP16-GEMM
shortcut and dispatch directly to the wna16 Triton kernel
(`fused_moe_kernel_gptq_awq`) for true fused-dequant W4A16/W8A16 GEMM.

The local helper `_fused_marlin_moe_impl` mirrors `fused_experts_impl`'s
orchestration (chunk loop, moe_align, two GEMMs, activation, reduction)
but deletes the INT4/INT8 dequant branch and forwards `block_shape` so
the wna16 path is actually taken.

MVP scope:
  - quant_type: GPTQ uint4b8 (INT4) and uint8b128 (INT8)
  - activation: SwiGLU / SiLU
  - act_order:  NOT supported (g_idx / sort_indices must be None)
  - FP8 input:  NOT supported
  - LoRA, clamp_limit, expert_map: NOT supported
"""

import functools
from enum import IntEnum
from typing import Any, Callable, NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl
from torch.utils.weak import WeakTensorKeyDictionary

from flag_gems import runtime
from flag_gems.fused.fused_moe import (
    MoEActivation,
    _get_config_dtype_str,
    _get_config_quant_dtype,
    apply_moe_activation,
    dispatch_fused_moe_kernel,
    moe_kernel_quantize_input,
    try_get_optimal_moe_config,
)
from flag_gems.fused.moe_align_block_size import (
    moe_align_block_size,
    moe_align_block_size_singleton,
    moe_align_block_size_small_grouped,
)
from flag_gems.fused.moe_sum import moe_sum
from flag_gems.fused.silu_and_mul import silu_and_mul_out
from flag_gems.utils import libentry, libtuner

# ----------------------------------------------------------------------------
# quant_type_id constants — mirror a subset of vLLM scalar_types ids.
# ----------------------------------------------------------------------------
# GPTQ INT4 (weight stored as w + 8, dequant subtracts 8)
QUANT_TYPE_UINT4B8 = 0
# INT8 (weight stored as w + 128)
QUANT_TYPE_UINT8B128 = 1
# MXFP4 (FP4 E2M1 weight + per-32 E8M0 scale). Mirrors vLLM scalar_types.float4_e2m1f.id.
QUANT_TYPE_FP4_E2M1 = 6
# MXFP4 block size (E8M0 scale shared by every 32 weights).
MXFP4_GROUP_SIZE = 32

_QUANT_TYPE_INT4 = {QUANT_TYPE_UINT4B8}
_QUANT_TYPE_INT8 = {QUANT_TYPE_UINT8B128}
_QUANT_TYPE_FP4 = {QUANT_TYPE_FP4_E2M1}
_SUPPORTED_QUANT_TYPES = _QUANT_TYPE_INT4 | _QUANT_TYPE_INT8 | _QUANT_TYPE_FP4
_FULL_HOPPER_MIN_SM_COUNT = 100


@functools.lru_cache(maxsize=1)
def _is_hopper() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


class _W4A16KernelPolicy(NamedTuple):
    block_m: int
    use_fused_gemm1_silu: bool
    move_router_weight_before_gemm2: bool


class _MXFP4KernelPolicy(NamedTuple):
    block_m: int
    use_fused_gemm1_silu: bool
    align_mode: "_MXFP4AlignMode"


class _DeviceInfo(NamedTuple):
    is_hopper: bool
    has_full_hopper_sm_count: bool


class _RouterWeightPlacement(IntEnum):
    none = 0
    before_silu = 1
    after_silu = 2


class _MXFP4AlignMode(IntEnum):
    default = 0
    singleton = 1
    small_grouped = 2


def _router_weight_placement(
    apply_router_weight_on_input: bool,
    move_router_weight_before_gemm2: bool,
) -> _RouterWeightPlacement:
    if apply_router_weight_on_input:
        return _RouterWeightPlacement.before_silu
    if move_router_weight_before_gemm2:
        return _RouterWeightPlacement.after_silu
    return _RouterWeightPlacement.none


@functools.lru_cache(maxsize=None)
def _get_device_info(device: torch.device) -> _DeviceInfo:
    if device.type != "cuda" or not torch.cuda.is_available():
        return _DeviceInfo(
            is_hopper=False,
            has_full_hopper_sm_count=False,
        )
    device_index = torch.cuda.current_device() if device.index is None else device.index
    props = torch.cuda.get_device_properties(device_index)
    is_hopper = props.major >= 9
    return _DeviceInfo(
        is_hopper=is_hopper,
        has_full_hopper_sm_count=(
            is_hopper and props.multi_processor_count >= _FULL_HOPPER_MIN_SM_COUNT
        ),
    )


def _routed_tokens_per_expert(M: int, E: int, top_k: int) -> int:
    return max(M * max(top_k, 1) // max(E, 1), 1)


def _select_block_m(
    M: int,
    E: int,
    top_k: int,
    cutoff: int,
) -> int:
    routed_tokens_per_expert = _routed_tokens_per_expert(M, E, top_k)
    if routed_tokens_per_expert <= cutoff:
        return 16
    if routed_tokens_per_expert <= 64:
        return 32
    return 64


def _select_w4a16_kernel_policy(
    device: torch.device,
    M: int,
    E: int,
    top_k: int,
    swap_ab: bool,
    apply_router_weight_on_input: bool,
) -> _W4A16KernelPolicy:
    device_info = _get_device_info(device)

    # Base tiling policy. Full Hopper uses a smaller cutoff because it has
    # enough SMs to keep smaller routed-token CTAs busy; reduced Hopper uses a
    # larger cutoff to avoid excessive tiny CTAs.
    if not swap_ab:
        block_m_cutoff = 16
    elif not device_info.is_hopper or device_info.has_full_hopper_sm_count:
        block_m_cutoff = 8
    else:
        block_m_cutoff = 16

    block_m = _select_block_m(M, E, top_k, block_m_cutoff)

    # Full Hopper keeps the fused GEMM1+SiLU path on broadly. Reduced Hopper
    # uses it for tiny decode and larger-token batches, while avoiding the
    # small-mid token range that regressed in H20 sweeps.
    is_full_hopper = device_info.is_hopper and device_info.has_full_hopper_sm_count
    is_reduced_hopper = (
        device_info.is_hopper and not device_info.has_full_hopper_sm_count
    )
    if is_full_hopper:
        use_fused_gemm1_silu = True
    elif is_reduced_hopper:
        use_fused_gemm1_silu = M <= 4 or M >= 64
    else:
        use_fused_gemm1_silu = False

    move_router_weight_before_gemm2 = (
        use_fused_gemm1_silu and not apply_router_weight_on_input and M >= 512
    )

    return _W4A16KernelPolicy(
        block_m=block_m,
        use_fused_gemm1_silu=use_fused_gemm1_silu,
        move_router_weight_before_gemm2=move_router_weight_before_gemm2,
    )


def _select_mxfp4_kernel_policy(
    device: torch.device,
    M: int,
    E: int,
    top_k: int,
    swap_ab: bool,
) -> _MXFP4KernelPolicy:
    device_info = _get_device_info(device)
    is_reduced_hopper = (
        device_info.is_hopper and not device_info.has_full_hopper_sm_count
    )
    routed_tokens_per_expert = _routed_tokens_per_expert(M, E, top_k)
    block_m = _select_block_m(M, E, top_k, 8 if swap_ab else 16)
    if is_reduced_hopper and M <= 128 and routed_tokens_per_expert < 8:
        block_m = 8

    if is_reduced_hopper and M == 1:
        align_mode = _MXFP4AlignMode.singleton
    elif is_reduced_hopper and ((M == 4 and E <= 512) or (M == 8 and E <= 8)):
        align_mode = _MXFP4AlignMode.small_grouped
    else:
        align_mode = _MXFP4AlignMode.default

    return _MXFP4KernelPolicy(
        block_m=block_m,
        use_fused_gemm1_silu=is_reduced_hopper and M > 1,
        align_mode=align_mode,
    )


def _align_mxfp4_tokens(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_m: int,
    align_mode: _MXFP4AlignMode,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if align_mode == _MXFP4AlignMode.singleton:
        return moe_align_block_size_singleton(topk_ids, block_m)
    if align_mode == _MXFP4AlignMode.small_grouped:
        return moe_align_block_size_small_grouped(topk_ids, num_experts, block_m)
    return moe_align_block_size(
        topk_ids=topk_ids,
        block_size=block_m,
        num_experts=num_experts,
        expert_map=None,
    )


# ============================================================================
# W4A16 (GPTQ uint4b8) fast path: tile-B + nibble-interleaved weight packing
# fed to a magic-number SIMD INT4->bf16/fp16 dequant + tl.dot kernel. This is
# the Hopper-gated short path taken by fused_marlin_moe for plain GPTQ uint4b8.
# ============================================================================
_W_PACK_CACHE: WeakTensorKeyDictionary = WeakTensorKeyDictionary()
_SCALE_PACK_CACHE: WeakTensorKeyDictionary = WeakTensorKeyDictionary()
_SCALE_PACK_CACHE_E8M0: WeakTensorKeyDictionary = WeakTensorKeyDictionary()


def _pack_w_interleave(w: torch.Tensor, block_size_k: int) -> torch.Tensor:
    assert w.dtype == torch.uint8
    assert w.ndim == 3
    assert (
        block_size_k % 8 == 0
    ), f"BLOCK_SIZE_K={block_size_k} must be multiple of 8 (8 logical K per int32)"
    E, N_out, K_half = w.shape
    K = K_half * 2
    B = block_size_k // 8
    assert K % (8 * B) == 0, f"K={K} must be divisible by BLOCK_SIZE_K={block_size_k}"
    num_groups = K // (8 * B)

    _NIBBLE_PERM = (0, 4, 1, 5, 2, 6, 3, 7)
    _BIT_SHIFTS = tuple(4 * p for p in _NIBBLE_PERM)
    shifts = torch.tensor(_BIT_SHIFTS, dtype=torch.int32, device=w.device)
    out = torch.empty(E, K // 8, N_out, dtype=torch.int32, device=w.device)

    for e in range(E):
        we = w[e]  # (N_out, K//2) uint8
        low = (we & 0xF).to(torch.uint8)
        high = ((we >> 4) & 0xF).to(torch.uint8)
        unpacked = torch.stack([low, high], dim=-1).reshape(N_out, K)
        tiled = unpacked.reshape(N_out, num_groups, 8, B).transpose(-1, -2)
        # (N_out, num_groups, B, 8)
        packed = (tiled.to(torch.int32) << shifts).sum(dim=-1, dtype=torch.int32)
        # (N_out, num_groups, B) -> (N_out, K//8)
        packed = packed.reshape(N_out, K // 8)
        out[e].copy_(packed.transpose(0, 1))
    return out  # (E, K//8, N_out)


def _pack_scale_transpose(s: torch.Tensor) -> torch.Tensor:
    assert s.ndim == 3
    return s.transpose(-2, -1).contiguous()


def _cached_pack_w(w: torch.Tensor, block_size_k: int, cached: bool) -> torch.Tensor:
    if not cached:
        return _pack_w_interleave(w, block_size_k)
    per_w = _W_PACK_CACHE.get(w)
    if per_w is None:
        per_w = {}
        _W_PACK_CACHE[w] = per_w
    packed = per_w.get(block_size_k)
    if packed is None:
        packed = _pack_w_interleave(w, block_size_k)
        per_w[block_size_k] = packed
    return packed


def _cached_pack_scale(s: torch.Tensor, cached: bool) -> torch.Tensor:
    if not cached:
        return _pack_scale_transpose(s)
    packed = _SCALE_PACK_CACHE.get(s)
    if packed is None:
        packed = _pack_scale_transpose(s)
        _SCALE_PACK_CACHE[s] = packed
    return packed


def w4a16_pack(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    *,
    cached: bool = True,
    pack_strategy: str = "interleave",
    block_size_k: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if pack_strategy != "interleave":
        raise NotImplementedError(
            f"pack_strategy={pack_strategy!r} not supported (only 'interleave')"
        )
    w1_packed = _cached_pack_w(w1, block_size_k, cached=cached)
    w2_packed = _cached_pack_w(w2, block_size_k, cached=cached)
    w1_scale_packed = (
        _cached_pack_scale(w1_scale, cached=cached) if w1_scale is not None else None
    )
    w2_scale_packed = (
        _cached_pack_scale(w2_scale, cached=cached) if w2_scale is not None else None
    )
    return w1_packed, w2_packed, w1_scale_packed, w2_scale_packed


def _pack_scale_e8m0(s: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
    # E8M0 (value = 2^(byte-127)) -> compute_dtype, transposed to (E, K/gs, N).
    s_u8 = s.view(torch.uint8) if s.dtype != torch.uint8 else s
    scale = torch.exp2((s_u8.to(torch.int32) - 127).to(torch.float32)).to(compute_dtype)
    return scale.transpose(-2, -1).contiguous()


def _cached_pack_scale_e8m0(s, compute_dtype, cached: bool) -> torch.Tensor:
    if not cached:
        return _pack_scale_e8m0(s, compute_dtype)
    packed = _SCALE_PACK_CACHE_E8M0.get(s)
    if packed is None:
        packed = _pack_scale_e8m0(s, compute_dtype)
        _SCALE_PACK_CACHE_E8M0[s] = packed
    return packed


def mxfp4_pack(
    w1, w2, w1_scale, w2_scale, compute_dtype, *, cached=True, block_size_k=128
):
    return (
        _cached_pack_w(w1, block_size_k, cached=cached),
        _cached_pack_w(w2, block_size_k, cached=cached),
        _cached_pack_scale_e8m0(w1_scale, compute_dtype, cached=cached),
        _cached_pack_scale_e8m0(w2_scale, compute_dtype, cached=cached),
    )


@triton.jit
def _dequant_int4_fp16(b, scales):
    x1, x2, x3, x4, x5, x6, x7, x8 = tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b32  r0, r1, r2, r3, r4, r5, r6, r8, r9, r10, r11, r12;
        .reg .b16  h0, h1, h2, h3, h4, h5, h6, h7;
        .reg .b16  s;
        mov.u32 r0, $8;
        shr.u32 r1, r0, 8;
        lop3.b32 r2, r0, 983055,     1677747200,  234;   // (r0 & 0x000F000F) | 0x64006400
        lop3.b32 r3, r0, 15728880,   1677747200,  234;   // (r0 & 0x00F000F0) | 0x64006400
        lop3.b32 r4, r1, 983055,     1677747200,  234;
        lop3.b32 r5, r1, 15728880,   1677747200,  234;
        mov.u32 r6,  1678271496;                          // 0x64086408 = (1032,1032)
        mov.u32 r8,   738208768;                          // 0x2C002C00 = (1/16,1/16)
        mov.u32 r9,  -729754496;                          // 0xD480D480 = (-72,-72)
        sub.f16x2     r10, r2, r6;
        sub.f16x2     r12, r4, r6;
        fma.rn.f16x2  r11, r3, r8, r9;
        fma.rn.f16x2  r4,  r5, r8, r9;
        mov.b32 {h0, h1}, r10;
        mov.b32 {h2, h3}, r11;
        mov.b32 {h4, h5}, r12;
        mov.b32 {h6, h7}, r4;
        mov.b16 s, $9;
        mul.f16 h0, h0, s;
        mul.f16 h1, h1, s;
        mul.f16 h2, h2, s;
        mul.f16 h3, h3, s;
        mul.f16 h4, h4, s;
        mul.f16 h5, h5, s;
        mul.f16 h6, h6, s;
        mul.f16 h7, h7, s;
        mov.b16 $0, h0;
        mov.b16 $1, h1;
        mov.b16 $2, h2;
        mov.b16 $3, h3;
        mov.b16 $4, h4;
        mov.b16 $5, h5;
        mov.b16 $6, h6;
        mov.b16 $7, h7;
        }
        """,
        constraints="=h,=h,=h,=h,=h,=h,=h,=h,r,h",
        args=[b, scales],
        dtype=(tl.float16,) * 8,
        is_pure=True,
        pack=1,
    )
    return x1, x2, x3, x4, x5, x6, x7, x8


@triton.jit
def _dequant_int4_bf16(b, scales):
    x1, x2, x3, x4, x5, x6, x7, x8 = tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b32  r0, r1, r2, r3, q0, q1, q2, q3, s0, s1, s2, s3, magic;
        .reg .b16  h0, h1, h2, h3, h4, h5, h6, h7;
        .reg .b16  s;
        mov.u32 r0, $8;
        shr.u32 r1, r0, 4;          // high nibble of bytes 0,2 -> bits 0-3
        shr.u32 r2, r0, 8;          // low  nibble of bytes 1,3 -> bits 0-3
        shr.u32 r3, r0, 12;         // high nibble of bytes 1,3 -> bits 0-3
        // (x & 0x000F000F) | 0x43004300 -> bf16x2 of (128+nibble, 128+nibble)
        lop3.b32 q0, r0, 983055, 1124090624, 234;
        lop3.b32 q1, r1, 983055, 1124090624, 234;
        lop3.b32 q2, r2, 983055, 1124090624, 234;
        lop3.b32 q3, r3, 983055, 1124090624, 234;
        mov.u32 magic, 1124614920;  // 0x43084308 = (136,136)
        sub.rn.bf16x2 s0, q0, magic;
        sub.rn.bf16x2 s1, q1, magic;
        sub.rn.bf16x2 s2, q2, magic;
        sub.rn.bf16x2 s3, q3, magic;
        mov.b32 {h0, h1}, s0;       // (n0-8, n4-8)
        mov.b32 {h2, h3}, s1;       // (n1-8, n5-8)
        mov.b32 {h4, h5}, s2;       // (n2-8, n6-8)
        mov.b32 {h6, h7}, s3;       // (n3-8, n7-8)
        mov.b16 s, $9;
        mul.rn.bf16 h0, h0, s;
        mul.rn.bf16 h1, h1, s;
        mul.rn.bf16 h2, h2, s;
        mul.rn.bf16 h3, h3, s;
        mul.rn.bf16 h4, h4, s;
        mul.rn.bf16 h5, h5, s;
        mul.rn.bf16 h6, h6, s;
        mul.rn.bf16 h7, h7, s;
        mov.b16 $0, h0;
        mov.b16 $1, h1;
        mov.b16 $2, h2;
        mov.b16 $3, h3;
        mov.b16 $4, h4;
        mov.b16 $5, h5;
        mov.b16 $6, h6;
        mov.b16 $7, h7;
        }
        """,
        constraints="=h,=h,=h,=h,=h,=h,=h,=h,r,h",
        args=[b, scales],
        dtype=(tl.bfloat16,) * 8,
        is_pure=True,
        pack=1,
    )
    return x1, x2, x3, x4, x5, x6, x7, x8


# FP4 (E2M1) -> bf16/fp16 SIMD dequant. Marlin bit trick
# bare = (shifted & 0x80008000) | ((shifted & 0x70007000) >> RIGHT_SHIFT), then x bias
# (2^126 / 2^14) restores the true value (subnormal 0.5 works for free). The per-32
# E8M0 scale is folded in: a BLOCK_SIZE_K=128 tile spans 4 groups, so outputs
# h0,h1 use s0; h2,h3 use s1; h4,h5 use s2; h6,h7 use s3.
@triton.jit
def _dequant_fp4_bf16(b, s0, s1, s2, s3):
    x1, x2, x3, x4, x5, x6, x7, x8 = tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b32  r0, r1, r2, r3, q0, q1, q2, q3, t, bias;
        .reg .b16  h0, h1, h2, h3, h4, h5, h6, h7, s;
        mov.u32 r0, $8;
        shr.u32 r1, r0, 4;
        shr.u32 r2, r0, 8;
        shr.u32 r3, r0, 12;
        and.b32 q0, r0, 983055;
        shl.b32 q0, q0, 12;
        and.b32 t, q0, 2147516416;
        and.b32 q0, q0, 1879076864;
        shr.b32 q0, q0, 6;
        or.b32 q0, q0, t;
        and.b32 q1, r1, 983055;
        shl.b32 q1, q1, 12;
        and.b32 t, q1, 2147516416;
        and.b32 q1, q1, 1879076864;
        shr.b32 q1, q1, 6;
        or.b32 q1, q1, t;
        and.b32 q2, r2, 983055;
        shl.b32 q2, q2, 12;
        and.b32 t, q2, 2147516416;
        and.b32 q2, q2, 1879076864;
        shr.b32 q2, q2, 6;
        or.b32 q2, q2, t;
        and.b32 q3, r3, 983055;
        shl.b32 q3, q3, 12;
        and.b32 t, q3, 2147516416;
        and.b32 q3, q3, 1879076864;
        shr.b32 q3, q3, 6;
        or.b32 q3, q3, t;
        mov.u32 bias, 2122350208;        // 0x7E807E80 = bf16x2 (2^126, 2^126)
        mul.rn.bf16x2 q0, q0, bias;
        mul.rn.bf16x2 q1, q1, bias;
        mul.rn.bf16x2 q2, q2, bias;
        mul.rn.bf16x2 q3, q3, bias;
        mov.b32 {h0, h1}, q0;
        mov.b32 {h2, h3}, q1;
        mov.b32 {h4, h5}, q2;
        mov.b32 {h6, h7}, q3;
        mov.b16 s, $9;
        mul.rn.bf16 h0, h0, s;
        mul.rn.bf16 h1, h1, s;
        mov.b16 s, $10;
        mul.rn.bf16 h2, h2, s;
        mul.rn.bf16 h3, h3, s;
        mov.b16 s, $11;
        mul.rn.bf16 h4, h4, s;
        mul.rn.bf16 h5, h5, s;
        mov.b16 s, $12;
        mul.rn.bf16 h6, h6, s;
        mul.rn.bf16 h7, h7, s;
        mov.b16 $0, h0;
        mov.b16 $1, h1;
        mov.b16 $2, h2;
        mov.b16 $3, h3;
        mov.b16 $4, h4;
        mov.b16 $5, h5;
        mov.b16 $6, h6;
        mov.b16 $7, h7;
        }
        """,
        constraints="=h,=h,=h,=h,=h,=h,=h,=h,r,h,h,h,h",
        args=[b, s0, s1, s2, s3],
        dtype=(tl.bfloat16,) * 8,
        is_pure=True,
        pack=1,
    )
    return x1, x2, x3, x4, x5, x6, x7, x8


@triton.jit
def _dequant_fp4_fp16(b, s0, s1, s2, s3):
    x1, x2, x3, x4, x5, x6, x7, x8 = tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b32  r0, r1, r2, r3, q0, q1, q2, q3, t, bias;
        .reg .b16  h0, h1, h2, h3, h4, h5, h6, h7, s;
        mov.u32 r0, $8;
        shr.u32 r1, r0, 4;
        shr.u32 r2, r0, 8;
        shr.u32 r3, r0, 12;
        and.b32 q0, r0, 983055;
        shl.b32 q0, q0, 12;
        and.b32 t, q0, 2147516416;
        and.b32 q0, q0, 1879076864;
        shr.b32 q0, q0, 3;
        or.b32 q0, q0, t;
        and.b32 q1, r1, 983055;
        shl.b32 q1, q1, 12;
        and.b32 t, q1, 2147516416;
        and.b32 q1, q1, 1879076864;
        shr.b32 q1, q1, 3;
        or.b32 q1, q1, t;
        and.b32 q2, r2, 983055;
        shl.b32 q2, q2, 12;
        and.b32 t, q2, 2147516416;
        and.b32 q2, q2, 1879076864;
        shr.b32 q2, q2, 3;
        or.b32 q2, q2, t;
        and.b32 q3, r3, 983055;
        shl.b32 q3, q3, 12;
        and.b32 t, q3, 2147516416;
        and.b32 q3, q3, 1879076864;
        shr.b32 q3, q3, 3;
        or.b32 q3, q3, t;
        mov.u32 bias, 1946186752;        // 0x74007400 = f16x2 (2^14, 2^14)
        mul.rn.f16x2 q0, q0, bias;
        mul.rn.f16x2 q1, q1, bias;
        mul.rn.f16x2 q2, q2, bias;
        mul.rn.f16x2 q3, q3, bias;
        mov.b32 {h0, h1}, q0;
        mov.b32 {h2, h3}, q1;
        mov.b32 {h4, h5}, q2;
        mov.b32 {h6, h7}, q3;
        mov.b16 s, $9;
        mul.f16 h0, h0, s;
        mul.f16 h1, h1, s;
        mov.b16 s, $10;
        mul.f16 h2, h2, s;
        mul.f16 h3, h3, s;
        mov.b16 s, $11;
        mul.f16 h4, h4, s;
        mul.f16 h5, h5, s;
        mov.b16 s, $12;
        mul.f16 h6, h6, s;
        mul.f16 h7, h7, s;
        mov.b16 $0, h0;
        mov.b16 $1, h1;
        mov.b16 $2, h2;
        mov.b16 $3, h3;
        mov.b16 $4, h4;
        mov.b16 $5, h5;
        mov.b16 $6, h6;
        mov.b16 $7, h7;
        }
        """,
        constraints="=h,=h,=h,=h,=h,=h,=h,=h,r,h,h,h,h",
        args=[b, s0, s1, s2, s3],
        dtype=(tl.float16,) * 8,
        is_pure=True,
        pack=1,
    )
    return x1, x2, x3, x4, x5, x6, x7, x8


@triton.jit
def _stack_along_dim0(a, b, X: tl.constexpr, Y: tl.constexpr):
    j = tl.join(a, b)  # (X, Y, 2)
    p = tl.permute(j, (2, 0, 1))  # (2, X, Y)
    return tl.reshape(p, (2 * X, Y))  # (2X, Y) block-concat


@triton.jit
def _stack_8(bs, K_PACK: tl.constexpr, N: tl.constexpr):
    s01 = _stack_along_dim0(bs[0], bs[1], K_PACK, N)  # (2*K_PACK, N)
    s23 = _stack_along_dim0(bs[2], bs[3], K_PACK, N)
    s45 = _stack_along_dim0(bs[4], bs[5], K_PACK, N)
    s67 = _stack_along_dim0(bs[6], bs[7], K_PACK, N)
    s0123 = _stack_along_dim0(s01, s23, 2 * K_PACK, N)  # (4*K_PACK, N)
    s4567 = _stack_along_dim0(s45, s67, 2 * K_PACK, N)
    return _stack_along_dim0(s0123, s4567, 4 * K_PACK, N)  # (8*K_PACK, N)


@triton.jit
def _write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    compute_type: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if SWAP_AB:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=compute_type)
        c_ptrs = c_ptr + stride_cm * offs_token[None, :] + stride_cn * offs_cn[:, None]
        c_mask = token_mask[None, :] & (offs_cn[:, None] < N)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_N": 64, "GROUP_SIZE_M": 1}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 1}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 4}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 256, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 256, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=2
        ),
    ],
    key=[
        "N",
        "K",
        "EM",
        "BLOCK_SIZE_M",
        "MUL_ROUTED_WEIGHT",
        "top_k",
    ],
)
@triton.jit
def _w4a16_moe_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsg,
    stride_bsn,
    BLOCK_SIZE_M: tl.constexpr,  # token tile (MMA M-dim, or N-dim if SWAP_AB)
    BLOCK_SIZE_N: tl.constexpr,  # weight tile (MMA N-dim, or M-dim if SWAP_AB)
    BLOCK_SIZE_K: tl.constexpr,  # logical-K tile (must match packing)
    GROUP_SIZE_M: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,  # = quant group_size (e.g. 128)
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    BLOCK_SIZE_K_PACK: tl.constexpr = BLOCK_SIZE_K // 8

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
            SWAP_AB,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_ak_pack = tl.arange(0, BLOCK_SIZE_K_PACK)
    offs_bk = tl.arange(0, BLOCK_SIZE_K_PACK)

    if SWAP_AB:
        a_base = a_ptr + (offs_token[None, :] // top_k * stride_am)
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_bn[:, None] * stride_bn
            + offs_bk[None, :] * stride_bk
        )
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        a_base = a_ptr + (offs_token[:, None] // top_k * stride_am)
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_bk[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    scale_base = b_scale_ptr + off_experts * stride_bse + offs_bn * stride_bsn

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        b_packed = tl.load(b_ptrs)
        scale_idx = k * BLOCK_SIZE_K // GROUP_SIZE_K
        scale = tl.load(scale_base + scale_idx * stride_bsg)
        scale_bc = scale[:, None] if SWAP_AB else scale[None, :]

        if compute_type == tl.float16:
            bs = _dequant_int4_fp16(b_packed, scale_bc)
        else:
            bs = _dequant_int4_bf16(b_packed, scale_bc)

        k_logical_base = k * BLOCK_SIZE_K
        for j in tl.static_range(8):
            k_off = k_logical_base + j * BLOCK_SIZE_K_PACK
            if SWAP_AB:
                a_j_ptrs = a_base + (k_off + offs_ak_pack[:, None]) * stride_ak
                a_j = tl.load(
                    a_j_ptrs, mask=token_mask[None, :], other=0.0
                )  # (K_PACK, M)
                accumulator = tl.dot(bs[j], a_j, acc=accumulator)  # (N, M)
            else:
                a_j_ptrs = a_base + (k_off + offs_ak_pack[None, :]) * stride_ak
                a_j = tl.load(
                    a_j_ptrs, mask=token_mask[:, None], other=0.0
                )  # (M, K_PACK)
                accumulator = tl.dot(a_j, bs[j], acc=accumulator)  # (M, N)

        b_ptrs += BLOCK_SIZE_K_PACK * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * (
            moe_weight[None, :] if SWAP_AB else moe_weight[:, None]
        )

    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if SWAP_AB:
        c_ptrs = c_ptr + stride_cm * offs_token[None, :] + stride_cn * offs_cn[:, None]
        c_mask = token_mask[None, :] & (offs_cn[:, None] < N)
    else:
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_N": 64, "GROUP_SIZE_M": 1}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 1}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 4}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 256, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 256, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=2
        ),
    ],
    key=[
        "N",
        "K",
        "EM",
        "BLOCK_SIZE_M",
        "APPLY_ROUTER_WEIGHT_BEFORE_SILU",
        "APPLY_ROUTER_WEIGHT_AFTER_SILU",
        "top_k",
    ],
)
@triton.jit
def _w4a16_moe_gemm_silu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsg,
    stride_bsn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    APPLY_ROUTER_WEIGHT_BEFORE_SILU: tl.constexpr,
    APPLY_ROUTER_WEIGHT_AFTER_SILU: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    BLOCK_SIZE_K_PACK: tl.constexpr = BLOCK_SIZE_K // 8

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
            SWAP_AB,
        )
        return

    offs_bn_gate = offs_cn % N
    offs_bn_up = offs_bn_gate + N
    offs_ak_pack = tl.arange(0, BLOCK_SIZE_K_PACK)
    offs_bk = tl.arange(0, BLOCK_SIZE_K_PACK)

    if SWAP_AB:
        a_base = a_ptr + (offs_token[None, :] // top_k * stride_am)
        b_ptrs_gate = (
            b_ptr
            + off_experts * stride_be
            + offs_bn_gate[:, None] * stride_bn
            + offs_bk[None, :] * stride_bk
        )
        b_ptrs_up = (
            b_ptr
            + off_experts * stride_be
            + offs_bn_up[:, None] * stride_bn
            + offs_bk[None, :] * stride_bk
        )
        acc_gate = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        a_base = a_ptr + (offs_token[:, None] // top_k * stride_am)
        b_ptrs_gate = (
            b_ptr
            + off_experts * stride_be
            + offs_bk[:, None] * stride_bk
            + offs_bn_gate[None, :] * stride_bn
        )
        b_ptrs_up = (
            b_ptr
            + off_experts * stride_be
            + offs_bk[:, None] * stride_bk
            + offs_bn_up[None, :] * stride_bn
        )
        acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    scale_base_gate = b_scale_ptr + off_experts * stride_bse + offs_bn_gate * stride_bsn
    scale_base_up = b_scale_ptr + off_experts * stride_bse + offs_bn_up * stride_bsn

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        scale_idx = k * BLOCK_SIZE_K // GROUP_SIZE_K
        scale_gate = tl.load(scale_base_gate + scale_idx * stride_bsg)
        scale_up = tl.load(scale_base_up + scale_idx * stride_bsg)
        scale_gate_bc = scale_gate[:, None] if SWAP_AB else scale_gate[None, :]
        scale_up_bc = scale_up[:, None] if SWAP_AB else scale_up[None, :]

        b_packed_gate = tl.load(b_ptrs_gate)
        b_packed_up = tl.load(b_ptrs_up)
        if compute_type == tl.float16:
            bs_gate = _dequant_int4_fp16(b_packed_gate, scale_gate_bc)
            bs_up = _dequant_int4_fp16(b_packed_up, scale_up_bc)
        else:
            bs_gate = _dequant_int4_bf16(b_packed_gate, scale_gate_bc)
            bs_up = _dequant_int4_bf16(b_packed_up, scale_up_bc)

        k_logical_base = k * BLOCK_SIZE_K
        for j in tl.static_range(8):
            k_off = k_logical_base + j * BLOCK_SIZE_K_PACK
            if SWAP_AB:
                a_j_ptrs = a_base + (k_off + offs_ak_pack[:, None]) * stride_ak
                a_j = tl.load(a_j_ptrs, mask=token_mask[None, :], other=0.0)
                acc_gate = tl.dot(bs_gate[j], a_j, acc=acc_gate)
                acc_up = tl.dot(bs_up[j], a_j, acc=acc_up)
            else:
                a_j_ptrs = a_base + (k_off + offs_ak_pack[None, :]) * stride_ak
                a_j = tl.load(a_j_ptrs, mask=token_mask[:, None], other=0.0)
                acc_gate = tl.dot(a_j, bs_gate[j], acc=acc_gate)
                acc_up = tl.dot(a_j, bs_up[j], acc=acc_up)

        b_ptrs_gate += BLOCK_SIZE_K_PACK * stride_bk
        b_ptrs_up += BLOCK_SIZE_K_PACK * stride_bk

    if APPLY_ROUTER_WEIGHT_BEFORE_SILU:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        routed_weight = moe_weight[None, :] if SWAP_AB else moe_weight[:, None]
        acc_gate = acc_gate * routed_weight
        acc_up = acc_up * routed_weight

    accumulator = (acc_gate * tl.sigmoid(acc_gate)) * acc_up
    if APPLY_ROUTER_WEIGHT_AFTER_SILU:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * (
            moe_weight[None, :] if SWAP_AB else moe_weight[:, None]
        )

    accumulator = accumulator.to(compute_type)
    if SWAP_AB:
        c_ptrs = c_ptr + stride_cm * offs_token[None, :] + stride_cn * offs_cn[:, None]
        c_mask = token_mask[None, :] & (offs_cn[:, None] < N)
    else:
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _invoke_w4a16_moe_gemm(
    A: torch.Tensor,  # (M, K) for GEMM1, (M*top_k, K) for GEMM2
    B: torch.Tensor,  # (E, K//8, N) int32
    C: torch.Tensor,  # (M, top_k, N) or (M*top_k, N) view
    B_scale: torch.Tensor,  # (E, K/gs, N) fp16/bf16
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    mul_routed_weight: bool,
    top_k: int,
    block_m: int,
    block_size_k: int,
    group_size: int,
    compute_type,  # tl.float16 or tl.bfloat16
    swap_ab: bool,
):
    M_a = A.size(0)
    K = A.size(1)
    N = B.size(2)
    EM = sorted_token_ids.size(0)
    if M_a < block_m:
        EM = min(EM, M_a * top_k * block_m)

    if C.ndim == 3:
        stride_cm = C.stride(1)
        stride_cn = C.stride(2)
    else:
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    _w4a16_moe_gemm_kernel[grid](
        A,
        B,
        C,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        A.size(0) * top_k,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        stride_cm,
        stride_cn,
        B_scale.stride(0),
        B_scale.stride(1),
        B_scale.stride(2),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_K=group_size,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        SWAP_AB=swap_ab,
    )


def _invoke_w4a16_moe_gemm_silu(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor,
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    router_weight: _RouterWeightPlacement,
    top_k: int,
    block_m: int,
    block_size_k: int,
    group_size: int,
    compute_type,
    swap_ab: bool,
):
    M_a = A.size(0)
    K = A.size(1)
    N = C.size(-1)
    EM = sorted_token_ids.size(0)
    if M_a < block_m:
        EM = min(EM, M_a * top_k * block_m)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    _w4a16_moe_gemm_silu_kernel[grid](
        A,
        B,
        C,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        A.size(0) * top_k,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(0),
        C.stride(1),
        B_scale.stride(0),
        B_scale.stride(1),
        B_scale.stride(2),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_K=group_size,
        APPLY_ROUTER_WEIGHT_BEFORE_SILU=(
            router_weight == _RouterWeightPlacement.before_silu
        ),
        APPLY_ROUTER_WEIGHT_AFTER_SILU=(
            router_weight == _RouterWeightPlacement.after_silu
        ),
        top_k=top_k,
        compute_type=compute_type,
        SWAP_AB=swap_ab,
    )


def fused_moe_w4a16_gptq(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str = "silu",
    group_size: int = 128,
    apply_router_weight_on_input: bool = False,
    inplace: bool = False,
    swap_ab: bool = True,
) -> torch.Tensor:
    assert activation == "silu"
    assert hidden_states.dtype in (torch.float16, torch.bfloat16)
    assert hidden_states.is_contiguous()
    assert w1.dtype == torch.uint8 and w2.dtype == torch.uint8
    assert w1.stride(-1) == 1 and w2.stride(-1) == 1

    M = hidden_states.size(0)
    K = hidden_states.size(1)
    E = w1.size(0)
    intermediate_size = w1.size(1) // 2
    top_k_num = topk_ids.size(1)

    assert w1.shape == (E, 2 * intermediate_size, K // 2)
    assert w2.shape == (E, K, intermediate_size // 2)
    assert K % group_size == 0
    assert intermediate_size % group_size == 0
    assert w1_scale.shape == (E, 2 * intermediate_size, K // group_size)
    assert w2_scale.shape == (E, K, intermediate_size // group_size)
    assert w1_scale.dtype == hidden_states.dtype
    assert w2_scale.dtype == hidden_states.dtype
    assert topk_weights.shape == topk_ids.shape

    block_size_k = group_size
    # Compute_type for the kernel.
    if hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    else:
        compute_type = tl.bfloat16

    w1_packed, w2_packed, w1_scale_packed, w2_scale_packed = w4a16_pack(
        w1,
        w2,
        w1_scale,
        w2_scale,
        block_size_k=block_size_k,
        cached=True,
    )

    policy = _select_w4a16_kernel_policy(
        hidden_states.device,
        M,
        E,
        top_k_num,
        swap_ab,
        apply_router_weight_on_input,
    )
    block_m = policy.block_m
    use_fused_gemm1_silu = policy.use_fused_gemm1_silu
    router_weight_placement = _router_weight_placement(
        apply_router_weight_on_input,
        policy.move_router_weight_before_gemm2,
    )
    mul_routed_weight_in_gemm2 = router_weight_placement == _RouterWeightPlacement.none

    cache13_size = M * top_k_num * K
    if not use_fused_gemm1_silu:
        cache13_size = max(cache13_size, M * top_k_num * 2 * intermediate_size)
    cache13 = torch.empty(
        cache13_size, device=hidden_states.device, dtype=hidden_states.dtype
    )
    intermediate_cache1 = None
    if not use_fused_gemm1_silu:
        intermediate_cache1 = cache13[: M * top_k_num * 2 * intermediate_size].view(
            M * top_k_num, 2 * intermediate_size
        )
    intermediate_cache3 = cache13[: M * top_k_num * K].view(M, top_k_num, K)
    intermediate_cache2 = torch.empty(
        (M * top_k_num, intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids=topk_ids,
        block_size=block_m,
        num_experts=E,
        expert_map=None,
    )

    if use_fused_gemm1_silu:
        _invoke_w4a16_moe_gemm_silu(
            A=hidden_states,
            B=w1_packed,
            C=intermediate_cache2,
            B_scale=w1_scale_packed,
            topk_weights=(
                topk_weights
                if router_weight_placement != _RouterWeightPlacement.none
                else None
            ),
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            router_weight=router_weight_placement,
            top_k=top_k_num,
            block_m=block_m,
            block_size_k=block_size_k,
            group_size=group_size,
            compute_type=compute_type,
            swap_ab=swap_ab,
        )
    else:
        assert intermediate_cache1 is not None
        _invoke_w4a16_moe_gemm(
            A=hidden_states,
            B=w1_packed,
            C=intermediate_cache1,
            B_scale=w1_scale_packed,
            topk_weights=topk_weights if apply_router_weight_on_input else None,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=apply_router_weight_on_input,
            top_k=top_k_num,
            block_m=block_m,
            block_size_k=block_size_k,
            group_size=group_size,
            compute_type=compute_type,
            swap_ab=swap_ab,
        )
        gate = intermediate_cache1[:, :intermediate_size]
        up = intermediate_cache1[:, intermediate_size:]
        silu_and_mul_out(gate, up, intermediate_cache2)

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    _invoke_w4a16_moe_gemm(
        A=intermediate_cache2,
        B=w2_packed,
        C=intermediate_cache3,
        B_scale=w2_scale_packed,
        topk_weights=topk_weights if mul_routed_weight_in_gemm2 else None,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=mul_routed_weight_in_gemm2,
        top_k=1,
        block_m=block_m,
        block_size_k=block_size_k,
        group_size=group_size,
        compute_type=compute_type,
        swap_ab=swap_ab,
    )

    moe_sum(intermediate_cache3, out_hidden_states)

    return out_hidden_states


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("fused_marlin_moe_mxfp4"),
    key=["N", "K", "EM_BUCKET", "BLOCK_SIZE_M", "SWAP_AB"],
    strategy=["align32", "align32", "align32", "align32", "default"],
    flagtune_op_name="fused_marlin_moe_mxfp4",
    flagtune_expand_op_name="fused_marlin_moe_mxfp4",
)
@triton.jit
def _mxfp4_moe_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsg,
    stride_bsn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    SWAP_AB: tl.constexpr,
    EM_BUCKET: tl.constexpr,
):
    BLOCK_SIZE_K_PACK: tl.constexpr = BLOCK_SIZE_K // 8

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
            SWAP_AB,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_ak_pack = tl.arange(0, BLOCK_SIZE_K_PACK)
    offs_bk = tl.arange(0, BLOCK_SIZE_K_PACK)

    if SWAP_AB:
        a_base = a_ptr + (offs_token[None, :] // top_k * stride_am)
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_bn[:, None] * stride_bn
            + offs_bk[None, :] * stride_bk
        )
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        a_base = a_ptr + (offs_token[:, None] // top_k * stride_am)
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_bk[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    scale_base = b_scale_ptr + off_experts * stride_bse + offs_bn * stride_bsn

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        b_packed = tl.load(b_ptrs)

        # One BLOCK_SIZE_K tile spans BLOCK_SIZE_K/GROUP_SIZE_K (=4) E8M0 groups.
        g0 = k * BLOCK_SIZE_K // GROUP_SIZE_K
        sc0 = tl.load(scale_base + (g0 + 0) * stride_bsg)
        sc1 = tl.load(scale_base + (g0 + 1) * stride_bsg)
        sc2 = tl.load(scale_base + (g0 + 2) * stride_bsg)
        sc3 = tl.load(scale_base + (g0 + 3) * stride_bsg)
        if SWAP_AB:
            s0, s1, s2, s3 = sc0[:, None], sc1[:, None], sc2[:, None], sc3[:, None]
        else:
            s0, s1, s2, s3 = sc0[None, :], sc1[None, :], sc2[None, :], sc3[None, :]

        if compute_type == tl.float16:
            bs = _dequant_fp4_fp16(b_packed, s0, s1, s2, s3)
        else:
            bs = _dequant_fp4_bf16(b_packed, s0, s1, s2, s3)

        k_logical_base = k * BLOCK_SIZE_K
        for j in tl.static_range(8):
            k_off = k_logical_base + j * BLOCK_SIZE_K_PACK
            if SWAP_AB:
                a_j_ptrs = a_base + (k_off + offs_ak_pack[:, None]) * stride_ak
                a_j = tl.load(a_j_ptrs, mask=token_mask[None, :], other=0.0)
                accumulator = tl.dot(bs[j], a_j, acc=accumulator)
            else:
                a_j_ptrs = a_base + (k_off + offs_ak_pack[None, :]) * stride_ak
                a_j = tl.load(a_j_ptrs, mask=token_mask[:, None], other=0.0)
                accumulator = tl.dot(a_j, bs[j], acc=accumulator)

        b_ptrs += BLOCK_SIZE_K_PACK * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * (
            moe_weight[None, :] if SWAP_AB else moe_weight[:, None]
        )

    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if SWAP_AB:
        c_ptrs = c_ptr + stride_cm * offs_token[None, :] + stride_cn * offs_cn[:, None]
        c_mask = token_mask[None, :] & (offs_cn[:, None] < N)
    else:
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("fused_marlin_moe_mxfp4_gemm_silu"),
    key=["N", "K", "BLOCK_SIZE_M", "SWAP_AB"],
    strategy=["align32", "align32", "align32", "default"],
    flagtune_op_name="fused_marlin_moe_mxfp4_gemm_silu",
    flagtune_expand_op_name="fused_marlin_moe_mxfp4_gemm_silu",
)
@triton.jit
def _mxfp4_moe_gemm_silu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsg,
    stride_bsn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    APPLY_ROUTER_WEIGHT_BEFORE_SILU: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    BLOCK_SIZE_K_PACK: tl.constexpr = BLOCK_SIZE_K // 8

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
            SWAP_AB,
        )
        return

    offs_bn_gate = offs_cn % N
    offs_bn_up = offs_bn_gate + N
    offs_ak_pack = tl.arange(0, BLOCK_SIZE_K_PACK)
    offs_bk = tl.arange(0, BLOCK_SIZE_K_PACK)

    if SWAP_AB:
        a_base = a_ptr + (offs_token[None, :] // top_k * stride_am)
        b_ptrs_gate = (
            b_ptr
            + off_experts * stride_be
            + offs_bn_gate[:, None] * stride_bn
            + offs_bk[None, :] * stride_bk
        )
        b_ptrs_up = (
            b_ptr
            + off_experts * stride_be
            + offs_bn_up[:, None] * stride_bn
            + offs_bk[None, :] * stride_bk
        )
        acc_gate = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        a_base = a_ptr + (offs_token[:, None] // top_k * stride_am)
        b_ptrs_gate = (
            b_ptr
            + off_experts * stride_be
            + offs_bk[:, None] * stride_bk
            + offs_bn_gate[None, :] * stride_bn
        )
        b_ptrs_up = (
            b_ptr
            + off_experts * stride_be
            + offs_bk[:, None] * stride_bk
            + offs_bn_up[None, :] * stride_bn
        )
        acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    scale_base_gate = b_scale_ptr + off_experts * stride_bse + offs_bn_gate * stride_bsn
    scale_base_up = b_scale_ptr + off_experts * stride_bse + offs_bn_up * stride_bsn

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        g0 = k * BLOCK_SIZE_K // GROUP_SIZE_K
        sc_gate0 = tl.load(scale_base_gate + (g0 + 0) * stride_bsg)
        sc_gate1 = tl.load(scale_base_gate + (g0 + 1) * stride_bsg)
        sc_gate2 = tl.load(scale_base_gate + (g0 + 2) * stride_bsg)
        sc_gate3 = tl.load(scale_base_gate + (g0 + 3) * stride_bsg)
        sc_up0 = tl.load(scale_base_up + (g0 + 0) * stride_bsg)
        sc_up1 = tl.load(scale_base_up + (g0 + 1) * stride_bsg)
        sc_up2 = tl.load(scale_base_up + (g0 + 2) * stride_bsg)
        sc_up3 = tl.load(scale_base_up + (g0 + 3) * stride_bsg)
        if SWAP_AB:
            sg0, sg1, sg2, sg3 = (
                sc_gate0[:, None],
                sc_gate1[:, None],
                sc_gate2[:, None],
                sc_gate3[:, None],
            )
            su0, su1, su2, su3 = (
                sc_up0[:, None],
                sc_up1[:, None],
                sc_up2[:, None],
                sc_up3[:, None],
            )
        else:
            sg0, sg1, sg2, sg3 = (
                sc_gate0[None, :],
                sc_gate1[None, :],
                sc_gate2[None, :],
                sc_gate3[None, :],
            )
            su0, su1, su2, su3 = (
                sc_up0[None, :],
                sc_up1[None, :],
                sc_up2[None, :],
                sc_up3[None, :],
            )

        b_packed_gate = tl.load(b_ptrs_gate)
        b_packed_up = tl.load(b_ptrs_up)
        if compute_type == tl.float16:
            bs_gate = _dequant_fp4_fp16(b_packed_gate, sg0, sg1, sg2, sg3)
            bs_up = _dequant_fp4_fp16(b_packed_up, su0, su1, su2, su3)
        else:
            bs_gate = _dequant_fp4_bf16(b_packed_gate, sg0, sg1, sg2, sg3)
            bs_up = _dequant_fp4_bf16(b_packed_up, su0, su1, su2, su3)

        k_logical_base = k * BLOCK_SIZE_K
        for j in tl.static_range(8):
            k_off = k_logical_base + j * BLOCK_SIZE_K_PACK
            if SWAP_AB:
                a_j_ptrs = a_base + (k_off + offs_ak_pack[:, None]) * stride_ak
                a_j = tl.load(a_j_ptrs, mask=token_mask[None, :], other=0.0)
                acc_gate = tl.dot(bs_gate[j], a_j, acc=acc_gate)
                acc_up = tl.dot(bs_up[j], a_j, acc=acc_up)
            else:
                a_j_ptrs = a_base + (k_off + offs_ak_pack[None, :]) * stride_ak
                a_j = tl.load(a_j_ptrs, mask=token_mask[:, None], other=0.0)
                acc_gate = tl.dot(a_j, bs_gate[j], acc=acc_gate)
                acc_up = tl.dot(a_j, bs_up[j], acc=acc_up)

        b_ptrs_gate += BLOCK_SIZE_K_PACK * stride_bk
        b_ptrs_up += BLOCK_SIZE_K_PACK * stride_bk

    if APPLY_ROUTER_WEIGHT_BEFORE_SILU:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        routed_weight = moe_weight[None, :] if SWAP_AB else moe_weight[:, None]
        acc_gate = acc_gate * routed_weight
        acc_up = acc_up * routed_weight

    accumulator = (acc_gate * tl.sigmoid(acc_gate)) * acc_up

    accumulator = accumulator.to(compute_type)
    if SWAP_AB:
        c_ptrs = c_ptr + stride_cm * offs_token[None, :] + stride_cn * offs_cn[:, None]
        c_mask = token_mask[None, :] & (offs_cn[:, None] < N)
    else:
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _invoke_mxfp4_moe_gemm(
    A,
    B,
    C,
    B_scale,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    *,
    mul_routed_weight: bool,
    top_k: int,
    block_m: int,
    block_size_k: int,
    group_size: int,
    compute_type,
    swap_ab: bool = False,
):
    M_a = A.size(0)
    K = A.size(1)
    N = B.size(2)
    EM = sorted_token_ids.size(0)
    if M_a < block_m:
        EM = min(EM, M_a * top_k * block_m)
    em_bucket = 0 if M_a == 1 else EM

    if C.ndim == 3:
        stride_cm = C.stride(1)
        stride_cn = C.stride(2)
    else:
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    _mxfp4_moe_gemm_kernel[grid](
        A,
        B,
        C,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        A.size(0) * top_k,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        stride_cm,
        stride_cn,
        B_scale.stride(0),
        B_scale.stride(1),
        B_scale.stride(2),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_K=group_size,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        SWAP_AB=swap_ab,
        EM_BUCKET=em_bucket,
    )


def _invoke_mxfp4_moe_gemm_silu(
    A,
    B,
    C,
    B_scale,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    *,
    apply_router_weight_before_silu: bool,
    top_k: int,
    block_m: int,
    block_size_k: int,
    group_size: int,
    compute_type,
    swap_ab: bool = False,
):
    M_a = A.size(0)
    K = A.size(1)
    N = C.size(-1)
    EM = sorted_token_ids.size(0)
    if M_a < block_m:
        EM = min(EM, M_a * top_k * block_m)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    _mxfp4_moe_gemm_silu_kernel[grid](
        A,
        B,
        C,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        A.size(0) * top_k,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(0),
        C.stride(1),
        B_scale.stride(0),
        B_scale.stride(1),
        B_scale.stride(2),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_K=group_size,
        APPLY_ROUTER_WEIGHT_BEFORE_SILU=apply_router_weight_before_silu,
        top_k=top_k,
        compute_type=compute_type,
        SWAP_AB=swap_ab,
    )


def fused_moe_mxfp4(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str = "silu",
    group_size: int = MXFP4_GROUP_SIZE,
    apply_router_weight_on_input: bool = False,
    inplace: bool = False,
    swap_ab: bool = True,
) -> torch.Tensor:
    """MXFP4 (W4A16) fused MoE. Weights: w1 (E, 2N, K//2) / w2 (E, K, N//2) uint8,
    two FP4 (E2M1) per byte; scales E8M0 (float8_e8m0fnu), per-32 group."""
    assert activation == "silu"
    assert hidden_states.dtype in (torch.float16, torch.bfloat16)
    assert hidden_states.is_contiguous()
    assert w1.dtype == torch.uint8 and w2.dtype == torch.uint8
    assert w1.stride(-1) == 1 and w2.stride(-1) == 1

    M = hidden_states.size(0)
    K = hidden_states.size(1)
    E = w1.size(0)
    intermediate_size = w1.size(1) // 2
    top_k_num = topk_ids.size(1)

    # BLOCK_SIZE_K=128 keeps tl.dot's K>=16 (K_PACK=16); a tile spans 4 E8M0 groups.
    block_size_k = 128
    assert w1.shape == (E, 2 * intermediate_size, K // 2)
    assert w2.shape == (E, K, intermediate_size // 2)
    assert K % block_size_k == 0
    assert intermediate_size % block_size_k == 0
    assert block_size_k % group_size == 0
    assert w1_scale.shape == (E, 2 * intermediate_size, K // group_size)
    assert w2_scale.shape == (E, K, intermediate_size // group_size)
    assert topk_weights.shape == topk_ids.shape

    compute_type = tl.float16 if hidden_states.dtype == torch.float16 else tl.bfloat16

    w1_packed, w2_packed, w1_scale_packed, w2_scale_packed = mxfp4_pack(
        w1,
        w2,
        w1_scale,
        w2_scale,
        hidden_states.dtype,
        block_size_k=block_size_k,
        cached=True,
    )

    policy = _select_mxfp4_kernel_policy(
        hidden_states.device,
        M,
        E,
        top_k_num,
        swap_ab,
    )
    block_m = policy.block_m
    use_fused_gemm1_silu = policy.use_fused_gemm1_silu

    cache13_size = M * top_k_num * K
    if not use_fused_gemm1_silu:
        cache13_size = max(cache13_size, M * top_k_num * 2 * intermediate_size)
    cache13 = torch.empty(
        cache13_size, device=hidden_states.device, dtype=hidden_states.dtype
    )
    intermediate_cache1 = None
    if not use_fused_gemm1_silu:
        intermediate_cache1 = cache13[: M * top_k_num * 2 * intermediate_size].view(
            M * top_k_num, 2 * intermediate_size
        )
    intermediate_cache3 = cache13[: M * top_k_num * K].view(M, top_k_num, K)
    intermediate_cache2 = torch.empty(
        (M * top_k_num, intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = _align_mxfp4_tokens(
        topk_ids,
        E,
        block_m,
        policy.align_mode,
    )

    if use_fused_gemm1_silu:
        _invoke_mxfp4_moe_gemm_silu(
            A=hidden_states,
            B=w1_packed,
            C=intermediate_cache2,
            B_scale=w1_scale_packed,
            topk_weights=topk_weights if apply_router_weight_on_input else None,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            apply_router_weight_before_silu=apply_router_weight_on_input,
            top_k=top_k_num,
            block_m=block_m,
            block_size_k=block_size_k,
            group_size=group_size,
            compute_type=compute_type,
            swap_ab=swap_ab,
        )
    else:
        assert intermediate_cache1 is not None
        _invoke_mxfp4_moe_gemm(
            A=hidden_states,
            B=w1_packed,
            C=intermediate_cache1,
            B_scale=w1_scale_packed,
            topk_weights=topk_weights if apply_router_weight_on_input else None,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=apply_router_weight_on_input,
            top_k=top_k_num,
            block_m=block_m,
            block_size_k=block_size_k,
            group_size=group_size,
            compute_type=compute_type,
            swap_ab=swap_ab,
        )

        gate = intermediate_cache1[:, :intermediate_size]
        up = intermediate_cache1[:, intermediate_size:]
        silu_and_mul_out(gate, up, intermediate_cache2)

    _invoke_mxfp4_moe_gemm(
        A=intermediate_cache2,
        B=w2_packed,
        C=intermediate_cache3,
        B_scale=w2_scale_packed,
        topk_weights=topk_weights if not apply_router_weight_on_input else None,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=not apply_router_weight_on_input,
        top_k=1,
        block_m=block_m,
        block_size_k=block_size_k,
        group_size=group_size,
        compute_type=compute_type,
        swap_ab=swap_ab,
    )

    out_hidden_states = hidden_states if inplace else torch.empty_like(hidden_states)
    moe_sum(intermediate_cache3, out_hidden_states)
    return out_hidden_states


# ----------------------------------------------------------------------------
# Phase-2 impl: copy of fused_experts_impl but with the dequant shortcut
# removed so the wna16 Triton kernel is actually invoked for W4A16/W8A16.
# ----------------------------------------------------------------------------
def _fused_marlin_moe_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Like fused_experts_impl, but:
      - drops all paths irrelevant to W4A16/W8A16 (no fp8, int8_w8a8, mxfp).
      - REMOVES the `w = w.to(fp16) * scale.unsqueeze(-1)` dequant shortcut.
      - forwards block_shape so the wna16 kernel uses the right group_size.
    """
    assert (
        activation == "silu"
    ), f"Only 'silu' activation is supported, got {activation}"
    assert (
        use_int4_w4a16 or use_int8_w8a16
    ), "_fused_marlin_moe_impl expects a quantized path"

    activation_enum = MoEActivation.from_str(activation)

    # Packed-aware shape check.
    # W4A16 (pack_factor=2): w1.size(2) == K // 2
    # W8A16 (pack_factor=1): w1.size(2) == K
    expected_packed_k = (
        hidden_states.size(1) // 2 if use_int4_w4a16 else hidden_states.size(1)
    )
    assert w1.size(2) == expected_packed_k, (
        f"w1 packed K mismatch: hidden_size={hidden_states.size(1)}, "
        f"use_int4_w4a16={use_int4_w4a16}, expected w1.size(2)={expected_packed_k}, "
        f"got {w1.size(2)}"
    )

    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens = hidden_states.size(0)
    E, N, _ = w1.size()
    K = w2.size(1)
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.size(1)

    CHUNK_SIZE: int = 16 * 1024
    M = min(num_tokens, CHUNK_SIZE)

    config_dtype = _get_config_dtype_str(
        use_fp8_w8a8=False,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        ocp_mx_scheme=None,
        dtype=hidden_states.dtype,
    )
    quant_dtype = _get_config_quant_dtype(
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        ocp_mx_scheme=None,
    )

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.size(),
        w2.size(),
        top_k_num,
        config_dtype,
        block_shape=block_shape,
        E=E,
    )
    config = get_config_func(M)
    config["SPLIT_K"] = 1

    # cache1 and cache3 share memory (non-overlapping lifetime)
    cache13 = torch.empty(
        M * top_k_num * max(N, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache13[: M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[: M * top_k_num * K].view(M, top_k_num, K)

    activation_out_dim = MoEActivation.adjust_N_for_activation(N, activation_enum)
    intermediate_cache2 = torch.empty(
        (M * top_k_num, activation_out_dim),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    out_hidden_states = hidden_states if inplace else torch.empty_like(hidden_states)

    # ★ Phase-2 KEY DIFFERENCE: the W4A16/W8A16 dequant shortcut that lived
    # here in `fused_experts_impl` is intentionally REMOVED. The wna16
    # Triton kernel will consume INT4 weights + scale directly.

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.size()

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[
                : tokens_in_chunk * topk_ids.size(1)
            ]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)
            config["SPLIT_K"] = 1

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        # Activation quantization is a no-op for W4A16/W8A16 (no input quant).
        qcurr_hidden_states, a1q_scale = moe_kernel_quantize_input(
            A=curr_hidden_states,
            A_scale=None,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape,
            ocp_mx_scheme=None,
        )

        # Use the routed-path (skip the SPARSITY_FACTOR shortcut, which is
        # explicitly disabled for quantized + block_shape configs anyway).
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            curr_topk_ids,
            config["BLOCK_SIZE_M"],
            global_num_experts,
            expert_map,
        )

        # ----- GEMM 1: hidden @ w1  (fused dequant on B inside the kernel) -----
        dispatch_fused_moe_kernel(
            qcurr_hidden_states,
            w1,
            intermediate_cache1,
            a1q_scale,
            w1_scale,
            w1_zp,
            curr_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            top_k_num,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            B_bias=w1_bias,
        )

        # ----- Activation: SwiGLU = silu(gate) * up -----
        apply_moe_activation(
            activation_enum, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            A=intermediate_cache2,
            A_scale=None,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape,
            ocp_mx_scheme=None,
        )

        if expert_map is not None:
            intermediate_cache3.zero_()

        # ----- GEMM 2: act @ w2  (fused dequant on B inside the kernel) -----
        dispatch_fused_moe_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            a2q_scale,
            w2_scale,
            w2_zp,
            curr_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            B_bias=w2_bias,
        )

        # ----- Reduce: sum topk-weighted expert outputs back per token -----
        moe_sum(
            intermediate_cache3.view(*intermediate_cache3.size()),
            out_hidden_states[begin_chunk_idx:end_chunk_idx],
        )

    return out_hidden_states


# ----------------------------------------------------------------------------
# Public entry point: vLLM-aligned wrapper.
# ----------------------------------------------------------------------------
def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: Optional[torch.Tensor],
    bias2: Optional[torch.Tensor],
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: Any = None,
    activation_func: Optional[Callable] = None,
    moe_sum: Optional[Callable] = None,
    expert_map: Optional[torch.Tensor] = None,
    input_global_scale1: Optional[torch.Tensor] = None,
    input_global_scale2: Optional[torch.Tensor] = None,
    global_scale1: Optional[torch.Tensor] = None,
    global_scale2: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    intermediate_cache13: Optional[torch.Tensor] = None,
    intermediate_cache2: Optional[torch.Tensor] = None,
    is_k_full: bool = True,
    output: Optional[torch.Tensor] = None,
    input_dtype: Optional[torch.dtype] = None,
    inplace: bool = False,
    clamp_limit: Optional[float] = None,
    group_size: int = 128,
) -> torch.Tensor:
    """Phase-2 entry point: dispatch to local wna16-using impl."""
    # ---- MVP guardrails --------------------------------------------------
    if quant_type_id not in _SUPPORTED_QUANT_TYPES:
        raise NotImplementedError(
            f"MVP supports quant_type_id in {_SUPPORTED_QUANT_TYPES}, "
            f"got {quant_type_id}"
        )
    if g_idx1 is not None or g_idx2 is not None:
        raise NotImplementedError("act_order (g_idx) not yet supported in MVP")
    if sort_indices1 is not None or sort_indices2 is not None:
        raise NotImplementedError("act_order (sort_indices) not yet supported in MVP")
    if input_dtype is not None:
        raise NotImplementedError("FP8 / INT8 input quantization not supported")
    if clamp_limit is not None:
        raise NotImplementedError("clamp_limit (GLM-4 swiglu) not supported")
    if input_global_scale1 is not None or input_global_scale2 is not None:
        raise NotImplementedError("input_global_scale not supported in MVP")
    if global_scale1 is not None or global_scale2 is not None:
        raise NotImplementedError("global_scale not supported in MVP")

    use_int4_w4a16 = quant_type_id in _QUANT_TYPE_INT4
    use_int8_w8a16 = quant_type_id in _QUANT_TYPE_INT8
    use_fp4_w4a16 = quant_type_id in _QUANT_TYPE_FP4

    activation_str = "silu"
    if activation is not None:
        for attr in ("value", "name"):
            v = getattr(activation, attr, None)
            if isinstance(v, str):
                activation_str = v.lower()
                break
        if isinstance(activation, str):
            activation_str = activation.lower()
    if activation_str != "silu":
        raise NotImplementedError(
            f"MVP only supports SiLU/SwiGLU activation, got {activation_str}"
        )

    if inplace and output is not None:
        raise ValueError("Cannot pass both inplace=True and output")

    if (
        # The magic-trick kernel's bf16 dequant uses sub.bf16x2/mul.bf16 PTX,
        # which require sm_90+; on pre-Hopper fall back to the generic wna16 kernel.
        _is_hopper()
        and use_int4_w4a16
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
        and w1.dtype == torch.uint8
        and w2.dtype == torch.uint8
        and bias1 is None
        and bias2 is None
        and w1_zeros is None
        and w2_zeros is None
        and expert_map is None
        and (global_num_experts == -1 or global_num_experts == w1.size(0))
        and group_size >= 128
        and w1_scale.dtype == hidden_states.dtype
        and w2_scale.dtype == hidden_states.dtype
    ):
        result = fused_moe_w4a16_gptq(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation_str,
            group_size=group_size,
            apply_router_weight_on_input=apply_router_weight_on_input,
            inplace=inplace,
        )
        if output is not None:
            output.copy_(result)
            return output
        return result

    # MXFP4 fast path: FP4 (E2M1) weights + per-32 E8M0 scale.
    if use_fp4_w4a16:
        if not (
            _is_hopper()
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
            and w1.dtype == torch.uint8
            and w2.dtype == torch.uint8
            and bias1 is None
            and bias2 is None
            and w1_zeros is None
            and w2_zeros is None
            and expert_map is None
            and (global_num_experts == -1 or global_num_experts == w1.size(0))
        ):
            raise NotImplementedError(
                "MXFP4 fast path requires Hopper, bf16/fp16 activations, uint8 "
                "packed weights, no bias/zeros/expert_map."
            )
        result = fused_moe_mxfp4(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation_str,
            group_size=group_size,
            apply_router_weight_on_input=apply_router_weight_on_input,
            inplace=inplace,
        )
        if output is not None:
            output.copy_(result)
            return output
        return result

    result = _fused_marlin_moe_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=inplace,
        activation=activation_str,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_int4_w4a16=use_int4_w4a16,
        use_int8_w8a16=use_int8_w8a16,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zeros,
        w2_zp=w2_zeros,
        w1_bias=bias1,
        w2_bias=bias2,
        # Critical for Phase 2: block_shape=[0, group_size] makes the
        # wna16 Triton kernel use the per-group scales correctly.
        block_shape=[0, group_size],
    )

    if output is not None:
        output.copy_(result)
        return output
    return result


__all__ = [
    "fused_marlin_moe",
    "QUANT_TYPE_UINT4B8",
    "QUANT_TYPE_UINT8B128",
    "QUANT_TYPE_FP4_E2M1",
]
