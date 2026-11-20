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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


def _check_supported_scales(scale: torch.Tensor, name: str) -> bool:
    if scale.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if scale.dtype == torch.uint8:
        return True
    raise TypeError(f"{name} must be a floating scale tensor or uint8 UE8M0 scales")


@triton.jit
def _ue8m0_to_f32(x):
    return tl.exp2(x.to(tl.float32) - 127.0)


@triton.jit
def _decode_e2m1(code):
    idx = code & 0x07
    mag = tl.where(
        idx == 0,
        0.0,
        tl.where(
            idx == 1,
            0.5,
            tl.where(
                idx == 2,
                1.0,
                tl.where(
                    idx == 3,
                    1.5,
                    tl.where(
                        idx == 4,
                        2.0,
                        tl.where(idx == 5, 3.0, tl.where(idx == 6, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    return tl.where((code & 0x08) != 0, -mag, mag)


@triton.jit
def _load_fp4_weight(
    packed_ptr,
    scale_ptr,
    expert,
    n_offsets,
    k_offsets,
    stride_e,
    stride_n,
    stride_kp,
    scale_stride_e,
    scale_stride_n,
    scale_stride_g,
    N: tl.constexpr,
    K: tl.constexpr,
    SCALE_IS_UE8M0: tl.constexpr,
):
    packed_offsets = (
        expert * stride_e
        + n_offsets[:, None] * stride_n
        + (k_offsets[None, :] // 2) * stride_kp
    )
    packed = tl.load(
        packed_ptr + packed_offsets,
        mask=(n_offsets[:, None] < N) & (k_offsets[None, :] < K),
        other=0,
    ).to(tl.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    code = tl.where((k_offsets[None, :] & 1) == 0, low, high)
    values = _decode_e2m1(code)

    scale_offsets = (
        expert * scale_stride_e
        + n_offsets[:, None] * scale_stride_n
        + (k_offsets[None, :] // 32) * scale_stride_g
    )
    raw_scale = tl.load(
        scale_ptr + scale_offsets,
        mask=(n_offsets[:, None] < N) & (k_offsets[None, :] < K),
        other=0,
    )
    scale = _ue8m0_to_f32(raw_scale) if SCALE_IS_UE8M0 else raw_scale.to(tl.float32)
    return values * scale


@triton.jit
def _fp8_fp4_mega_moe_l1_kernel(
    x_ptr,
    x_scale_ptr,
    topk_idx_ptr,
    l1_w_ptr,
    l1_s_ptr,
    l1_out_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    TOP_K: tl.constexpr,
    stride_xm,
    stride_xh,
    stride_xsm,
    stride_xsg,
    stride_topkm,
    stride_topkk,
    stride_w_e,
    stride_w_n,
    stride_w_kp,
    stride_s_e,
    stride_s_n,
    stride_s_g,
    stride_out_m,
    stride_out_k,
    stride_out_n,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_IS_UE8M0: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_topk = tl.program_id(1)
    pid_n = tl.program_id(2)

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    expert = tl.load(topk_idx_ptr + pid_m * stride_topkm + pid_topk * stride_topkk)
    valid_expert = expert >= 0
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k0 in range(0, H, BLOCK_K):
        ks = k0 + k_offsets
        x = tl.load(
            x_ptr + pid_m * stride_xm + ks * stride_xh,
            mask=ks < H,
            other=0.0,
        ).to(tl.float32)
        x_scale = tl.load(
            x_scale_ptr + pid_m * stride_xsm + (ks // 32) * stride_xsg,
            mask=ks < H,
            other=0.0,
        ).to(tl.float32)
        x = x * x_scale

        w = _load_fp4_weight(
            l1_w_ptr,
            l1_s_ptr,
            expert,
            n_offsets,
            ks,
            stride_w_e,
            stride_w_n,
            stride_w_kp,
            stride_s_e,
            stride_s_n,
            stride_s_g,
            2 * I,
            H,
            SCALE_IS_UE8M0,
        )
        acc += tl.sum(w * x[None, :], axis=1)

    out_ptrs = (
        l1_out_ptr
        + pid_m * stride_out_m
        + pid_topk * stride_out_k
        + n_offsets * stride_out_n
    )
    tl.store(out_ptrs, acc, mask=valid_expert & (n_offsets < 2 * I))


@triton.jit
def _fp8_fp4_mega_moe_l2_kernel(
    topk_idx_ptr,
    topk_weights_ptr,
    l1_out_ptr,
    l2_w_ptr,
    l2_s_ptr,
    y_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    TOP_K: tl.constexpr,
    stride_topkm,
    stride_topkk,
    stride_twm,
    stride_twk,
    stride_l1_m,
    stride_l1_k,
    stride_l1_n,
    stride_w_e,
    stride_w_h,
    stride_w_ip,
    stride_s_e,
    stride_s_h,
    stride_s_g,
    stride_ym,
    stride_yh,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
    SCALE_IS_UE8M0: tl.constexpr,
    ACTIVATION_CLAMP: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    i_offsets = tl.arange(0, BLOCK_I)
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for tk in range(0, TOP_K):
        expert = tl.load(topk_idx_ptr + pid_m * stride_topkm + tk * stride_topkk)
        valid_expert = expert >= 0
        route_w = tl.load(
            topk_weights_ptr + pid_m * stride_twm + tk * stride_twk,
            mask=valid_expert,
            other=0.0,
        )
        expert_acc = tl.zeros([BLOCK_H], dtype=tl.float32)

        for i0 in range(0, I, BLOCK_I):
            is_ = i0 + i_offsets
            gate = tl.load(
                l1_out_ptr + pid_m * stride_l1_m + tk * stride_l1_k + is_ * stride_l1_n,
                mask=(is_ < I) & valid_expert,
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                l1_out_ptr
                + pid_m * stride_l1_m
                + tk * stride_l1_k
                + (I + is_) * stride_l1_n,
                mask=(is_ < I) & valid_expert,
                other=0.0,
            ).to(tl.float32)
            if ACTIVATION_CLAMP >= 0.0:
                gate = tl.minimum(tl.maximum(gate, -ACTIVATION_CLAMP), ACTIVATION_CLAMP)
                up = tl.minimum(tl.maximum(up, -ACTIVATION_CLAMP), ACTIVATION_CLAMP)
            act = (gate / (1.0 + tl.exp(-gate))) * up

            w = _load_fp4_weight(
                l2_w_ptr,
                l2_s_ptr,
                expert,
                h_offsets,
                is_,
                stride_w_e,
                stride_w_h,
                stride_w_ip,
                stride_s_e,
                stride_s_h,
                stride_s_g,
                H,
                I,
                SCALE_IS_UE8M0,
            )
            expert_acc += tl.sum(w * act[None, :], axis=1)

        acc += expert_acc * route_w

    y_ptrs = y_ptr + pid_m * stride_ym + h_offsets * stride_yh
    tl.store(y_ptrs, acc, mask=h_offsets < H)


def fp8_fp4_mega_moe(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    l1_weights: torch.Tensor,
    l1_scales: torch.Tensor,
    l2_weights: torch.Tensor,
    l2_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    activation_clamp: float | None = None,
) -> torch.Tensor:
    """Functional Triton implementation of a local FP8 x FP4 MegaMoE.

    This is a correctness-oriented single-rank implementation.  It consumes
    unpacked routing tensors directly instead of DeepGEMM's symmetric buffer.
    FP4 weights are expected in DeepGEMM's simple packed E2M1 format before
    transform_weights_for_mega_moe:

      * l1_weights: [num_experts, 2 * intermediate, hidden // 2]
      * l1_scales : [num_experts, 2 * intermediate, hidden // 32]
      * l2_weights: [num_experts, hidden, intermediate // 2]
      * l2_scales : [num_experts, hidden, intermediate // 32]
    """
    if x_fp8.ndim != 2:
        raise ValueError("x_fp8 must be [num_tokens, hidden]")
    if topk_idx.shape != topk_weights.shape:
        raise ValueError("topk_idx and topk_weights must have the same shape")
    if topk_idx.shape[0] != x_fp8.shape[0]:
        raise ValueError("routing tensors must have the same token count as x_fp8")
    if l1_weights.ndim != 3 or l2_weights.ndim != 3:
        raise ValueError("l1_weights and l2_weights must be 3D packed FP4 tensors")

    num_tokens, hidden = x_fp8.shape
    top_k = topk_idx.shape[1]
    num_experts, l1_n, l1_k_half = l1_weights.shape
    l2_experts, l2_h, l2_i_half = l2_weights.shape
    intermediate = l2_i_half * 2

    if l2_experts != num_experts or l2_h != hidden:
        raise ValueError("l2 weight shape is inconsistent with x/l1 weights")
    if l1_n != 2 * intermediate or l1_k_half * 2 != hidden:
        raise ValueError("l1 weight shape must be [E, 2 * I, H // 2]")
    if hidden % 32 != 0 or intermediate % 32 != 0:
        raise ValueError("hidden and intermediate must be multiples of 32")
    if x_scale.shape != (num_tokens, hidden // 32):
        raise ValueError("x_scale must be [num_tokens, hidden // 32]")
    if top_k > 8:
        raise ValueError("this Triton fallback supports top_k <= 8")

    scale_is_ue8m0 = _check_supported_scales(l1_scales, "l1_scales")
    if _check_supported_scales(l2_scales, "l2_scales") != scale_is_ue8m0:
        raise TypeError(
            "l1_scales and l2_scales must use the same scale representation"
        )

    if out is None:
        out = torch.empty(
            (num_tokens, hidden), device=x_fp8.device, dtype=torch.bfloat16
        )
    if out.shape != (num_tokens, hidden):
        raise ValueError("out must be [num_tokens, hidden]")

    l1_out = torch.empty(
        (num_tokens, top_k, 2 * intermediate),
        device=x_fp8.device,
        dtype=torch.float32,
    )

    block_n = 16
    block_k = 32
    l1_grid = (num_tokens, top_k, triton.cdiv(2 * intermediate, block_n))
    with torch_device_fn.device(x_fp8.device):
        _fp8_fp4_mega_moe_l1_kernel[l1_grid](
            x_fp8,
            x_scale,
            topk_idx,
            l1_weights,
            l1_scales,
            l1_out,
            num_tokens,
            hidden,
            intermediate,
            top_k,
            x_fp8.stride(0),
            x_fp8.stride(1),
            x_scale.stride(0),
            x_scale.stride(1),
            topk_idx.stride(0),
            topk_idx.stride(1),
            l1_weights.stride(0),
            l1_weights.stride(1),
            l1_weights.stride(2),
            l1_scales.stride(0),
            l1_scales.stride(1),
            l1_scales.stride(2),
            l1_out.stride(0),
            l1_out.stride(1),
            l1_out.stride(2),
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            SCALE_IS_UE8M0=scale_is_ue8m0,
        )

        block_h = 16
        block_i = 32
        l2_grid = (num_tokens, triton.cdiv(hidden, block_h))
        _fp8_fp4_mega_moe_l2_kernel[l2_grid](
            topk_idx,
            topk_weights,
            l1_out,
            l2_weights,
            l2_scales,
            out,
            num_tokens,
            hidden,
            intermediate,
            top_k,
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_weights.stride(0),
            topk_weights.stride(1),
            l1_out.stride(0),
            l1_out.stride(1),
            l1_out.stride(2),
            l2_weights.stride(0),
            l2_weights.stride(1),
            l2_weights.stride(2),
            l2_scales.stride(0),
            l2_scales.stride(1),
            l2_scales.stride(2),
            out.stride(0),
            out.stride(1),
            BLOCK_H=block_h,
            BLOCK_I=block_i,
            SCALE_IS_UE8M0=scale_is_ue8m0,
            ACTIVATION_CLAMP=(
                -1.0 if activation_clamp is None else float(activation_clamp)
            ),
        )
    return out


def _unpack_fp4_e2m1_torch(packed: torch.Tensor) -> torch.Tensor:
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=packed.device,
        dtype=torch.float32,
    )
    p = packed.to(torch.uint8)
    lo = p & 0x0F
    hi = (p >> 4) & 0x0F
    codes = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        device=packed.device,
        dtype=torch.uint8,
    )
    codes[..., 0::2] = lo
    codes[..., 1::2] = hi
    mag = values[(codes & 0x07).long()]
    sign = (codes & 0x08) != 0
    return torch.where(sign & ((codes & 0x07) != 0), -mag, mag)


def _expand_scale_torch(scale: torch.Tensor, width: int) -> torch.Tensor:
    if scale.dtype == torch.uint8:
        scale = torch.exp2(scale.to(torch.float32) - 127.0)
    else:
        scale = scale.to(torch.float32)
    return torch.repeat_interleave(scale, 32, dim=-1)[..., :width]


def fp8_fp4_mega_moe_torch_ref(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    l1_weights: torch.Tensor,
    l1_scales: torch.Tensor,
    l2_weights: torch.Tensor,
    l2_scales: torch.Tensor,
    activation_clamp: float | None = None,
) -> torch.Tensor:
    x = x_fp8.float() * _expand_scale_torch(x_scale, x_fp8.shape[-1])
    l1 = _unpack_fp4_e2m1_torch(l1_weights) * _expand_scale_torch(
        l1_scales, l1_weights.shape[-1] * 2
    )
    l2 = _unpack_fp4_e2m1_torch(l2_weights) * _expand_scale_torch(
        l2_scales, l2_weights.shape[-1] * 2
    )

    num_tokens, hidden = x.shape
    intermediate = l2.shape[-1]
    out = torch.zeros((num_tokens, hidden), device=x.device, dtype=torch.float32)
    for token in range(num_tokens):
        for slot in range(topk_idx.shape[1]):
            expert = int(topk_idx[token, slot].item())
            if expert < 0:
                continue
            l1_out = torch.matmul(l1[expert], x[token])
            gate = l1_out[:intermediate]
            up = l1_out[intermediate:]
            if activation_clamp is not None:
                gate = gate.clamp(-activation_clamp, activation_clamp)
                up = up.clamp(-activation_clamp, activation_clamp)
            act = torch.nn.functional.silu(gate) * up
            out[token] += topk_weights[token, slot].float() * torch.matmul(
                l2[expert], act
            )
    return out
