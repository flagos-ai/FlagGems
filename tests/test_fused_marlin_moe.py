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
Precision tests for fused_marlin_moe (FlagGems Phase 2).

Phase 2 routes through the wna16 Triton kernel (fused_moe_kernel_gptq_awq)
for true fused-dequant W4A16 GEMM. Inputs are therefore real GPTQ-quantized
weights produced by vLLM's quantize_weights, not unit-scale FP16 stand-ins.

Oracle: dequantized weights run through a naive PyTorch SwiGLU MoE reference.
The wrapper sees packed uint8 weights; the reference sees the matching
fp16/bf16 w_ref returned by quantize_weights so quantization round-off is
shared by both sides.
"""

import pytest
import torch

import flag_gems
from flag_gems.fused.fused_marlin_moe import (
    QUANT_TYPE_FP4_E2M1,
    QUANT_TYPE_UINT4B8,
    QUANT_TYPE_UINT8B128,
    fused_marlin_moe,
)

from . import conftest as cfg


def _is_hopper():
    # The W4A16 fast-path kernel's bf16 dequant uses sm_90-only PTX
    # (sub.bf16x2 / mul.bf16); the fast path is gated to Hopper.
    if flag_gems.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    return 90 <= sm < 100


# -----------------------------------------------------------------------------
# Local GPTQ uint4b8 quantization helper (self-contained, no vllm dependency).
# Matches the layout produced by vllm.quantize_weights(..., uint4b8): values
# in [-7, 7] shifted to unsigned [1, 15] and packed two-per-byte into uint8.
# -----------------------------------------------------------------------------
QUANT_TYPE_UINT4B8_TAG = "uint4b8"


def _gptq_quantize_uint4b8(w_2d, group_size):
    """
    Symmetric per-group INT4 quantization with +8 offset (GPTQ uint4b8 convention).

    Self-contained replacement for vllm.quantize_weights(w, scalar_types.uint4b8,
    group_size, False, False). Produces unpacked integer codes (each cell a
    nibble in [0, 15]) plus the exact dequantized FP reference, both compatible
    with the layout fused_moe_kernel_gptq_awq consumes.

    Args:
        w_2d: (out_dim, in_dim), fp16 or bf16.
        group_size: int, must divide in_dim.

    Returns:
        w_ref:  (out_dim, in_dim), same dtype.  Dequantized reference values.
        w_q_unsigned: (out_dim, in_dim), uint8.  Each cell a nibble in [0, 15].
        scales: (out_dim, in_dim // group_size), same dtype as w_2d.
    """
    out_dim, in_dim = w_2d.shape
    assert in_dim % group_size == 0
    ng = in_dim // group_size

    w_grouped = w_2d.reshape(out_dim, ng, group_size).to(torch.float32)
    max_abs = w_grouped.abs().amax(dim=-1, keepdim=True)
    # scale = max_abs / 7  (symmetric INT4 range [-7, 7] after +8 offset -> [1, 15])
    scales_fp = (max_abs / 7.0).clamp(min=1e-8)

    w_q_signed = torch.round(w_grouped / scales_fp).clamp(-7, 7)
    w_ref_grouped = (w_q_signed * scales_fp).to(w_2d.dtype)
    w_q_unsigned = (w_q_signed + 8).clamp(0, 15).to(torch.uint8)

    w_ref = w_ref_grouped.reshape(out_dim, in_dim)
    w_q_unsigned = w_q_unsigned.reshape(out_dim, in_dim)
    scales = scales_fp.squeeze(-1).to(w_2d.dtype)
    return w_ref, w_q_unsigned, scales


QUANT_TYPE_UINT8B128_TAG = "uint8b128"


def _gptq_quantize_uint8b128(w_2d, group_size):
    """
    Symmetric per-group INT8 quantization with +128 offset (GPTQ uint8b128).

    Sister function to _gptq_quantize_uint4b8. Self-contained replacement for
    vllm.quantize_weights(w, scalar_types.uint8b128, group_size, False, False).
    Produces unpacked integer codes (each cell a byte in [1, 255], i.e.
    signed [-127, 127] shifted by +128) plus the exact dequantized FP
    reference, in the layout fused_moe_kernel_gptq_awq's W8A16 branch consumes
    (no nibble packing — one byte per element).

    Args:
        w_2d: (out_dim, in_dim), fp16 or bf16.
        group_size: int, must divide in_dim.

    Returns:
        w_ref:  (out_dim, in_dim), same dtype.  Dequantized reference values.
        w_q_unsigned: (out_dim, in_dim), uint8.  Each cell in [1, 255].
        scales: (out_dim, in_dim // group_size), same dtype as w_2d.
    """
    out_dim, in_dim = w_2d.shape
    assert in_dim % group_size == 0
    ng = in_dim // group_size

    w_grouped = w_2d.reshape(out_dim, ng, group_size).to(torch.float32)
    max_abs = w_grouped.abs().amax(dim=-1, keepdim=True)
    # scale = max_abs / 127 (symmetric INT8 range [-127, 127] after +128 -> [1, 255])
    scales_fp = (max_abs / 127.0).clamp(min=1e-8)

    w_q_signed = torch.round(w_grouped / scales_fp).clamp(-127, 127)
    w_ref_grouped = (w_q_signed * scales_fp).to(w_2d.dtype)
    w_q_unsigned = (w_q_signed + 128).clamp(0, 255).to(torch.uint8)

    w_ref = w_ref_grouped.reshape(out_dim, in_dim)
    w_q_unsigned = w_q_unsigned.reshape(out_dim, in_dim)
    scales = scales_fp.squeeze(-1).to(w_2d.dtype)
    return w_ref, w_q_unsigned, scales


# -----------------------------------------------------------------------------
# Shape configs.
# Tuple format: (num_tokens, num_experts, hidden_size, intermediate_size, topk)
# Hard requirement: hidden_size and intermediate_size are multiples of 128
# (the wna16 group_size). Smallest legal hidden = 128.

# -----------------------------------------------------------------------------
QUICK_CONFIGS = [
    (1, 8, 128, 256, 2),
    (4, 8, 128, 256, 2),
    (16, 8, 256, 512, 2),
    (32, 8, 128, 256, 4),
]

if cfg.QUICK_MODE:
    FULL_CONFIGS = QUICK_CONFIGS[:2]
else:
    FULL_CONFIGS = QUICK_CONFIGS + [
        (64, 8, 256, 512, 2),
        (128, 16, 128, 256, 4),
        # Mixtral-8x7B-like
        (1, 8, 4096, 14336, 2),
        (16, 8, 4096, 14336, 2),
        (64, 8, 4096, 14336, 2),
        # DeepSeek-V3-like (TP=8 shard)
        (1, 256, 7168, 2048, 8),
        (16, 256, 7168, 2048, 8),
        (64, 256, 7168, 2048, 8),
        # Qwen3-5-397B-A17B
        (1, 512, 4096, 1024, 10),
        (16, 512, 4096, 1024, 10),
        (64, 512, 4096, 1024, 10),
        # DeepSeek-V4-Flash
        (1, 256, 4096, 2048, 6),
        (16, 256, 4096, 2048, 6),
        (64, 256, 4096, 2048, 6),
    ]

GROUP_SIZE = 128


def _quantize_moe_weight(w_fp, group_size):
    """
    Apply vLLM's per-expert GPTQ quantization, returning the packed uint8
    weight and bf16/fp16 dequantized reference, in the layout fused MoE
    kernels consume.

    Args:
        w_fp: (E, out_dim, in_dim), fp16 or bf16.
    Returns:
        w_q:    (E, out_dim, in_dim // 2), uint8   (INT4 packed two-per-byte)
        w_ref:  (E, out_dim, in_dim), same dtype as w_fp  (dequantized values)
        scales: (E, out_dim, in_dim // group_size), same dtype as w_fp
    """
    E, out_dim, in_dim = w_fp.shape
    assert (
        in_dim % group_size == 0
    ), f"in_dim={in_dim} not divisible by group_size={group_size}"

    w_q = torch.empty(E, out_dim, in_dim // 2, device=w_fp.device, dtype=torch.uint8)
    w_ref = torch.empty_like(w_fp)
    scales = torch.empty(
        E,
        out_dim,
        in_dim // group_size,
        device=w_fp.device,
        dtype=w_fp.dtype,
    )
    for e in range(E):
        # Self-contained GPTQ uint4b8 quantization (no vllm dependency).
        ref_e, q_e_unpacked, sc_e = _gptq_quantize_uint4b8(w_fp[e], group_size)
        # Pack two nibbles per byte; low nibble = even, high nibble = odd.
        q_e_packed = q_e_unpacked[:, 1::2] * 16 + q_e_unpacked[:, ::2]
        w_q[e] = q_e_packed
        w_ref[e] = ref_e
        scales[e] = sc_e
    return w_q, w_ref, scales


def _quantize_moe_weight_int8(w_fp, group_size):
    """
    Per-expert GPTQ uint8b128 quantization. Sister of _quantize_moe_weight
    (which is INT4 packed). INT8 weights are one byte per element — no
    nibble packing — so the output K-dim is in_dim, not in_dim // 2.

    Args:
        w_fp: (E, out_dim, in_dim), fp16 or bf16.

    Returns:
        w_q:    (E, out_dim, in_dim), uint8   (each cell in [1, 255])
        w_ref:  (E, out_dim, in_dim), same dtype as w_fp
        scales: (E, out_dim, in_dim // group_size), same dtype as w_fp
    """
    E, out_dim, in_dim = w_fp.shape
    assert (
        in_dim % group_size == 0
    ), f"in_dim={in_dim} not divisible by group_size={group_size}"
    w_q = torch.empty(E, out_dim, in_dim, device=w_fp.device, dtype=torch.uint8)
    w_ref = torch.empty_like(w_fp)
    scales = torch.empty(
        E,
        out_dim,
        in_dim // group_size,
        device=w_fp.device,
        dtype=w_fp.dtype,
    )
    for e in range(E):
        ref_e, q_e, sc_e = _gptq_quantize_uint8b128(w_fp[e], group_size)
        w_q[e] = q_e
        w_ref[e] = ref_e
        scales[e] = sc_e
    return w_q, w_ref, scales


def _make_inputs(
    num_tokens, num_experts, hidden_size, intermediate_size, topk, dtype, device
):
    """
    Build all tensors for one test case.

    Returns:
        hidden_states          (M, K)        fp16/bf16
        w1_q, w2_q             packed uint8     -> wrapper input
        w1_ref, w2_ref         fp16/bf16        -> reference GEMM input
        topk_weights, topk_ids
        w1_scale, w2_scale     3D scales matching w1_q/w2_q
    """
    torch.manual_seed(0)
    # Match vLLM's test_fused_marlin_moe magnitude (test_moe.py): A, w1, w2 are
    # all scaled by 1/10 so output magnitudes stay small enough for the fixed
    # atol=4e-2 check (and the INT4 quant grid stays well-conditioned).
    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device, dtype=dtype) / 10.0
    )

    w1_fp = (
        torch.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        / 10.0
    )
    w2_fp = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )
        / 10.0
    )

    w1_q, w1_ref, w1_scale = _quantize_moe_weight(w1_fp, GROUP_SIZE)
    w2_q, w2_ref, w2_scale = _quantize_moe_weight(w2_fp, GROUP_SIZE)

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    return (
        hidden_states,
        w1_q,
        w2_q,
        w1_ref,
        w2_ref,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
    )


def _make_inputs_int8(
    num_tokens, num_experts, hidden_size, intermediate_size, topk, dtype, device
):
    """
    Build all tensors for one W8A16 test case. Sister of _make_inputs.

    Returns:
        hidden_states          (M, K)        fp16/bf16
        w1_q, w2_q             unpacked uint8     -> wrapper input
        w1_ref, w2_ref         fp16/bf16          -> reference GEMM input
        topk_weights, topk_ids
        w1_scale, w2_scale     3D scales matching w1_q/w2_q
    """
    torch.manual_seed(0)
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    w1_fp = (
        torch.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        / 10.0
    )
    w2_fp = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )
        / 10.0
    )

    w1_q, w1_ref, w1_scale = _quantize_moe_weight_int8(w1_fp, GROUP_SIZE)
    w2_q, w2_ref, w2_scale = _quantize_moe_weight_int8(w2_fp, GROUP_SIZE)

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    return (
        hidden_states,
        w1_q,
        w2_q,
        w1_ref,
        w2_ref,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
    )


# -----------------------------------------------------------------------------
# MXFP4 (E2M1 weight + per-32 E8M0 scale) round-to-nearest quantization.
# -----------------------------------------------------------------------------
MXFP4_GROUP_SIZE = 32
_E2M1_POS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1_MID = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
_E2M1_MAX = 6.0


def _quantize_mxfp4(w_2d, group_size):
    """Round-to-nearest MXFP4. Returns w_ref (dequant), nibbles (uint8 [0,15]),
    scale_e8m0 (out, in // group_size)."""
    out_dim, in_dim = w_2d.shape
    ng = in_dim // group_size
    device = w_2d.device
    wg = w_2d.reshape(out_dim, ng, group_size).to(torch.float32)
    amax = wg.abs().amax(dim=-1, keepdim=True)
    # E8M0 scale = 2^ceil(log2(amax / 6)) so amax / scale <= 6 (FP4 max).
    exp = torch.ceil(torch.log2((amax / _E2M1_MAX).clamp(min=1e-30))).clamp(-127, 127)
    scale = torch.exp2(exp)
    e8m0_byte = (exp + 127.0).to(torch.uint8)
    wn = wg / scale
    sign = wn < 0
    a = wn.abs().clamp(max=_E2M1_MAX)
    mag = torch.bucketize(a, torch.tensor(_E2M1_MID, device=device))  # 0..7
    q = torch.tensor(_E2M1_POS, device=device)[mag]
    ref = torch.where(sign, -q, q) * scale
    nibbles = (sign.to(torch.uint8) * 8 + mag.to(torch.uint8)).reshape(out_dim, in_dim)
    w_ref = ref.reshape(out_dim, in_dim).to(w_2d.dtype)
    scale_e8m0 = e8m0_byte.squeeze(-1).view(torch.float8_e8m0fnu)
    return w_ref, nibbles, scale_e8m0


def _quantize_moe_weight_mxfp4(w_fp, group_size):
    """Per-expert MXFP4 in FlagGems layout (packed two nibbles/byte + E8M0 scale)."""
    E, out_dim, in_dim = w_fp.shape
    w_q = torch.empty(E, out_dim, in_dim // 2, device=w_fp.device, dtype=torch.uint8)
    w_ref = torch.empty_like(w_fp)
    scales = torch.empty(
        E,
        out_dim,
        in_dim // group_size,
        device=w_fp.device,
        dtype=torch.float8_e8m0fnu,
    )
    for e in range(E):
        ref_e, nib_e, sc_e = _quantize_mxfp4(w_fp[e], group_size)
        w_q[e] = nib_e[:, 1::2] * 16 + nib_e[:, ::2]
        w_ref[e] = ref_e
        scales[e] = sc_e
    return w_q, w_ref, scales


def _make_inputs_mxfp4(
    num_tokens, num_experts, hidden_size, intermediate_size, topk, dtype, device
):
    torch.manual_seed(0)
    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device, dtype=dtype) / 10.0
    )
    w1_fp = (
        torch.randn(
            num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
        )
        / 10.0
    )
    w2_fp = (
        torch.randn(
            num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
        )
        / 10.0
    )
    w1_q, w1_ref, w1_scale = _quantize_moe_weight_mxfp4(w1_fp, MXFP4_GROUP_SIZE)
    w2_q, w2_ref, w2_scale = _quantize_moe_weight_mxfp4(w2_fp, MXFP4_GROUP_SIZE)

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).to(dtype)
    return (
        hidden_states,
        w1_q,
        w2_q,
        w1_ref,
        w2_ref,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
    )


def compute_max_diff(output, output_ref):
    """vLLM's Marlin accuracy metric (mean relative error), from
    vllm/tests/kernels/utils.py; test_marlin_gemm.py asserts it < 0.04."""
    return torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref)
    )


def _reference_swiglu_moe(
    hidden_states,
    w1_ref,
    w2_ref,
    topk_weights,
    topk_ids,
    apply_router_weight_on_input=False,
):
    """fp32 dequant-SwiGLU MoE ground truth (weights cast per-expert to avoid a
    full fp32 copy of the (E, *, *) tensors)."""
    M, K = hidden_states.shape
    _, two_N, _ = w1_ref.shape
    N = two_N // 2
    topk = topk_ids.shape[1]
    hs = hidden_states.float()
    tw = topk_weights.float()
    out = torch.zeros(M, K, device=hidden_states.device, dtype=torch.float32)
    for m in range(M):
        x = hs[m]
        for k in range(topk):
            e = topk_ids[m, k].item()
            route_weight = tw[m, k]
            route_input = route_weight * x if apply_router_weight_on_input else x
            gate_up = w1_ref[e].float() @ route_input
            gate, up = gate_up[:N], gate_up[N:]
            act = torch.nn.functional.silu(gate) * up
            y = w2_ref[e].float() @ act
            out[m] += y if apply_router_weight_on_input else route_weight * y
    return out


@pytest.mark.skipif(
    not _is_hopper(),
    reason="W4A16 fast path uses Hopper-only bf16 SIMD PTX (sm_90+)",
)
@pytest.mark.parametrize("config", FULL_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("apply_router_weight_on_input", [False, True])
def test_fused_marlin_moe_vs_ref(config, dtype, apply_router_weight_on_input):
    """Compare fused_marlin_moe (packed INT4) against PyTorch reference (dequant)."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device

    hs, w1_q, w2_q, w1_ref, w2_ref, tw, ti, w1s, w2s = _make_inputs(
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        topk,
        dtype,
        device,
    )

    result = fused_marlin_moe(
        hidden_states=hs,
        w1=w1_q,
        w2=w2_q,
        bias1=None,
        bias2=None,
        w1_scale=w1s,
        w2_scale=w2s,
        topk_weights=tw,
        topk_ids=ti,
        quant_type_id=QUANT_TYPE_UINT4B8,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    ref = _reference_swiglu_moe(
        hs,
        w1_ref,
        w2_ref,
        tw,
        ti,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    torch.cuda.synchronize()

    max_diff = compute_max_diff(result.float(), ref)
    assert max_diff < 0.04, f"max_diff={max_diff:.4f}"


@pytest.mark.parametrize("config", QUICK_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_marlin_moe_vs_ref_int8(config, dtype):
    """Compare fused_marlin_moe (unpacked INT8) against PyTorch reference (dequant)."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device

    hs, w1_q, w2_q, w1_ref, w2_ref, tw, ti, w1s, w2s = _make_inputs_int8(
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        topk,
        dtype,
        device,
    )
    result = fused_marlin_moe(
        hidden_states=hs,
        w1=w1_q,
        w2=w2_q,
        bias1=None,
        bias2=None,
        w1_scale=w1s,
        w2_scale=w2s,
        topk_weights=tw,
        topk_ids=ti,
        quant_type_id=QUANT_TYPE_UINT8B128,
    )
    ref = _reference_swiglu_moe(hs, w1_ref, w2_ref, tw, ti)
    torch.cuda.synchronize()
    # INT8 should be tighter than INT4; same vLLM Marlin metric.
    max_diff = compute_max_diff(result.float(), ref)
    assert max_diff < 0.04, f"max_diff={max_diff:.4f}"


@pytest.mark.skipif(
    not _is_hopper(),
    reason="MXFP4 fast path uses Hopper-only bf16/fp16 SIMD PTX (sm_90+)",
)
@pytest.mark.parametrize("config", FULL_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_marlin_moe_mxfp4_vs_ref(config, dtype):
    """Compare fused_marlin_moe (MXFP4) against PyTorch reference (dequant)."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device

    hs, w1_q, w2_q, w1_ref, w2_ref, tw, ti, w1s, w2s = _make_inputs_mxfp4(
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        topk,
        dtype,
        device,
    )
    result = fused_marlin_moe(
        hidden_states=hs,
        w1=w1_q,
        w2=w2_q,
        bias1=None,
        bias2=None,
        w1_scale=w1s,
        w2_scale=w2s,
        topk_weights=tw,
        topk_ids=ti,
        quant_type_id=QUANT_TYPE_FP4_E2M1,
        group_size=MXFP4_GROUP_SIZE,
    )
    ref = _reference_swiglu_moe(hs, w1_ref, w2_ref, tw, ti)
    torch.cuda.synchronize()

    max_diff = compute_max_diff(result.float(), ref)
    assert max_diff < 0.04, f"max_diff={max_diff:.4f}"


# -----------------------------------------------------------------------------
# MVP guardrails: features the wrapper rejects must raise NotImplementedError.
# -----------------------------------------------------------------------------


def _minimal_args(device=flag_gems.device, dtype=torch.bfloat16):
    """Smallest valid arg bundle, used to probe rejection paths."""
    M, K, N, E, topk = 4, 128, 256, 4, 2
    return _make_inputs(M, E, K, N, topk, dtype, device)


def test_rejects_unsupported_quant_type():
    hs, w1_q, w2_q, _, _, tw, ti, w1s, w2s = _minimal_args()
    with pytest.raises(NotImplementedError, match="quant_type_id"):
        fused_marlin_moe(
            hidden_states=hs,
            w1=w1_q,
            w2=w2_q,
            bias1=None,
            bias2=None,
            w1_scale=w1s,
            w2_scale=w2s,
            topk_weights=tw,
            topk_ids=ti,
            quant_type_id=999,
        )


def test_rejects_act_order():
    hs, w1_q, w2_q, _, _, tw, ti, w1s, w2s = _minimal_args()
    g_idx = torch.zeros(8, dtype=torch.long, device=hs.device)
    with pytest.raises(NotImplementedError, match="act_order"):
        fused_marlin_moe(
            hidden_states=hs,
            w1=w1_q,
            w2=w2_q,
            bias1=None,
            bias2=None,
            w1_scale=w1s,
            w2_scale=w2s,
            topk_weights=tw,
            topk_ids=ti,
            quant_type_id=QUANT_TYPE_UINT4B8,
            g_idx1=g_idx,
        )


def test_rejects_fp8_input_dtype():
    hs, w1_q, w2_q, _, _, tw, ti, w1s, w2s = _minimal_args()
    with pytest.raises(NotImplementedError, match="FP8"):
        fused_marlin_moe(
            hidden_states=hs,
            w1=w1_q,
            w2=w2_q,
            bias1=None,
            bias2=None,
            w1_scale=w1s,
            w2_scale=w2s,
            topk_weights=tw,
            topk_ids=ti,
            quant_type_id=QUANT_TYPE_UINT4B8,
            input_dtype=torch.float8_e4m3fn,
        )
