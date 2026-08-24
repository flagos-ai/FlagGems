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

import pytest
import torch

import flag_gems
from flag_gems.fused.fused_marlin_moe import QUANT_TYPE_FLOAT8_E4M3FN, fused_marlin_moe

GROUP_SIZE = 128
QUICK_CONFIGS = [
    (1, 8, 128, 256, 2),
    (4, 8, 128, 256, 2),
    (16, 8, 256, 512, 2),
    (32, 8, 128, 256, 4),
]


def _is_hopper():
    if flag_gems.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    return 90 <= sm < 100


def _quantize_moe_weight_fp8(w_fp, group_size):
    """Quantize expert weights to E4M3 with one scale per K group."""
    num_experts, out_dim, in_dim = w_fp.shape
    assert in_dim % group_size == 0

    fp8_dtype = torch.float8_e4m3fn
    fp8_info = torch.finfo(fp8_dtype)
    num_groups = in_dim // group_size
    w_q = torch.empty(num_experts, out_dim, in_dim, device=w_fp.device, dtype=fp8_dtype)
    w_ref = torch.empty_like(w_fp)
    scales = torch.empty(
        num_experts,
        out_dim,
        num_groups,
        device=w_fp.device,
        dtype=w_fp.dtype,
    )
    for expert in range(num_experts):
        w_grouped = w_fp[expert].reshape(out_dim, num_groups, group_size).float()
        scales_fp = (w_grouped.abs().amax(dim=-1, keepdim=True) / fp8_info.max).clamp(
            min=1e-8
        )
        q_expert = (
            (w_grouped / scales_fp).clamp(fp8_info.min, fp8_info.max).to(fp8_dtype)
        )
        w_q[expert] = q_expert.reshape(out_dim, in_dim)
        w_ref[expert] = (
            (q_expert.float() * scales_fp).to(w_fp.dtype).reshape(out_dim, in_dim)
        )
        scales[expert] = scales_fp.squeeze(-1).to(w_fp.dtype)
    return w_q, w_ref, scales.contiguous()


def _make_inputs(config, dtype):
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device
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
    w1_q, w1_ref, w1_scale = _quantize_moe_weight_fp8(w1_fp, GROUP_SIZE)
    w2_q, w2_ref, w2_scale = _quantize_moe_weight_fp8(w2_fp, GROUP_SIZE)

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return (
        hidden_states,
        w1_q,
        w2_q,
        w1_ref,
        w2_ref,
        topk_weights.to(dtype),
        topk_ids,
        w1_scale,
        w2_scale,
    )


def _reference_swiglu_moe(hidden_states, w1_ref, w2_ref, topk_weights, topk_ids):
    num_tokens, hidden_size = hidden_states.shape
    intermediate_size = w1_ref.shape[1] // 2
    topk = topk_ids.shape[1]
    hidden_fp32 = hidden_states.float()
    weights_fp32 = topk_weights.float()
    output = torch.zeros(
        num_tokens, hidden_size, device=hidden_states.device, dtype=torch.float32
    )
    for token in range(num_tokens):
        for slot in range(topk):
            expert = topk_ids[token, slot].item()
            gate_up = w1_ref[expert].float() @ hidden_fp32[token]
            gate = gate_up[:intermediate_size]
            up = gate_up[intermediate_size:]
            inter = torch.nn.functional.silu(gate) * up
            output[token] += weights_fp32[token, slot] * (
                w2_ref[expert].float() @ inter
            )
    return output


@pytest.mark.skipif(not _is_hopper(), reason="W(fp8)A16 fast path requires Hopper")
@pytest.mark.parametrize("config", QUICK_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_marlin_moe_vs_ref_fp8_weight(config, dtype):
    """Compare W(fp8)A16 fused MoE against a dequantized FP32 reference."""
    hs, w1_q, w2_q, w1_ref, w2_ref, tw, ti, w1s, w2s = _make_inputs(config, dtype)
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
        quant_type_id=QUANT_TYPE_FLOAT8_E4M3FN,
    )
    reference = _reference_swiglu_moe(hs, w1_ref, w2_ref, tw, ti)
    torch.cuda.synchronize()
    relative_error = torch.mean(torch.abs(result.float() - reference)) / torch.mean(
        torch.abs(reference)
    )
    assert relative_error < 0.04, f"relative_error={relative_error:.4f}"
