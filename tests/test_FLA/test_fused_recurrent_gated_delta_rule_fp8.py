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

import math

import pytest
import torch

import flag_gems


def _fp8_decode_available() -> bool:
    return (
        flag_gems.device == "cuda"
        and torch.cuda.is_available()
        and hasattr(torch, "float8_e4m3fn")
        and torch.cuda.get_device_capability()[0] >= 9
    )


pytestmark = [
    pytest.mark.fused_recurrent_gated_delta_rule,
    pytest.mark.skipif(
        not _fp8_decode_available(), reason="FP8 GDN decode requires SM90 or newer"
    ),
]

H = 4
HV = 8
K = 128
V = 128


def _make_packed_inputs(num_sequences: int):
    mixed = torch.randn(
        num_sequences,
        2 * H * K + HV * V,
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )
    q, k, v = torch.split(mixed, (H * K, H * K, HV * V), dim=-1)
    q = q.view(1, num_sequences, H, K)
    k = k.view(1, num_sequences, H, K)
    v = (0.125 * v).view(1, num_sequences, HV, V)
    g = torch.empty(
        1,
        num_sequences,
        HV,
        device=flag_gems.device,
        dtype=torch.bfloat16,
    ).uniform_(math.log(0.98), math.log(0.995))
    beta = torch.rand(
        1,
        num_sequences,
        HV,
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )
    cu_seqlens = torch.arange(
        num_sequences + 1, device=flag_gems.device, dtype=torch.long
    )
    state_indices = torch.arange(
        num_sequences, device=flag_gems.device, dtype=torch.long
    )
    return q, k, v, g, beta, cu_seqlens, state_indices


def _run_bf16(q, k, v, g, beta, state, cu_seqlens, state_indices):
    return flag_gems.fused_recurrent_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=K**-0.5,
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=True,
    )


def _run_reference(q, k, v, g, beta, state):
    packed = q.shape[0] == 1
    q_float = q.float()
    k_float = k.float()
    q_float *= torch.rsqrt((q_float * q_float).sum(dim=-1, keepdim=True) + 1e-6)
    k_float *= torch.rsqrt((k_float * k_float).sum(dim=-1, keepdim=True) + 1e-6)
    q_float *= K**-0.5

    value_heads_per_key_head = HV // H
    q_float = q_float.repeat_interleave(value_heads_per_key_head, dim=2)
    k_float = k_float.repeat_interleave(value_heads_per_key_head, dim=2)
    if packed:
        q_float, k_float = q_float[0], k_float[0]
        v_float, g_float, beta_float = v[0], g[0], beta[0]
    else:
        q_float, k_float = q_float[:, 0], k_float[:, 0]
        v_float, g_float, beta_float = v[:, 0], g[:, 0], beta[:, 0]

    state = state.float() * torch.exp(g_float.float())[:, :, None, None]
    residual = v_float.float() - torch.einsum("nhk,nhkv->nhv", k_float, state)
    residual *= beta_float.float()[:, :, None]
    state += k_float[:, :, :, None] * residual[:, :, None, :]
    output = torch.einsum("nhk,nhkv->nhv", q_float, state)
    output = output[None] if packed else output[:, None]
    return output.to(q.dtype), state


def test_gdn_state_fp8_round_trip():
    torch.manual_seed(0)
    state = 0.125 * torch.randn(
        2, HV, K, V, device=flag_gems.device, dtype=torch.bfloat16
    )
    state_fp8 = flag_gems.quantize_gdn_state_fp8(state.contiguous())
    actual = flag_gems.dequantize_gdn_state_fp8(state_fp8, output_dtype=torch.float32)
    error = (actual - state.float()).abs()
    assert error.max().item() < 0.04
    assert error.mean().item() < 0.004


@pytest.mark.parametrize("num_sequences", [4, 64, 128])
def test_fused_recurrent_gated_delta_rule_fp8_w8a16_decode_accuracy(
    num_sequences,
):
    torch.manual_seed(1)
    q, k, v, g, beta, cu_seqlens, state_indices = _make_packed_inputs(num_sequences)
    initial_state = 0.02 * torch.randn(
        num_sequences,
        HV,
        K,
        V,
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )
    state_fp8 = flag_gems.quantize_gdn_state_fp8(initial_state.contiguous())
    state_ref = flag_gems.dequantize_gdn_state_fp8(
        state_fp8, output_dtype=torch.bfloat16
    )

    expected, state_ref = _run_reference(q, k, v, g, beta, state_ref)
    actual, state_fp8 = flag_gems.fused_recurrent_gated_delta_rule_fp8_w8a16_decode(
        q,
        k,
        v,
        g,
        beta,
        state_fp8,
        K**-0.5,
        cu_seqlens,
        state_indices,
        True,
    )
    actual_state = flag_gems.dequantize_gdn_state_fp8(state_fp8)

    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-3, rtol=5e-2)
    torch.testing.assert_close(actual_state, state_ref.float(), atol=2e-2, rtol=2.5e-1)


def test_fused_recurrent_gated_delta_rule_fp8_w8a16_decode_repeated_updates():
    torch.manual_seed(2)
    num_sequences = 2
    state_ref = torch.zeros(
        num_sequences,
        HV,
        K,
        V,
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )
    state_fp8 = flag_gems.quantize_gdn_state_fp8(state_ref)
    max_output_error = 0.0
    mean_output_error = 0.0

    for _ in range(64):
        q, k, v, g, beta, cu_seqlens, state_indices = _make_packed_inputs(num_sequences)
        expected, state_ref = _run_bf16(
            q, k, v, g, beta, state_ref, cu_seqlens, state_indices
        )
        actual, state_fp8 = flag_gems.fused_recurrent_gated_delta_rule_fp8_w8a16_decode(
            q,
            k,
            v,
            g,
            beta,
            state_fp8,
            K**-0.5,
            cu_seqlens,
            state_indices,
            True,
        )
        error = (actual.float() - expected.float()).abs()
        max_output_error = max(max_output_error, error.max().item())
        mean_output_error += error.mean().item()

    actual_state = flag_gems.dequantize_gdn_state_fp8(state_fp8)
    state_error = (actual_state - state_ref.float()).abs()
    assert max_output_error < 0.015
    assert mean_output_error / 64 < 0.002
    assert state_error.mean().item() < 0.02


def test_fused_recurrent_gated_delta_rule_fp8_w8a16_nonpacked_accuracy():
    torch.manual_seed(3)
    q, k, v, g, beta, _, _ = _make_packed_inputs(64)
    q = q.transpose(0, 1).contiguous()
    k = k.transpose(0, 1).contiguous()
    v = v.transpose(0, 1).contiguous()
    g = g.transpose(0, 1).contiguous()
    beta = beta.transpose(0, 1).contiguous()
    initial_state = 0.02 * torch.randn(
        64, HV, K, V, device=flag_gems.device, dtype=torch.bfloat16
    )
    state_fp8 = flag_gems.quantize_gdn_state_fp8(initial_state.contiguous())
    state_ref = flag_gems.dequantize_gdn_state_fp8(state_fp8)

    expected, state_ref = _run_reference(q, k, v, g, beta, state_ref)
    actual, state_fp8 = flag_gems.fused_recurrent_gated_delta_rule_fp8_w8a16_decode(
        q,
        k,
        v,
        g,
        beta,
        state_fp8,
        K**-0.5,
    )
    actual_state = flag_gems.dequantize_gdn_state_fp8(state_fp8)

    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-3, rtol=5e-2)
    torch.testing.assert_close(actual_state, state_ref, atol=2e-2, rtol=2.5e-1)
