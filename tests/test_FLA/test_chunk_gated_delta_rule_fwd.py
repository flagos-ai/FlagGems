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
import torch.nn.functional as F

import flag_gems


def naive_chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, initial_state):
    """
    Naive reference implementation of chunk_gated_delta_rule_fwd.
    Implements the gated delta rule recurrence token-by-token:
        S_t = exp(g_t) * S_{t-1} + beta_t * k_t^T (v_t - k_t @ S_{t-1})
        o_t = q_t @ S_t * scale
    """
    B, T, H, K = q.shape
    V = v.shape[-1]

    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    beta = beta.float()

    S = (
        initial_state.float().clone()
        if initial_state is not None
        else torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)
    )
    outputs = []

    for t in range(T):
        q_t = q[:, t, :, :]  # (B, H, K)
        k_t = k[:, t, :, :]  # (B, H, K)
        v_t = v[:, t, :, :]  # (B, H, V)
        g_t = g[:, t, :]  # (B, H)
        beta_t = beta[:, t, :]  # (B, H)

        # Gating: S = exp(g_t) * S
        gate = torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        S = gate * S

        # Delta rule: S += beta_t * k_t^T @ (v_t - k_t @ S)
        # k_t: (B, H, K) -> (B, H, K, 1)
        # v_t: (B, H, V) -> (B, H, 1, V)
        kS = torch.einsum("bhk,bhkv->bhv", k_t, S)  # (B, H, V)
        delta = v_t - kS  # (B, H, V)
        # outer product: k_t^T @ delta -> (B, H, K, V)
        update = torch.einsum("bhk,bhv->bhkv", k_t, delta) * beta_t.unsqueeze(
            -1
        ).unsqueeze(-1)
        S = S + update

        # Output: o_t = q_t @ S * scale
        o_t = torch.einsum("bhk,bhkv->bhv", q_t, S) * scale  # (B, H, V)
        outputs.append(o_t)

    o = torch.stack(outputs, dim=1)  # (B, T, H, V)
    return o, S


@pytest.mark.chunk_gated_delta_rule_fwd
@pytest.mark.xfail(
    reason="Triton 3.6.0 compilation error on Hopper: 'ttng.warp_group_dot' op pipeliner issue"
)
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("T", [64, 128])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("K", [64])
@pytest.mark.parametrize("V", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_chunk_gated_delta_rule_fwd_accuracy(B, T, H, K, V, dtype):
    device = flag_gems.device
    torch.manual_seed(42)

    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    scale = K**-0.5
    initial_state = torch.zeros(B, H, K, V, device=device, dtype=dtype)

    ref_o, ref_final_state = naive_chunk_gated_delta_rule_fwd(
        q, k, v, g, beta, scale, initial_state
    )

    result = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=None,
    )
    # result is (g_cumsum, o, A, final_state, w_or_None, h_or_None, v_new_or_None)
    res_o = result[1]
    res_final_state = result[3]

    torch.testing.assert_close(res_o.float(), ref_o, rtol=1e-1, atol=2e-1)
    torch.testing.assert_close(
        res_final_state.float(), ref_final_state, rtol=1.5, atol=1.0
    )


@pytest.mark.chunk_gated_delta_rule_fwd
@pytest.mark.xfail(
    reason="Triton 3.6.0 compilation error on Hopper: 'ttng.warp_group_dot' op pipeliner issue"
)
@pytest.mark.parametrize("T", [64, 128, 256])
def test_chunk_gated_delta_rule_fwd_no_initial_state(T):
    device = flag_gems.device
    dtype = torch.bfloat16
    torch.manual_seed(0)

    B, H, K, V = 1, 4, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    scale = K**-0.5

    ref_o, _ = naive_chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, None)

    result = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    res_o = result[1]

    torch.testing.assert_close(res_o.float(), ref_o, rtol=1e-1, atol=2e-1)


@pytest.mark.chunk_gated_delta_rule_fwd
@pytest.mark.xfail(
    reason="Triton 3.6.0 compilation error on Hopper: 'ttng.warp_group_dot' op pipeliner issue"
)
@pytest.mark.parametrize("T", [64, 128])
def test_chunk_gated_delta_rule_fwd_with_cu_seqlens(T):
    device = flag_gems.device
    dtype = torch.bfloat16
    torch.manual_seed(1)

    B, H, K, V = 1, 4, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    scale = K**-0.5
    initial_state = torch.zeros(B, H, K, V, device=device, dtype=dtype)
    cu_seqlens = torch.arange(T + 1, device=device, dtype=torch.long)

    result = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    # Verify output shapes are correct
    res_o = result[1]
    res_final_state = result[3]
    assert res_o.shape == (B, T, H, V)
    assert res_final_state.shape[1:] == (H, K, V)
