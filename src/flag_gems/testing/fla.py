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
