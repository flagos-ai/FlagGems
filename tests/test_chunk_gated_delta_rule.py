import pytest
import torch

import flag_gems


def _reference_chunk_gated_delta_rule(q, k, v, beta, g):
    """Pure-PyTorch eager reference matching the recurrence in the kernel.

    For each timestep t:
        proj  = k_t @ h
        v_new = v_t - proj
        h     = exp(g_t) * h + beta_t * outer(k_t, v_new)
        o_t   = q_t @ h
    """
    B, H, L, D_k = q.shape
    D_v = v.shape[-1]
    scale = D_k**-0.5
    q_scaled = q.float() * scale

    h = torch.zeros(B, H, D_k, D_v, dtype=torch.float32, device=q.device)
    o = torch.zeros(B, H, L, D_v, dtype=torch.float32, device=q.device)

    for t in range(L):
        g_t = g[:, :, t].float().clamp(-10.0, 10.0)
        decay = torch.exp(g_t)
        beta_t = beta[:, :, t].float()
        q_t = q_scaled[:, :, t]
        k_t = k[:, :, t].float()
        v_t = v[:, :, t].float()

        proj = torch.einsum("bhk,bhkv->bhv", k_t, h)
        v_new = v_t - proj
        decay_e = decay.unsqueeze(-1).unsqueeze(-1)
        beta_e = beta_t.unsqueeze(-1).unsqueeze(-1)
        outer = torch.einsum("bhk,bhv->bhkv", k_t, v_new)
        h = decay_e * h + beta_e * outer

        o[:, :, t] = torch.einsum("bhk,bhkv->bhv", q_t, h)

    return o.to(q.dtype)


@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(
    "params",
    [
        {"B": 1, "H": 1, "L": 32, "D_k": 8, "D_v": 8, "BT": 16},
        {"B": 2, "H": 4, "L": 64, "D_k": 16, "D_v": 16, "BT": 32},
        {"B": 1, "H": 2, "L": 24, "D_k": 8, "D_v": 16, "BT": 16},
    ],
)
def test_accuracy_chunk_gated_delta_rule(params, dtype):
    from flag_gems.ops.chunk_gated_delta_rule import chunk_gated_delta_rule

    B, H, L, D_k, D_v = (
        params["B"],
        params["H"],
        params["L"],
        params["D_k"],
        params["D_v"],
    )
    BT = params["BT"]

    torch.manual_seed(0)
    q = torch.randn(B, H, L, D_k, dtype=dtype, device=flag_gems.device)
    k = torch.randn(B, H, L, D_k, dtype=dtype, device=flag_gems.device)
    v = torch.randn(B, H, L, D_v, dtype=dtype, device=flag_gems.device)
    beta = torch.sigmoid(torch.randn(B, H, L, dtype=dtype, device=flag_gems.device))
    g = torch.randn(B, H, L, dtype=dtype, device=flag_gems.device) * 0.01

    o_gems, _ = chunk_gated_delta_rule(q, k, v, beta, g, BT=BT)
    o_ref = _reference_chunk_gated_delta_rule(q, k, v, beta, g)

    assert o_gems.shape == (B, H, L, D_v)
    assert not torch.isnan(o_gems).any(), "Output contains NaN"
    assert not torch.isinf(o_gems).any(), "Output contains Inf"
    torch.testing.assert_close(o_gems, o_ref, rtol=1e-2, atol=1e-3)
