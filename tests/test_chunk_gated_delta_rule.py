import pytest
import torch

import flag_gems

from . import accuracy_utils as utils  # noqa: F401


@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(
    "params",
    [
        {"B": 1, "H": 1, "L": 32, "D_k": 8, "D_v": 8, "BT": 16},
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

    q = torch.randn(B, H, L, D_k, dtype=dtype, device=flag_gems.device)
    k = torch.randn(B, H, L, D_k, dtype=dtype, device=flag_gems.device)
    v = torch.randn(B, H, L, D_v, dtype=dtype, device=flag_gems.device)
    beta = torch.sigmoid(torch.randn(B, H, L, dtype=dtype, device=flag_gems.device))
    g = torch.randn(B, H, L, dtype=dtype, device=flag_gems.device) * 0.01

    o, final_state = chunk_gated_delta_rule(q, k, v, beta, g, BT=BT)
    assert o.shape == (B, H, L, D_v)
    assert not torch.isnan(o).any(), "Output contains NaN"
    assert not torch.isinf(o).any(), "Output contains Inf"
