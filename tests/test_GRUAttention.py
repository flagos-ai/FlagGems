import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# GRUAttention test - scaled dot-product attention
ATTENTION_SHAPES = [(1, 2, 4, 8), (2, 4, 8, 16)]


@pytest.mark.GRUAttention
@pytest.mark.parametrize("shape", ATTENTION_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_GRUAttention(shape, dtype):
    """Test for GRUAttention operator.

    This implements an attention mechanism:
    - Computes attention scores between query and key
    - Applies softmax to get attention weights
    - Returns weighted sum of values
    """
    B, H, N, D = shape

    # Create query, key, value tensors
    query = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    key = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    value = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # Reference implementation using standard attention with scaling
    ref_query = utils.to_reference(query)
    ref_key = utils.to_reference(key)
    ref_value = utils.to_reference(value)

    scale = 1.0 / (D**0.5)

    # Compute attention manually (reference)
    # q @ k^T: (B, H, N, D) @ (B, H, D, N) -> (B, H, N, N)
    ref_qk = torch.einsum("bhnd,bhmd->bhnm", ref_query, ref_key) * scale
    ref_attn_weights = torch.softmax(ref_qk, dim=-1)
    ref_out = torch.einsum("bhnm,bhmd->bhnd", ref_attn_weights, ref_value)

    # Run with FlagGems
    with flag_gems.use_gems():
        res_out = flag_gems.GRUAttention(query, key, value, scale=scale)

    # Compare outputs
    utils.gems_assert_close(res_out, ref_out, dtype)
