import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.CrossAttention
@pytest.mark.parametrize(
    "batch, num_head, q_seq_len, kv_seq_len, head_size",
    [
        (2, 4, 64, 64, 64),
        (2, 4, 128, 128, 64),
        (2, 4, 64, 128, 64),
        (2, 4, 128, 64, 64),
        (1, 2, 32, 32, 32),
    ],
)
# Only float32 used due to memory constraints with large attention shapes
@pytest.mark.parametrize("dtype", [torch.float32])
def test_CrossAttention(batch, num_head, q_seq_len, kv_seq_len, head_size, dtype):
    """Test CrossAttention operator against PyTorch reference."""
    # Create inputs with different Q and KV sequence lengths
    q = torch.randn(
        batch, num_head, q_seq_len, head_size, dtype=dtype, device=flag_gems.device
    )
    k = torch.randn(
        batch, num_head, kv_seq_len, head_size, dtype=dtype, device=flag_gems.device
    )
    v = torch.randn(
        batch, num_head, kv_seq_len, head_size, dtype=dtype, device=flag_gems.device
    )

    # Reference implementation using scaled_dot_product_attention
    # (which supports different Q and KV sequences)
    q_ref, k_ref, v_ref = (
        utils.to_reference(q, True),
        utils.to_reference(k, True),
        utils.to_reference(v, True),
    )
    scale = 1.0 / (head_size**0.5)
    ref_out = torch.nn.functional.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, scale=scale
    )

    # Test our implementation
    with flag_gems.use_gems():
        res_out = flag_gems.CrossAttention(q, k, v, scale=scale)

    utils.gems_assert_close(res_out, ref_out, dtype)
