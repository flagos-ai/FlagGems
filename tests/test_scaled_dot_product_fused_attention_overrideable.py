import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .accuracy_utils import gems_assert_close


@pytest.mark.scaled_dot_product_fused_attention_overrideable
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("num_head", [4, 8])
@pytest.mark.parametrize("seq_len", [16, 32, 128])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("is_causal", [False, True])
def test_scaled_dot_product_fused_attention_overrideable(
    batch, num_head, seq_len, head_dim, dtype, is_causal
):
    """Test accuracy of _scaled_dot_product_fused_attention_overrideable."""
    device = flag_gems.device
    q = torch.randn(
        batch, num_head, seq_len, head_dim, dtype=dtype, device=device
    ).uniform_(-0.1, 0.1)
    k = torch.randn(
        batch, num_head, seq_len, head_dim, dtype=dtype, device=device
    ).uniform_(-0.1, 0.1)
    v = torch.randn(
        batch, num_head, seq_len, head_dim, dtype=dtype, device=device
    ).uniform_(-0.1, 0.1)

    # Reference from torch scaled_dot_product_attention
    ref_output = utils.to_reference(
        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    )

    # Get result from our implementation
    with flag_gems.use_gems():
        (
            output,
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            philox_seed,
            philox_offset,
            debug_attn_mask,
        ) = flag_gems._scaled_dot_product_fused_attention_overrideable(
            q, k, v, is_causal=is_causal
        )

    # Compare outputs
    gems_assert_close(output, ref_output, dtype)

    # Verify return values shape and content
    assert cum_seq_q.shape == (batch, 2)
    assert cum_seq_k.shape == (batch, 2)
    assert max_q == seq_len
    assert max_k == seq_len
    assert philox_seed.shape == (1,)
    assert philox_offset.shape == (1,)
