import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .accuracy_utils import gems_assert_close


@pytest.mark.scaled_dot_product_attention_math_for_mps
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("num_head", [4, 8])
@pytest.mark.parametrize("seq_len", [16, 32, 128])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("is_causal", [False, True])
def test_scaled_dot_product_attention_math_for_mps(
    batch, num_head, seq_len, head_dim, dtype, is_causal
):
    """Test accuracy of _scaled_dot_product_attention_math_for_mps."""
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

    # Reference: the math definition returns both the attention output and the
    # full softmax attention-weight matrix. The aten op only dispatches on
    # MPS/Meta, so compute the reference math directly.
    scale = 1.0 / (head_dim**0.5)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if is_causal:
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    ref_weights = torch.softmax(scores, dim=-1)
    ref_output = torch.matmul(ref_weights, v.float())
    ref_output = utils.to_reference(ref_output.to(dtype))
    ref_weights = utils.to_reference(ref_weights.to(dtype))

    with flag_gems.use_gems():
        output, attn_weights = flag_gems._scaled_dot_product_attention_math_for_mps(
            q, k, v, is_causal=is_causal
        )

    # output and attn_weights both reduce over the key/sequence dimension
    # (softmax normalization + weighted value sum), so scale atol accordingly.
    gems_assert_close(output, ref_output, dtype, reduce_dim=seq_len)
    gems_assert_close(attn_weights, ref_weights, dtype, reduce_dim=seq_len)
