import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

device = flag_gems.device

# torch._scaled_dot_product_attention_math is exposed as an aten dispatch op;
# call it via torch.ops.aten so flag_gems.use_gems() can intercept it and to
# avoid depending on the private torch._... alias across PyTorch versions.
sdpa_math = torch.ops.aten._scaled_dot_product_attention_math


# Cases cover: (1) equal q/kv seq lens with and without causal masking, (2)
# differing q/kv seq lens, (3) an explicit float attn_mask, and (4) custom
# scale values. attn_mask is only used when is_causal=False because the aten
# op rejects attn_mask+is_causal=True.
@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.parametrize(
    "batch, num_head, q_seq_len, kv_seq_len, head_size, use_mask, is_causal, scale",
    [
        # equal q/kv seq lens, both causal settings
        (2, 4, 8, 8, 16, False, False, None),
        (2, 4, 8, 8, 16, False, True, None),
        (2, 4, 32, 32, 32, False, False, None),
        (2, 4, 32, 32, 32, False, True, None),
        (4, 8, 64, 64, 64, False, False, None),
        (4, 8, 64, 64, 64, False, True, None),
        (2, 4, 128, 128, 128, False, False, None),
        (2, 4, 128, 128, 128, False, True, None),
        (1, 2, 256, 256, 64, False, False, None),
        (1, 2, 256, 256, 64, False, True, None),
        # differing q/kv seq lens (causal=False only)
        (2, 4, 32, 64, 32, False, False, None),
        (4, 8, 64, 128, 64, False, False, None),
        (2, 4, 128, 64, 64, False, False, None),
        # explicit float attn_mask
        (2, 4, 32, 32, 32, True, False, None),
        (4, 8, 64, 64, 64, True, False, None),
        # custom scale values
        (2, 4, 32, 32, 32, False, False, 0.5),
        (2, 4, 32, 32, 32, False, False, 1.0),
        (2, 4, 32, 32, 32, False, False, 2.0),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_scaled_dot_product_attention_math(
    batch,
    num_head,
    q_seq_len,
    kv_seq_len,
    head_size,
    use_mask,
    is_causal,
    scale,
    dtype,
):
    utils.init_seed(1234567890)

    q = torch.randn(batch, num_head, q_seq_len, head_size, dtype=dtype, device=device)
    k = torch.randn(batch, num_head, kv_seq_len, head_size, dtype=dtype, device=device)
    v = torch.randn(batch, num_head, kv_seq_len, head_size, dtype=dtype, device=device)
    attn_mask = (
        torch.randn(q_seq_len, kv_seq_len, dtype=dtype, device=device)
        if use_mask
        else None
    )

    ref_q = utils.to_reference(q)
    ref_k = utils.to_reference(k)
    ref_v = utils.to_reference(v)
    ref_mask = utils.to_reference(attn_mask)

    ref_out, ref_weights = sdpa_math(
        ref_q, ref_k, ref_v, attn_mask=ref_mask, is_causal=is_causal, scale=scale
    )

    with flag_gems.use_gems():
        res_out, res_weights = sdpa_math(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale
        )

    utils.gems_assert_close(res_out, ref_out, dtype, atol=3e-2)
    utils.gems_assert_close(res_weights, ref_weights, dtype, atol=3e-2)
