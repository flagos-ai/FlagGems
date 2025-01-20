import numpy as np
import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference


@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize("batch", [8, 16])
@pytest.mark.parametrize("num_head", [1, 8])
@pytest.mark.parametrize("q_seq_len", [17, 64, 128])
@pytest.mark.parametrize("kv_seq_len", [128, 2048])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("add_bias", [True, False])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_dot_product_attention(
    batch, num_head, q_seq_len, kv_seq_len, head_size, add_bias, is_causal, dtype
):
    np.random.seed(0)
    np_query = np.random.uniform(
        -0.05, 0.05, (batch, num_head, q_seq_len, head_size)
    ).astype(np.float32)
    np_key = np.random.uniform(
        -0.05, 0.05, (batch, num_head, kv_seq_len, head_size)
    ).astype(np.float32)
    np_value = np.random.uniform(
        -0.05, 0.05, (batch, num_head, kv_seq_len, head_size)
    ).astype(np.float32)
    np_attn_bias = np.random.uniform(
        -0.05, 0.05, (batch, num_head, q_seq_len, kv_seq_len)
    ).astype(np.float32)

    query = torch.tensor(np_query, device="cuda", dtype=dtype)
    key = torch.tensor(np_key, device="cuda", dtype=dtype)
    value = torch.tensor(np_value, device="cuda", dtype=dtype)
    if add_bias:
        attn_bias = torch.tensor(np_attn_bias, device="cuda", dtype=dtype)
    else:
        attn_bias = None

    ref_query = to_reference(query, False)
    ref_key = to_reference(key, False)
    ref_value = to_reference(value, False)
    ref_attn_bias = to_reference(attn_bias, False) if add_bias else None

    scale = float(1.0 / np.sqrt(head_size))

    if is_causal:
        torch_result = torch.nn.functional.scaled_dot_product_attention(
            ref_query,
            ref_key,
            ref_value,
            scale=scale,
            is_causal=is_causal,
        )
    else:
        torch_result = torch.nn.functional.scaled_dot_product_attention(
            ref_query,
            ref_key,
            ref_value,
            attn_mask=ref_attn_bias,
            scale=scale,
            is_causal=is_causal,
        )

    with flag_gems.use_gems():
        if is_causal:
            flaggem_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, scale=scale, is_causal=is_causal
            )
        else:
            flaggem_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_bias, scale=scale, is_causal=is_causal
            )
    gems_assert_close(flaggem_result, torch_result, dtype)
