import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Small shapes for attention accuracy tests:
# (batch, heads, seq_len, head_dim) combinations covering common
# attention patterns from single-batch single-head to multi-batch multi-head
ATTENTION_SHAPES = [
    (1, 2, 8, 16),
    (2, 4, 16, 32),
    (4, 8, 32, 64),
]


@pytest.mark.scaled_dot_product_efficient_attention
@pytest.mark.parametrize("shape", ATTENTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_scaled_dot_product_efficient_attention(shape, dtype):
    batch, num_heads, seq_len, head_dim = shape

    query = torch.randn(
        batch, num_heads, seq_len, head_dim, dtype=dtype, device=flag_gems.device
    )
    key = torch.randn(
        batch, num_heads, seq_len, head_dim, dtype=dtype, device=flag_gems.device
    )
    value = torch.randn(
        batch, num_heads, seq_len, head_dim, dtype=dtype, device=flag_gems.device
    )

    ref_out = utils.to_reference(
        torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=False
        )
    )

    with flag_gems.use_gems():
        res_out, _, _, _ = flag_gems._scaled_dot_product_efficient_attention(
            query, key, value, None, False
        )

    utils.gems_assert_close(res_out, ref_out, dtype, atol=3e-4)


@pytest.mark.scaled_dot_product_efficient_attention
@pytest.mark.parametrize("shape", ATTENTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_scaled_dot_product_efficient_attention_causal(shape, dtype):
    batch, num_heads, seq_len, head_dim = shape

    query = torch.randn(
        batch, num_heads, seq_len, head_dim, dtype=dtype, device=flag_gems.device
    )
    key = torch.randn(
        batch, num_heads, seq_len, head_dim, dtype=dtype, device=flag_gems.device
    )
    value = torch.randn(
        batch, num_heads, seq_len, head_dim, dtype=dtype, device=flag_gems.device
    )

    ref_out = utils.to_reference(
        torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )
    )

    with flag_gems.use_gems():
        res_out, _, _, _ = flag_gems._scaled_dot_product_efficient_attention(
            query, key, value, None, False, is_causal=True
        )

    utils.gems_assert_close(res_out, ref_out, dtype, atol=3e-4)


@pytest.mark.scaled_dot_product_efficient_attention
@pytest.mark.parametrize("shape", ATTENTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_scaled_dot_product_efficient_attention_logsumexp(shape, dtype):
    batch, num_heads, seq_len, head_dim = shape

    query = torch.randn(
        batch, num_heads, seq_len, head_dim, dtype=dtype, device=flag_gems.device
    )
    key = torch.randn(
        batch, num_heads, seq_len, head_dim, dtype=dtype, device=flag_gems.device
    )
    value = torch.randn(
        batch, num_heads, seq_len, head_dim, dtype=dtype, device=flag_gems.device
    )

    ref_out = utils.to_reference(
        torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=False
        )
    )

    with flag_gems.use_gems():
        (
            res_out,
            res_log_sumexp,
            _,
            _,
        ) = flag_gems._scaled_dot_product_efficient_attention(
            query, key, value, None, True
        )

    utils.gems_assert_close(res_out, ref_out, dtype, atol=3e-4)
