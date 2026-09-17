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


def _get_atol_for_dtype(dtype):
    if dtype == torch.bfloat16:
        return 2e-3
    else:
        return 3e-4


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
        res_out, _, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention(
            query,
            key,
            value,
            attn_bias=None,
            compute_log_sumexp=False,
            dropout_p=0.0,
            is_causal=False,
        )

    utils.gems_assert_close(res_out, ref_out, dtype, atol=_get_atol_for_dtype(dtype))


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
        res_out, _, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention(
            query,
            key,
            value,
            attn_bias=None,
            compute_log_sumexp=False,
            dropout_p=0.0,
            is_causal=True,
        )

    utils.gems_assert_close(res_out, ref_out, dtype, atol=_get_atol_for_dtype(dtype))


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

    # Reference log_sumexp from the native aten op (run on device, outside
    # `use_gems()`). It is padded to ceil(seq_len / 32) * 32, so slice to
    # seq_len before comparing.
    _, ref_log_sumexp, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention(
        query,
        key,
        value,
        attn_bias=None,
        compute_log_sumexp=True,
        dropout_p=0.0,
        is_causal=False,
    )
    ref_log_sumexp = ref_log_sumexp[:, :, :seq_len]
    ref_log_sumexp = utils.to_reference(ref_log_sumexp)

    with flag_gems.use_gems():
        (
            res_out,
            res_log_sumexp,
            _,
            _,
        ) = torch.ops.aten._scaled_dot_product_efficient_attention(
            query,
            key,
            value,
            attn_bias=None,
            compute_log_sumexp=True,
            dropout_p=0.0,
            is_causal=False,
        )

    utils.gems_assert_close(res_out, ref_out, dtype, atol=_get_atol_for_dtype(dtype))
    # log_sumexp is always float32 regardless of input dtype.
    utils.gems_assert_close(
        res_log_sumexp, ref_log_sumexp, torch.float32, atol=_get_atol_for_dtype(dtype)
    )
