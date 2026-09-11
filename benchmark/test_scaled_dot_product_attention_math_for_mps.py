import pytest
import torch

import flag_gems

from . import base, consts


class ScaledDotProductAttentionMathForMpsBenchmark(base.GenericBenchmark):
    """
    benchmark for _scaled_dot_product_attention_math_for_mps
    """

    # Attention requires 4D shapes (B, H, S, D) — override generic 1D/2D/3D defaults
    DEFAULT_SHAPES = [
        (2, 8, 64, 64),
        (4, 8, 128, 64),
        (2, 8, 128, 128),
        (1, 16, 256, 64),
    ]

    def set_more_shapes(self):
        return None


@pytest.mark.scaled_dot_product_attention_math_for_mps
@pytest.mark.parametrize("is_causal", [False, True])
def test_scaled_dot_product_attention_math_for_mps(is_causal):
    """Benchmark for _scaled_dot_product_attention_math_for_mps."""

    def attention_kwargs(shape, dtype, device):
        # shape: (batch, num_heads, seq_len, head_size)
        query = torch.randn(shape, device=device, dtype=dtype)
        key = torch.randn(shape, device=device, dtype=dtype)
        value = torch.randn(shape, device=device, dtype=dtype)
        yield (query, key, value, None, 0.0, is_causal, None)

    def torch_ref(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        dropout_mask=None,
        scale=None,
    ):
        head_dim = query.shape[-1]
        scale_factor = 1.0 / (head_dim**0.5) if scale is None else scale
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        if is_causal:
            seq_q, seq_k = query.shape[-2], key.shape[-2]
            mask = torch.ones(seq_q, seq_k, dtype=torch.bool, device=query.device).tril(
                diagonal=seq_k - seq_q
            )
            scores = scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights

    def gems_op(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        dropout_mask=None,
        scale=None,
    ):
        return flag_gems._scaled_dot_product_attention_math_for_mps(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            dropout_mask,
            scale=scale,
        )

    bench = ScaledDotProductAttentionMathForMpsBenchmark(
        op_name="scaled_dot_product_attention_math_for_mps",
        input_fn=attention_kwargs,
        torch_op=torch_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(gems_op)
    bench.run()
