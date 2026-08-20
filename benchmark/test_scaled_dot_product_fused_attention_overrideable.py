import pytest
import torch

import flag_gems

from . import base, consts


class ScaledDotProductFusedAttentionOverrideableBenchmark(base.GenericBenchmark):
    """
    benchmark for _scaled_dot_product_fused_attention_overrideable
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


@pytest.mark.scaled_dot_product_fused_attention_overrideable
@pytest.mark.parametrize("is_causal", [False, True])
def test_scaled_dot_product_fused_attention_overrideable(is_causal):
    """Benchmark for _scaled_dot_product_fused_attention_overrideable."""

    def attention_kwargs(shape, dtype, device):
        # shape: (batch, num_heads, seq_len, head_size)
        query = torch.randn(shape, device=device, dtype=dtype)
        key = torch.randn(shape, device=device, dtype=dtype)
        value = torch.randn(shape, device=device, dtype=dtype)
        yield (query, key, value, None, 0.0, is_causal, False, None)

    def torch_ref(
        query,
        key,
        value,
        attn_bias=None,
        dropout_p=0.0,
        is_causal=False,
        return_debug_mask=False,
        scale=None,
    ):
        return flag_gems.ops.scaled_dot_product_attention_forward(
            query, key, value, is_causal=is_causal
        )

    def gems_op(
        query,
        key,
        value,
        attn_bias=None,
        dropout_p=0.0,
        is_causal=False,
        return_debug_mask=False,
        scale=None,
    ):
        return flag_gems._scaled_dot_product_fused_attention_overrideable(
            query, key, value, attn_bias, dropout_p, is_causal, return_debug_mask, scale
        )

    bench = ScaledDotProductFusedAttentionOverrideableBenchmark(
        op_name="scaled_dot_product_fused_attention_overrideable",
        input_fn=attention_kwargs,
        torch_op=torch_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(gems_op)
    bench.run()
