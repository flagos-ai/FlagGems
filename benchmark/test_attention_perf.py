import pytest
import torch

import flag_gems

from .performance_utils import GenericBenchmark, vendor_name


class AttentionBenchmark(GenericBenchmark):
    """
    benchmark for attention
    """

    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (batch, num_heads, seq_len, head_size).
        return None


@pytest.mark.skipif(vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.skipif(
    flag_gems.device == "musa" or vendor_name == "hygon", reason="RuntimeError"
)
@pytest.mark.attention
@pytest.mark.parametrize("dropout_p", [0.0, 0.25])
@pytest.mark.parametrize("is_causal", [True, False])
def test_perf_scaled_dot_product_attention(dropout_p, is_causal):
    def scaled_dot_product_attention_kwargs(shape, dtype, device):
        query = torch.randn(shape, device=device, dtype=dtype)
        key = torch.randn(shape, device=device, dtype=dtype)
        value = torch.randn(shape, device=device, dtype=dtype)
        yield query, key, value, dropout_p, is_causal

    def sdpa_flash(query, key, value, dropout_p=dropout_p, is_causal=is_causal):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

    bench = AttentionBenchmark(
        op_name="scaled_dot_product_attention",
        input_fn=scaled_dot_product_attention_kwargs,
        # torch_op=torch.nn.functional.scaled_dot_product_attention,
        torch_op=sdpa_flash,
        dtypes=[
            torch.float16,
            torch.bfloat16,
        ],
    )
    bench.run()
