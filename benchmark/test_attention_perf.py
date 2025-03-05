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
def test_perf_scaled_dot_product_attention():
    def scaled_dot_product_attention_kwargs(shape, dtype, device):
        query = torch.randn(shape, device=device, dtype=dtype)
        key = torch.randn(shape, device=device, dtype=dtype)
        value = torch.randn(shape, device=device, dtype=dtype)
        yield query, key, value, None, 0.0, True

    bench = AttentionBenchmark(
        op_name="scaled_dot_product_attention",
        input_fn=scaled_dot_product_attention_kwargs,
        torch_op=torch.nn.functional.scaled_dot_product_attention,
        dtypes=[
            torch.float16,
            torch.bfloat16,
        ],
    )
    bench.run()
