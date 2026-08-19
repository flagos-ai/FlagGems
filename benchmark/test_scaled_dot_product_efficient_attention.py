import pytest
import torch

import flag_gems

from . import base, consts

# Attention benchmark shapes: (batch, heads, seq_len, head_dim)
# Cover small-to-medium configurations typical in unit and integration tests
ATTENTION_BENCHMARK_SHAPES = [
    (1, 2, 8, 16),
    (2, 4, 16, 32),
    (4, 8, 32, 64),
    (8, 16, 64, 64),
]


class ScaledDotProductEfficientAttentionBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = ATTENTION_BENCHMARK_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            batch, num_heads, seq_len, head_dim = shape
            query = torch.randn(
                batch, num_heads, seq_len, head_dim, dtype=cur_dtype, device=self.device
            )
            key = torch.randn(
                batch, num_heads, seq_len, head_dim, dtype=cur_dtype, device=self.device
            )
            value = torch.randn(
                batch, num_heads, seq_len, head_dim, dtype=cur_dtype, device=self.device
            )
            yield query, key, value, None, False


@pytest.mark.scaled_dot_product_efficient_attention
def test_scaled_dot_product_efficient_attention():
    bench = ScaledDotProductEfficientAttentionBenchmark(
        op_name="scaled_dot_product_efficient_attention",
        torch_op=flag_gems._scaled_dot_product_efficient_attention,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
