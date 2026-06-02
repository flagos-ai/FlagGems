import pytest
import torch

from . import base

# CrossAttention benchmark
CROSS_ATTENTION_SHAPES = [
    # (batch, num_heads, seq_len_q, seq_len_kv, head_dim)
    (2, 4, 64, 64, 64),
    (2, 4, 128, 128, 64),
    (2, 8, 256, 256, 64),
    (4, 8, 512, 512, 64),
]


class CrossAttentionBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CROSS_ATTENTION_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            batch, num_heads, seq_q, seq_kv, head_dim = shape
            q = torch.randn(
                batch, num_heads, seq_q, head_dim, dtype=cur_dtype, device=self.device
            )
            k = torch.randn(
                batch, num_heads, seq_kv, head_dim, dtype=cur_dtype, device=self.device
            )
            v = torch.randn(
                batch, num_heads, seq_kv, head_dim, dtype=cur_dtype, device=self.device
            )
            scale = 1.0 / (head_dim**0.5)
            yield q, k, v, scale


@pytest.mark.CrossAttention
def test_CrossAttention():
    def CrossAttention_torch_op(q, k, v, scale):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)

    bench = CrossAttentionBenchmark(
        op_name="CrossAttention",
        torch_op=CrossAttention_torch_op,
        # float32 only: attention kernels are memory-intensive with large shapes
        dtypes=[torch.float32],
    )
    bench.run()
