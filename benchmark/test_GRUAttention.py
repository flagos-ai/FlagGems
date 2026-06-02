import pytest
import torch

from . import base

# GRUAttention benchmark
GRU_ATTENTION_SHAPES = [
    (1, 2, 4, 8),
    (2, 4, 8, 16),
    (4, 8, 16, 32),
    (8, 16, 32, 64),
]


class GRUAttentionBenchmark(base.Benchmark):
    """Benchmark for GRUAttention operator."""

    def __init__(self, op_name, torch_op, dtypes=None, **kwargs):
        super().__init__(op_name, torch_op, dtypes, **kwargs)
        self.scale = None  # Will be set based on shape

    def set_shapes(self, shape_file_path=None):
        self.shapes = GRU_ATTENTION_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            B, H, N, D = shape
            self.scale = 1.0 / (D**0.5)
            query = torch.randn(B, H, N, D, dtype=cur_dtype, device=self.device)
            key = torch.randn(B, H, N, D, dtype=cur_dtype, device=self.device)
            value = torch.randn(B, H, N, D, dtype=cur_dtype, device=self.device)
            yield query, key, value, self.scale

    def get_tflops(self, op, *args, **kwargs):
        # Attention is compute-intensive: 2 * B * H * N * N * D for QK and attn @ V
        q = args[0]
        B, H, N, D = q.shape
        return 2 * B * H * N * N * D


def torch_GRUAttention(q, k, v, scale=None):
    """Reference implementation using einsum."""
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    qk = torch.einsum("bhnd,bhmd->bhnm", q, k) * scale
    attn_weights = torch.softmax(qk, dim=-1)
    return torch.einsum("bhnm,bhmd->bhnd", attn_weights, v)


@pytest.mark.GRUAttention
def test_GRUAttention():
    bench = GRUAttentionBenchmark(
        op_name="GRUAttention",
        torch_op=torch_GRUAttention,
        # float32 and float16 are the supported dtypes for this kernel
        dtypes=[torch.float32, torch.float16],
    )
    bench.run()
