import pytest
import torch

import flag_gems

from . import base, consts


class EmbeddingDenseBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (32, 2048, 128, 8192),
            (16, 2048, 256, 16384),
            (8, 4096, 256, 32768),
        ]


def _input_fn(shape, dtype, device):
    B, M, D, num_weights = shape

    grad_output = torch.randn((B, M, D), device=device, dtype=dtype)
    indices = torch.randint(0, num_weights, (B, M), device=device, dtype=torch.long)

    def inject_padding_idx(cur_indices: torch.Tensor, padding_idx: int) -> torch.Tensor:
        if padding_idx < 0:
            return cur_indices
        mask = torch.rand((B, M), device=device) < 0.25
        return torch.where(mask, torch.full_like(cur_indices, padding_idx), cur_indices)

    test_cases = [(-1, False), (0, True), (5, False)]
    for padding_idx, scale_grad_by_freq in test_cases:
        cur_indices = inject_padding_idx(indices, padding_idx)
        yield grad_output, cur_indices, num_weights, padding_idx, scale_grad_by_freq


@pytest.mark.skipif(
    (not torch.cuda.is_available()) or (flag_gems.device != "cuda"),
    reason="CUDA backend is not available for this benchmark.",
)
@pytest.mark.embedding_dense_backward
def test_embedding_dense_backward():
    bench = EmbeddingDenseBackwardBenchmark(
        input_fn=_input_fn,
        op_name="embedding_dense_backward",
        torch_op=torch.ops.aten.embedding_dense_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
