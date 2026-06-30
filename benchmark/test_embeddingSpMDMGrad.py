import pytest
import torch

import flag_gems

from . import base, consts

# Worktree original code: embeddingSpMDMGrad benchmark
EMBEDDING_SPMDM_GRAD_SHAPES = [
    (2, 4, 8, 16),
    (4, 8, 32, 64),
    (1, 3, 64, 128),
]


class EmbeddingSpMDMGradBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = EMBEDDING_SPMDM_GRAD_SHAPES


def embedding_sp_mdm_grad_input_fn(shape, dtype, device):
    B, M, D, num_weights = shape

    grad_output = torch.randn((B, M, D), device=device, dtype=dtype)
    indices = torch.randint(0, num_weights, (B, M), device=device, dtype=torch.long)

    # Test with different padding_idx values
    for padding_idx in [-1, 0]:
        yield grad_output, indices, num_weights, padding_idx


@pytest.mark.embeddingSpMDMGrad
def test_embeddingSpMDMGrad():
    bench = EmbeddingSpMDMGradBenchmark(
        input_fn=embedding_sp_mdm_grad_input_fn,
        op_name="embeddingSpMDMGrad",
        torch_op=flag_gems.embeddingSpMDMGrad,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
