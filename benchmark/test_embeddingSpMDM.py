import pytest
import torch

import flag_gems

from . import base, consts

# embeddingSpMDM benchmark configurations
# shapes: (batch_size, embedding_dim, output_dim)
EMBEDDING_SPMDM_SHAPES = [
    (1, 64, 128),
    (8, 128, 256),
    (32, 256, 512),
    (64, 512, 1024),
    (128, 256, 128),
    (256, 512, 256),
]


class EmbeddingSpMDMBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = EMBEDDING_SPMDM_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            batch_size, embedding_dim, output_dim = shape
            num_embeddings = 10000  # vocab size

            weight = torch.randn(
                num_embeddings, embedding_dim, dtype=cur_dtype, device=self.device
            )
            indices = torch.randint(
                0, num_embeddings, (batch_size,), device=self.device
            )
            dense = torch.randn(
                embedding_dim, output_dim, dtype=cur_dtype, device=self.device
            )

            yield weight, indices, dense

    def get_tflops(self, op, *args, **kwargs):
        weight, indices, dense = args
        # FLOPs: 2 * batch_size * embedding_dim * output_dim (for matrix multiplication)
        batch_size = indices.shape[0]
        embedding_dim = weight.shape[1]
        output_dim = dense.shape[1]
        return 2 * batch_size * embedding_dim * output_dim


@pytest.mark.embeddingSpMDM
def test_embeddingSpMDM():
    bench = EmbeddingSpMDMBenchmark(
        op_name="embeddingSpMDM",
        torch_op=flag_gems.embeddingSpMDM,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
