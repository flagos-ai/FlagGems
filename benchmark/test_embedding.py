import pytest
import torch

from .attri_util import FLOAT_DTYPES, BenchLevel
from .performance_utils import Config, GenericBenchmark2DOnly


class EmbeddingBenchmark(GenericBenchmark2DOnly):
    def set_more_shapes(self):
        return None


def embedding_backward_input_fn(shape, dtype, device):
    num_embeddings, embedding_dim = shape
    indices = torch.randint(0, num_embeddings, (num_embeddings,), device=device)
    weight = torch.randn((num_embeddings, embedding_dim), device=device, dtype=dtype)
    yield indices, weight

    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        indices_2d = torch.randint(
            0,
            num_embeddings,
            (num_embeddings, num_embeddings),
            device=device,
        )
        yield indices_2d, weight


@pytest.mark.embedding_backward
def test_embedding_backward():
    bench = EmbeddingBenchmark(
        input_fn=embedding_backward_input_fn,
        op_name="embedding_backward",
        torch_op=torch.nn.functional.embedding,
        dtypes=[torch.float32, torch.float16],
        is_backward=True,
    )
    bench.run()
