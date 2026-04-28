import pytest
import torch

from . import base, consts


class EmbeddingBenchmark(base.GenericBenchmark2DOnly):
    def set_more_shapes(self):
        # TODO: add more shapes
        return []


def embedding_input_fn(shape, dtype, device):
    num_embeddings, embedding_dim = shape
    indices = torch.randint(0, num_embeddings, (num_embeddings,), device=device)
    weight = torch.randn((num_embeddings, embedding_dim), device=device, dtype=dtype)
    yield {"input": indices, "weight": weight},

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        indices_2d = torch.randint(
            0,
            num_embeddings,
            (num_embeddings, num_embeddings),
            device=device,
        )

        yield {"input": indices_2d, "weight": weight},


def embedding_backward_input_fn(shape, dtype, device):
    for forward_args in embedding_input_fn(shape, dtype, device):
        input = forward_args[0]["input"]
        weight = forward_args[0]["weight"]

        weight.requires_grad_(True)
        # import pudb; pudb.set_trace()
        # output = torch.nn.functional.embedding(input, weight)
        # grad_output = torch.randn_like(output)
        yield input, weight


@pytest.mark.embedding
def test_embedding():
    # Note(Zhengzekang): triton do not support bfloat16 atomic add which is used in embedding grad.
    bench = EmbeddingBenchmark(
        input_fn=embedding_input_fn,
        op_name="embedding",
        torch_op=torch.nn.functional.embedding,
        dtypes=[
            torch.float32,
            torch.float16,
        ],
    )
    bench.run()


@pytest.mark.embedding_backward
def test_embedding_backward():
    # Note(Zhengzekang): triton do not support bfloat16 atomic add which is used in embedding grad.
    bench = EmbeddingBenchmark(
        input_fn=embedding_backward_input_fn,
        op_name="embedding_backward",
        torch_op=torch.nn.functional.embedding,
        dtypes=[
            torch.float32,
            torch.float16,
        ],
        is_backward=True,
    )
    bench.run()
