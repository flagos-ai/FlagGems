import torch

from .performance_utils import (
    FLOAT_DTYPES,
    INT_DTYPES,
    POINTWISE_BATCH,
    SIZES,
    Benchmark,
    unary_int_arg,
)


def test_perf_embedding():
    def embedding_kwargs(dtype, batch, size):
        input = torch.randint(0, batch, (batch,), device="cuda")
        weight = torch.randn((batch + 1, size), device="cuda", dtype=dtype)
        return {"input": input, "weight": weight}

    bench = Benchmark(
        op_name="embedding",
        torch_op=torch.nn.functional.embedding,
        arg_func=None,
        dtypes=[
            torch.float32,
            torch.float16,
        ],  # Note(Zhengzekang): triton do not support bfloat16 atomic add which is used in embedding grad.
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=embedding_kwargs,
    )
    bench.run()


def test_perf_topk():
    def topk_kwargs(dtype, batch, size):
        x = torch.randn((batch, size), device="cuda", dtype=dtype)
        return {"x": x, "k": 5, "dim": -1}

    bench = Benchmark(
        op_name="topk",
        torch_op=torch.topk,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=topk_kwargs,
    )
    bench.run()


def test_perf_resolve_neg():
    def resolve_neg_arg(dtype, batch, size):
        x = torch.randn(size=(batch, size), dtype=dtype, device="cuda")
        y = x.conj()
        z = y.imag
        return (z,)

    bench = Benchmark(
        op_name="resolve_neg",
        torch_op=torch.resolve_neg,
        arg_func=resolve_neg_arg,
        dtypes=[torch.cfloat],
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_resolve_conj():
    def resolve_conj_arg(dtype, batch, size):
        x = torch.randn(size=(size, batch), dtype=dtype, device="cuda")
        return (x.conj(),)

    bench = Benchmark(
        op_name="resolve_conj",
        torch_op=torch.resolve_conj,
        arg_func=resolve_conj_arg,
        dtypes=[torch.cfloat],
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_unique():
    def unique_kwargs(dtype, batch, size):
        return {"sorted": True, "return_inverse": True, "return_counts": False}

    bench = Benchmark(
        op_name="unique",
        torch_op=torch.unique,
        arg_func=unary_int_arg,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=unique_kwargs,
    )
    bench.run()
