import torch

from .performance_utils import (
    FLOAT_DTYPES,
    POINTWISE_BATCH,
    SIZES,
    Benchmark,
    unary_arg,
)


def test_perf_rand():
    def rand_kwargs(dtype, batch, size):
        return {"size": (batch, size), "dtype": dtype, "device": "cuda"}

    bench = Benchmark(
        op_name="rand",
        torch_op=torch.rand,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=rand_kwargs,
    )
    bench.run()


def test_perf_randn():
    def randn_kwargs(dtype, batch, size):
        return {"size": (batch, size), "dtype": dtype, "device": "cuda"}

    bench = Benchmark(
        op_name="randn",
        torch_op=torch.randn,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=randn_kwargs,
    )
    bench.run()


def test_perf_rand_like():
    bench = Benchmark(
        op_name="rand_like",
        torch_op=torch.rand_like,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_normal():
    def normal_arg(dtype, batch, size):
        loc = torch.full(size=(size, batch), fill_value=3.0, dtype=dtype, device="cuda")
        scale = torch.full(
            size=(size, batch), fill_value=10.0, dtype=dtype, device="cuda"
        )
        return loc, scale

    bench = Benchmark(
        op_name="distributions.normal.Normal",
        torch_op=torch.distributions.normal.Normal,
        arg_func=normal_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_uniform():
    bench = Benchmark(
        op_name="uniform_",
        torch_op=torch.Tensor.uniform_,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_exponential_():
    bench = Benchmark(
        op_name="exponential_",
        torch_op=torch.Tensor.exponential_,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()
