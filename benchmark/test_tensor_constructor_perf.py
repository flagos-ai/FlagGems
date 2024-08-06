import torch

from .performance_utils import (
    FLOAT_DTYPES,
    POINTWISE_BATCH,
    SIZES,
    Benchmark,
    unary_arg,
)


def test_perf_ones():
    def ones_kwargs(dtype, batch, size):
        return {"size": (batch, size), "dtype": dtype, "device": "cuda"}

    bench = Benchmark(
        op_name="ones",
        torch_op=torch.ones,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=ones_kwargs,
    )
    bench.run()


def test_perf_zeros():
    def zeros_kwargs(dtype, batch, size):
        return {"size": (batch, size), "dtype": dtype, "device": "cuda"}

    bench = Benchmark(
        op_name="zeros",
        torch_op=torch.zeros,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=zeros_kwargs,
    )
    bench.run()


def test_perf_full():
    def full_kwargs(dtype, batch, size):
        return {
            "size": (batch, size),
            "fill_value": 3.1415926,
            "dtype": dtype,
            "device": "cuda",
        }

    bench = Benchmark(
        op_name="full",
        torch_op=torch.full,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=full_kwargs,
    )
    bench.run()


def test_perf_ones_like():
    bench = Benchmark(
        op_name="ones_like",
        torch_op=torch.ones_like,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_zeros_like():
    bench = Benchmark(
        op_name="zeros_like",
        torch_op=torch.zeros_like,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_full_like():
    def full_kwargs(dtype, batch, size):
        return {
            "input": torch.randn([batch, size], dtype=dtype, device="cuda"),
            "fill_value": 3.1415926,
        }

    bench = Benchmark(
        op_name="full_like",
        torch_op=torch.full_like,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=full_kwargs,
    )
    bench.run()
