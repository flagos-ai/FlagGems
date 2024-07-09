import pytest
import torch

from .performance_utils import (
    FLOAT_DTYPES,
    POINTWISE_BATCH,
    SIZES,
    Benchmark,
    unary_arg,
)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_rand(dtype):
    def rand_kwargs(dtype, batch, size):
        return {"size": (batch, size), "dtype": dtype, "device": "cuda"}

    bench = Benchmark(
        op_name="rand",
        torch_op=torch.rand,
        arg_func=None,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=rand_kwargs,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_randn(dtype):
    def randn_kwargs(dtype, batch, size):
        return {"size": (batch, size), "dtype": dtype, "device": "cuda"}

    bench = Benchmark(
        op_name="randn",
        torch_op=torch.randn,
        arg_func=None,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=randn_kwargs,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_rand_like(dtype):
    bench = Benchmark(
        op_name="rand_like",
        torch_op=torch.rand_like,
        arg_func=unary_arg,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()
