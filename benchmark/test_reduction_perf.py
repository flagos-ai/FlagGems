import torch
import pytest
import flag_gems
from .performance_utils import *


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_all(dtype):
    bench = Benchmark(
        op_name="all",
        torch_op=torch.all,
        arg_func=unary_arg,
        dtype=dtype,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_amax(dtype):
    bench = Benchmark(
        op_name="amax",
        torch_op=torch.amax,
        arg_func=unary_arg,
        dtype=dtype,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_any(dtype):
    bench = Benchmark(
        op_name="any",
        torch_op=torch.any,
        arg_func=unary_arg,
        dtype=dtype,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_argmax(dtype):
    bench = Benchmark(
        op_name="argmax",
        torch_op=torch.argmax,
        arg_func=unary_arg,
        dtype=dtype,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_cross_entropy_loss(dtype):
    bench = Benchmark(
        op_name="cross_entropy_loss",
        torch_op=torch.nn.CrossEntropyLoss(),
        arg_func=cross_entropy_loss_args,
        dtype=dtype,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_cumsum(dtype):
    bench = Benchmark(
        op_name="cumsum",
        torch_op=torch.cumsum,
        arg_func=cumsum_args,
        dtype=dtype,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_groupnorm(dtype):
    bench = Benchmark(
        op_name="groupnorm",
        torch_op=torch.nn.functional.group_norm,
        arg_func=group_norm_args,
        dtype=dtype,
        batch=BLAS_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_layernorm(dtype):
    bench = Benchmark(
        op_name="layernorm",
        torch_op=torch.layer_norm,
        arg_func=layer_norm_args,
        dtype=dtype,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_log_softmax(dtype):
    bench = Benchmark(
        op_name="log_softmax",
        torch_op=torch.nn.functional.log_softmax,
        arg_func=unary_arg,
        dtype=dtype,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_max(dtype):
    bench = Benchmark(
        op_name="max",
        torch_op=torch.max,
        arg_func=unary_arg,
        dtype=dtype,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_mean(dtype):
    bench = Benchmark(
        op_name="mean",
        torch_op=torch.mean,
        arg_func=unary_arg,
        dtype=dtype,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_min(dtype):
    bench = Benchmark(
        op_name="min",
        torch_op=torch.min,
        arg_func=unary_arg,
        dtype=dtype,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_prod(dtype):
    bench = Benchmark(
        op_name="prod",
        torch_op=torch.prod,
        arg_func=unary_arg,
        dtype=dtype,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_softmax(dtype):
    bench = Benchmark(
        op_name="softmax",
        torch_op=torch.nn.functional.softmax,
        arg_func=unary_arg,
        dtype=dtype,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_sum(dtype):
    bench = Benchmark(
        op_name="sum",
        torch_op=torch.sum,
        arg_func=unary_arg,
        dtype=dtype,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_var_mean(dtype):
    bench = Benchmark(
        op_name="var_mean",
        torch_op=torch.var_mean,
        arg_func=unary_arg,
        dtype=dtype,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_vector_norm(dtype):
    bench = Benchmark(
        op_name="vector_norm",
        torch_op=torch.linalg.vector_norm,
        arg_func=unary_arg,
        dtype=dtype,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()
