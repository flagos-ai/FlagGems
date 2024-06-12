import torch
import pytest
import flag_gems
from .performance_utils import *


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_abs(dtype):
    bench = Benchmark(
        op_name="abs",
        torch_op=torch.abs,
        arg_func=unary_arg,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_add(dtype):
    bench = Benchmark(
        op_name="add",
        torch_op=torch.add,
        arg_func=binary_args,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_perf_bitwiseand(dtype):
    bench = Benchmark(
        op_name="bitwiseand",
        torch_op=torch.bitwise_and,
        arg_func=binary_int_args,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_perf_bitwisenot(dtype):
    bench = Benchmark(
        op_name="bitwisenot",
        torch_op=torch.bitwise_not,
        arg_func=unary_int_arg,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_perf_bitwiseor(dtype):
    bench = Benchmark(
        op_name="bitwiseor",
        torch_op=torch.bitwise_or,
        arg_func=binary_int_args,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_clamp(dtype):
    bench = Benchmark(
        op_name="clamp",
        torch_op=torch.clamp,
        arg_func=ternary_args,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_cos(dtype):
    bench = Benchmark(
        op_name="cos",
        torch_op=torch.cos,
        arg_func=unary_arg,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_div(dtype):
    bench = Benchmark(
        op_name="div",
        torch_op=torch.div,
        arg_func=binary_args,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_dropout(dtype):
    bench = Benchmark(
        op_name="dropout",
        torch_op=torch.nn.Dropout(p=0.5),
        arg_func=unary_arg,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_eq(dtype):
    bench = Benchmark(
        op_name="eq",
        torch_op=torch.eq,
        arg_func=binary_args,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_exp(dtype):
    bench = Benchmark(
        op_name="exp",
        torch_op=torch.exp,
        arg_func=unary_arg,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_ge(dtype):
    bench = Benchmark(
        op_name="ge",
        torch_op=torch.ge,
        arg_func=binary_args,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_gelu(dtype):
    bench = Benchmark(
        op_name="gelu",
        torch_op=torch.nn.functional.gelu,
        arg_func=unary_arg,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_gt(dtype):
    bench = Benchmark(
        op_name="gt",
        torch_op=torch.gt,
        arg_func=binary_args,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_isinf(dtype):
    bench = Benchmark(
        op_name="isinf",
        torch_op=torch.isinf,
        arg_func=unary_arg,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_isnan(dtype):
    bench = Benchmark(
        op_name="isnan",
        torch_op=torch.isnan,
        arg_func=unary_arg,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_le(dtype):
    bench = Benchmark(
        op_name="le",
        torch_op=torch.le,
        arg_func=binary_args,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_lt(dtype):
    bench = Benchmark(
        op_name="lt",
        torch_op=torch.lt,
        arg_func=binary_args,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_mul(dtype):
    bench = Benchmark(
        op_name="mul",
        torch_op=torch.mul,
        arg_func=binary_args,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_ne(dtype):
    bench = Benchmark(
        op_name="ne",
        torch_op=torch.ne,
        arg_func=binary_args,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_neg(dtype):
    bench = Benchmark(
        op_name="neg",
        torch_op=torch.neg,
        arg_func=unary_arg,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_pow(dtype):
    bench = Benchmark(
        op_name="pow",
        torch_op=torch.pow,
        arg_func=binary_args,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_reciprocal(dtype):
    bench = Benchmark(
        op_name="reciprocal",
        torch_op=torch.reciprocal,
        arg_func=unary_arg,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_relu(dtype):
    bench = Benchmark(
        op_name="relu",
        torch_op=torch.nn.functional.relu,
        arg_func=unary_arg,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_rsqrt(dtype):
    bench = Benchmark(
        op_name="rsqrt",
        torch_op=torch.rsqrt,
        arg_func=unary_arg,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_sigmoid(dtype):
    bench = Benchmark(
        op_name="sigmoid",
        torch_op=torch.sigmoid,
        arg_func=unary_arg,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_silu(dtype):
    bench = Benchmark(
        op_name="silu",
        torch_op=torch.nn.functional.silu,
        arg_func=unary_arg,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_sin(dtype):
    bench = Benchmark(
        op_name="sin",
        torch_op=torch.sin,
        arg_func=unary_arg,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_sub(dtype):
    bench = Benchmark(
        op_name="sub",
        torch_op=torch.sub,
        arg_func=binary_args,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_tanh(dtype):
    bench = Benchmark(
        op_name="tanh",
        torch_op=torch.tanh,
        arg_func=unary_arg,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_triu(dtype):
    bench = Benchmark(
        op_name="triu",
        torch_op=torch.triu,
        arg_func=unary_arg,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()

@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_where(dtype):
    bench = Benchmark(
        op_name="where",
        torch_op=torch.where,
        arg_func=where_args,
        dtype=dtype,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()