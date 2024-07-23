import torch

from .performance_utils import (
    FLOAT_DTYPES,
    INT_DTYPES,
    POINTWISE_BATCH,
    SIZES,
    Benchmark,
    binary_args,
    binary_int_args,
    ternary_args,
    unary_arg,
    unary_int_arg,
)


def test_perf_abs():
    bench = Benchmark(
        op_name="abs",
        torch_op=torch.abs,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_add():
    bench = Benchmark(
        op_name="add",
        torch_op=torch.add,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_bitwiseand():
    bench = Benchmark(
        op_name="bitwiseand_int",
        torch_op=torch.bitwise_and,
        arg_func=binary_int_args,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_bitwisenot():
    bench = Benchmark(
        op_name="bitwisenot_int",
        torch_op=torch.bitwise_not,
        arg_func=unary_int_arg,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_bitwiseor():
    bench = Benchmark(
        op_name="bitwiseor_int",
        torch_op=torch.bitwise_or,
        arg_func=binary_int_args,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_clamp():
    bench = Benchmark(
        op_name="clamp",
        torch_op=torch.clamp,
        arg_func=ternary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_cos():
    bench = Benchmark(
        op_name="cos",
        torch_op=torch.cos,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_div():
    bench = Benchmark(
        op_name="div",
        torch_op=torch.div,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_dropout():
    bench = Benchmark(
        op_name="dropout",
        torch_op=torch.nn.Dropout(p=0.5),
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_eq():
    bench = Benchmark(
        op_name="eq",
        torch_op=torch.eq,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_exp():
    bench = Benchmark(
        op_name="exp",
        torch_op=torch.exp,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_ge():
    bench = Benchmark(
        op_name="ge",
        torch_op=torch.ge,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_gelu():
    bench = Benchmark(
        op_name="gelu",
        torch_op=torch.nn.functional.gelu,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_gt():
    bench = Benchmark(
        op_name="gt",
        torch_op=torch.gt,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_isinf():
    bench = Benchmark(
        op_name="isinf",
        torch_op=torch.isinf,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_isnan():
    bench = Benchmark(
        op_name="isnan",
        torch_op=torch.isnan,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_le():
    bench = Benchmark(
        op_name="le",
        torch_op=torch.le,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_lt():
    bench = Benchmark(
        op_name="lt",
        torch_op=torch.lt,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_mul():
    bench = Benchmark(
        op_name="mul",
        torch_op=torch.mul,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_ne():
    bench = Benchmark(
        op_name="ne",
        torch_op=torch.ne,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_neg():
    bench = Benchmark(
        op_name="neg",
        torch_op=torch.neg,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_pow():
    bench = Benchmark(
        op_name="pow",
        torch_op=torch.pow,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_reciprocal():
    bench = Benchmark(
        op_name="reciprocal",
        torch_op=torch.reciprocal,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_relu():
    bench = Benchmark(
        op_name="relu",
        torch_op=torch.nn.functional.relu,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_rsqrt():
    bench = Benchmark(
        op_name="rsqrt",
        torch_op=torch.rsqrt,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_sigmoid():
    bench = Benchmark(
        op_name="sigmoid",
        torch_op=torch.sigmoid,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_silu():
    bench = Benchmark(
        op_name="silu",
        torch_op=torch.nn.functional.silu,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_sin():
    bench = Benchmark(
        op_name="sin",
        torch_op=torch.sin,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_sub():
    bench = Benchmark(
        op_name="sub",
        torch_op=torch.sub,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_tanh():
    bench = Benchmark(
        op_name="tanh",
        torch_op=torch.tanh,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_triu():
    bench = Benchmark(
        op_name="triu",
        torch_op=torch.triu,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_where():
    def where_args(dtype, batch, size):
        inp1 = torch.randn([batch, size], dtype=dtype, device="cuda")
        inp2 = torch.randn([batch, size], dtype=dtype, device="cuda")
        condition = inp1 > 0
        return condition, inp1, inp2

    bench = Benchmark(
        op_name="where",
        torch_op=torch.where,
        arg_func=where_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_isclose():
    bench = Benchmark(
        op_name="isclose",
        torch_op=torch.isclose,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_isclose_int():
    bench = Benchmark(
        op_name="isclose_int",
        torch_op=torch.isclose,
        arg_func=binary_int_args,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_allclose():
    bench = Benchmark(
        op_name="allclose",
        torch_op=torch.allclose,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_allclose_int():
    bench = Benchmark(
        op_name="allclose_int",
        torch_op=torch.allclose,
        arg_func=binary_int_args,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_erf():
    bench = Benchmark(
        op_name="erf",
        torch_op=torch.erf,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_isfinite():
    bench = Benchmark(
        op_name="isfinite",
        torch_op=torch.isfinite,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_isfinite_int():
    bench = Benchmark(
        op_name="isfinite_int",
        torch_op=torch.isfinite,
        arg_func=unary_int_arg,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_flip():
    def flip_kwargs(dtype, batch, size):
        return {"dims": [0, 1]}

    bench = Benchmark(
        op_name="flip",
        torch_op=torch.flip,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=flip_kwargs,
    )
    bench.run()


def test_perf_flip_int():
    def flip_kwargs(dtype, batch, size):
        return {"dims": [0, 1]}

    bench = Benchmark(
        op_name="flip",
        torch_op=torch.flip,
        arg_func=unary_int_arg,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=flip_kwargs,
    )
    bench.run()
