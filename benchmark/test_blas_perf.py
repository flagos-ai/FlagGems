import torch
import pytest
import flag_gems
from .performance_utils import *


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_addmm(dtype):
    bench = Benchmark(
        op_name="addmm",
        torch_op=torch.addmm,
        arg_func=addmm_args,
        dtype=dtype,
        batch=DEFAULT_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_bmm(dtype):
    bench = Benchmark(
        op_name="bmm",
        torch_op=torch.bmm,
        arg_func=bmm_args,
        dtype=dtype,
        batch=BLAS_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_mm(dtype):
    bench = Benchmark(
        op_name="mm",
        torch_op=torch.mm,
        arg_func=mm_args,
        dtype=dtype,
        batch=DEFAULT_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_mv(dtype):
    bench = Benchmark(
        op_name="mv",
        torch_op=torch.mv,
        arg_func=mv_args,
        dtype=dtype,
        batch=BLAS_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_outer(dtype):
    bench = Benchmark(
        op_name="outer",
        torch_op=torch.outer,
        arg_func=outer_args,
        dtype=dtype,
        batch=DEFAULT_BATCH,
        sizes=SIZES,
    )
    bench.run()
