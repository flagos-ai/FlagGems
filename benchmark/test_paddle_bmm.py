from typing import Generator

import pytest
import flag_gems 
import torch

from benchmark.attri_util import (
    COMPLEX_DTYPES,
    DEFAULT_METRICS,
    FLOAT_DTYPES,
    PADDLE_FLOAT_DTYPES,
    BenchLevel,
    model_shapes,
)
from benchmark.conftest import Config
from benchmark.performance_utils import Benchmark, GenericBenchmark2DOnly, vendor_name


class BlasBenchmark(Benchmark):
    """
    benchmark for blas
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def get_input_iter(self, cur_dtype) -> Generator:
        for b, m, n, k in self.shapes:
            yield from self.input_fn(b, m, n, k, cur_dtype, self.device, False)

        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            for b, m, n, k in self.shapes:
                yield from self.input_fn(b, m, n, k, cur_dtype, self.device, True)

    def set_more_shapes(self):
        large_k_shapes = [
            (8, 1848, 1536, 151936),
            (8, 1848, 1536, 128256),
            (8, 1848, 1536, 152064),
        ]

        model_shaps = model_shapes()
        return large_k_shapes + model_shaps

    def get_tflops(self, op, *args, **kwargs):
        total_flops = 0
        # shape(m,k)(k,n)
        # total_flops mxnx2k
        if self.op_name == "mm":
            total_flops = args[0].shape[0] * args[0].shape[1] * args[1].shape[1] * 2
        # shape(m,n)(n,p)
        # total_flops mxpx(2n+1)
        elif self.op_name == "addmm":
            total_flops = (
                args[0].shape[0] * args[1].shape[1] * (args[1].shape[0] * 2 + 1)
            )
        # total_flops bxnxpx2m
        elif self.op_name == "bmm":
            total_flops = (
                args[0].shape[0]
                * args[0].shape[1]
                * args[1].shape[2]
                * 2
                * args[0].shape[2]
            )
        return total_flops



def addmm_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    bias = torch.randn([m, n], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=cur_dtype, device=device)
        yield bias, inp1, inp2.t(),
    else:
        inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
        yield bias, inp1, inp2,


def bmm_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([b, m, k], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([b, n, k], dtype=cur_dtype, device=device)
        yield inp1, inp2.transpose(1, 2)
    else:
        inp2 = torch.randn([b, k, n], dtype=cur_dtype, device=device)
        yield inp1, inp2

@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(
            "bmm",
            torch.bmm,
            bmm_input_fn,
            marks=pytest.mark.bmm,
        ),
    ],
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="temp disable for updating",
)
def test_blas_benchmark(op_name, torch_op, input_fn):
    bench = BlasBenchmark(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=PADDLE_FLOAT_DTYPES
    )
    bench.run()



