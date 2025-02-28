import itertools
from typing import Generator

import pytest
import torch

import flag_gems

from .attri_util import (
    COMPLEX_DTYPES,
    DEFAULT_METRICS,
    FLOAT_DTYPES,
    BenchLevel,
    llama_shapes,
)
from .conftest import Config
from .performance_utils import Benchmark, vendor_name


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
            yield from self.input_fn(b, m, n, k, cur_dtype, self.device)
        # llama shapes
        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            for m, n, k, _ in llama_shapes():
                yield from self.input_fn(1, m, n, k, cur_dtype, self.device)

    def set_more_shapes(self):
        split_k_shapes = [
            (1, m, m, k)
            for m in [16 * i for i in range(1, 5)]
            for k in [4096 * i for i in range(1, 9)]
        ]
        # 'mv' operations only involve M and N dimensions.
        # Shapes with large K values are not suitable for these two operations.
        if self.op_name not in ["mv"]:
            # B=1 or 4, M= 13, N= 2 , K=2^6..2^15
            large_k_shapes = list(
                itertools.product([1, 4], [13], [2], [2**i for i in range(6, 15)])
            )
            return large_k_shapes + split_k_shapes
        return split_k_shapes

    def get_tflops(self, op, *args, **kwargs):
        """This method is currently not really implemented and serves as a placeholder.
        A proper implementation will be developed in the future."""
        total_flops = 0
        # shape(m,k)(k,n)
        # total_flops mxnx2k
        if self.op_name == "mm":
            total_flops = args[0].shape[0] * args[0].shape[1] * args[1].shape[1] * 2
        # shape(m,n)(n,p)
        # total_flops mxpx(2n+1)
        if self.op_name == "addmm":
            total_flops = (
                args[0].shape[0] * args[1].shape[1] * (args[1].shape[0] * 2 + 1)
            )
        # shape(b,n,m), (b,m,p)
        # total_flops bxnxpx2m
        if self.op_name == "bmm":
            total_flops = (
                args[0].shape[0]
                * args[0].shape[1]
                * args[1].shape[2]
                * 2
                * args[0].shape[2]
            )
        # shape(n,m)(m,)
        # total_flops n*2m
        if self.op_name == "mv":
            total_flops = args[0].shape[0] * 2 * args[0].shape[1]

        return total_flops


def addmm_input_fn(b, m, n, k, cur_dtype, device):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
    bias = torch.randn([m, n], dtype=cur_dtype, device=device)
    yield bias, inp1, inp2,


def bmm_input_fn(b, m, n, k, cur_dtype, device):
    inp1 = torch.randn([b, m, k], dtype=cur_dtype, device=device)
    inp2 = torch.randn([b, k, n], dtype=cur_dtype, device=device)
    yield inp1, inp2


def mm_input_fn(b, m, n, k, cur_dtype, device):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
    yield inp1, inp2


def mv_input_fn(b, m, n, k, cur_dtype, device):
    inp1 = torch.randn([m, n], dtype=cur_dtype, device=device)
    inp2 = torch.randn([n], dtype=cur_dtype, device=device)
    yield inp1, inp2


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(
            "addmm",
            torch.addmm,
            addmm_input_fn,
            marks=pytest.mark.addmm,
        ),
        pytest.param(
            "bmm",
            torch.bmm,
            bmm_input_fn,
            marks=pytest.mark.bmm,
        ),
        pytest.param(
            "mm",
            torch.Tensor.mm,
            mm_input_fn,
            marks=pytest.mark.mm,
        ),
        pytest.param(
            "mv",
            torch.Tensor.mv,
            mv_input_fn,
            marks=pytest.mark.mv,
        ),
    ],
)
def test_blas_benchmark(op_name, torch_op, input_fn):
    bench = BlasBenchmark(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=FLOAT_DTYPES
    )
    bench.run()


class OuterBenchmark(BlasBenchmark):
    """
    benchmark for outer
    """

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        for m, n in self.shapes:
            yield from self.input_fn(m, n, cur_dtype, self.device)


@pytest.mark.outer
def test_outer_benchmark():
    def outer_input_fn(m, n, cur_dtype, device):
        inp1 = torch.randn([m], dtype=cur_dtype, device=device)
        inp2 = torch.randn([n], dtype=cur_dtype, device=device)
        yield inp1, inp2

    bench = OuterBenchmark(
        input_fn=outer_input_fn,
        op_name="outer",
        torch_op=torch.Tensor.outer,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


class VdotBenchmark(BlasBenchmark):
    """
    benchmark for vdot
    """

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            m = shape[0]
            yield from self.input_fn(m, cur_dtype, self.device)


@pytest.mark.skipif(vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.skipif(flag_gems.device == "musa", reason="Segmentation fault")
@pytest.mark.vdot
def test_vdot_benchmark():
    def vdot_input_fn(m, cur_dtype, device):
        inp1 = torch.randn([m], dtype=cur_dtype, device=device)
        inp2 = torch.randn([m], dtype=cur_dtype, device=device)
        yield inp1, inp2

    bench = VdotBenchmark(
        input_fn=vdot_input_fn,
        op_name="vdot",
        torch_op=torch.Tensor.vdot,
        dtypes=COMPLEX_DTYPES + FLOAT_DTYPES,
    )
    bench.run()
