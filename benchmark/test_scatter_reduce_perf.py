import pytest
import torch

from benchmark.performance_utils import GenericBenchmark2DOnly
from flag_gems.utils import shape_utils


def scatter_reduce_input_fn_factory(reduce):
    def inner(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        dim = -1
        size_dim = shape[dim]
        index = torch.randint(0, size_dim, shape, dtype=torch.long, device=device)
        src = torch.randn(shape, dtype=dtype, device=device)
        yield inp, dim, index, src, {"reduce": reduce}

    return inner


def scatter_reduce_gbps(bench_fn_args, latency):
    inp, dim, index, src = bench_fn_args[:4]
    io_amount = sum([shape_utils.size_in_bytes(item) for item in [index, src, inp]])
    return io_amount * 1e-9 / (latency * 1e-3)


class ScatterReduceBenchmark(GenericBenchmark2DOnly):
    def set_more_metrics(self):
        return ["gbps"]

    def get_gbps(self, args, latency):
        return scatter_reduce_gbps(args, latency)


@pytest.mark.scatter_reduce_
def test_perf_scatter_reduce_sum():
    bench = ScatterReduceBenchmark(
        op_name="scatter_reduce_.sum",
        torch_op=torch.Tensor.scatter_reduce_,
        input_fn=scatter_reduce_input_fn_factory("sum"),
        dtypes=[torch.float16, torch.float32],
        is_inplace=True,
    )
    bench.run()


@pytest.mark.scatter_reduce_
def test_perf_scatter_reduce_amax():
    bench = ScatterReduceBenchmark(
        op_name="scatter_reduce_.amax",
        torch_op=torch.Tensor.scatter_reduce_,
        input_fn=scatter_reduce_input_fn_factory("amax"),
        dtypes=[torch.float16, torch.float32],
        is_inplace=True,
    )
    bench.run()


@pytest.mark.scatter_reduce_
def test_perf_scatter_reduce_amin():
    bench = ScatterReduceBenchmark(
        op_name="scatter_reduce_.amin",
        torch_op=torch.Tensor.scatter_reduce_,
        input_fn=scatter_reduce_input_fn_factory("amin"),
        dtypes=[torch.float16, torch.float32],
        is_inplace=True,
    )
    bench.run()


@pytest.mark.scatter_reduce_
def test_perf_scatter_reduce_mean():
    bench = ScatterReduceBenchmark(
        op_name="scatter_reduce_.mean",
        torch_op=torch.Tensor.scatter_reduce_,
        input_fn=scatter_reduce_input_fn_factory("mean"),
        dtypes=[torch.float16, torch.float32],
        is_inplace=True,
    )
    bench.run()
