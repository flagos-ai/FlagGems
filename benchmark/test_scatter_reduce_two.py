import pytest
import torch

import flag_gems
from flag_gems.utils import shape_utils

from . import base, consts

FLOAT16_FLOAT32_DTYPES = [torch.float16, torch.float32]


class ScatterReducePublicBenchmark(base.GenericBenchmark2DOnly):
    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        if flag_gems.vendor_name == "kunlunxin":
            return []

        shapes = super().set_more_shapes()
        return [
            shape
            for shape in shapes
            if len(shape) == 2 and shape[0] > 16 and shape[1] > 16
        ]


def scatter_reduce_two_input_fn_factory(reduce, include_self=True, out=False):
    def inner(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        dim = -1
        src_shape = list(shape)
        src_shape[dim] = max(1, src_shape[dim] // 2)
        src = torch.randn(src_shape, dtype=dtype, device=device)
        index = torch.randint(0, shape[dim], src_shape, dtype=torch.long, device=device)
        kwargs = {"reduce": reduce, "include_self": include_self}
        if out:
            kwargs["out"] = torch.empty_like(inp)
        yield inp, dim, index, src, kwargs

    return inner


def scatter_reduce_bench_dtypes(reduce):
    if reduce in ("sum", "mean"):
        return FLOAT16_FLOAT32_DTYPES
    return consts.FLOAT_DTYPES


def gather_scatter_gbps(bench_fn_args, latency):
    inp, dim, index = bench_fn_args[:3]
    data_shape = list(inp.shape)
    data_shape[dim] = index.shape[dim]
    data = torch.empty(data_shape, dtype=inp.dtype, device=inp.device)
    io_amount = sum([shape_utils.size_in_bytes(item) for item in [index, data, data]])
    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize(
    "reduce, include_self",
    [("sum", True), ("mean", False), ("prod", True), ("amax", False), ("amin", True)],
)
def test_scatter_reduce_two(reduce, include_self):
    bench = ScatterReducePublicBenchmark(
        op_name="scatter_reduce_two",
        torch_op=torch.scatter_reduce,
        input_fn=scatter_reduce_two_input_fn_factory(reduce, include_self),
        get_gbps=gather_scatter_gbps,
        dtypes=scatter_reduce_bench_dtypes(reduce),
    )
    bench.run()


@pytest.mark.scatter_reduce_two_
@pytest.mark.parametrize(
    "reduce, include_self",
    [("sum", True), ("mean", True), ("prod", True), ("amax", True), ("amin", True)],
)
def test_scatter_reduce_two_inplace(reduce, include_self):
    bench = ScatterReducePublicBenchmark(
        op_name="scatter_reduce_two_",
        torch_op=torch.Tensor.scatter_reduce_,
        input_fn=scatter_reduce_two_input_fn_factory(reduce, include_self),
        get_gbps=gather_scatter_gbps,
        dtypes=scatter_reduce_bench_dtypes(reduce),
        is_inplace=True,
    )
    bench.run()


@pytest.mark.scatter_reduce_two_out
@pytest.mark.parametrize(
    "reduce, include_self",
    [("sum", False), ("mean", True), ("amax", True), ("amin", False)],
)
def test_scatter_reduce_two_out(reduce, include_self):
    bench = ScatterReducePublicBenchmark(
        op_name="scatter_reduce_two_out",
        torch_op=torch.scatter_reduce,
        input_fn=scatter_reduce_two_input_fn_factory(reduce, include_self, out=True),
        get_gbps=gather_scatter_gbps,
        dtypes=scatter_reduce_bench_dtypes(reduce),
    )
    bench.run()
