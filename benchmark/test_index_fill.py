import pytest
import torch

import flag_gems
from flag_gems.utils import shape_utils

from . import base, consts


class IndexFillBenchmark(base.GenericBenchmark2DOnly):
    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        if flag_gems.vendor_name == "kunlunxin":
            return []

        return [
            shape
            for shape in super().set_more_shapes()
            if len(shape) == 2 and shape[0] > 16 and shape[1] > 16
        ]


def index_fill_gbps(bench_fn_args, latency):
    inp = bench_fn_args[0]
    index = bench_fn_args[2]
    io_amount = sum(shape_utils.size_in_bytes(item) for item in [inp, index, inp])
    return io_amount * 1e-9 / (latency * 1e-3)


def index_fill_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    dim = 0 if len(shape) == 1 else 1
    index_len = max(inp.size(dim) // 2, 1)
    index = torch.randperm(inp.size(dim), device=device)[:index_len]
    value = 1.25
    yield inp, dim, index, value


@pytest.mark.index_fill
def test_index_fill():
    bench = IndexFillBenchmark(
        op_name="index_fill",
        torch_op=torch.index_fill,
        input_fn=index_fill_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        get_gbps=index_fill_gbps,
    )
    bench.run()


def index_fill__input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    dim = 0 if len(shape) == 1 else 1
    index_len = max(inp.size(dim) // 2, 1)
    index = torch.randperm(inp.size(dim), device=device)[:index_len]
    value = 1.25
    yield inp, dim, index, value


@pytest.mark.index_fill_
def test_index_fill_():
    bench = IndexFillBenchmark(
        op_name="index_fill_",
        torch_op=torch.Tensor.index_fill_,
        input_fn=index_fill__input_fn,
        dtypes=consts.FLOAT_DTYPES,
        get_gbps=index_fill_gbps,
    )
    bench.run()
