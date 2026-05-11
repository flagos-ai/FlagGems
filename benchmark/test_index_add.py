import pytest
import torch

import flag_gems
from flag_gems.utils import shape_utils

from . import base


class TensorSelectBenchmark(base.GenericBenchmark2DOnly):
    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # Speed Up Benchmark Test, Big Shape Will Cause Timeout
        if flag_gems.vendor_name == "kunlunxin":
            return []

        shapes = super().set_more_shapes()
        shapes = [
            # this filter is for scatter
            shape
            for shape in shapes
            if len(shape) == 2 and shape[0] > 16 and shape[1] > 16
        ]
        return shapes


def index_add_gbps(bench_fn_args, latency):
    index = bench_fn_args[2]
    src = bench_fn_args[3]
    io_amount = sum([shape_utils.size_in_bytes(item) for item in [index, src, src]])

    return io_amount * 1e-9 / (latency * 1e-3)


def index_add_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    dim = 0 if len(shape) == 1 else 1
    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max // 2 if index_max >= 2 else 1
    index = torch.randperm(index_len, device=device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=device)
    yield inp, dim, index, src


@pytest.mark.index_add
def test_index_add():
    bench = TensorSelectBenchmark(
        op_name="index_add",
        torch_op=torch.index_add,
        input_fn=index_add_input_fn,
        dtypes=[torch.float16, torch.float32],
        get_gbps=index_add_gbps,
    )
    bench.run()


def index_add__input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    dim = 0 if len(shape) == 1 else 1
    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max // 2 if index_max >= 2 else 1
    index = torch.randperm(index_len, device=device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=device)
    yield inp, dim, index, src


@pytest.mark.index_add_
def test_index_add_():
    bench = TensorSelectBenchmark(
        op_name="index_add_",
        torch_op=torch.Tensor.index_add_,
        input_fn=index_add__input_fn,
        dtypes=[torch.float16, torch.float32],
        get_gbps=index_add_gbps,
    )
    bench.run()
