import random

import pytest
import torch

import flag_gems
from flag_gems.utils import shape_utils

from . import attri_util as attr_utils
from . import performance_utils as utils


class TensorSelectBenchmark(utils.GenericBenchmark2DOnly):
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


def _input_fn(shape, dtype, device):
    dim = 0 if len(shape) == 1 else 1
    index = random.randint(0, shape[dim] - 1)
    inp = torch.randn(shape, dtype=dtype, device=device)

    src_shape = list(inp.shape)
    del src_shape[dim]
    src = torch.randn(src_shape, dtype=dtype, device=device)

    yield inp, src, dim, index


def _get_gbps(bench_fn_args, latency):
    inp = bench_fn_args[0]
    src = bench_fn_args[1]
    io_amount = sum([shape_utils.size_in_bytes(item) for item in [inp, src, src]])

    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.select_scatter
def test_select_scatter():
    bench = TensorSelectBenchmark(
        op_name="select_scatter",
        torch_op=torch.select_scatter,
        input_fn=_input_fn,
        dtypes=attr_utils.FLOAT_DTYPES,
        get_gbps=_get_gbps,
    )

    bench.run()
