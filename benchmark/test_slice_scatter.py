import pytest
import torch

import flag_gems
from flag_gems.utils import shape_utils

from . import base, consts


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


def _input_fn(shape, dtype, device):
    dim = 0 if len(shape) == 1 else 1
    start = 0
    end = shape[dim]
    step = 2

    inp = torch.randn(shape, dtype=dtype, device=device)

    range = end - start
    valid_shape = list(inp.shape)
    if end < start:
        range = 0
    elif (end - start) > valid_shape[dim]:
        range = valid_shape[dim]
        start = 0
        end = valid_shape[dim]

    valid_shape[dim] = (range + (step - 1)) // step
    src = torch.randn(valid_shape, dtype=dtype, device=device)
    yield inp, src, dim, start, end, step


def _get_gbps(bench_fn_args, latency):
    inp, mask, src = bench_fn_args
    io_amount = sum([shape_utils.size_in_bytes(item) for item in [inp, mask, src, inp]])

    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.slice_scatter
def test_slice_scatter():
    bench = TensorSelectBenchmark(
        op_name="slice_scatter",
        torch_op=torch.slice_scatter,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        get_gbps=_get_gbps,
    )
    bench.run()
