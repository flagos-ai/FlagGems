import pytest
import torch

import flag_gems
from flag_gems.utils import shape_utils

from . import attri_util as consts
from . import performance_utils as base
from . import utils


# TODO(Qiming): Move this to an abstraction layer
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


def _input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    mask = utils.generate_tensor_input(shape, cur_dtype, device) < 0.3
    src = utils.generate_tensor_input(shape, cur_dtype, device)

    yield inp, mask, src


def _get_gbps(bench_fn_args, latency):
    inp, mask, src = bench_fn_args
    io_amount = sum([shape_utils.size_in_bytes(item) for item in [inp, mask, src, inp]])

    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.masked_scatter
def test_masked_scatter():
    bench = TensorSelectBenchmark(
        op_name="masked_scatter",
        torch_op=torch.masked_scatter,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        get_gbps=_get_gbps,
    )
    bench.run()


@pytest.mark.masked_scatter_
def test_masked_scatter_inplace():
    bench = TensorSelectBenchmark(
        op_name="masked_scatter_",
        torch_op=torch.Tensor.masked_scatter_,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        get_gbps=_get_gbps,
        is_inplace=True,
    )
    bench.run()
