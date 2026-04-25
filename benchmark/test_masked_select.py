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


def _input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    mask = utils.generate_tensor_input(shape, cur_dtype, device) < 0.3

    yield inp, mask


def _get_gbps(bench_fn_args, latency):
    mask = bench_fn_args[1]
    io_amount = sum([shape_utils.size_in_bytes(item) for item in [mask]])
    io_amount += 2 * int(torch.sum(mask))
    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.masked_select
def test_masked_select():
    bench = TensorSelectBenchmark(
        op_name="masked_select",
        input_fn=_input_fn,
        torch_op=torch.masked_select,
        dtypes=attr_utils.FLOAT_DTYPES,
        get_gbps=_get_gbps,
    )

    bench.run()
