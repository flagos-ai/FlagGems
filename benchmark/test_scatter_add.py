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
    inp = torch.randn(shape, dtype=dtype, device=device)

    dim = -1
    size_dim = shape[dim]
    index_shape = list(shape)
    index_shape[dim] = 2 * shape[dim]
    index = torch.randint(0, size_dim, index_shape, dtype=torch.long, device=device)
    yield inp, dim, index


def _get_gbps(bench_fn_args, latency):
    inp, dim, index = bench_fn_args[:3]
    data_shape = list(inp.shape)
    data_shape[dim] = index.shape[dim]
    data = torch.empty(data_shape, dtype=inp.dtype, device=inp.device)
    io_amount = sum([shape_utils.size_in_bytes(item) for item in [index, data, data]])
    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.scatter_add_
def test_scatter_add_inplace():
    def scatter_input_fn(shape, dtype, device):
        input_gen = _input_fn(shape, dtype, device)
        inp, dim, index = next(input_gen)
        src_shape = list(size + 16 for size in index.shape)
        src = torch.randn(src_shape, dtype=dtype, device=device)

        yield inp, dim, index, src

    bench = TensorSelectBenchmark(
        op_name="scatter_add_",
        torch_op=torch.Tensor.scatter_add_,
        input_fn=scatter_input_fn,
        get_gbps=_get_gbps,
        dtypes=attr_utils.FLOAT_DTYPES,
    )
    bench.run()
