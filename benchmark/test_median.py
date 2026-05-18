import pytest
import torch

from . import base, consts, utils


def _median_flat_input(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp,


def _median_dim_input(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"dim": -1},


class MedianFlatBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return [
            (1024 * 1024,),
            (1024, 1),
            (1024, 512),
            (64, 1024, 64),
            (8, 256 * 1024),
        ]


class MedianDimBenchmark(base.GenericBenchmark2DOnly):
    def set_more_shapes(self):
        return [
            (1024, 1),
            (1024, 512),
            (16, 128 * 1024),
            (8, 256 * 1024),
        ]


@pytest.mark.median
def test_perf_median_flat():
    bench = MedianFlatBenchmark(
        input_fn=_median_flat_input,
        op_name="median",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()


@pytest.mark.median
def test_perf_median_dim():
    bench = MedianDimBenchmark(
        input_fn=_median_dim_input,
        op_name="median_dim",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()
