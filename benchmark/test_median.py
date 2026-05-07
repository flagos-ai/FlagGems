import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


def median_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    yield inp, {"dim": -1}

    if utils.Config.bench_level == utils.BenchLevel.COMPREHENSIVE:
        yield inp, {"dim": 0}
        yield inp, {"dim": -1, "keepdim": True}


def median_full_input_fn(shape, cur_dtype, device):
    """torch.median(x) — whole-tensor scalar reduction."""
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    yield inp,


@pytest.mark.median
def test_median():
    bench = utils.GenericBenchmark2DOnly(
        op_name="median",
        input_fn=median_input_fn,
        torch_op=torch.median,
        dtypes=attr_utils.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.median
def test_median_full():
    bench = utils.GenericBenchmark2DOnly(
        op_name="median_full",
        input_fn=median_full_input_fn,
        torch_op=torch.median,
        dtypes=attr_utils.FLOAT_DTYPES,
    )
    bench.run()
