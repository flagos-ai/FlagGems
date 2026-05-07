import pytest
import torch

from . import base, consts, utils


def median_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"dim": -1}

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield inp, {"dim": 0}
        yield inp, {"dim": -1, "keepdim": True}


def median_full_input_fn(shape, dtype, device):
    """torch.median(x) — whole-tensor scalar reduction."""
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield (inp,)


@pytest.mark.median
def test_median():
    bench = base.GenericBenchmark2DOnly(
        op_name="median",
        input_fn=median_input_fn,
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.median
def test_median_full():
    bench = base.GenericBenchmark2DOnly(
        op_name="median_full",
        input_fn=median_full_input_fn,
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
