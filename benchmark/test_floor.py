import pytest
import torch

from . import base, consts


@pytest.mark.floor
def test_floor():
    bench = base.UnaryPointwiseBenchmark(
        op_name="floor", torch_op=torch.floor, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.floor_
def test_floor_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="floor_",
        torch_op=torch.Tensor.floor_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.floor_out
def test_floor_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="floor_out",
        torch_op=torch.floor,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
