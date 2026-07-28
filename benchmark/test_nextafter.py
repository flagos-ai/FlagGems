import pytest
import torch

from . import base, consts


@pytest.mark.nextafter
def test_nextafter():
    bench = base.BinaryPointwiseBenchmark(
        op_name="nextafter",
        torch_op=torch.nextafter,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.nextafter_
def test_nextafter_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="nextafter_",
        torch_op=torch.Tensor.nextafter_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
