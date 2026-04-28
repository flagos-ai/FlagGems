import pytest
import torch

from . import base, consts


@pytest.mark.sigmoid
def test_sigmoid():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sigmoid", torch_op=torch.sigmoid, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.sigmoid_
def test_sigmoid_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sigmoid_",
        torch_op=torch.sigmoid_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
