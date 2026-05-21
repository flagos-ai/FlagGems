import pytest
import torch

from . import base, consts


@pytest.mark.relu
def test_relu():
    bench = base.UnaryPointwiseBenchmark(
        op_name="relu", torch_op=torch.nn.functional.relu, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.relu_
def test_relu_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="relu_",
        torch_op=torch.nn.functional.relu_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
