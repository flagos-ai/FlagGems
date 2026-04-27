import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.relu
def test_relu():
    bench = base.UnaryPointwiseBenchmark(
        op_name="relu", torch_op=torch.nn.functional.relu, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.relu_
def test_relu_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="relu_",
        torch_op=torch.nn.functional.relu_,
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
