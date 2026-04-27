import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.sigmoid
def test_sigmoid():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sigmoid", torch_op=torch.sigmoid, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.sigmoid_
def test_sigmoid_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sigmoid_",
        torch_op=torch.sigmoid_,
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
