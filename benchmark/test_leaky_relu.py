import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.leaky_relu
def test_leaky_relu():
    bench = base.UnaryPointwiseBenchmark(
        op_name="leaky_relu",
        torch_op=torch.nn.functional.leaky_relu,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.leaky_relu_
def test_leaky_relu_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="atan_",
        torch_op=torch.nn.functional.leaky_relu_,
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
