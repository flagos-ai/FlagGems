import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.square
def test_square():
    bench = base.UnaryPointwiseBenchmark(
        op_name="square", torch_op=torch.square, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.square_
def test_square_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="square_",
        torch_op=torch.square_,
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.square_out
def test_square_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="square_out",
        torch_op=torch.square,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
