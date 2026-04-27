import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.ceil
def test_ceil():
    bench = base.UnaryPointwiseBenchmark(
        op_name="ceil", torch_op=torch.ceil, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.ceil_
def test_ceil_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="ceil_",
        torch_op=torch.ceil_,
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.ceil_out
def test_ceil_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="ceil_out",
        torch_op=torch.ceil,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
