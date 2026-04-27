import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.logical_and
def test_logical_and():
    bench = base.BinaryPointwiseBenchmark(
        op_name="logical_and",
        torch_op=torch.logical_and,
        dtypes=attrs.INT_DTYPES + attrs.BOOL_DTYPES,
    )
    bench.run()


@pytest.mark.logical_and_
def test_logical_and_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="logical_and_",
        torch_op=lambda a, b: a.logical_and_(b),
        dtypes=attrs.INT_DTYPES + attrs.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()
