import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.logical_or
def test_logical_or():
    bench = base.BinaryPointwiseBenchmark(
        op_name="logical_or",
        torch_op=torch.logical_or,
        dtypes=attrs.INT_DTYPES + attrs.BOOL_DTYPES,
    )
    bench.run()


@pytest.mark.logical_or_
def test_logical_or_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="logical_or_",
        torch_op=lambda a, b: a.logical_or_(b),
        dtypes=attrs.INT_DTYPES + attrs.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()
