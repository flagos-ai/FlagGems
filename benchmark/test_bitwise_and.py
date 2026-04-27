import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.bitwise_and
def test_bitwise_and():
    bench = base.BinaryPointwiseBenchmark(
        op_name="bitwise_and",
        torch_op=torch.bitwise_and,
        dtypes=attrs.INT_DTYPES + attrs.BOOL_DTYPES,
    )
    bench.run()


@pytest.mark.bitwise_and_
def test_bitwise_and_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="bitwise_and_",
        torch_op=lambda a, b: a.bitwise_and_(b),
        dtypes=attrs.INT_DTYPES + attrs.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()
