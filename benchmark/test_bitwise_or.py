import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.bitwise_or
def test_bitwise_or():
    bench = base.BinaryPointwiseBenchmark(
        op_name="bitwise_or",
        torch_op=torch.bitwise_or,
        dtypes=attrs.INT_DTYPES + attrs.BOOL_DTYPES,
    )
    bench.run()


@pytest.mark.bitwise_or_
def test_bitwise_or_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="bitwise_or_",
        torch_op=lambda a, b: a.bitwise_or_(b),
        dtypes=attrs.INT_DTYPES + attrs.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()
