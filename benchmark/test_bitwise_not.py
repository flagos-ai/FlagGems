import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.bitwise_not
def test_bitwise_not():
    bench = base.UnaryPointwiseBenchmark(
        op_name="bitwise_not", torch_op=torch.bitwise_not, dtypes=attrs.INT_DTYPES
    )
    bench.run()


def test_bitwise_not_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="bitwise_not_",
        torch_op=lambda a: a.bitwise_not_(),
        dtypes=attrs.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()
