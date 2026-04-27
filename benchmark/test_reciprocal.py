import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.reciprocal
def test_reciprocal():
    bench = base.UnaryPointwiseBenchmark(
        op_name="reciprocal", torch_op=torch.reciprocal, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.reciprocal_
def test_reciprocal_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="reciprocal_",
        torch_op=torch.reciprocal_,
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
