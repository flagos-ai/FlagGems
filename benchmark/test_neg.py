import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.neg
def test_neg():
    bench = base.UnaryPointwiseBenchmark(
        op_name="neg", torch_op=torch.neg, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.neg_
def test_neg_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="neg_", torch_op=torch.neg_, dtypes=attrs.FLOAT_DTYPES, is_inplace=True
    )
    bench.run()
