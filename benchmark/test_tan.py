import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.tan
def test_tan():
    bench = base.UnaryPointwiseBenchmark(
        op_name="tan", torch_op=torch.tan, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.tan_
def test_tan_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="tan_", torch_op=torch.tan_, dtypes=attrs.FLOAT_DTYPES, is_inplace=True
    )
    bench.run()
