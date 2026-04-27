import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.sqrt
def test_sqrt():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sqrt", torch_op=torch.sqrt, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.sqrt_
def test_sqrt_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sqrt_",
        torch_op=torch.sqrt_,
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
