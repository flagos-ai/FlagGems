import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.sin
def test_sin():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sin", torch_op=torch.sin, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.sin_
def test_sin_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sin_", torch_op=torch.sin_, dtypes=attrs.FLOAT_DTYPES, is_inplace=True
    )
    bench.run()
