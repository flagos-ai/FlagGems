import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.cos
def test_cos():
    bench = base.UnaryPointwiseBenchmark(
        op_name="cos", torch_op=torch.cos, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.cos_
def test_cos_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="cos_", torch_op=torch.cos_, dtypes=attrs.FLOAT_DTYPES, is_inplace=True
    )
    bench.run()
