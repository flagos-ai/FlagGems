import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.angle
def test_angle():
    bench = base.UnaryPointwiseBenchmark(
        op_name="angle",
        torch_op=torch.angle,
        dtypes=attrs.COMPLEX_DTYPES
        + [torch.float32]
        + attrs.INT_DTYPES
        + attrs.BOOL_DTYPES,
    )
    bench.run()
