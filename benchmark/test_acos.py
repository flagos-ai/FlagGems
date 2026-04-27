import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.acos
def test_acos():
    bench = base.UnaryPointwiseBenchmark(
        op_name="acos", torch_op=torch.acos, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()
