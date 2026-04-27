import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.isnan
def test_isnan():
    bench = base.UnaryPointwiseBenchmark(
        op_name="isnan", torch_op=torch.isnan, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()
