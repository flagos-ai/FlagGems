import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.logical_not
def test_logical_not():
    bench = base.UnaryPointwiseBenchmark(
        op_name="logical_not", torch_op=torch.logical_not, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()
