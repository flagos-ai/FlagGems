import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.isfinite
def test_isfinite():
    bench = base.UnaryPointwiseBenchmark(
        op_name="isfinite", torch_op=torch.isfinite, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()
