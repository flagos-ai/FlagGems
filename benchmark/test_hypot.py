import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.hypot
def test_hypot():
    bench = base.BinaryPointwiseBenchmark(
        op_name="hypot",
        torch_op=torch.hypot,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
