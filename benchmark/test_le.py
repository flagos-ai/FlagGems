import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.le
def test_le():
    bench = base.BinaryPointwiseBenchmark(
        op_name="le",
        torch_op=torch.le,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
