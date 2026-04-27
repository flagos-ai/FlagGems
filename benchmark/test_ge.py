import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.ge
def test_ge():
    bench = base.BinaryPointwiseBenchmark(
        op_name="ge",
        torch_op=torch.ge,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
