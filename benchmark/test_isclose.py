import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.isclose
def test_isclose():
    bench = base.BinaryPointwiseBenchmark(
        op_name="isclose",
        torch_op=torch.isclose,
        dtypes=attrs.FLOAT_DTYPES + attrs.INT_DTYPES,
    )
    bench.run()
