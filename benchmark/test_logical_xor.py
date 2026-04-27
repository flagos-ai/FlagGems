import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.logical_xor
def test_logical_xor():
    bench = base.BinaryPointwiseBenchmark(
        op_name="logical_xor",
        torch_op=torch.logical_xor,
        dtypes=attrs.INT_DTYPES + attrs.BOOL_DTYPES,
    )
    bench.run()
