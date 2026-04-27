import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.allclose
def test_allclose():
    bench = base.BinaryPointwiseBenchmark(
        op_name="allclose",
        torch_op=torch.allclose,
        dtypes=attrs.FLOAT_DTYPES + attrs.INT_DTYPES,
    )
    bench.run()
