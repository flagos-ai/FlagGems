import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.lt
def test_lt():
    bench = base.BinaryPointwiseBenchmark(
        op_name="lt",
        torch_op=torch.lt,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
