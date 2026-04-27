import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.equal
def test_equal():
    bench = base.BinaryPointwiseBenchmark(
        op_name="equal",
        torch_op=torch.sub,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
