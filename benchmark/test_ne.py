import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.ne
def test_ne():
    bench = base.BinaryPointwiseBenchmark(
        op_name="ne",
        torch_op=torch.ne,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
