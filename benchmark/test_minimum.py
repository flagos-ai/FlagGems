import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.minimum
def test_minimum():
    bench = base.BinaryPointwiseBenchmark(
        op_name="minimum",
        torch_op=torch.minimum,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
