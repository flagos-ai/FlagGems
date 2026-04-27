import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.maximum
def test_maximum():
    bench = base.BinaryPointwiseBenchmark(
        op_name="maximum",
        torch_op=torch.maximum,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
