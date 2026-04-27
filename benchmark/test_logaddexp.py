import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.logaddexp
def test_logaddexp():
    bench = base.BinaryPointwiseBenchmark(
        op_name="logaddexp",
        torch_op=torch.logaddexp,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
