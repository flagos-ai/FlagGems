import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.fmin
def test_fmin():
    bench = base.BinaryPointwiseBenchmark(
        op_name="fmin",
        torch_op=torch.fmin,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
