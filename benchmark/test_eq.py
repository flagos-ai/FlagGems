import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.eq
def test_eq():
    bench = base.BinaryPointwiseBenchmark(
        op_name="eq",
        torch_op=torch.eq,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
