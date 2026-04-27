import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.gcd
def test_gcd():
    bench = base.BinaryPointwiseBenchmark(
        op_name="gcd",
        torch_op=torch.gcd,
        dtypes=attrs.INT_DTYPES,
    )
    bench.run()
