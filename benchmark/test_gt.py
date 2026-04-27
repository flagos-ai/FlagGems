import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.gt
def test_gt():
    bench = base.BinaryPointwiseBenchmark(
        op_name="gt",
        torch_op=torch.gt,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
