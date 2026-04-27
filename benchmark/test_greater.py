import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.greater
def test_greater():
    bench = base.BinaryPointwiseBenchmark(
        op_name="greater",
        torch_op=torch.greater,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
