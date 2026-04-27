import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.absolute
def test_absolute():
    bench = base.UnaryPointwiseBenchmark(
        op_name="absolute", torch_op=torch.absolute, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()
