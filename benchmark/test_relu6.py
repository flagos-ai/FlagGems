import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.relu6
def test_relu6():
    bench = base.UnaryPointwiseBenchmark(
        op_name="relu6", torch_op=torch.nn.functional.relu6, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()
