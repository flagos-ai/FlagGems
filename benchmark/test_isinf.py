import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.isinf
def test_isinf():
    bench = base.UnaryPointwiseBenchmark(
        op_name="isinf", torch_op=torch.isinf, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()
