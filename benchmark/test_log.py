import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.log
def test_log():
    bench = base.UnaryPointwiseBenchmark(
        op_name="log", torch_op=torch.log, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()
