import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.log_sigmoid
def test_log_sigmoid():
    bench = base.UnaryPointwiseBenchmark(
        op_name="log_sigmoid",
        torch_op=torch.nn.functional.logsigmoid,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
