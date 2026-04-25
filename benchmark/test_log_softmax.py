import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


@pytest.mark.log_softmax
def test_log_softmax():
    bench = utils.GenericBenchmark2DOnly(
        op_name="log_softmax",
        input_fn=utils.unary_input_fn,
        torch_op=torch.nn.functional.log_softmax,
        dtypes=attr_utils.FLOAT_DTYPES,
    )
    bench.run()
