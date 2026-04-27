import pytest
import torch

from . import attri_util as consts
from . import performance_utils as base
from . import utils


@pytest.mark.log_softmax
def test_log_softmax():
    bench = base.GenericBenchmark2DOnly(
        op_name="log_softmax",
        input_fn=utils.unary_input_fn,
        torch_op=torch.nn.functional.log_softmax,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
