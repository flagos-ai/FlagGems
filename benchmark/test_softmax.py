import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


@pytest.mark.softmax
def test_softmax():
    bench = utils.UnaryReductionBenchmark(
        op_name="softmax",
        torch_op=torch.nn.functional.softmax,
        dtypes=attr_utils.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.softmax_backward
def test_softmax_backward():
    bench = utils.UnaryReductionBenchmark(
        op_name="softmax",
        torch_op=torch.nn.functional.softmax,
        dtypes=attr_utils.FLOAT_DTYPES,
        is_backward=True,
    )

    bench.run()
