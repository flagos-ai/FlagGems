import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


@pytest.mark.sum
def test_sum():
    bench = utils.UnaryReductionBenchmark(
        op_name="sum", torch_op=torch.sum, dtypes=attr_utils.FLOAT_DTYPES
    )
    bench.run()
