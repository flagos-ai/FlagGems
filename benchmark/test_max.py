import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


@pytest.mark.max
def test_max():
    bench = utils.UnaryReductionBenchmark(
        op_name="max", torch_op=torch.max, dtypes=attr_utils.FLOAT_DTYPES
    )
    bench.run()
