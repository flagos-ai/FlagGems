import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


@pytest.mark.min
def test_min():
    bench = utils.UnaryReductionBenchmark(
        op_name="min", torch_op=torch.min, dtypes=attr_utils.FLOAT_DTYPES
    )
    bench.run()
