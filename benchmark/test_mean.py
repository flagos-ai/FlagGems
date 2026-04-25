import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


@pytest.mark.mean
def test_mean():
    bench = utils.UnaryReductionBenchmark(
        op_name="mean", torch_op=torch.mean, dtypes=attr_utils.FLOAT_DTYPES
    )
    bench.run()
