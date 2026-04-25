import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


@pytest.mark.test_amax
def test_amax():
    bench = utils.UnaryReductionBenchmark(
        op_name="amax", torch_op=torch.amax, dtypes=attr_utils.FLOAT_DTYPES
    )
    bench.run()
