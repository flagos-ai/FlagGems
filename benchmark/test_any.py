import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


@pytest.mark.any
def test_any():
    bench = utils.UnaryReductionBenchmark(
        op_name="any", torch_op=torch.any, dtypes=attr_utils.FLOAT_DTYPES
    )
    bench.run()
