import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


@pytest.mark.std
def test_std():
    bench = utils.UnaryReductionBenchmark(
        op_name="std", torch_op=torch.std, dtypes=attr_utils.FLOAT_DTYPES
    )
    bench.run()
