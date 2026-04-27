import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


@pytest.mark.prod
def test_prod():
    bench = utils.UnaryReductionBenchmark(
        op_name="prod", torch_op=torch.prod, dtypes=attr_utils.FLOAT_DTYPES
    )
    bench.run()
