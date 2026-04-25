import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


@pytest.mark.all
def test_all():
    bench = utils.UnaryReductionBenchmark(
        op_name="all", torch_op=torch.all, dtypes=attr_utils.FLOAT_DTYPES
    )
    bench.run()
