import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


@pytest.mark.var_mean
def test_var_mean():
    bench = utils.UnaryReductionBenchmark(
        op_name="var_mean", torch_op=torch.var_mean, dtypes=attr_utils.FLOAT_DTYPES
    )
    bench.run()
