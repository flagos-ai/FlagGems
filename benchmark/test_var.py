import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


@pytest.mark.var
def test_var():
    bench = utils.UnaryReductionBenchmark(
        op_name="var", torch_op=torch.var, dtypes=attr_utils.FLOAT_DTYPES
    )
    bench.run()
