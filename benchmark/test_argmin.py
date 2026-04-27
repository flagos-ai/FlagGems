import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


@pytest.mark.argmin
def test_argmin():
    bench = utils.UnaryReductionBenchmark(
        op_name="argmin", torch_op=torch.argmin, dtypes=attr_utils.FLOAT_DTYPES
    )
    bench.run()
