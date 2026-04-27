import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


@pytest.mark.argmax
def test_argmax():
    bench = utils.UnaryReductionBenchmark(
        op_name="argmax", torch_op=torch.argmax, dtypes=attr_utils.FLOAT_DTYPES
    )
    bench.run()
