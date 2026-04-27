import pytest
import torch

from . import attri_util as consts
from . import performance_utils as base
from . import utils


@pytest.mark.tril
def test_tril():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=utils.unary_input_fn,
        op_name="tril",
        torch_op=torch.tril,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
