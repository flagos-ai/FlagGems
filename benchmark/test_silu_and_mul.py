import pytest
import torch

import flag_gems

from . import attri_util as consts
from . import performance_utils as base
from . import utils


@pytest.mark.silu_and_mul
def test_silu_and_mul():
    def torch_op(x, y):
        return torch.mul(torch.nn.functional.silu(x), y)

    bench = base.GenericBenchmark(
        input_fn=utils.binary_input_fn,
        op_name="silu_and_mul",
        gems_op=flag_gems.silu_and_mul,
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
