import pytest
import torch

import flag_gems

from . import base, consts, utils


@pytest.mark.exponential
def test_exponential_outplace():
    bench = base.GenericBenchmark(
        op_name="exponential",
        input_fn=utils.unary_input_fn,
        torch_op=torch.Tensor.exponential_,
        gems_op=flag_gems.exponential,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
