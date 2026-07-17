import pytest
import torch

import flag_gems

from . import base, consts, utils


@pytest.mark.exponential_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_exponential_inplace():
    bench = base.GenericBenchmark(
        op_name="exponential_",
        input_fn=utils.unary_input_fn,
        torch_op=torch.Tensor.exponential_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
