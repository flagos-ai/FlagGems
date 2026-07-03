import pytest
import torch

import flag_gems

from . import base, consts, utils


@pytest.mark.gelu_and_mul
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_gelu_and_mul():
    def torch_op(x, y):
        return torch.mul(torch.nn.functional.gelu(x), y)

    bench = base.GenericBenchmark(
        input_fn=utils.binary_input_fn,
        op_name="gelu_and_mul",
        torch_op=torch_op,
        gems_op=flag_gems.gelu_and_mul,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
