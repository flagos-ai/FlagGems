import pytest
import torch

import flag_gems

from . import base, consts, utils


@pytest.mark.uniform_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_uniform_inplace():
    bench = base.GenericBenchmark(
        input_fn=utils.unary_input_fn,
        op_name="uniform_",
        torch_op=torch.Tensor.uniform_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
