import pytest
import torch

import flag_gems

from . import base, utils


@pytest.mark.rand_like
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_rand_like():
    bench = base.GenericBenchmark(
        op_name="rand_like", input_fn=utils.unary_input_fn, torch_op=torch.rand_like
    )
    bench.run()
