import pytest
import torch

import flag_gems

from . import base, consts, utils


@pytest.mark.silu_and_mul
def test_silu_and_mul_with_clamp():
    limit = 7.0

    def gems_op(x, y):
        return flag_gems.silu_and_mul_with_clamp(x, y, limit)

    def torch_op(x, y):
        gate = torch.clamp(x, max=limit)
        up = torch.clamp(y, min=-limit, max=limit)
        return torch.mul(torch.nn.functional.silu(gate), up)

    bench = base.GenericBenchmark(
        input_fn=utils.binary_input_fn,
        op_name="silu_and_mul_with_clamp",
        gems_op=gems_op,
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
