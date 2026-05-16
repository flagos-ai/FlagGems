import pytest
import torch

import flag_gems

from . import base, consts, utils


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


@pytest.mark.silu_and_mul_out
def test_silu_and_mul_out():
    def gems_op(x, y):
        out = torch.empty_like(x)
        return flag_gems.silu_and_mul_out(x, y, out)

    def torch_op(x, y):
        out = torch.empty_like(x)
        return torch.mul(torch.nn.functional.silu(x), y, out=out)

    bench = base.GenericBenchmark(
        input_fn=utils.binary_input_fn,
        op_name="silu_and_mul_out",
        gems_op=gems_op,
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
