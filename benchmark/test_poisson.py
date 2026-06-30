import pytest
import torch

import flag_gems

from . import base, consts


def _input_fn(shape, dtype, device):
    inp = torch.rand(shape, dtype=dtype, device=device)
    yield (inp,)


@pytest.mark.poisson
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_poisson():
    bench = base.GenericBenchmark2DOnly(
        op_name="poisson",
        torch_op=torch.poisson,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
