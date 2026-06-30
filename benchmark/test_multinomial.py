import pytest
import torch

import flag_gems

from . import base, consts


def _input_fn(shape, dtype, device):
    dist = torch.rand(shape, dtype=dtype, device=device)
    n_samples = 10000
    yield dist, n_samples, True,


@pytest.mark.multinomial
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_multinomial_with_replacement():
    bench = base.GenericBenchmark2DOnly(
        input_fn=_input_fn,
        op_name="multinomial",
        torch_op=torch.multinomial,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
