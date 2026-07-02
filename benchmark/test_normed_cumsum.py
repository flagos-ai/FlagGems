import pytest
import torch

import flag_gems

from . import base, consts


def torch_normed_cumsum(inp, dim=-1):
    return torch.cumsum(inp, dim=dim) / inp.sum(dim=dim, keepdim=True)


def input_fn(shape, dtype, device):
    inp = torch.rand(shape, dtype=dtype, device=device) + 0.1
    dim = -1
    yield inp, dim


@pytest.mark.normed_cumsum
def test_normed_cumsum():
    bench = base.GenericBenchmark(
        input_fn=input_fn,
        op_name="normed_cumsum",
        torch_op=torch_normed_cumsum,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.normed_cumsum)
    bench.run()
