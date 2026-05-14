import pytest
import torch

import flag_gems

from . import base, consts


def normed_cumsum_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    dim = -1 if len(shape) > 1 else 0
    yield inp, dim


@pytest.mark.normed_cumsum
def test_normed_cumsum():
    bench = base.GenericBenchmark(
        op_name="normed_cumsum",
        torch_op=lambda inp, dim: torch.cumsum(inp, dim=dim)
        / torch.sum(inp, dim=dim, keepdim=True),
        gems_op=flag_gems.normed_cumsum,
        input_fn=normed_cumsum_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
