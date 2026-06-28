import pytest
import torch

from flag_gems.ops import normed_cumsum

from . import base, consts


def normed_cumsum_ref(inp, dim=-1):
    cumsum = torch.cumsum(inp, dim=dim)
    total = inp.sum(dim=dim, keepdim=True)
    return cumsum / total


def input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device).abs() + 1e-4
    if inp.ndim > 1:
        yield inp, 1
    else:
        yield (inp,)


@pytest.mark.normed_cumsum
def test_normed_cumsum_perf():
    bench = base.GenericBenchmark2DOnly(
        op_name="normed_cumsum",
        input_fn=input_fn,
        torch_op=normed_cumsum_ref,
        gems_op=normed_cumsum,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
