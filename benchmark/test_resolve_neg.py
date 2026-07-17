import pytest
import torch

import flag_gems

from . import base


def _input_fn(shape, dtype, device):
    x = torch.randn(size=shape, dtype=dtype, device=device)
    if flag_gems.vendor_name == "mthreads":
        yield x.conj(),
    else:
        yield x.conj().imag,


@pytest.mark.resolve_neg
def test_resolve_neg():
    bench = base.GenericBenchmarkExcluse1D(
        op_name="resolve_neg",
        input_fn=_input_fn,
        dtypes=[torch.cfloat],
        torch_op=torch.resolve_neg,
    )
    bench.run()
