import pytest
import torch

from . import performance_utils as utils


def _input_fn(shape, dtype, device):
    x = torch.randn(size=shape, dtype=dtype, device=device)
    yield x.conj(),


@pytest.mark.resolve_conj
def test_resolve_conj():
    bench = utils.GenericBenchmarkExcluse1D(
        op_name="resolve_conj",
        input_fn=_input_fn,
        torch_op=torch.resolve_conj,
        dtypes=[torch.cfloat],
    )
    bench.run()
