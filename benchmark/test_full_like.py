import pytest
import torch

from . import performance_utils as base


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield {"input": inp, "fill_value": 3.1415926},


@pytest.mark.full_like
def test_full_like():
    bench = base.GenericBenchmark(
        op_name="full_like", input_fn=_input_fn, torch_op=torch.full_like
    )
    bench.run()
