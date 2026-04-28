import pytest
import torch

from . import base


def _input_fn(shape, dtype, device):
    yield {"size": shape, "dtype": dtype, "device": device},


@pytest.mark.randn
def test_randn():
    bench = base.GenericBenchmark(
        op_name="randn", input_fn=_input_fn, torch_op=torch.randn
    )
    bench.run()
