import pytest
import torch

from . import base


def _input_fn(shape, dtype, device):
    yield {"s": 0.01, "dtype": dtype, "device": device},


@pytest.mark.scalar_tensor
def test_scalar_tensor():
    bench = base.GenericBenchmark(
        op_name="scalar_tensor", input_fn=_input_fn, torch_op=torch.scalar_tensor
    )
    bench.run()
