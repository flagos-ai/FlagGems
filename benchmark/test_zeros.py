import pytest
import torch

from . import performance_utils as base


def _input_fn(shape, dtype, device):
    yield {"size": shape, "dtype": dtype, "device": device},


@pytest.mark.zeros
def test_zeros():
    bench = base.GenericBenchmark(
        op_name="zeros", input_fn=_input_fn, torch_op=torch.zeros
    )
    bench.run()
