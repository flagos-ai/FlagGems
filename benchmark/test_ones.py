import pytest
import torch

from . import performance_utils as base


def _input_fn(shape, dtype, device):
    yield {"size": shape, "dtype": dtype, "device": device},


@pytest.mark.ones
def test_ones():
    bench = base.GenericBenchmark(
        op_name="ones", input_fn=_input_fn, torch_op=torch.ones
    )
    bench.run()
