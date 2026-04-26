import pytest
import torch

from . import performance_utils as base


def _input_fn(shape, dtype, device):
    yield {"size": shape, "fill_value": 3.1415926, "dtype": dtype, "device": device},


@pytest.mark.full
def test_full():
    bench = base.GenericBenchmark(
        op_name="full", input_fn=_input_fn, torch_op=torch.full
    )
    bench.run()
