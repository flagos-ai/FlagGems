import pytest
import torch

from . import performance_utils as base


def _input_fn(shape, dtype, device):
    yield {"size": shape, "dtype": dtype, "device": device},


@pytest.mark.rand
def test_rand():
    bench = base.GenericBenchmark(
        op_name="rand", input_fn=_input_fn, torch_op=torch.zeros_like
    )
    bench.run()
