import pytest
import torch

from . import base


def _input_fn(shape, dtype, device):
    yield {"high": 100, "size": shape, "dtype": dtype, "device": device},


@pytest.mark.randint
def test_randint():
    bench = base.GenericBenchmark(
        op_name="randint", input_fn=_input_fn, torch_op=torch.randint
    )
    bench.run()
