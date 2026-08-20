import pytest
import torch

from . import base


def flatten_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    # Flatten all dimensions (default behavior)
    yield inp, {"start_dim": 0, "end_dim": -1}


@pytest.mark.flatten
def test_flatten():
    bench = base.GenericBenchmark(
        input_fn=flatten_input_fn,
        op_name="flatten.using_ints",
        torch_op=torch.flatten,
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.run()
