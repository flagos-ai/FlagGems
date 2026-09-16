import pytest
import torch

from . import base


def unfold_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    # Unfold along dim 0 with size=4, step=2
    yield inp, {"dimension": 0, "size": 4, "step": 2}


@pytest.mark.unfold
def test_unfold_view():
    bench = base.GenericBenchmark(
        input_fn=unfold_input_fn,
        op_name="unfold",
        torch_op=torch.Tensor.unfold,
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.run()
