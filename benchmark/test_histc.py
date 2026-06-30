import pytest
import torch

from . import base


def _input_fn(shape, dtype, device):
    inp = torch.rand(shape, dtype=dtype, device=device) * 10
    yield inp, {"bins": 100, "min": 0, "max": 10}


@pytest.mark.histc
def test_histc():
    bench = base.GenericBenchmark2DOnly(
        input_fn=_input_fn,
        op_name="histc",
        torch_op=torch.histc,
        dtypes=[torch.float32],
    )
    bench.run()
