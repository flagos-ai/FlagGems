import pytest
import torch

from . import base


def cudnn_is_acceptable_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield inp, {}


@pytest.mark.cudnn_is_acceptable
def test_cudnn_is_acceptable():
    bench = base.GenericBenchmark(
        input_fn=cudnn_is_acceptable_input_fn,
        op_name="cudnn_is_acceptable",
        torch_op=torch.ops.aten.cudnn_is_acceptable.default,
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.run()
