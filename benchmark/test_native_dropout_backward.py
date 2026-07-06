import pytest
import torch

from . import base, consts


def native_dropout_backward_input_fn(shape, dtype, device):
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    mask = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    scale = 2.0
    yield grad_output, mask, scale


@pytest.mark.native_dropout_backward
def test_native_dropout_backward():
    bench = base.GenericBenchmark(
        input_fn=native_dropout_backward_input_fn,
        op_name="native_dropout_backward",
        torch_op=torch.ops.aten.native_dropout_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
