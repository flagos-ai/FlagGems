import pytest
import torch

from . import base, consts


def mse_loss_backward_input_fn(shape, cur_dtype, device):
    grad_output = base.generate_tensor_input(shape, cur_dtype, device)
    inp = base.generate_tensor_input(shape, cur_dtype, device)
    target = base.generate_tensor_input(shape, cur_dtype, device)
    yield grad_output, inp, target, {"reduction": 1}


@pytest.mark.mse_loss_backward
def test_mse_loss_backward():
    bench = base.GenericBenchmark2DOnly(
        input_fn=mse_loss_backward_input_fn,
        op_name="mse_loss_backward",
        torch_op=torch.ops.aten.mse_loss_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
