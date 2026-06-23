import pytest
import torch

from . import base, consts


@pytest.mark.soft_margin_loss_backward
def test_soft_margin_loss_backward():
    def soft_margin_loss_backward_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        target = (torch.randint(0, 2, shape, device=device).to(dtype) * 2) - 1
        grad_output = torch.ones(shape, dtype=dtype, device=device)
        yield grad_output, inp, target, 1

    bench = base.GenericBenchmark(
        input_fn=soft_margin_loss_backward_input_fn,
        op_name="soft_margin_loss_backward",
        torch_op=torch.ops.aten.soft_margin_loss_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
