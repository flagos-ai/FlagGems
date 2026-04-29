import pytest
import torch

from . import base, consts


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    target = (torch.randint(0, 2, shape, device=device).to(dtype) * 2) - 1
    yield inp, target


@pytest.mark.soft_margin_loss
def test_soft_margin_loss():
    bench = base.GenericBenchmark(
        input_fn=_input_fn,
        op_name="soft_margin_loss",
        torch_op=torch.ops.aten.soft_margin_loss,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
