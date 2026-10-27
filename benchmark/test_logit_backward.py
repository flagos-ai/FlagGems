import pytest
import torch

from . import base, consts


@pytest.mark.logit_backward
def test_logit_backward():
    bench = base.UnaryPointwiseBenchmark(
        op_name="logit_backward",
        torch_op=lambda a: torch.ops.aten.logit_backward(
            torch.ones_like(a), torch.rand_like(a), 1e-6
        ),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
