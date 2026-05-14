import pytest
import torch

from . import base


@pytest.mark.alpha_dropout
def test_alpha_dropout():
    bench = base.UnaryPointwiseBenchmark(
        op_name="alpha_dropout",
        torch_op=torch.alpha_dropout,
        dtypes=[torch.float32],
    )
    bench.run()
