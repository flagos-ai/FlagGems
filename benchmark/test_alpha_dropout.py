import pytest
import torch

from . import base, consts


@pytest.mark.alpha_dropout
def test_alpha_dropout():
    bench = base.UnaryPointwiseBenchmark(
        op_name="alpha_dropout",
        torch_op=torch.nn.AlphaDropout(p=0.5),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
