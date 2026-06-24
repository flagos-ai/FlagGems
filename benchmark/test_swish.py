import pytest
import torch

from . import base, consts


@pytest.mark.swish
def test_swish():
    bench = base.UnaryPointwiseBenchmark(
        op_name="swish",
        torch_op=torch.nn.functional.silu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
