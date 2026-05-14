import pytest
import torch

from . import base


@pytest.mark.cross_attention
def test_cross_attention():
    bench = base.UnaryPointwiseBenchmark(
        op_name="cross_attention",
        torch_op=torch.cross_attention,
        dtypes=[torch.float32],
    )
    bench.run()
