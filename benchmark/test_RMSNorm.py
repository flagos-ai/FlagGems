import pytest
import torch

from . import base


@pytest.mark.RMSNorm
def test_RMSNorm():
    bench = base.UnaryPointwiseBenchmark(
        op_name="RMSNorm",
        torch_op=torch.RMSNorm,
        dtypes=[torch.float32],
    )
    bench.run()
