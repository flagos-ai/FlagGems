import pytest
import torch

from . import base


@pytest.mark.igammac_
def test_igammac_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="igammac_",
        torch_op=torch.igammac_,
        dtypes=[torch.float32],
    )
    bench.run()
