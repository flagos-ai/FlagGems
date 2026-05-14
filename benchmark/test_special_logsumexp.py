import pytest
import torch

from . import base


@pytest.mark.special_logsumexp
def test_special_logsumexp():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_logsumexp",
        torch_op=torch.special_logsumexp,
        dtypes=[torch.float32],
    )
    bench.run()
