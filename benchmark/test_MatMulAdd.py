import pytest
import torch

from . import base


@pytest.mark.MatMulAdd
def test_MatMulAdd():
    bench = base.UnaryPointwiseBenchmark(
        op_name="MatMulAdd",
        torch_op=torch.MatMulAdd,
        dtypes=[torch.float32],
    )
    bench.run()
