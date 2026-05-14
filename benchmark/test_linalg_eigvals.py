import pytest
import torch

from . import base


@pytest.mark.linalg_eigvals
def test_linalg_eigvals():
    bench = base.UnaryPointwiseBenchmark(
        op_name="linalg_eigvals",
        torch_op=torch.linalg_eigvals,
        dtypes=[torch.float32],
    )
    bench.run()
