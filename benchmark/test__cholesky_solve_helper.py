import pytest
import torch

from . import base


@pytest.mark._cholesky_solve_helper
def test__cholesky_solve_helper():
    bench = base.UnaryPointwiseBenchmark(
        op_name="_cholesky_solve_helper",
        torch_op=torch._cholesky_solve_helper,
        dtypes=[torch.float32],
    )
    bench.run()
