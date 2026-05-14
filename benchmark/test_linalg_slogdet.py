import pytest
import torch

from . import base


@pytest.mark.linalg_slogdet
def test_linalg_slogdet():
    bench = base.UnaryPointwiseBenchmark(
        op_name="linalg_slogdet",
        torch_op=torch.linalg_slogdet,
        dtypes=[torch.float32],
    )
    bench.run()
