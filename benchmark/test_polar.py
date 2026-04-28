import pytest
import torch

from . import base


@pytest.mark.polar
def test_polar():
    bench = base.BinaryPointwiseBenchmark(
        op_name="polar",
        torch_op=torch.polar,
        dtypes=[torch.float32],
    )
    bench.run()
