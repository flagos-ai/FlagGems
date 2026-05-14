import pytest
import torch

from . import base


@pytest.mark._euclidean_dist
def test__euclidean_dist():
    bench = base.UnaryPointwiseBenchmark(
        op_name="_euclidean_dist",
        torch_op=torch._euclidean_dist,
        dtypes=[torch.float32],
    )
    bench.run()
