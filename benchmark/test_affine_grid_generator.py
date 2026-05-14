import pytest
import torch

from . import base


@pytest.mark.affine_grid_generator
def test_affine_grid_generator():
    bench = base.UnaryPointwiseBenchmark(
        op_name="affine_grid_generator",
        torch_op=torch.affine_grid_generator,
        dtypes=[torch.float32],
    )
    bench.run()
