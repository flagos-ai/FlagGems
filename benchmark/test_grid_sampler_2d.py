import pytest
import torch

from . import base, consts, utils

# Common image processing shapes for grid sampling benchmarks
# (N, C, H, W) tuples
_GRID_SHAPES = [
    (1, 3, 112, 112),
    (2, 64, 32, 32),
    (4, 256, 16, 16),
    (8, 3, 64, 64),
]


class GridSampler2dBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = _GRID_SHAPES
        self.shape_desc = "N, C, H, W"


def _input_fn(shape, dtype, device):
    N, C, H, W = shape
    input_t = utils.generate_tensor_input((N, C, H, W), dtype, device)
    grid = torch.rand(N, H, W, 2, dtype=dtype, device=device)
    yield input_t, grid, 0, 0, False


@pytest.mark.grid_sampler_2d
def test_grid_sampler_2d():
    bench = GridSampler2dBenchmark(
        op_name="grid_sampler_2d",
        input_fn=_input_fn,
        torch_op=torch.ops.aten.grid_sampler_2d,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
