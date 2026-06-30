import pytest
import torch

from . import base, consts

GRID_SAMPLER_SHAPES = [
    (1, 1, 32, 32),
    (1, 3, 64, 64),
    (2, 1, 128, 128),
]


class GridSampler2DBenchmark(base.Benchmark):
    """Benchmark wrapper for grid_sampler_2d, a non-pointwise sampling operator."""

    def __init__(self, op_name, torch_op, dtypes=None, **kwargs):
        super().__init__(op_name, torch_op, dtypes, **kwargs)
        self.shapes = GRID_SAMPLER_SHAPES

    def set_shapes(self, shape_file_path=None):
        """Override to prevent loading from shape file."""

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            N, C, H, W = shape
            OH, OW = H // 2, W // 2
            inp = torch.randn(size=shape, device=self.device, dtype=cur_dtype)
            grid = torch.randn(N, OH, OW, 2, device=self.device, dtype=cur_dtype)
            yield inp, grid, 0, 0, True


@pytest.mark.grid_sampler_2d
def test_grid_sampler_2d():
    bench = GridSampler2DBenchmark(
        op_name="grid_sampler_2d",
        torch_op=torch.grid_sampler_2d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
