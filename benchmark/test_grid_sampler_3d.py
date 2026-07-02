import pytest
import torch

from . import base, consts

# Benchmark shapes covering a range of typical 3D grid sampling workloads
GRID_SAMPLER_3D_SHAPES = [
    (1, 3, 8, 8, 8),
    (2, 8, 16, 16, 16),
    (4, 16, 32, 32, 32),
    (8, 32, 64, 64, 64),
]


class GridSampler3DBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = GRID_SAMPLER_3D_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            N, C, ID, IH, IW = shape
            # Output size is half the input for representativeness
            OD, OH, OW = ID // 2, IH // 2, IW // 2
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            grid = torch.randn(N, OD, OH, OW, 3, dtype=cur_dtype, device=self.device)
            grid = grid * 1.5  # Cover more spatial range
            yield inp, grid, 0, 0, False  # bilinear, zeros, align_corners=False


@pytest.mark.grid_sampler_3d
def test_grid_sampler_3d():
    bench = GridSampler3DBenchmark(
        op_name="grid_sampler_3d",
        torch_op=torch.ops.aten.grid_sampler_3d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
