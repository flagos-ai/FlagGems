import pytest
import torch

from . import base

GRID_SAMPLER_2D_BACKWARD_SHAPES = [
    (2, 3, 8, 8, 4, 4),
    (4, 16, 32, 32, 16, 16),
]


class GridSampler2DBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = GRID_SAMPLER_2D_BACKWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            N, C, H, W, OH, OW = shape
            input = torch.randn(N, C, H, W, dtype=cur_dtype, device=self.device)
            grid = torch.randn(N, OH, OW, 2, dtype=cur_dtype, device=self.device)
            grad_output = torch.randn(N, C, OH, OW, dtype=cur_dtype, device=self.device)
            # grid_sampler_2d_backward(grad_output, input, grid,
            #   interpolation_mode, padding_mode, align_corners, output_mask)
            yield grad_output, input, grid, 0, 0, False, (True, True)


@pytest.mark.grid_sampler_2d_backward
def test_grid_sampler_2d_backward():
    bench = GridSampler2DBackwardBenchmark(
        op_name="grid_sampler_2d_backward",
        torch_op=torch.ops.aten.grid_sampler_2d_backward,
        dtypes=[torch.float32],
    )
    bench.run()
