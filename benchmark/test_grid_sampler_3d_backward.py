from typing import Generator

import pytest
import torch

from . import base


class GridSampler3dBackwardBenchmark(base.Benchmark):
    """Benchmark for grid_sampler_3d_backward operator."""

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        # (N, C, iD, iH, iW, oD, oH, oW) configurations
        # Small to large covering typical 3D grid sampling workloads
        self.shapes = [
            (1, 3, 8, 8, 8, 4, 4, 4),
            (2, 16, 16, 16, 16, 8, 8, 8),
            (4, 32, 16, 16, 16, 8, 8, 8),
            (2, 64, 32, 32, 32, 16, 16, 16),
            (1, 128, 32, 32, 32, 16, 16, 16),
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for config in self.shapes:
            N, C, iD, iH, iW, oD, oH, oW = config
            grad_output = torch.randn(
                N, C, oD, oH, oW, device=self.device, dtype=cur_dtype
            )
            input_tensor = torch.randn(
                N, C, iD, iH, iW, device=self.device, dtype=cur_dtype
            )
            # Grid values in [-1, 1]
            grid = (
                torch.rand(N, oD, oH, oW, 3, device=self.device, dtype=cur_dtype) * 2
                - 1
            )
            yield (
                grad_output,
                input_tensor,
                grid,
                0,  # interpolation_mode: bilinear
                0,  # padding_mode: zeros
                True,  # align_corners
                [True, True],  # output_mask
            )


@pytest.mark.grid_sampler_3d_backward
def test_grid_sampler_3d_backward():
    bench = GridSampler3dBackwardBenchmark(
        op_name="grid_sampler_3d_backward",
        torch_op=torch.ops.aten.grid_sampler_3d_backward.default,
        # PyTorch grid_sampler_3d_backward only supports float32/float64 on CUDA
        dtypes=[torch.float32],
    )
    bench.run()
