import pytest
import torch

from . import base


class GridSamplerBenchmark(base.Benchmark):
    def __init__(self, op_name, torch_op, dtypes, interp_mode, pad_mode, align_corners):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
        self.interp_mode = interp_mode
        self.pad_mode = pad_mode
        self.align_corners = align_corners

    def set_shapes(self, shape_file_path=None):
        # (N, C, H_in, W_in, H_out, W_out)
        self.shapes = [
            (2, 16, 64, 64, 48, 48),
            (4, 32, 128, 128, 96, 96),
            (1, 64, 256, 256, 192, 192),
            (1, 64, 512, 512, 384, 384),
            (1, 64, 768, 768, 576, 576),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype):
        for N, C, H_in, W_in, H_out, W_out in self.shapes:
            x = torch.randn((N, C, H_in, W_in), dtype=cur_dtype, device=self.device)
            grid = (
                torch.rand((N, H_out, W_out, 2), dtype=cur_dtype, device=self.device)
                * 2.0
                - 1.0
            )
            yield x, grid, self.interp_mode, self.pad_mode, self.align_corners


class GridSampler3DBenchmark(base.Benchmark):
    def __init__(self, op_name, torch_op, dtypes, interp_mode, pad_mode, align_corners):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
        self.interp_mode = interp_mode
        self.pad_mode = pad_mode
        self.align_corners = align_corners

    def set_shapes(self, shape_file_path=None):
        # (N, C, D_in, H_in, W_in, D_out, H_out, W_out)
        self.shapes = [
            (2, 8, 16, 16, 16, 12, 12, 12),
            (1, 32, 48, 48, 48, 36, 36, 36),
            (2, 32, 128, 128, 128, 64, 64, 64),
            (1, 16, 256, 256, 256, 128, 128, 128),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype):
        for N, C, D_in, H_in, W_in, D_out, H_out, W_out in self.shapes:
            x = torch.randn(
                (N, C, D_in, H_in, W_in), dtype=cur_dtype, device=self.device
            )
            grid = (
                torch.rand(
                    (N, D_out, H_out, W_out, 3), dtype=cur_dtype, device=self.device
                )
                * 2.0
                - 1.0
            )
            yield x, grid, self.interp_mode, self.pad_mode, self.align_corners


@pytest.mark.grid_sampler_2d
@pytest.mark.parametrize(
    "interp_name,interp_mode",
    [("bilinear", 0), ("nearest", 1), ("bicubic", 2)],
)
@pytest.mark.parametrize(
    "pad_name,pad_mode",
    [("zeros", 0), ("reflection", 2)],
)
def test_perf_grid_sampler_2d(interp_name, interp_mode, pad_name, pad_mode):
    bench = GridSamplerBenchmark(
        op_name=f"grid_sampler_2d_{interp_name}_{pad_name}",
        torch_op=torch.ops.aten.grid_sampler_2d,
        dtypes=[torch.float16, torch.float32],
        interp_mode=interp_mode,
        pad_mode=pad_mode,
        align_corners=False,
    )
    bench.run()


@pytest.mark.grid_sampler_3d
@pytest.mark.parametrize(
    "interp_name,interp_mode",
    [("bilinear", 0), ("nearest", 1)],
)
@pytest.mark.parametrize(
    "pad_name,pad_mode",
    [("zeros", 0), ("reflection", 2)],
)
def test_perf_grid_sampler_3d(interp_name, interp_mode, pad_name, pad_mode):
    bench = GridSampler3DBenchmark(
        op_name=f"grid_sampler_3d_{interp_name}_{pad_name}",
        torch_op=torch.ops.aten.grid_sampler_3d,
        dtypes=[torch.float16, torch.float32],
        interp_mode=interp_mode,
        pad_mode=pad_mode,
        align_corners=False,
    )
    bench.run()
