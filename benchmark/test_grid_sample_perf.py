import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark

GRID_SAMPLE_2D_SHAPES = [
    (1, 3, 128, 128),
    (1, 3, 512, 512),
    (1, 3, 1024, 1024),
    (4, 16, 128, 128),
    (8, 32, 64, 64),
    (2, 64, 256, 256),
    (16, 64, 256, 256),
    (1, 128, 512, 512),
]

GRID_SAMPLE_3D_SHAPES = [
    (1, 3, 16, 32, 32),
    (1, 3, 16, 64, 64),
    (1, 3, 32, 64, 64),
    (2, 8, 16, 32, 32),
    (4, 16, 8, 32, 32),
    (2, 16, 32, 64, 64),
]


class GridSample2DBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        return None


class GridSample3DBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        return None


@pytest.mark.grid_sampler_2d
def test_perf_grid_sampler_2d():
    def grid_sampler_2d_input_fn(shape, dtype, device):
        N, C, IH, IW = shape
        inp = torch.randn(shape, device=device, dtype=dtype)
        OH, OW = IH, IW
        grid = torch.rand((N, OH, OW, 2), device=device, dtype=dtype) * 2 - 1
        yield {
            "input": inp,
            "grid": grid,
            "mode": "bilinear",
            "padding_mode": "zeros",
            "align_corners": False,
        },

    bench = GridSample2DBenchmark(
        input_fn=grid_sampler_2d_input_fn,
        op_name="grid_sampler_2d",
        torch_op=torch.nn.functional.grid_sample,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.grid_sampler_3d
def test_perf_grid_sampler_3d():
    def grid_sampler_3d_input_fn(shape, dtype, device):
        N, C, ID, IH, IW = shape
        inp = torch.randn(shape, device=device, dtype=dtype)
        OD, OH, OW = ID, IH, IW
        grid = torch.rand((N, OD, OH, OW, 3), device=device, dtype=dtype) * 2 - 1
        yield {
            "input": inp,
            "grid": grid,
            "mode": "bilinear",
            "padding_mode": "zeros",
            "align_corners": False,
        },

    bench = GridSample3DBenchmark(
        input_fn=grid_sampler_3d_input_fn,
        op_name="grid_sampler_3d",
        torch_op=torch.nn.functional.grid_sample,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
