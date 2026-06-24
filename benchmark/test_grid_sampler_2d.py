import sys

import pytest
import torch

import flag_gems

from . import base, consts  # noqa: F401

# Representative shapes for grid_sampler_2d: small (1,1,32,32),
# medium (1,3,64,64), and larger (2,1,128,128) with different aspect ratios
GRID_SAMPLER_SHAPES = [
    (1, 1, 32, 32),
    (1, 3, 64, 64),
    (2, 1, 128, 128),
]


# Custom benchmark: runs inline without bench.run() due to complex op signature
@pytest.mark.grid_sampler_2d
def test_grid_sampler_2d():
    for shape in GRID_SAMPLER_SHAPES:
        for dtype in consts.FLOAT_DTYPES:
            N, C, H, W = shape
            OH, OW = H // 2, W // 2

            input = torch.randn(size=shape, device=flag_gems.device, dtype=dtype)
            grid = torch.randn(N, OH, OW, 2, device=flag_gems.device, dtype=dtype)

            # Warmup
            with flag_gems.use_gems():
                for _ in range(10):
                    _ = torch.grid_sampler_2d(input, grid, 0, 0, True)

            # Benchmark
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            with flag_gems.use_gems():
                start.record()
                for _ in range(100):
                    _ = torch.grid_sampler_2d(input, grid, 0, 0, True)
                end.record()
                torch.cuda.synchronize()
                gems_time = start.elapsed_time(end) / 100

            # Reference
            ref_input = input.cpu().to(dtype)
            ref_grid = grid.cpu().to(dtype)
            start.record()
            for _ in range(100):
                _ = torch.grid_sampler_2d(ref_input, ref_grid, 0, 0, True)
            end.record()
            torch.cuda.synchronize()
            torch_time = start.elapsed_time(end) / 100

            speedup = torch_time / gems_time
            sys.stdout.write(
                f"SUCCESS  {torch_time:.3f}  {gems_time:.3f}  {speedup:.3f}"
                f"  {{shape=torch.Size({shape}) dtype={str(dtype).split('.')[-1]}}}\n"
            )
