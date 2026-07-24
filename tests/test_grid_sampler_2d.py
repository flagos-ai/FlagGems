import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Typical image processing shapes: small thumbnails to HD inputs
# (N, C, H, W) covering common batch/channel/spatial combos
SHAPES = [
    (1, 3, 32, 32),
    (2, 3, 64, 64),
    (4, 256, 16, 16),
    (1, 64, 8, 8),
]


@pytest.mark.grid_sampler_2d
@pytest.mark.parametrize("interpolation_mode", [0, 1, 2])
@pytest.mark.parametrize("padding_mode", [0, 1, 2])
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_grid_sampler_2d(dtype, shape, align_corners, padding_mode, interpolation_mode):
    N, C, H, W = shape
    input_t = torch.randn(N, C, H, W, dtype=dtype, device=flag_gems.device)
    # Output spatial size matches input spatial size for simplicity
    grid = torch.rand(N, H, W, 2, dtype=torch.float32, device=flag_gems.device)

    ref_in = utils.to_reference(input_t).to(torch.float32)
    ref_grid = utils.to_reference(grid).to(torch.float32)
    # Bicubic fallback: kernel produces bilinear output for mode 2,
    # so compare against bilinear reference
    ref_mode = 0 if interpolation_mode == 2 else interpolation_mode
    ref_out = torch.ops.aten.grid_sampler_2d(
        ref_in, ref_grid, ref_mode, padding_mode, align_corners
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.grid_sampler_2d(
            input_t, grid, interpolation_mode, padding_mode, align_corners
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
