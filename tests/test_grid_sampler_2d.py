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

# Supporting string inputs for interpolation and padding modes.
INTERPOLATION_MODES = ["bilinear", "nearest", "bicubic"]
PADDING_MODES = ["zeros", "border", "reflection"]

_INTERP_CODE = {"bilinear": 0, "nearest": 1, "bicubic": 2}
_PAD_CODE = {"zeros": 0, "border": 1, "reflection": 2}


@pytest.mark.grid_sampler_2d
@pytest.mark.parametrize("interpolation_mode", INTERPOLATION_MODES)
@pytest.mark.parametrize("padding_mode", PADDING_MODES)
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
    # Reference uses the same interpolation mode as the kernel under,
    # the bicubic path which uses Keys cubic convolution (a=-0.75).
    ref_code = _INTERP_CODE[interpolation_mode]
    pad_code = _PAD_CODE[padding_mode]
    ref_out = torch.ops.aten.grid_sampler_2d(
        ref_in, ref_grid, ref_code, pad_code, align_corners
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.grid_sampler_2d(
            input_t, grid, ref_code, pad_code, align_corners
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
