import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.grid_sampler_2d
@pytest.mark.parametrize("shape", [(1, 1, 4, 4), (1, 3, 8, 8)])  # Simple shapes only
# kernel 当前仅支持 float32
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("interpolation_mode", [0])  # Bilinear only for now
@pytest.mark.parametrize("align_corners", [True])
def test_grid_sampler_2d(shape, dtype, interpolation_mode, align_corners):
    N, C, H, W = shape
    OH, OW = H // 2, W // 2

    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grid = torch.randn(N, OH, OW, 2, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input)
    ref_grid = utils.to_reference(grid)

    ref_out = torch.grid_sampler_2d(
        ref_input, ref_grid, interpolation_mode, 0, align_corners  # zeros padding
    )
    with flag_gems.use_gems():
        res_out = torch.grid_sampler_2d(
            input, grid, interpolation_mode, 0, align_corners
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
