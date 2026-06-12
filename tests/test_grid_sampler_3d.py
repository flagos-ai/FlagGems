import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Input shapes covering various batch/channel/spatial sizes
GRID_SAMPLER_3D_SHAPES = [
    (1, 3, 4, 4, 4),
    (2, 8, 8, 8, 8),
    (1, 16, 16, 16, 16),
    (4, 8, 8, 16, 16),
    (2, 4, 16, 16, 16),
]


@pytest.mark.grid_sampler_3d
@pytest.mark.parametrize("input_shape", GRID_SAMPLER_3D_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("interpolation_mode", [0, 1])  # 0=bilinear, 1=nearest
@pytest.mark.parametrize("padding_mode", [0, 1, 2])  # 0=zeros, 1=border, 2=reflection
@pytest.mark.parametrize("align_corners", [True, False])
def test_grid_sampler_3d(
    input_shape, dtype, interpolation_mode, padding_mode, align_corners
):
    N, C, ID, IH, IW = input_shape
    # Fixed small output size to keep test runtime manageable
    OD, OH, OW = 2, 2, 2

    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    grid = torch.randn(N, OD, OH, OW, 3, dtype=dtype, device=flag_gems.device)
    grid = grid * 1.5  # Extend grid range to cover more spatial regions

    # Reference (PyTorch) computation in float32 for accuracy
    ref_inp = utils.to_reference(inp, True)
    ref_grid = utils.to_reference(grid, True)

    ref_out = torch.ops.aten.grid_sampler_3d(
        ref_inp, ref_grid, interpolation_mode, padding_mode, align_corners
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.grid_sampler_3d(
            inp, grid, interpolation_mode, padding_mode, align_corners
        )

    # Relaxed tolerance for lower-precision dtypes
    if dtype == torch.float32:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-5)
    elif dtype == torch.float16:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=5e-3)
    else:  # bfloat16
        utils.gems_assert_close(res_out, ref_out, dtype, atol=2e-2)
