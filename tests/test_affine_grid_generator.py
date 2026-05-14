import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shape configurations for affine_grid_generator: (N, H, W)
AFFINE_GRID_SHAPES = (
    [(1, 4, 4)]
    if QUICK_MODE
    else [
        (1, 2, 2),
        (1, 4, 4),
        (2, 8, 8),
        (4, 16, 16),
        (1, 32, 32),
        (2, 64, 64),
        (1, 128, 128),
    ]
)


@pytest.mark.affine_grid_generator
@pytest.mark.parametrize("shape", AFFINE_GRID_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("align_corners", [False, True])
def test_accuracy_affine_grid_generator(shape, dtype, align_corners):
    N, H, W = shape
    # theta: (N, 2, 3) affine transformation matrix
    theta = torch.randn((N, 2, 3), dtype=dtype, device=flag_gems.device)
    # size: [N, C, H, W] - we use C=3 for typical RGB images
    size = [N, 3, H, W]

    ref_theta = to_reference(theta)

    ref_out = torch.affine_grid_generator(ref_theta, size, align_corners)
    with flag_gems.use_gems():
        res_out = torch.affine_grid_generator(theta, size, align_corners)

    # Use dtype-appropriate tolerance since float16/bfloat16 have coarser resolution
    if dtype == torch.float16:
        gems_assert_close(res_out, ref_out, dtype, atol=1e-3)
    elif dtype == torch.bfloat16:
        gems_assert_close(res_out, ref_out, dtype, atol=2e-2)
    else:
        gems_assert_close(res_out, ref_out, dtype)
