import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.grid_sampler_3d_backward
@pytest.mark.parametrize("interpolation_mode", [0, 1])  # bilinear, nearest
@pytest.mark.parametrize("padding_mode", [0, 1, 2])  # zeros, border, reflection
@pytest.mark.parametrize("align_corners", [True, False])
# PyTorch grid_sampler_3d_backward only supports float32 on CUDA
@pytest.mark.parametrize("dtype", [torch.float32])
def test_grid_sampler_3d_backward(
    interpolation_mode, padding_mode, align_corners, dtype
):
    # Fixed shape: (N, C, D, H, W) for input, (N, oD, oH, oW, 3) for grid
    N, C, D, H, W = 2, 3, 8, 8, 8
    oD, oH, oW = 4, 4, 4
    # Create input and grid
    input_tensor = torch.randn(N, C, D, H, W, device=flag_gems.device, dtype=dtype)
    # Grid values in [-1, 1]
    grid = torch.rand(N, oD, oH, oW, 3, device=flag_gems.device, dtype=dtype) * 2 - 1

    # Create grad_output matching forward output shape
    grad_output = torch.randn(N, C, oD, oH, oW, device=flag_gems.device, dtype=dtype)

    # Reference
    ref_input = utils.to_reference(input_tensor)
    ref_grid = utils.to_reference(grid)
    ref_grad_output = utils.to_reference(grad_output)

    ref_out = torch.ops.aten.grid_sampler_3d_backward.default(
        ref_grad_output,
        ref_input,
        ref_grid,
        interpolation_mode,
        padding_mode,
        align_corners,
        [True, True],
    )

    # Gems result
    with flag_gems.use_gems():
        res_out = torch.ops.aten.grid_sampler_3d_backward.default(
            grad_output,
            input_tensor,
            grid,
            interpolation_mode,
            padding_mode,
            align_corners,
            [True, True],
        )

    # Compare grad_input
    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    # Compare grad_grid
    utils.gems_assert_close(res_out[1], ref_out[1], dtype)
