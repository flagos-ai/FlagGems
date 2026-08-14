import pytest
import torch

import flag_gems

GRID_SAMPLER_SHAPES = [
    (2, 3, 8, 8, 4, 4),
]
GRID_SAMPLER_INTERP_MODE = [0]  # 0=bilinear
GRID_SAMPLER_PADDING_MODE = [0]  # 0=zeros
GRID_SAMPLER_ALIGN_CORNERS = [True, False]


@pytest.mark.grid_sampler_2d_backward
@pytest.mark.parametrize("shape", GRID_SAMPLER_SHAPES)
@pytest.mark.parametrize("interp_mode", GRID_SAMPLER_INTERP_MODE)
@pytest.mark.parametrize("padding_mode", GRID_SAMPLER_PADDING_MODE)
@pytest.mark.parametrize("align_corners", GRID_SAMPLER_ALIGN_CORNERS)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_grid_sampler_2d_backward(
    shape, interp_mode, padding_mode, align_corners, dtype
):
    N, C, H, W, OH, OW = shape
    torch.manual_seed(42)
    input = torch.randn(N, C, H, W, dtype=dtype, device=flag_gems.device)
    grid = torch.randn(N, OH, OW, 2, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(N, C, OH, OW, dtype=dtype, device=flag_gems.device)

    # Compute gradients
    with flag_gems.use_gems():
        grad_input, grad_grid = torch.ops.aten.grid_sampler_2d_backward(
            grad_output,
            input,
            grid,
            interp_mode,
            padding_mode,
            align_corners,
            (True, True),
        )

    # Check shapes
    assert (
        grad_input.shape == input.shape
    ), f"Expected {input.shape}, got {grad_input.shape}"
    assert (
        grad_grid.shape == grid.shape
    ), f"Expected {grid.shape}, got {grad_grid.shape}"

    # Check that we get reasonable gradients (not all zeros or NaN)
    assert not torch.isnan(grad_input).any(), "grad_input contains NaN"
    assert not torch.isnan(grad_grid).any(), "grad_grid contains NaN"

    # With random input and grid, we should have some non-zero gradients
    assert grad_input.abs().sum() > 0, "grad_input is all zeros"
    assert grad_grid.abs().sum() > 0, "grad_grid is all zeros"


@pytest.mark.grid_sampler_2d_backward
@pytest.mark.parametrize("shape", GRID_SAMPLER_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_grid_sampler_2d_backward_grad_input_only(shape, dtype):
    N, C, H, W, OH, OW = shape
    torch.manual_seed(42)
    input = torch.randn(N, C, H, W, dtype=dtype, device=flag_gems.device)
    grid = torch.randn(N, OH, OW, 2, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(N, C, OH, OW, dtype=dtype, device=flag_gems.device)

    # Compute only grad_input
    with flag_gems.use_gems():
        grad_input, grad_grid = torch.ops.aten.grid_sampler_2d_backward(
            grad_output, input, grid, 0, 0, False, (True, False)
        )

    # Check shapes
    assert grad_input.shape == input.shape
    assert grad_grid is None

    # Check validity
    assert not torch.isnan(grad_input).any()
    assert grad_input.abs().sum() > 0


@pytest.mark.grid_sampler_2d_backward
@pytest.mark.parametrize("shape", GRID_SAMPLER_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_grid_sampler_2d_backward_grad_grid_only(shape, dtype):
    N, C, H, W, OH, OW = shape
    torch.manual_seed(42)
    input = torch.randn(N, C, H, W, dtype=dtype, device=flag_gems.device)
    grid = torch.randn(N, OH, OW, 2, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(N, C, OH, OW, dtype=dtype, device=flag_gems.device)

    # Compute only grad_grid
    with flag_gems.use_gems():
        grad_input, grad_grid = torch.ops.aten.grid_sampler_2d_backward(
            grad_output, input, grid, 0, 0, False, (False, True)
        )

    # Check shapes
    assert grad_input is None
    assert grad_grid.shape == grid.shape

    # Check validity
    assert not torch.isnan(grad_grid).any()
    assert grad_grid.abs().sum() > 0
