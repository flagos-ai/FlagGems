import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Typical image processing shapes: small thumbnails to HD inputs
# (N, C, H, W) covering common batch/channel/spatial combos, including a
# non-square input (H != W) so the IH != IW indexing paths are exercised.
SHAPES = [
    (1, 3, 32, 32),
    (2, 3, 64, 64),
    (4, 256, 16, 16),
    (1, 64, 8, 8),
    (1, 3, 16, 8),
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
    # Output spatial size matches input spatial size for simplicity.
    # Grid values span beyond [-1, 1] so coordinates land out of bounds,
    # exercising the zeros / border / reflection padding branches (a grid
    # confined to [0, 1] via torch.rand never goes out of bounds and leaves
    # those branches uncovered, including the multi-period reflection path).
    grid = (
        torch.rand(N, H, W, 2, dtype=torch.float32, device=flag_gems.device) * 2.4 - 1.2
    )

    ref_in = utils.to_reference(input_t).to(torch.float32)
    ref_grid = utils.to_reference(grid).to(torch.float32)
    # Reference uses the same interpolation mode as the kernel under,
    # the bicubic path which uses Keys cubic convolution (a=-0.75).
    ref_code = _INTERP_CODE[interpolation_mode]
    pad_code = _PAD_CODE[padding_mode]
    ref_out = torch.ops.aten.grid_sampler_2d(
        ref_in, ref_grid, ref_code, pad_code, align_corners
    )

    res_out = flag_gems.grid_sampler_2d(
        input_t, grid, ref_code, pad_code, align_corners
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.grid_sampler_2d
@pytest.mark.parametrize("interpolation_mode", INTERPOLATION_MODES)
@pytest.mark.parametrize("padding_mode", PADDING_MODES)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_grid_sampler_2d_oh_ne_iw(
    dtype, align_corners, padding_mode, interpolation_mode
):
    # Output spatial size differs from the input spatial size (OH != IH,
    # OW != IW), exercising the independent output-/input-dimension indexing
    # paths in the kernel which the same-size shapes above leave uncovered.
    N, C, IH, IW = 1, 2, 8, 16
    OH, OW = 5, 7
    input_t = torch.randn(N, C, IH, IW, dtype=dtype, device=flag_gems.device)
    grid = (
        torch.rand(N, OH, OW, 2, dtype=torch.float32, device=flag_gems.device) * 2.4
        - 1.2
    )

    ref_in = utils.to_reference(input_t).to(torch.float32)
    ref_grid = utils.to_reference(grid).to(torch.float32)
    ref_code = _INTERP_CODE[interpolation_mode]
    pad_code = _PAD_CODE[padding_mode]
    ref_out = torch.ops.aten.grid_sampler_2d(
        ref_in, ref_grid, ref_code, pad_code, align_corners
    )

    res_out = flag_gems.grid_sampler_2d(
        input_t, grid, ref_code, pad_code, align_corners
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.grid_sampler_2d
@pytest.mark.parametrize("padding_mode", PADDING_MODES)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_grid_sampler_2d_nearest_half_integer(dtype, align_corners, padding_mode):
    # Nearest interpolation must round half-integers to the nearest even index
    # (matching PyTorch's nearbyint). A random grid almost never lands a
    # coordinate within 1e-6 of k + 0.5, so the x_is_half / y_is_half branch is
    # otherwise never exercised.
    #
    # The pixel-space half-integer x = k + 0.5 must be exactly representable so
    # the kernel's is_half guard and torch's nearbyint see the same value.
    #   align_corners=False: x = gx * (W/2) + (W-1)/2  -> exact when W is even.
    #   align_corners=True:  x = (gx + 1) * (W-1)/2   -> exact when W-1 is a
    #     power of two. W = 5 satisfies both (W odd with W-1 = 4 = 2^2, and the
    #     half-integers themselves are exact in fp32), so both align_corners
    #     modes produce exact half-integer pixel coords.
    N, C, H, W = 1, 1, 5, 5
    input_t = torch.randn(N, C, H, W, dtype=dtype, device=flag_gems.device)

    ks = torch.arange(0, W, dtype=torch.float32, device=flag_gems.device)
    x_pix = ks + 0.5  # exact half-integers in [0.5, W-0.5]
    if align_corners:
        gx = x_pix / ((W - 1) / 2) - 1
        gy = torch.full((N, H, W), 0.0, device=flag_gems.device)
    else:
        gx = (x_pix - (W - 1) / 2) / (W / 2)
        gy = torch.full((N, H, W), ((H - 1) / 2) / (H / 2), device=flag_gems.device)
    gx = gx.view(1, 1, W).expand(N, H, W)
    grid = torch.stack([gx, gy], dim=-1).contiguous()

    ref_in = utils.to_reference(input_t).to(torch.float32)
    ref_grid = utils.to_reference(grid).to(torch.float32)
    ref_code = _INTERP_CODE["nearest"]
    pad_code = _PAD_CODE[padding_mode]
    ref_out = torch.ops.aten.grid_sampler_2d(
        ref_in, ref_grid, ref_code, pad_code, align_corners
    )

    res_out = flag_gems.grid_sampler_2d(
        input_t, grid, ref_code, pad_code, align_corners
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.grid_sampler_2d
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_grid_sampler_2d_dim_one_reflection(dtype, align_corners):
    # A spatial dimension of size 1 makes the reflection period 2*(n-1) == 0,
    # hitting the degenerate guard in _reflect_int_index / _reflect_coord that
    # maps every (out-of-bound) coordinate to index 0.
    N, C, H, W = 1, 2, 1, 8
    input_t = torch.randn(N, C, H, W, dtype=dtype, device=flag_gems.device)
    grid = (
        torch.rand(N, H, W, 2, dtype=torch.float32, device=flag_gems.device) * 2.4 - 1.2
    )

    ref_in = utils.to_reference(input_t).to(torch.float32)
    ref_grid = utils.to_reference(grid).to(torch.float32)
    ref_code = _INTERP_CODE["bilinear"]
    pad_code = _PAD_CODE["reflection"]
    ref_out = torch.ops.aten.grid_sampler_2d(
        ref_in, ref_grid, ref_code, pad_code, align_corners
    )

    res_out = flag_gems.grid_sampler_2d(
        input_t, grid, ref_code, pad_code, align_corners
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
