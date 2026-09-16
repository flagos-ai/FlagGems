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

    res_out = flag_gems.grid_sampler_3d(
        inp, grid, interpolation_mode, padding_mode, align_corners
    )

    # Tolerances aligned with the 2D grid_sample test suite (ATOL_DICT).
    # float16/bfloat16 values measured on NVIDIA H20 with output size (2,2,2):
    # float16 worst ~9e-4, bfloat16 worst ~7e-3.
    if dtype == torch.float32:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-5)
    elif dtype == torch.float16:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)
    else:  # bfloat16
        utils.gems_assert_close(res_out, ref_out, dtype, atol=0.016)


# Pairs of (input_shape, output_size) covering downsampling, upsampling,
# non-square outputs, and the minimum 1x1x1 case. The fixed (2, 2, 2) output
# in test_grid_sampler_3d above is intentionally limited for CI runtime; this
# parametrization broadens output-size coverage per review feedback.
# float16/bfloat16 excluded: insufficient grid precision for nearest +
# reflection on larger output sizes (low-precision grid coordinates can round
# to a different pixel than the upcast reference), mirroring the dtype
# restriction used by test_upsample_bilinear2d_aa.
GRID_SAMPLER_3D_OUTPUT_SIZE_CONFIGS = [
    ((1, 3, 4, 4, 4), (1, 1, 1)),  # minimum output
    ((1, 3, 4, 4, 4), (2, 2, 2)),  # small output
    ((2, 8, 8, 8, 8), (4, 4, 4)),  # downsample
    ((2, 8, 8, 8, 8), (8, 8, 8)),  # same size
    ((2, 8, 8, 8, 8), (16, 16, 16)),  # upsample
    ((2, 8, 8, 8, 8), (16, 8, 4)),  # non-square output
]


@pytest.mark.grid_sampler_3d
@pytest.mark.parametrize(
    "input_shape, output_size", GRID_SAMPLER_3D_OUTPUT_SIZE_CONFIGS
)
# float32 only: see comment on GRID_SAMPLER_3D_OUTPUT_SIZE_CONFIGS above.
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("interpolation_mode", [0, 1])  # 0=bilinear, 1=nearest
@pytest.mark.parametrize("padding_mode", [0, 1, 2])  # 0=zeros, 1=border, 2=reflection
@pytest.mark.parametrize("align_corners", [True, False])
def test_grid_sampler_3d_varied_output_sizes(
    input_shape, output_size, dtype, interpolation_mode, padding_mode, align_corners
):
    N, C, ID, IH, IW = input_shape
    OD, OH, OW = output_size

    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    grid = torch.randn(N, OD, OH, OW, 3, dtype=dtype, device=flag_gems.device)
    grid = grid * 1.5  # Extend grid range to cover more spatial regions

    # Reference (PyTorch) computation upcast to float64 for accuracy
    ref_inp = utils.to_reference(inp, True)
    ref_grid = utils.to_reference(grid, True)

    ref_out = torch.ops.aten.grid_sampler_3d(
        ref_inp, ref_grid, interpolation_mode, padding_mode, align_corners
    )

    res_out = flag_gems.grid_sampler_3d(
        inp, grid, interpolation_mode, padding_mode, align_corners
    )

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-5)
