import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    PRIMARY_FLOAT_DTYPES,
    gems_assert_close,
    to_reference,
)
from .conftest import QUICK_MODE

device = flag_gems.device

# 4D shapes: (N, C, H_in, W_in) -> grid: (N, H_out, W_out, 2)
GRID_SAMPLE_4D_SHAPES = [
    # (N, C, H_in, W_in, H_out, W_out)
    (1, 1, 4, 4, 4, 4),  # minimal
    (2, 3, 8, 8, 4, 4),  # small downscale
    (1, 1, 1, 1, 1, 1),  # single pixel
    (2, 16, 32, 32, 16, 16),  # medium
    (1, 3, 64, 64, 32, 32),  # larger
    (2, 3, 7, 11, 5, 9),  # odd/prime dimensions
    (1, 1, 128, 128, 64, 64),  # large
    (4, 8, 17, 31, 13, 23),  # all prime dims
]

GRID_SAMPLE_4D_SHAPES_QUICK = [
    (2, 3, 8, 8, 4, 4),
    (1, 1, 32, 32, 16, 16),
    (2, 3, 7, 11, 5, 9),
]

# 5D shapes: (N, C, D_in, H_in, W_in) -> grid: (N, D_out, H_out, W_out, 3)
GRID_SAMPLE_5D_SHAPES = [
    # (N, C, D_in, H_in, W_in, D_out, H_out, W_out)
    (1, 1, 4, 4, 4, 2, 2, 2),  # minimal
    (2, 3, 4, 4, 4, 3, 3, 3),  # small
    (1, 1, 1, 1, 1, 1, 1, 1),  # single voxel
    (2, 8, 8, 8, 8, 4, 4, 4),  # medium
    (1, 3, 5, 7, 11, 3, 5, 9),  # odd/prime
    (2, 4, 16, 16, 16, 8, 8, 8),  # larger
]

GRID_SAMPLE_5D_SHAPES_QUICK = [
    (2, 3, 4, 4, 4, 3, 3, 3),
    (1, 3, 5, 7, 11, 3, 5, 9),
]

GRID_SAMPLE_MODES_4D = ["bilinear", "nearest", "bicubic"]
GRID_SAMPLE_MODES_5D = ["bilinear", "nearest"]
GRID_SAMPLE_PADDING_MODES = ["zeros", "border", "reflection"]


FLOAT_DTYPES = [torch.float32] if QUICK_MODE else FLOAT_DTYPES


def _make_grid_sample_inputs_4d(shape_tuple, dtype, grid_range=1.0):
    N, C, IH, IW, OH, OW = shape_tuple
    inp = torch.randn(N, C, IH, IW, device=device, dtype=dtype)
    grid = torch.randn(N, OH, OW, 2, device=device, dtype=dtype) * grid_range
    return inp, grid


def _make_grid_sample_inputs_5d(shape_tuple, dtype, grid_range=1.0):
    N, C, ID, IH, IW, OD, OH, OW = shape_tuple
    inp = torch.randn(N, C, ID, IH, IW, device=device, dtype=dtype)
    grid = torch.randn(N, OD, OH, OW, 3, device=device, dtype=dtype) * grid_range
    return inp, grid


@pytest.mark.grid_sampler_2d
@pytest.mark.parametrize(
    "shape",
    GRID_SAMPLE_4D_SHAPES_QUICK if QUICK_MODE else GRID_SAMPLE_4D_SHAPES,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("mode", GRID_SAMPLE_MODES_4D)
@pytest.mark.parametrize("padding_mode", GRID_SAMPLE_PADDING_MODES)
@pytest.mark.parametrize("align_corners", [True, False])
def test_accuracy_grid_sampler_2d(shape, dtype, mode, padding_mode, align_corners):
    inp, grid = _make_grid_sample_inputs_4d(shape, dtype)

    if dtype == torch.bfloat16 and mode == "bicubic" and padding_mode == "reflection":
        ref_out = torch.nn.functional.grid_sample(
            inp.float(),
            grid.float(),
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        ).to(dtype)
    else:
        ref_out = torch.nn.functional.grid_sample(
            inp,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.grid_sample(
            inp,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

    gems_assert_close(res_out, to_reference(ref_out), dtype)


@pytest.mark.grid_sampler_2d
@pytest.mark.parametrize(
    "shape",
    GRID_SAMPLE_4D_SHAPES_QUICK if QUICK_MODE else GRID_SAMPLE_4D_SHAPES,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("mode", GRID_SAMPLE_MODES_4D)
@pytest.mark.parametrize("padding_mode", GRID_SAMPLE_PADDING_MODES)
def test_accuracy_grid_sampler_2d_extreme_grid(shape, dtype, mode, padding_mode):
    inp, grid = _make_grid_sample_inputs_4d(shape, dtype, grid_range=3.0)

    if dtype == torch.bfloat16:
        ref_out = torch.nn.functional.grid_sample(
            inp.float(),
            grid.float(),
            mode=mode,
            padding_mode=padding_mode,
            align_corners=False,
        ).to(dtype)
    else:
        ref_out = torch.nn.functional.grid_sample(
            inp,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=False,
        )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.grid_sample(
            inp,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=False,
        )

    gems_assert_close(res_out, to_reference(ref_out), dtype)


@pytest.mark.grid_sampler_3d
@pytest.mark.parametrize(
    "shape",
    GRID_SAMPLE_5D_SHAPES_QUICK if QUICK_MODE else GRID_SAMPLE_5D_SHAPES,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("mode", GRID_SAMPLE_MODES_5D)
@pytest.mark.parametrize("padding_mode", GRID_SAMPLE_PADDING_MODES)
@pytest.mark.parametrize("align_corners", [True, False])
def test_accuracy_grid_sampler_3d(shape, dtype, mode, padding_mode, align_corners):
    inp, grid = _make_grid_sample_inputs_5d(shape, dtype)

    ref_out = torch.nn.functional.grid_sample(
        inp,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.grid_sample(
            inp,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

    gems_assert_close(res_out, to_reference(ref_out), dtype)


@pytest.mark.grid_sampler_2d
@pytest.mark.parametrize("dtype", PRIMARY_FLOAT_DTYPES)
def test_accuracy_grid_sampler_2d_nan_grid(dtype):
    N, C, IH, IW = 1, 3, 8, 8
    OH, OW = 4, 4
    inp = torch.randn(N, C, IH, IW, device=device, dtype=dtype)
    grid = torch.randn(N, OH, OW, 2, device=device, dtype=dtype)
    grid_nan = grid.clone()
    grid_nan[0, 0, 0, 0] = float("nan")
    grid_nan[0, 1, 1, 1] = float("nan")
    grid_ref = grid.clone()
    grid_ref[0, 0, 0, 0] = -1.0
    grid_ref[0, 1, 1, 1] = -1.0

    ref_out = torch.nn.functional.grid_sample(
        inp,
        grid_ref,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.grid_sample(
            inp,
            grid_nan,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

    gems_assert_close(res_out, to_reference(ref_out), dtype)
