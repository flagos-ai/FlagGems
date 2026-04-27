import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.grid_sample
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border"])
@pytest.mark.parametrize("align_corners", [True, False])
def test_accuracy_grid_sample(dtype, mode, padding_mode, align_corners):
    N, C, IH, IW = 2, 3, 8, 8
    OH, OW = 4, 4
    inp = torch.randn(N, C, IH, IW, dtype=dtype, device=flag_gems.device)
    grid = torch.randn(N, OH, OW, 2, dtype=dtype, device=flag_gems.device) * 0.5
    ref_inp = utils.to_reference(inp, True)
    ref_grid = utils.to_reference(grid, True)

    ref_out = torch.nn.functional.grid_sample(
        ref_inp,
        ref_grid,
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

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.grid_sample
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        {"input": (1, 1, 4, 4), "grid": (1, 2, 2, 2)},
        {"input": (2, 3, 16, 16), "grid": (2, 8, 8, 2)},
        {"input": (1, 1, 32, 32), "grid": (1, 16, 16, 2)},
        {"input": (4, 8, 64, 64), "grid": (4, 32, 32, 2)},
    ],
)
def test_accuracy_grid_sample_various_sizes(dtype, shape):
    inp = torch.randn(shape["input"], dtype=dtype, device=flag_gems.device)
    grid = torch.randn(shape["grid"], dtype=dtype, device=flag_gems.device) * 0.5
    ref_inp = utils.to_reference(inp, True)
    ref_grid = utils.to_reference(grid, True)

    ref_out = torch.nn.functional.grid_sample(ref_inp, ref_grid, align_corners=True)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.grid_sample(inp, grid, align_corners=True)

    utils.gems_assert_close(res_out, ref_out, dtype)
