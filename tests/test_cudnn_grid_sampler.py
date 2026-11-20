# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

# (N, C, H_in, W_in, H_out, W_out) tuples for the 2D cuDNN grid sampler.
if cfg.QUICK_MODE:
    # Single minimal case for --quick CI runs.
    GRID_SAMPLER_SHAPES = [
        (1, 3, 8, 8, 8, 8),
    ]
else:
    # Includes degenerate 1x1 inputs/grids, odd and non-square sizes, and
    # upsampled outputs.
    GRID_SAMPLER_SHAPES = [
        (1, 1, 1, 1, 1, 1),
        (1, 1, 2, 2, 4, 3),
        (1, 3, 8, 8, 8, 8),
        (2, 3, 16, 24, 16, 24),
        (2, 8, 32, 32, 32, 32),
        (4, 3, 17, 23, 33, 41),
        (2, 16, 64, 64, 128, 128),
        (1, 3, 5, 7, 1, 1),
    ]

# cudnn does not support bfloat16 for this operator (CUDNN_STATUS_BAD_PARAM);
# float16 and float32 are supported (cudnn computes in float32 internally).
GRID_SAMPLER_DTYPES = [torch.float16, torch.float32]

# ``aten::cudnn_grid_sampler`` == ``grid_sample`` with these fixed options, so
# the reference is taken from the portable ``aten::grid_sampler_2d`` instead of
# the CUDA-only ``cudnn_grid_sampler``. That keeps the whole file runnable on
# every backend and under ``--ref cpu``.
BILINEAR, ZEROS_PADDING, ALIGN_CORNERS = 0, 0, True


def _grid_sample_reference(inp, grid):
    return torch.ops.aten.grid_sampler_2d(
        inp, grid, BILINEAR, ZEROS_PADDING, ALIGN_CORNERS
    )


def _cpu_grid_sample_reference(inp, grid):
    # The CUDA ``grid_sampler_2d`` kernel maps non-finite grid coordinates to
    # zero while the CPU kernel (and cuDNN) propagate NaN, so the non-finite
    # cases are checked against the CPU reference on every backend.
    return _grid_sample_reference(inp.cpu(), grid.cpu()).to(inp.device)


def _grid_sample_reference_out(inp, grid, out):
    return torch.ops.aten.grid_sampler_2d.out(
        inp, grid, BILINEAR, ZEROS_PADDING, ALIGN_CORNERS, out=out
    )


def _make_grid(N, H_out, W_out, dtype, device, mode="in_range"):
    """Build a grid tensor for the grid sampler.

    ``in_range`` samples the interior, ``full`` covers the whole [-1, 1] range
    (touching pixel centers), and ``out_of_range`` pushes part of the samples
    outside the input image so the zero-padding path is exercised.
    """
    grid = torch.randn(N, H_out, W_out, 2, dtype=torch.float32, device=device)
    if mode == "in_range":
        grid = torch.clamp(grid, -0.9, 0.9)
    elif mode == "full":
        grid = grid * 0.5
        grid = torch.clamp(grid, -1.0, 1.0)
    else:
        grid = grid * 1.5
    return grid.to(dtype)


def _make_inputs(N, C, H_in, W_in, H_out, W_out, dtype, mode):
    inp = torch.randn(N, C, H_in, W_in, dtype=dtype, device=flag_gems.device)
    grid = _make_grid(N, H_out, W_out, dtype, flag_gems.device, mode)
    # to_reference upcasts to float64, matching the reference out-buffer dtype.
    ref_inp = utils.to_reference(inp, True)
    ref_grid = utils.to_reference(grid, True)
    return inp, grid, ref_inp, ref_grid


@pytest.mark.cudnn_grid_sampler
@pytest.mark.parametrize("shape", GRID_SAMPLER_SHAPES)
@pytest.mark.parametrize("dtype", GRID_SAMPLER_DTYPES)
@pytest.mark.parametrize("grid_mode", ["in_range", "full", "out_of_range"])
def test_cudnn_grid_sampler(shape, dtype, grid_mode):
    N, C, H_in, W_in, H_out, W_out = shape
    inp, grid, ref_inp, ref_grid = _make_inputs(
        N, C, H_in, W_in, H_out, W_out, dtype, grid_mode
    )

    ref_out = _grid_sample_reference(ref_inp, ref_grid)
    res_out = flag_gems.cudnn_grid_sampler(inp, grid)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


@pytest.mark.cudnn_grid_sampler
@pytest.mark.parametrize("dtype", GRID_SAMPLER_DTYPES)
def test_cudnn_grid_sampler_non_contiguous(dtype):
    N, C, H_in, W_in, H_out, W_out = 2, 4, 12, 16, 20, 24
    inp_base = torch.randn(N, C, W_in, H_in, dtype=dtype, device=flag_gems.device)
    grid_base = torch.randn(N, W_out, H_out, 2, dtype=dtype, device=flag_gems.device)
    inp = inp_base.transpose(2, 3)
    grid = grid_base.transpose(1, 2).clamp(-0.9, 0.9)

    ref_inp = utils.to_reference(inp, True)
    ref_grid = utils.to_reference(grid, True)
    ref_out = _grid_sample_reference(ref_inp, ref_grid)
    res_out = flag_gems.cudnn_grid_sampler(inp, grid)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


@pytest.mark.cudnn_grid_sampler
@pytest.mark.parametrize("dtype", GRID_SAMPLER_DTYPES)
def test_cudnn_grid_sampler_nan_inf_grid(dtype):
    N, C, H_in, W_in, H_out, W_out = 2, 3, 8, 8, 5, 6
    inp = torch.randn(N, C, H_in, W_in, dtype=dtype, device=flag_gems.device)
    grid = _make_grid(N, H_out, W_out, dtype, flag_gems.device, "in_range")
    grid[0, 0, 0, 0] = float("nan")
    grid[0, 1, 1, 1] = float("inf")
    grid[0, 2, 2, 0] = float("-inf")

    ref_inp = utils.to_reference(inp, True)
    ref_grid = utils.to_reference(grid, True)
    ref_out = _cpu_grid_sample_reference(ref_inp, ref_grid)
    res_out = flag_gems.cudnn_grid_sampler(inp, grid)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.cudnn_grid_sampler
@pytest.mark.parametrize("dtype", GRID_SAMPLER_DTYPES)
def test_cudnn_grid_sampler_special_values(dtype):
    # Constant input: the interpolated value must equal that constant inside the
    # image and zero outside of it.
    N, C, H_in, W_in, H_out, W_out = 2, 3, 7, 9, 11, 13
    inp = torch.full((N, C, H_in, W_in), 2.5, dtype=dtype, device=flag_gems.device)
    grid = _make_grid(N, H_out, W_out, dtype, flag_gems.device, "full")

    ref_inp = utils.to_reference(inp, True)
    ref_grid = utils.to_reference(grid, True)
    ref_out = _grid_sample_reference(ref_inp, ref_grid)
    res_out = flag_gems.cudnn_grid_sampler(inp, grid)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


@pytest.mark.cudnn_grid_sampler
@pytest.mark.parametrize("dtype", GRID_SAMPLER_DTYPES)
@pytest.mark.parametrize(
    "bad_grid",
    [
        torch.rand(3, 5, 6, 2),  # batch mismatch
        torch.rand(2, 5, 6, 3),  # last dimension is not 2
        torch.rand(2, 5, 2),  # not 4-D
    ],
    ids=["batch", "last_dim", "ndim"],
)
def test_cudnn_grid_sampler_invalid_grid(dtype, bad_grid):
    # A grid the kernel cannot index must raise instead of silently reading out
    # of bounds, matching the cuDNN argument checks.
    inp = torch.randn(2, 3, 8, 8, dtype=dtype, device=flag_gems.device)
    grid = bad_grid.to(dtype=dtype, device=flag_gems.device)

    with pytest.raises((ValueError, RuntimeError)):
        flag_gems.cudnn_grid_sampler(inp, grid)


@pytest.mark.cudnn_grid_sampler_out
@pytest.mark.parametrize("shape", GRID_SAMPLER_SHAPES)
@pytest.mark.parametrize("dtype", GRID_SAMPLER_DTYPES)
def test_cudnn_grid_sampler_out(shape, dtype):
    N, C, H_in, W_in, H_out, W_out = shape
    inp, grid, ref_inp, ref_grid = _make_inputs(
        N, C, H_in, W_in, H_out, W_out, dtype, "out_of_range"
    )
    # to_reference upcasts to float64, so the reference out buffer must match it.
    ref_out = torch.empty(
        (N, C, H_out, W_out), dtype=ref_inp.dtype, device=ref_inp.device
    )
    out = torch.empty((N, C, H_out, W_out), dtype=dtype, device=flag_gems.device)

    _grid_sample_reference_out(ref_inp, ref_grid, ref_out)
    res_out = flag_gems.cudnn_grid_sampler_out(inp, grid, out=out)

    assert res_out.data_ptr() == out.data_ptr()
    utils.gems_assert_close(out, ref_out, dtype, atol=1e-3)


@pytest.mark.cudnn_grid_sampler_out
@pytest.mark.parametrize("dtype", GRID_SAMPLER_DTYPES)
@pytest.mark.parametrize("zero_numel", [True, False], ids=["empty_out", "wrong_shape"])
def test_cudnn_grid_sampler_out_resize(dtype, zero_numel):
    # ATen's resize_output(): an out tensor with the wrong shape is resized in
    # place, warning first unless it had zero elements.
    N, C, H_in, W_in, H_out, W_out = 2, 3, 8, 8, 5, 6
    inp, grid, ref_inp, ref_grid = _make_inputs(
        N, C, H_in, W_in, H_out, W_out, dtype, "in_range"
    )
    out = torch.empty(
        0 if zero_numel else 2 * 3 * 8 * 8, dtype=dtype, device=inp.device
    )
    ref_out = torch.empty(
        (N, C, H_out, W_out), dtype=ref_inp.dtype, device=ref_inp.device
    )

    _grid_sample_reference_out(ref_inp, ref_grid, ref_out)
    if zero_numel:
        # An empty out is resized without a deprecation warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            res_out = flag_gems.cudnn_grid_sampler_out(inp, grid, out=out)
    else:
        with pytest.warns(UserWarning):
            res_out = flag_gems.cudnn_grid_sampler_out(inp, grid, out=out)

    assert tuple(res_out.shape) == (N, C, H_out, W_out)
    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


@pytest.mark.cudnn_grid_sampler_out
@pytest.mark.parametrize("dtype", GRID_SAMPLER_DTYPES)
def test_cudnn_grid_sampler_out_strided(dtype):
    # A strided out cannot be written with flat offsets, so the kernel must
    # stage through a contiguous temporary and copy into the view.
    N, C, H_in, W_in, H_out, W_out = 2, 3, 8, 8, 5, 6
    inp, grid, ref_inp, ref_grid = _make_inputs(
        N, C, H_in, W_in, H_out, W_out, dtype, "in_range"
    )
    out_base = torch.empty((N, C, W_out, H_out), dtype=dtype, device=inp.device)
    out = out_base.transpose(2, 3)
    ref_out = torch.empty(
        (N, C, H_out, W_out), dtype=ref_inp.dtype, device=ref_inp.device
    )

    _grid_sample_reference_out(ref_inp, ref_grid, ref_out)
    res_out = flag_gems.cudnn_grid_sampler_out(inp, grid, out=out)

    assert res_out.data_ptr() == out.data_ptr()
    assert not out.is_contiguous()
    utils.gems_assert_close(out, ref_out, dtype, atol=1e-3)


@pytest.mark.cudnn_grid_sampler_out
@pytest.mark.parametrize("dtype", GRID_SAMPLER_DTYPES)
def test_cudnn_grid_sampler_out_dtype_mismatch(dtype):
    inp = torch.randn(2, 3, 8, 8, dtype=dtype, device=flag_gems.device)
    grid = _make_grid(2, 5, 6, dtype, flag_gems.device)
    mismatched = torch.float32 if dtype != torch.float32 else torch.float16
    out = torch.empty(2, 3, 5, 6, dtype=mismatched, device=inp.device)

    with pytest.raises(ValueError):
        flag_gems.cudnn_grid_sampler_out(inp, grid, out=out)
