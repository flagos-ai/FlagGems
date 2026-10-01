# Copyright 2026 FlagOS Contributors
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

import pytest
import torch

import flag_gems

from . import test_utils as tu

# aten::_grid_sampler_2d_cpu_fallback is a CompositeExplicitAutograd composite
# whose body hardcodes `using scalar_t = float` and reads host memory. Its real
# contract is CPU + float32 + rank 4; the spec dimensions it does not have are
# recorded here rather than skipped silently:
#   * device -- an accelerator input makes the native kernel dereference host
#     storage and the CUDA backend dies with SIGSEGV instead of raising, so the
#     oracle and the candidate both receive CPU tensors.
#   * dtype -- float64 / float16 / bfloat16 / int8 / uint8 / int32 / int64 /
#     bool / float8_e4m3fn / float8_e5m2 on either operand raise "RuntimeError:
#     expected scalar type Float but found <T>", so only float32 is supported.
#   * shape -- check_grid_sampler_2d requires input.dim() == grid.dim() == 4,
#     so only the spec's 4-D shape level applies; the local sizes add this
#     operator's own boundaries (1x1 image, single channel, wide images).
#   * no scalar operand and no broadcasting -- the validator fixes
#     grid.size(0) == input.size(0) and grid.size(-1) == 2.
_CPU = torch.device("cpu")

SUPPORTED_DTYPES = [torch.float32]

# aten GridSamplerInterpolation (bilinear 0 / nearest 1 / bicubic 2) and
# GridSamplerPadding (zeros 0 / border 1 / reflection 2). An out-of-range value
# is accepted natively (the scalar loop falls through to the bilinear/zeros
# branch), so there is no invalid-mode negative case to assert.
_INTERPOLATIONS = (0, 1, 2)
_PADDINGS = (0, 1, 2)
_ALIGN_CORNERS = (False, True)

_IMAGE_SHAPES = [
    (16, 128, 64, 60),
    (2, 3, 19, 7),
    (1, 1, 1, 1),
    (1, 3, 2, 2),
    (2, 4, 7, 5),
    (3, 8, 17, 13),
    (2, 16, 63, 65),
    (5, 1, 9, 8),
]
# Quick retains the small image and single-pixel spatial boundaries.
IMAGE_SHAPES = tu.selected_cases(_IMAGE_SHAPES, quick=_IMAGE_SHAPES[1:4])

# Mode selection is a cheap workload, so every combination stays in both levels.
MODE_CASES = [
    (interpolation_mode, padding_mode, align_corners)
    for interpolation_mode in _INTERPOLATIONS
    for padding_mode in _PADDINGS
    for align_corners in _ALIGN_CORNERS
]

# (in_h, in_w, out_h, out_w): identical, down- and up-sampled grids, a single
# output point and an aspect-ratio change.
OUT_SIZE_CASES = [
    (8, 6, 8, 6),
    (8, 6, 3, 4),
    (8, 6, 17, 13),
    (5, 9, 1, 1),
    (2, 2, 4, 4),
    (11, 7, 6, 11),
]

# Operand layouts the native strided path must accept.
LAYOUT_CASES = ["narrowed_input", "channels_last_input", "strided_grid"]

# Only the input's spatial dimensions must be non-empty, so an empty batch,
# empty channels and empty grid rows/columns are valid native workloads.
EMPTY_CASES = [
    ("empty_batch", (0, 3, 4, 5), (0, 4, 5, 2)),
    ("empty_channels", (2, 0, 4, 5), (2, 4, 5, 2)),
    ("empty_grid_rows", (2, 3, 4, 5), (2, 0, 5, 2)),
    ("empty_grid_columns", (2, 3, 4, 5), (2, 4, 0, 2)),
]

# .out is a real callable overload; it resizes a mismatched buffer (with a
# deprecation warning), so the buffer is created with the exact output shape.
OUT_CASES = [(0, 0, False), (1, 1, False), (2, 2, True)]

# nan-only, inf-only and nan+inf payloads for the supported dtype, in every
# interpolation mode. Positive specials stay out of quick.
SPECIAL_CASES = tu.selected_cases(
    [
        (dtype, scenario, interpolation_mode)
        for dtype, scenario in tu.special_value_cases(SUPPORTED_DTYPES)
        for interpolation_mode in _INTERPOLATIONS
    ],
    quick=[],
)

# autograd through the native composite is defined for every mode pair; the
# rows cover bilinear/nearest/bicubic against the distinct padding and
# align_corners paths. Backward stays out of quick.
BACKWARD_CASES = tu.selected_cases(
    [
        (0, 0, False),
        (0, 0, True),
        (1, 1, False),
        (1, 2, True),
        (2, 0, False),
        (2, 2, True),
    ],
    quick=[],
)

# Argument shapes the native validator rejects. Every row is kept in both levels.
BAD_CALLS = [
    ("input_rank_3", (2, 4, 5), (2, 4, 5, 1), torch.float32, torch.float32),
    ("input_rank_5", (2, 3, 4, 5, 6), (2, 4, 5, 6, 3), torch.float32, torch.float32),
    ("grid_rank_3", (2, 3, 4, 5), (4, 5, 2), torch.float32, torch.float32),
    ("grid_last_dim_1", (2, 3, 4, 5), (2, 4, 5, 1), torch.float32, torch.float32),
    ("grid_last_dim_3", (2, 3, 4, 5), (2, 4, 5, 3), torch.float32, torch.float32),
    ("batch_mismatch", (2, 3, 4, 5), (3, 4, 5, 2), torch.float32, torch.float32),
    ("empty_input_height", (2, 3, 0, 5), (2, 4, 5, 2), torch.float32, torch.float32),
    ("empty_input_width", (2, 3, 4, 0), (2, 4, 5, 2), torch.float32, torch.float32),
    ("float64_input", (2, 3, 4, 5), (2, 4, 5, 2), torch.float64, torch.float64),
    ("float16_input", (2, 3, 4, 5), (2, 4, 5, 2), torch.float16, torch.float16),
    ("int32_input", (2, 3, 4, 5), (2, 4, 5, 2), torch.int32, torch.float32),
    ("int64_grid", (2, 3, 4, 5), (2, 4, 5, 2), torch.float32, torch.int64),
    ("bool_grid", (2, 3, 4, 5), (2, 4, 5, 2), torch.float32, torch.bool),
]


def _cpu_values(dtype, shape, value_range):
    """Values in ``value_range`` on the operator's only valid device.

    ``tu.make_input`` allocates on ``flag_gems.device``; this composite's body is
    a CPU kernel, so the shared range resolution is reused and the result moved
    to the CPU the native contract requires.
    """
    return tu.make_input(dtype, shape, value_range).to(_CPU)


def _coord_grid(n, out_h, out_w, span=1.5):
    """(N, OH, OW, 2) coordinates spanning in-bounds, boundary and out-of-bounds
    values, so zeros, border and reflection take distinct sampling paths."""
    xs = torch.linspace(-span, span, out_w, dtype=torch.float32, device=_CPU)
    ys = torch.linspace(-span, span, out_h, dtype=torch.float32, device=_CPU)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    shift = torch.arange(n, dtype=torch.float32, device=_CPU) - (n - 1) / 2
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0) + (0.05 * shift).view(n, 1, 1, 1)
    return grid.contiguous()


def _pixel_center_grid(n, h, w):
    """Coordinates of every pixel centre with align_corners=False, so an H x W
    output samples each pixel exactly and reaches the stored special values."""
    xs = (2.0 * torch.arange(w, dtype=torch.float32, device=_CPU) + 1.0) / w - 1.0
    ys = (2.0 * torch.arange(h, dtype=torch.float32, device=_CPU) + 1.0) / h - 1.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    stacked = torch.stack((xx, yy), dim=-1).unsqueeze(0)
    return stacked.expand(n, h, w, 2).contiguous()


def _special_image(dtype, scenario, shape):
    """``shape``-sized image whose first pixels carry the shared special
    payload, which the pixel-centre grid then samples."""
    flat = _cpu_values(dtype, shape, ["-1", "1"]).flatten()
    payload = tu.make_special_input(dtype, scenario).to(_CPU)
    flat[: payload.numel()] = payload
    return flat.view(shape)


def _layout_inputs(layout):
    """Operands that are not plain contiguous tensors: a narrowed image with a
    storage offset and row gaps, a channels-last image, and a grid stepping over
    every other row and column."""
    image = _cpu_values(torch.float32, (3, 4, 8, 6), ["-1", "1"])
    if layout == "narrowed_input":
        return image[:, :, 2:7, 1:5], _coord_grid(3, 7, 5)
    if layout == "channels_last_input":
        return image.to(memory_format=torch.channels_last), _coord_grid(3, 8, 6)
    return image, _coord_grid(3, 8, 12)[:, :, ::2]


@pytest.mark.grid_sampler_2d_cpu_fallback
@pytest.mark.parametrize("shape", IMAGE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__grid_sampler_2d_cpu_fallback_values(shape, value_range, dtype):
    n, _, h, w = shape
    inp = _cpu_values(dtype, shape, value_range)
    grid = _coord_grid(n, h, w)
    ref_inp = tu.to_reference(inp)
    ref_grid = tu.to_reference(grid)

    ref_out = torch.ops.aten._grid_sampler_2d_cpu_fallback(
        ref_inp, ref_grid, 0, 0, False
    )
    res_out = flag_gems._grid_sampler_2d_cpu_fallback(inp, grid, 0, 0, False)

    tu.assert_result_close(res_out, ref_out)
    # The operator is out of place: the candidate must not touch its operands.
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(grid, ref_grid)


@pytest.mark.grid_sampler_2d_cpu_fallback
@pytest.mark.parametrize("interpolation_mode,padding_mode,align_corners", MODE_CASES)
def test__grid_sampler_2d_cpu_fallback_modes(
    interpolation_mode, padding_mode, align_corners
):
    inp = _cpu_values(torch.float32, (2, 4, 9, 7), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    # A grid larger than the image resamples the whole image in every mode.
    grid = _coord_grid(2, 11, 13)

    ref_out = torch.ops.aten._grid_sampler_2d_cpu_fallback(
        ref_inp, grid, interpolation_mode, padding_mode, align_corners
    )
    res_out = flag_gems._grid_sampler_2d_cpu_fallback(
        inp, grid, interpolation_mode, padding_mode, align_corners
    )

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.grid_sampler_2d_cpu_fallback
@pytest.mark.parametrize("interpolation_mode", _INTERPOLATIONS)
@pytest.mark.parametrize("in_h,in_w,out_h,out_w", OUT_SIZE_CASES)
def test__grid_sampler_2d_cpu_fallback_output_size(
    in_h, in_w, out_h, out_w, interpolation_mode
):
    inp = _cpu_values(torch.float32, (2, 3, in_h, in_w), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    grid = _coord_grid(2, out_h, out_w, span=1.2)

    ref_out = torch.ops.aten._grid_sampler_2d_cpu_fallback(
        ref_inp, grid, interpolation_mode, 0, False
    )
    res_out = flag_gems._grid_sampler_2d_cpu_fallback(
        inp, grid, interpolation_mode, 0, False
    )

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.grid_sampler_2d_cpu_fallback
@pytest.mark.parametrize("layout", LAYOUT_CASES)
def test__grid_sampler_2d_cpu_fallback_strided_operands(layout):
    inp, grid = _layout_inputs(layout)
    ref_inp = tu.to_reference(inp)
    ref_grid = tu.to_reference(grid)

    ref_out = torch.ops.aten._grid_sampler_2d_cpu_fallback(
        ref_inp, ref_grid, 0, 1, False
    )
    res_out = flag_gems._grid_sampler_2d_cpu_fallback(inp, grid, 0, 1, False)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.grid_sampler_2d_cpu_fallback
@pytest.mark.parametrize("name,in_shape,grid_shape", EMPTY_CASES)
def test__grid_sampler_2d_cpu_fallback_empty(name, in_shape, grid_shape):
    inp = _cpu_values(torch.float32, in_shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    grid = _coord_grid(grid_shape[0], grid_shape[1], grid_shape[2])

    ref_out = torch.ops.aten._grid_sampler_2d_cpu_fallback(ref_inp, grid, 0, 1, False)
    res_out = flag_gems._grid_sampler_2d_cpu_fallback(inp, grid, 0, 1, False)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.grid_sampler_2d_cpu_fallback
@pytest.mark.parametrize("interpolation_mode,padding_mode,align_corners", OUT_CASES)
def test__grid_sampler_2d_cpu_fallback_out(
    interpolation_mode, padding_mode, align_corners
):
    inp = _cpu_values(torch.float32, (2, 3, 9, 7), ["-1", "1"])
    grid = _coord_grid(2, 6, 5)
    ref_out = torch.empty((2, 3, 6, 5), dtype=torch.float32, device=_CPU)
    out = torch.empty_like(ref_out)

    torch.ops.aten._grid_sampler_2d_cpu_fallback.out(
        inp, grid, interpolation_mode, padding_mode, align_corners, out=ref_out
    )
    res = flag_gems._grid_sampler_2d_cpu_fallback(
        inp, grid, interpolation_mode, padding_mode, align_corners, out=out
    )

    # The .out schema writes into and returns the caller's buffer.
    assert res is out
    tu.assert_result_close(out, ref_out)


@pytest.mark.grid_sampler_2d_cpu_fallback
@pytest.mark.parametrize("dtype,scenario,interpolation_mode", SPECIAL_CASES)
def test__grid_sampler_2d_cpu_fallback_special_values(
    dtype, scenario, interpolation_mode
):
    inp = _special_image(dtype, scenario, (2, 2, 4, 6))
    ref_inp = tu.to_reference(inp)
    grid = _pixel_center_grid(2, 4, 6)

    ref_out = torch.ops.aten._grid_sampler_2d_cpu_fallback(
        ref_inp, grid, interpolation_mode, 0, False
    )
    res_out = flag_gems._grid_sampler_2d_cpu_fallback(
        inp, grid, interpolation_mode, 0, False
    )

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.grid_sampler_2d_cpu_fallback
@pytest.mark.parametrize(
    "interpolation_mode,padding_mode,align_corners", BACKWARD_CASES
)
def test__grid_sampler_2d_cpu_fallback_backward(
    interpolation_mode, padding_mode, align_corners
):
    # Both paths differentiate the operator through its original leaves with the
    # same upstream gradient, so the gradients are comparable case by case.
    inp = _cpu_values(torch.float32, (2, 3, 8, 7), ["-1", "1"]).requires_grad_(True)
    grid = _coord_grid(2, 6, 5).requires_grad_(True)
    ref_inp = tu.to_reference(inp).requires_grad_(True)
    ref_grid = tu.to_reference(grid).requires_grad_(True)
    upstream = torch.linspace(0.25, 1.0, 2 * 3 * 6 * 5).reshape(2, 3, 6, 5)

    res_out = flag_gems._grid_sampler_2d_cpu_fallback(
        inp, grid, interpolation_mode, padding_mode, align_corners
    )
    ref_out = torch.ops.aten._grid_sampler_2d_cpu_fallback(
        ref_inp, ref_grid, interpolation_mode, padding_mode, align_corners
    )
    tu.assert_result_close(res_out, ref_out)
    grad_inp, grad_grid = torch.autograd.grad(
        res_out, (inp, grid), grad_outputs=upstream
    )
    ref_grad_inp, ref_grad_grid = torch.autograd.grad(
        ref_out, (ref_inp, ref_grid), grad_outputs=upstream
    )

    tu.assert_result_close(grad_inp, ref_grad_inp)
    tu.assert_result_close(grad_grid, ref_grad_grid)


@pytest.mark.grid_sampler_2d_cpu_fallback
@pytest.mark.parametrize("case,in_shape,grid_shape,in_dtype,grid_dtype", BAD_CALLS)
def test__grid_sampler_2d_cpu_fallback_invalid(
    case, in_shape, grid_shape, in_dtype, grid_dtype
):
    inp = _cpu_values(in_dtype, in_shape, ["-1", "1"])
    grid = _cpu_values(grid_dtype, grid_shape, ["-1", "1"])

    # Only the candidate's rejection is asserted; the native probes that
    # established these errors are generation-time evidence.
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._grid_sampler_2d_cpu_fallback(inp, grid, 0, 0, False)


@pytest.mark.grid_sampler_2d_cpu_fallback
@pytest.mark.parametrize("mode", OUT_CASES)
def test__grid_sampler_2d_cpu_fallback_strided_out_guard(mode):
    inp = _cpu_values(torch.float32, (2, 3, 9, 7), ["-1", "1"])
    grid = _coord_grid(2, 6, 5)
    ref_inp, ref_grid = tu.to_reference(inp), tu.to_reference(grid)
    parent = torch.full((2, 3, 8, 12), 123.0, device=_CPU)
    ref_parent = parent.clone()
    out = parent[:, :, 1:7, 1:11:2]
    ref_out = ref_parent[:, :, 1:7, 1:11:2]
    torch.ops.aten._grid_sampler_2d_cpu_fallback.out(
        ref_inp, ref_grid, *mode, out=ref_out
    )
    returned = flag_gems._grid_sampler_2d_cpu_fallback(inp, grid, *mode, out=out)
    assert returned is out
    tu.assert_result_close(parent, ref_parent)
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(grid, ref_grid)
