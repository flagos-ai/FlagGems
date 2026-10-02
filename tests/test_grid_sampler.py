# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# aten::grid_sampler(input, grid, interpolation_mode, padding_mode, align_corners)
#   interpolation_mode: 0 bilinear (4-D) / trilinear (5-D), 1 nearest,
#                       2 bicubic (4-D only)
#   padding_mode:       0 zeros, 1 border, 2 reflection
# The schema declares no default argument and exposes no `out` overload, and the
# operator is fixed-rank with no scalar operand: the 2-D path takes a 4-D input
# (N, C, IH, IW) with a 4-D grid (N, OH, OW, 2), the 3-D path a 5-D input
# (N, C, ID, IH, IW) with a 5-D grid (N, OD, OH, OW, 3), and input and grid must
# agree on rank and batch size. There is therefore no broadcast or
# omitted-argument workload.
#
# Sampling is floating point only: probed on the active backend, bool plus the six
# non-float dtypes int8, uint8, int32, int64, float8_e4m3fn and float8_e5m2 raise
# RuntimeError ('grid_sampler_2d_cuda'/'grid_sampler_3d_cuda' not implemented for
# that type), so they are covered by the negative dtype workload instead of being
# dropped. float32, float16, bfloat16 and float64 are implemented.
_SUPPORTED_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
] + ([torch.float64] if utils.fp64_is_supported else [])

# The spec's rank-varying shapes mapped onto the two ranks the operator accepts:
# a single element, the quick shape and the large levels. The last two 2-D rows
# return a zero-element output (empty grid / zero channels), which the kernels
# accept natively.
_2D_SHAPES = [
    ((1, 1, 1, 1), (1, 1, 1, 2)),
    ((1, 2, 2, 2), (1, 3, 3, 2)),
    ((2, 3, 19, 7), (2, 19, 7, 2)),
    ((1, 2, 1024, 1024), (1, 64, 64, 2)),
    ((2, 3, 64, 60), (2, 32, 32, 2)),
    ((2, 3, 8, 8), (2, 0, 4, 2)),
    ((2, 0, 7, 5), (2, 4, 4, 2)),
]
_3D_SHAPES = [
    ((2, 2, 20, 16, 15), (2, 8, 8, 8, 3)),
    ((1, 2, 7, 32, 29), (1, 6, 6, 6, 3)),
]
_SHAPE_ROWS = _2D_SHAPES + _3D_SHAPES
_QUICK_SHAPE_ROWS = _2D_SHAPES[:3] + _2D_SHAPES[-2:] + _3D_SHAPES[:1]

_EMPTY_SHAPES = [
    ((0, 2, 3, 4), (0, 2, 2, 2)),
    ((2, 2, 3, 4), (2, 2, 0, 2)),
    ((0, 2, 3, 4, 5), (0, 2, 2, 2, 3)),
    ((2, 0, 3, 4, 5), (2, 2, 2, 2, 3)),
    ((2, 2, 3, 4, 5), (2, 0, 2, 2, 3)),
    ((2, 2, 3, 4, 5), (2, 2, 0, 2, 3)),
    ((2, 2, 3, 4, 5), (2, 2, 2, 0, 3)),
]
_SHAPE_ROWS += _EMPTY_SHAPES
_QUICK_SHAPE_ROWS += _EMPTY_SHAPES


def _coordinate_grid(shape, dtype, span=1.0):
    # Grid coordinates are normalized to [-1, 1] by definition, so they are not
    # drawn from the image value ranges. `span` widens them: coordinates outside
    # [-1, 1] are what select the zeros / border / reflection padding branch.
    return (
        torch.rand(shape, dtype=torch.float32, device=flag_gems.device) * (2 * span)
        - span
    ).to(dtype)


@pytest.mark.grid_sampler
@pytest.mark.parametrize(
    "shape_row", tu.selected_cases(_SHAPE_ROWS, quick=_QUICK_SHAPE_ROWS)
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_grid_sampler_value_range(shape_row, value_range, dtype):
    inp_shape, grid_shape = shape_row
    inp = tu.make_input(dtype, inp_shape, value_range)
    grid = _coordinate_grid(grid_shape, dtype)
    ref_inp = tu.to_reference(inp)
    ref_grid = tu.to_reference(grid)

    # Bilinear sampling with zeros padding and align_corners=False is a convex
    # combination of in-range samples, so the dtype-extreme ranges stay finite.
    ref_out = torch.ops.aten.grid_sampler(ref_inp, ref_grid, 0, 0, False)
    res_out = flag_gems.grid_sampler(inp, grid, 0, 0, False)

    tu.assert_result_close(res_out, ref_out)


# Every interpolation_mode / padding_mode / align_corners combination on both
# ranks. The 5-D rows keep modes 0/1 because 3-D sampling has no bicubic path
# (probed: mode 2 with a 5-D input raises 'bicubic interpolation only supports 4D
# input'). Out-of-range mode/padding integers are not rejected (probed: mode
# 3/-1 behave like bicubic, padding 3/-1 like zeros), so the invalid-parameter
# negatives below use unsupported dtypes and invalid shapes instead.
_MODE_VALUES = [
    (mode, padding, align)
    for mode in (0, 1, 2)
    for padding in (0, 1, 2)
    for align in (False, True)
]
_MODE_SHAPES = [
    ((2, 3, 19, 7), (2, 19, 7, 2)),
    ((2, 2, 20, 16, 15), (2, 8, 8, 8, 3)),
]
# Every row is a cheap parameter branch on small operands, so quick keeps the
# whole matrix.
_MODE_ROWS = [
    (inp_shape, grid_shape, mode, padding, align)
    for inp_shape, grid_shape in _MODE_SHAPES
    for mode, padding, align in _MODE_VALUES
    if not (len(inp_shape) == 5 and mode == 2)
]


@pytest.mark.grid_sampler
@pytest.mark.parametrize(
    "inp_shape,grid_shape,interpolation_mode,padding_mode,align_corners", _MODE_ROWS
)
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_grid_sampler_modes(
    inp_shape, grid_shape, interpolation_mode, padding_mode, align_corners, dtype
):
    inp = tu.make_input(dtype, inp_shape, ("-1", "1"))
    grid = _coordinate_grid(grid_shape, dtype, span=2.0)
    ref_inp = tu.to_reference(inp)
    ref_grid = tu.to_reference(grid)

    ref_out = torch.ops.aten.grid_sampler(
        ref_inp, ref_grid, interpolation_mode, padding_mode, align_corners
    )
    res_out = flag_gems.grid_sampler(
        inp, grid, interpolation_mode, padding_mode, align_corners
    )

    tu.assert_result_close(res_out, ref_out)


_CENTRE_COORDS = [2.0 * i / 4.0 - 1.0 for i in range(5)]


def _pixel_centre_grid(shape, dtype):
    # Coordinates landing exactly on the pixel centres of a length-5 axis with
    # align_corners=True; every other axis has extent 1, so index 0 is exact.
    rows = (
        [(x, 0.0) for x in _CENTRE_COORDS]
        if shape[-1] == 2
        else [(x, 0.0, 0.0) for x in _CENTRE_COORDS]
    )
    return torch.tensor(rows, dtype=dtype, device=flag_gems.device).reshape(shape)


# Nearest sampling (mode 1, zeros padding, align_corners=True) at exact pixel
# centres selects samples without arithmetic, so the payloads must survive
# bit-exactly: the comparison is exact and matching NaNs are allowed. FP8 rows
# are absent from both the shared matrix and the operator; they are covered by
# the negative dtype workload.
_SPECIAL_SHAPES = [
    ((1, 1, 1, 5), (1, 1, 5, 2)),
    ((1, 1, 1, 1, 5), (1, 1, 1, 5, 3)),
]
_SPECIAL_ROWS = tu.selected_cases(
    [
        (dtype, scenario, inp_shape, grid_shape)
        for dtype, scenario in tu.special_value_cases(_SUPPORTED_DTYPES)
        for inp_shape, grid_shape in _SPECIAL_SHAPES
    ],
    quick=[],
)


@pytest.mark.grid_sampler
@pytest.mark.parametrize("dtype,scenario,inp_shape,grid_shape", _SPECIAL_ROWS)
def test_grid_sampler_special_values(dtype, scenario, inp_shape, grid_shape):
    inp = tu.make_special_input(dtype, scenario).reshape(inp_shape)
    grid = _pixel_centre_grid(grid_shape, dtype)
    ref_inp = tu.to_reference(inp)
    ref_grid = tu.to_reference(grid)

    ref_out = torch.ops.aten.grid_sampler(ref_inp, ref_grid, 1, 0, True)
    res_out = flag_gems.grid_sampler(inp, grid, 1, 0, True)

    tu.assert_result_equal(res_out, ref_out)


# Gradients reach both operands (grid_sampler_2d_backward / _3d_backward), so the
# forward result and both operand gradients are compared. Two upstream gradients
# per row: 'unit' makes the input gradient a sum of interpolation weights and
# isolates weight accumulation, 'random' mixes signs so neighbouring pixels'
# contributions cancel.
_BACKWARD_UPSTREAM_KINDS = ["unit", "random"]
_BACKWARD_ROWS = tu.selected_cases(
    [
        row
        for row in [
            ((2, 3, 16, 16), (2, 8, 8, 2), 0, 0, False, torch.float32),
            ((2, 3, 16, 16), (2, 8, 8, 2), 1, 1, True, torch.float32),
            ((2, 3, 16, 16), (2, 8, 8, 2), 2, 2, False, torch.float32),
            ((2, 3, 16, 16), (2, 8, 8, 2), 0, 2, True, torch.float16),
            ((2, 3, 16, 16), (2, 8, 8, 2), 1, 0, False, torch.bfloat16),
            ((2, 3, 16, 16), (2, 8, 8, 2), 0, 1, True, torch.float64),
            ((2, 2, 8, 8, 8), (2, 4, 4, 4, 3), 0, 0, False, torch.float32),
            ((2, 2, 8, 8, 8), (2, 4, 4, 4, 3), 1, 2, True, torch.float32),
            ((2, 2, 8, 8, 8), (2, 4, 4, 4, 3), 0, 1, False, torch.float16),
            ((2, 2, 8, 8, 8), (2, 4, 4, 4, 3), 1, 0, True, torch.bfloat16),
            ((2, 2, 8, 8, 8), (2, 4, 4, 4, 3), 0, 2, False, torch.float64),
        ]
        if row[-1] != torch.float64 or utils.fp64_is_supported
    ],
    quick=[],
)


@pytest.mark.grid_sampler
@pytest.mark.parametrize("upstream_kind", _BACKWARD_UPSTREAM_KINDS)
@pytest.mark.parametrize(
    "inp_shape,grid_shape,interpolation_mode,padding_mode,align_corners,dtype",
    _BACKWARD_ROWS,
)
def test_grid_sampler_backward(
    inp_shape,
    grid_shape,
    interpolation_mode,
    padding_mode,
    align_corners,
    dtype,
    upstream_kind,
):
    inp = tu.make_input(dtype, inp_shape, ("-1", "1")).requires_grad_()
    grid = _coordinate_grid(grid_shape, dtype).requires_grad_()
    ref_inp = tu.to_reference(inp)
    ref_grid = tu.to_reference(grid)
    grad_shape = (inp_shape[0], inp_shape[1]) + tuple(grid_shape[1:-1])
    if upstream_kind == "unit":
        grad_out = torch.ones(grad_shape, dtype=dtype, device=flag_gems.device)
    else:
        grad_out = tu.make_input(dtype, grad_shape, ("-1", "1"))
    ref_grad_out = tu.to_reference(grad_out)

    ref_out = torch.ops.aten.grid_sampler(
        ref_inp, ref_grid, interpolation_mode, padding_mode, align_corners
    )
    ref_inp_grad, ref_grid_grad = torch.autograd.grad(
        ref_out, (ref_inp, ref_grid), grad_outputs=ref_grad_out
    )

    res_out = flag_gems.grid_sampler(
        inp, grid, interpolation_mode, padding_mode, align_corners
    )
    res_inp_grad, res_grid_grad = torch.autograd.grad(
        res_out, (inp, grid), grad_outputs=grad_out
    )

    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(res_inp_grad, ref_inp_grad)
    tu.assert_result_close(res_grid_grad, ref_grid_grad)


# Index arithmetic must honour the operands' real strides and storage offsets,
# not just their shapes; transposed / sliced inputs and strided / offset grids
# are valid native calls.
_INPUT_LAYOUT_ROWS = [
    (True, 0, 0),
    (False, 1, 2),
    (True, 2, 3),
]
_GRID_LAYOUT_KINDS = ["strided", "offset"]


def _layout_input(dtype, transpose_last_two, row_offset, col_offset):
    # Views of one base image: the transpose makes both spatial strides
    # non-standard, the slice adds a storage offset on top of that layout.
    base = tu.make_input(dtype, (2, 3, 12, 16), ("-1", "1"))
    inp = base.transpose(2, 3) if transpose_last_two else base
    if row_offset or col_offset:
        inp = inp[:, :, row_offset : row_offset + 8, col_offset : col_offset + 8]
    return inp


def _layout_grid(grid_shape, dtype, kind):
    wide = _coordinate_grid(
        (grid_shape[0], grid_shape[1] * 2, grid_shape[2] * 2, grid_shape[3]),
        dtype,
        span=2.0,
    )
    if kind == "strided":
        return wide[:, ::2, ::2, :]
    return wide[:, 2 : 2 + grid_shape[1], 2 : 2 + grid_shape[2], :]


@pytest.mark.grid_sampler
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
@pytest.mark.parametrize(
    "transpose_last_two,row_offset,col_offset",
    _INPUT_LAYOUT_ROWS,
)
def test_grid_sampler_input_layout(transpose_last_two, row_offset, col_offset, dtype):
    inp = _layout_input(dtype, transpose_last_two, row_offset, col_offset)
    grid = _coordinate_grid((2, 4, 4, 2), dtype, span=2.0)
    ref_inp = tu.to_reference(inp)
    ref_grid = tu.to_reference(grid)

    ref_out = torch.ops.aten.grid_sampler(ref_inp, ref_grid, 0, 0, False)
    res_out = flag_gems.grid_sampler(inp, grid, 0, 0, False)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.grid_sampler
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
@pytest.mark.parametrize("grid_layout", _GRID_LAYOUT_KINDS)
def test_grid_sampler_grid_layout(grid_layout, dtype):
    inp = tu.make_input(dtype, (2, 3, 12, 16), ("-1", "1"))
    grid = _layout_grid((2, 4, 4, 2), dtype, grid_layout)
    ref_inp = tu.to_reference(inp)
    ref_grid = tu.to_reference(grid)

    ref_out = torch.ops.aten.grid_sampler(ref_inp, ref_grid, 0, 0, False)
    res_out = flag_gems.grid_sampler(inp, grid, 0, 0, False)

    tu.assert_result_close(res_out, ref_out)


# Invalid inputs, both kept in quick: unsupported dtype (RuntimeError 'not
# implemented for ...') or an explicit shape complaint (input/grid rank, grid
# last dimension, batch size). Only the candidate's exception is asserted, and
# the values stay in [0, 1] so the dtype conversion cannot introduce inf/nan.
_UNSUPPORTED_DTYPES = [
    torch.bool,
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.int32,
    torch.int64,
]
_NEGATIVE_SHAPES = [
    ((2, 3, 8, 8), (2, 4, 4, 2)),
    ((2, 2, 4, 4, 4), (2, 4, 4, 4, 3)),
]
_NEGATIVE_DTYPE_ROWS = [
    (dtype, inp_shape, grid_shape)
    for dtype in _UNSUPPORTED_DTYPES
    for inp_shape, grid_shape in _NEGATIVE_SHAPES
]


@pytest.mark.grid_sampler
@pytest.mark.parametrize("dtype,inp_shape,grid_shape", _NEGATIVE_DTYPE_ROWS)
def test_grid_sampler_unsupported_dtype(dtype, inp_shape, grid_shape):
    inp = tu.make_input(torch.float32, inp_shape, ("0", "1")).to(dtype)
    grid = _coordinate_grid(grid_shape, torch.float32).to(dtype)

    with pytest.raises((ValueError, RuntimeError, TypeError, AssertionError)):
        flag_gems.grid_sampler(inp, grid, 0, 0, False)


_NEGATIVE_SHAPE_ROWS = [
    ((2, 3, 8), (2, 4, 4, 2)),
    ((2, 3, 4, 4, 4), (2, 4, 4, 2)),
    ((2, 3, 8, 8), (2, 4, 4, 3)),
    ((2, 3, 8, 8), (2, 4, 4, 2, 1)),
    ((2, 3, 8, 8), (2, 4, 2)),
    ((2, 3, 8, 8), (3, 4, 4, 2)),
]


@pytest.mark.grid_sampler
@pytest.mark.parametrize("inp_shape,grid_shape", _NEGATIVE_SHAPE_ROWS)
def test_grid_sampler_invalid_shape(inp_shape, grid_shape):
    inp = tu.make_input(torch.float32, inp_shape, ("-1", "1"))
    grid = _coordinate_grid(grid_shape, torch.float32)

    with pytest.raises((ValueError, RuntimeError, TypeError, AssertionError)):
        flag_gems.grid_sampler(inp, grid, 0, 0, False)
