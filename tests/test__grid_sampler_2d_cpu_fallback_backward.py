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

# aten::_grid_sampler_2d_cpu_fallback_backward(grad_output, input, grid,
# interpolation_mode, padding_mode, align_corners) -> (grad_input, grad_grid).
# Measured native contract: `.default` is the only overload; it is a
# CompositeImplicitAutograd whose kernel is CPU-only, so accelerator operands
# segfault inside at::native::grid_sampler_2d_cpu_fallback_backward and the
# operands are built on CPU - the operator's real contract - instead of on
# flag_gems.device; float32 is the only accepted dtype ("expected scalar type
# Float but found <Dtype>" / "not implemented for '<Dtype>'"), so the required
# 9-dtype grid collapses to it; input and grid must be rank-4 with matching batch
# and output sizes, so the shared 0/1/3/5-dim levels and broadcast do not apply;
# autograd reports "does not require grad" because the composite registers no
# backward; interpolation_mode/padding_mode are not validated, so the valid enum
# is swept rather than faked as a negative case.

_INTERPOLATION_NAMES = {0: "bilinear", 1: "nearest", 2: "bicubic"}
_PADDING_NAMES = {0: "zeros", 1: "border", 2: "reflection"}


def _mode_id(mode):
    interpolation_mode, padding_mode, align_corners = mode
    return (
        f"{_INTERPOLATION_NAMES[interpolation_mode]}-"
        f"{_PADDING_NAMES[padding_mode]}-ac{align_corners}"
    )


# Rows are (N, C, H, W, out_H, out_W): input (N, C, H, W), grad_output
# (N, C, out_H, out_W) and grid (N, out_H, out_W, 2). They keep the 4-dim spec
# level (16, 128, 64, 60) and add rank-4 resampling pairs whose out_H/out_W
# differ from H/W.
_IMAGE_SHAPE_ROWS = [
    (0, 2, 3, 4, 2, 2),
    (1, 2, 3, 4, 0, 2),
    (1, 1, 1, 1, 1, 1),
    (1, 3, 4, 4, 2, 3),
    (2, 3, 8, 6, 5, 7),
    (3, 2, 16, 16, 9, 11),
    (1, 4, 32, 24, 17, 13),
    (16, 128, 64, 60, 2, 3),
    (2, 64, 12, 40, 25, 7),
]
_QUICK_IMAGE_SHAPES = _IMAGE_SHAPE_ROWS[:5]
_IMAGE_SHAPES = tu.selected_cases(_IMAGE_SHAPE_ROWS, quick=_QUICK_IMAGE_SHAPES)

# One mode per interpolation/padding family plus both align_corners values; the
# full enum sweep is the separate mode workload below.
_MAIN_MODE_ROWS = [
    (0, 0, False),
    (1, 0, False),
    (2, 0, False),
    (0, 1, True),
    (1, 2, True),
    (2, 2, False),
]
_MAIN_MODES = _MAIN_MODE_ROWS

# Every native-valid (interpolation_mode, padding_mode, align_corners) triple,
# crossed with both coordinate spans. These are cheap call-form cases with no
# extra allocation, so they stay in the quick level as well. Span 1.0 keeps the
# grid inside [-1, 1] (in-bounds interpolation); span 2.0 pushes the samples
# outside the image so the padding branches run.
_MODE_ROWS = [
    (interpolation_mode, padding_mode, align_corners)
    for interpolation_mode in (0, 1, 2)
    for padding_mode in (0, 1, 2)
    for align_corners in (False, True)
]
_GRID_SPAN_ROWS = [("in-range", 1.0), ("out-of-bounds", 2.0)]

# Positive special-value inputs are the one dimension kept default-only.
_SPECIAL_SCENARIOS = tu.special_value_cases([torch.float32])
_SPECIAL_SLOTS = tu.selected_cases(["grad_output", "input", "grid"], quick=[])

_SMALL_SHAPE = (2, 3, 8, 6, 5, 7)


def _cpu_range_input(shape, value_range):
    """Fill one operand from a spec value range on CPU.

    tu.make_input allocates on flag_gems.device, which this host-only operator
    cannot accept, so the shared range symbols are resolved through
    tu.resolve_bound and filled on CPU instead.
    """
    low = tu.resolve_bound(value_range[0], torch.float32)
    high = tu.resolve_bound(value_range[1], torch.float32)
    if low == high:
        return torch.full(shape, low, dtype=torch.float32, device="cpu")
    return torch.testing.make_tensor(
        shape, dtype=torch.float32, device="cpu", low=low, high=high
    )


def _make_inputs(shape, value_range, grid_span=1.0):
    """Build the (grad_output, input, grid) operands of one workload.

    The value range drives the data operands; the grid holds normalized sampling
    coordinates, so it always spans [-1, 1] and ``grid_span`` > 1 scales it
    outwards to reach the padding branches.
    """
    n, c, h, w, out_h, out_w = shape
    grad_output = _cpu_range_input((n, c, out_h, out_w), value_range)
    inp = _cpu_range_input((n, c, h, w), value_range)
    grid = _cpu_range_input((n, out_h, out_w, 2), ["-1", "1"]) * grid_span
    return grad_output, inp, grid


@pytest.mark.grid_sampler_2d_cpu_fallback_backward
@pytest.mark.parametrize("shape", _IMAGE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("mode", _MAIN_MODES, ids=_mode_id)
def test__grid_sampler_2d_cpu_fallback_backward(shape, value_range, mode):
    interpolation_mode, padding_mode, align_corners = mode
    grad_output, inp, grid = _make_inputs(shape, value_range)
    ref_grad_output = tu.to_reference(grad_output)
    ref_inp = tu.to_reference(inp)
    ref_grid = tu.to_reference(grid)

    (
        ref_grad_input,
        ref_grad_grid,
    ) = torch.ops.aten._grid_sampler_2d_cpu_fallback_backward.default(
        ref_grad_output,
        ref_inp,
        ref_grid,
        interpolation_mode,
        padding_mode,
        align_corners,
    )

    res_grad_input, res_grad_grid = flag_gems._grid_sampler_2d_cpu_fallback_backward(
        grad_output, inp, grid, interpolation_mode, padding_mode, align_corners
    )

    # The candidate has no accelerator kernel, so both gradients must stay on the
    # device of their own inputs.
    assert res_grad_input.device == inp.device
    assert res_grad_grid.device == grid.device
    tu.assert_result_close(res_grad_input, ref_grad_input)
    tu.assert_result_close(res_grad_grid, ref_grad_grid)


@pytest.mark.grid_sampler_2d_cpu_fallback_backward
@pytest.mark.parametrize(
    "grid_span", _GRID_SPAN_ROWS, ids=[row[0] for row in _GRID_SPAN_ROWS]
)
@pytest.mark.parametrize("mode", _MODE_ROWS, ids=_mode_id)
def test__grid_sampler_2d_cpu_fallback_backward_modes(grid_span, mode):
    _, span = grid_span
    interpolation_mode, padding_mode, align_corners = mode
    grad_output, inp, grid = _make_inputs(_SMALL_SHAPE, ["-1", "1"], span)
    ref_grad_output = tu.to_reference(grad_output)
    ref_inp = tu.to_reference(inp)
    ref_grid = tu.to_reference(grid)

    (
        ref_grad_input,
        ref_grad_grid,
    ) = torch.ops.aten._grid_sampler_2d_cpu_fallback_backward.default(
        ref_grad_output,
        ref_inp,
        ref_grid,
        interpolation_mode,
        padding_mode,
        align_corners,
    )

    res_grad_input, res_grad_grid = flag_gems._grid_sampler_2d_cpu_fallback_backward(
        grad_output, inp, grid, interpolation_mode, padding_mode, align_corners
    )

    tu.assert_result_close(res_grad_input, ref_grad_input)
    tu.assert_result_close(res_grad_grid, ref_grad_grid)


@pytest.mark.grid_sampler_2d_cpu_fallback_backward
@pytest.mark.parametrize("slot", _SPECIAL_SLOTS)
@pytest.mark.parametrize(
    "dtype,scenario", _SPECIAL_SCENARIOS, ids=[row[1] for row in _SPECIAL_SCENARIOS]
)
def test__grid_sampler_2d_cpu_fallback_backward_special_values(slot, dtype, scenario):
    grad_output, inp, grid = _make_inputs(_SMALL_SHAPE, ["-1", "1"])
    operands = {"grad_output": grad_output, "input": inp, "grid": grid}
    # The shared payload generator materializes on flag_gems.device; this
    # host-only operator needs the same five values on CPU.
    payload = tu.make_special_input(dtype, scenario).to("cpu")
    operands[slot].view(-1)[: payload.numel()] = payload

    ref_grad_output = tu.to_reference(grad_output)
    ref_inp = tu.to_reference(inp)
    ref_grid = tu.to_reference(grid)

    (
        ref_grad_input,
        ref_grad_grid,
    ) = torch.ops.aten._grid_sampler_2d_cpu_fallback_backward.default(
        ref_grad_output, ref_inp, ref_grid, 0, 0, False
    )

    res_grad_input, res_grad_grid = flag_gems._grid_sampler_2d_cpu_fallback_backward(
        grad_output, inp, grid, 0, 0, False
    )

    # assert_result_close matches NaNs, so replacing a NaN or Inf result with a
    # finite value fails here.
    tu.assert_result_close(res_grad_input, ref_grad_input)
    tu.assert_result_close(res_grad_grid, ref_grad_grid)


@pytest.mark.grid_sampler_2d_cpu_fallback_backward
def test__grid_sampler_2d_cpu_fallback_backward_rejects_unsupported_input_dtype():
    # Native rejects anything but float32 ("expected scalar type Float but found
    # Double"); the candidate must reject the same input.
    grad_output, inp, grid = _make_inputs(_SMALL_SHAPE, ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems._grid_sampler_2d_cpu_fallback_backward(
            grad_output, inp.double(), grid, 0, 0, False
        )


@pytest.mark.grid_sampler_2d_cpu_fallback_backward
def test__grid_sampler_2d_cpu_fallback_backward_rejects_unsupported_grid_dtype():
    grad_output, inp, grid = _make_inputs(_SMALL_SHAPE, ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems._grid_sampler_2d_cpu_fallback_backward(
            grad_output, inp, grid.half(), 0, 0, False
        )


@pytest.mark.grid_sampler_2d_cpu_fallback_backward
def test__grid_sampler_2d_cpu_fallback_backward_rejects_grid_rank():
    # Native: "expected 4D input and grid with same number of dimensions".
    grad_output, inp, grid = _make_inputs(_SMALL_SHAPE, ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems._grid_sampler_2d_cpu_fallback_backward(
            grad_output, inp, grid[:, 0], 0, 0, False
        )


@pytest.mark.grid_sampler_2d_cpu_fallback_backward
def test__grid_sampler_2d_cpu_fallback_backward_rejects_grid_coordinate_size():
    # Native: "expected grid to have size 2 in last dimension".
    grad_output, inp, grid = _make_inputs(_SMALL_SHAPE, ["-1", "1"])
    bad_grid = _cpu_range_input(grid.shape[:-1] + (3,), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems._grid_sampler_2d_cpu_fallback_backward(
            grad_output, inp, bad_grid, 0, 0, False
        )


@pytest.mark.grid_sampler_2d_cpu_fallback_backward
def test__grid_sampler_2d_cpu_fallback_backward_rejects_batch_size_mismatch():
    # Native: "expected grid and input to have same batch size".
    grad_output, inp, grid = _make_inputs(_SMALL_SHAPE, ["-1", "1"])
    n, _, _, _, out_h, out_w = _SMALL_SHAPE
    bad_grid = _cpu_range_input((n + 1, out_h, out_w, 2), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems._grid_sampler_2d_cpu_fallback_backward(
            grad_output, inp, bad_grid, 0, 0, False
        )


@pytest.mark.grid_sampler_2d_cpu_fallback_backward
def test__grid_sampler_2d_cpu_fallback_backward_rejects_missing_argument():
    # align_corners is a required positional of the only overload; the missing
    # value surfaces either as a Python TypeError or as ATen's RuntimeError
    # "... is missing value for argument 'align_corners'".
    grad_output, inp, grid = _make_inputs(_SMALL_SHAPE, ["-1", "1"])
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._grid_sampler_2d_cpu_fallback_backward(grad_output, inp, grid, 0, 0)


@pytest.mark.grid_sampler_2d_cpu_fallback_backward
@pytest.mark.parametrize(
    "layout", ["input_step", "grad_transpose", "grid_offset", "grid_expand"]
)
@pytest.mark.parametrize("mode", _MODE_ROWS, ids=_mode_id)
def test__grid_sampler_2d_cpu_fallback_backward_layout(layout, mode):
    grad, inp, grid = _make_inputs(_SMALL_SHAPE, ["-1", "1"])
    n, c, h, w, oh, ow = _SMALL_SHAPE
    if layout == "input_step":
        inp = torch.randn(n, c, h, w * 2)[..., ::2]
    elif layout == "grad_transpose":
        grad = torch.randn(n, c, ow, oh).transpose(-1, -2)
    elif layout == "grid_offset":
        backing = torch.rand(n, oh, ow * 2 + 1, 2) * 2 - 1
        grid = backing[:, :, 1::2]
    else:
        grid = (torch.rand(1, oh, ow, 2) * 2 - 1).expand(n, -1, -1, -1)
    refs = [tu.to_reference(t) for t in (grad, inp, grid)]
    expected = torch.ops.aten._grid_sampler_2d_cpu_fallback_backward(*refs, *mode)
    actual = flag_gems._grid_sampler_2d_cpu_fallback_backward(grad, inp, grid, *mode)
    for result, reference in zip(actual, expected):
        tu.assert_result_close(result, reference)
    for operand, reference in zip((grad, inp, grid), refs):
        tu.assert_result_equal(operand, reference)


@pytest.mark.grid_sampler_2d_cpu_fallback_backward
def test__grid_sampler_2d_cpu_fallback_backward_no_autograd_graph():
    operands = [t.requires_grad_() for t in _make_inputs(_SMALL_SHAPE, ["-1", "1"])]
    refs = [t.detach().clone().requires_grad_() for t in operands]
    expected = torch.ops.aten._grid_sampler_2d_cpu_fallback_backward(*refs, 0, 0, False)
    actual = flag_gems._grid_sampler_2d_cpu_fallback_backward(*operands, 0, 0, False)
    for result, reference in zip(actual, expected):
        assert result.requires_grad == reference.requires_grad
        tu.assert_result_close(result, reference)
