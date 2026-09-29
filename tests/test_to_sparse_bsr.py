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

from . import accuracy_utils as utils
from . import test_utils as tu

# `to_sparse_bsr` compresses the last two non-dense dimensions of a dense tensor
# into a block matrix, so the shared grid's rank-0 and rank-1 shapes cannot
# express a block matrix and are covered as negative cases instead (native on
# this backend: rank 0 -> IndexError "Dimension specified as -2 but tensor has
# no dimensions", rank 1 -> IndexError "Dimension out of range (expected to be
# in range of [-1, 0], but got -2)"). The operator takes a single tensor operand
# plus a required `blocksize` list, so the spec's broadcast and tensor/scalar
# dimensions do not apply either. The block geometry is read back from
# `values()`: this build has no `blocksize()` accessor.

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

# The nine required dtypes, minus the ones the runtime reports as unavailable,
# plus bool and (when supported) float64 for extra storage coverage. The
# capability flags are plain runtime attributes of tests/accuracy_utils.py, so
# building this list never executes the operator.
_DTYPES = [
    dtype
    for dtype in tu.REQUIRED_DTYPES
    if (dtype not in _FP8_DTYPES or utils.fp8_is_supported)
    and (dtype != torch.bfloat16 or utils.bf16_is_supported)
    and (dtype != torch.int64 or utils.int64_is_supported)
] + [torch.bool]
if utils.fp64_is_supported:
    _DTYPES.append(torch.float64)

# FP8 is excluded from the gradient check: the sparse-to-dense backward reduces
# with index_add, which has no Float8 kernel on this backend (measured:
# RuntimeError: "index_add" not implemented for 'Float8_e4m3fn').
_BACKWARD_DTYPES = [
    dtype
    for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64)
    if (dtype != torch.bfloat16 or utils.bf16_is_supported)
    and (dtype != torch.float64 or utils.fp64_is_supported)
]

_SPECIAL_DTYPES = [dtype for dtype in _DTYPES if dtype.is_floating_point]

# Block-occupancy patterns of the dense fixtures. `band` stores two blocks per
# block row, shifted by the batch index; `interior` keeps only interior block
# rows and columns, `edges` only the corners, `zero` stores nothing. The view
# patterns feed a non-contiguous or stride-0 dense source through the
# conversion. Every pattern stores the same number of blocks in each batch
# element, which ATen requires ("Expect the same number of specified elements
# per batch").
_VIEW_PATTERNS = ("transpose", "offset", "step", "channels_last", "expand")

# (shape, blocksize, dense_dim, pattern) rows. `dense_dim` None omits the
# argument, which is the schema default the candidate must implement.
_ROWS = [
    # Spec grid shapes that can express a block matrix.
    ((1024, 1024), (4, 4), None, "band"),
    ((20, 320, 15), (1, 1), None, "band"),  # odd extents: only 1x1 divides them
    ((16, 128, 64, 60), (4, 4), None, "band"),
    ((16, 128, 64, 60), (4, 2), None, "band"),  # rectangular block
    ((8, 8), (8, 8), None, "band"),  # one block covering the whole matrix
    ((16, 7, 57, 32, 29), (1, 1), None, "band"),
    ((4, 8, 8), (2, 2), None, "band"),  # batch dimension in front of the matrix
    ((16, 128, 64, 60), (4, 4), 0, "band"),  # dense_dim passed explicitly
    ((16, 128, 64, 60), (4, 4), 1, "band"),
    ((16, 128, 64, 60), (4, 4), 2, "band"),
    ((0, 8), (2, 2), None, "band"),  # no block row at all
    # Empty block columns, empty dense tail and a payload with no stored block.
    # The zero extent sits outside the batch dimensions, because ATen rejects an
    # empty batch product (see the negative cases at the end of this file).
    ((8, 0), (2, 2), None, "band"),
    ((4, 8, 0), (2, 2), 1, "band"),
    ((2, 8, 8), (2, 2), None, "zero"),
    # Skewed block grids (64x8 and 8x64 blocks) and restricted occupancy.
    ((256, 32), (4, 4), None, "interior"),
    ((32, 256), (4, 4), None, "band"),
    ((128, 64), (8, 8), None, "edges"),
    # Rank-5 input with dense_dim 3, square and rectangular blocks.
    ((16, 7, 57, 32, 29), (1, 1), 3, "band"),
    ((16, 7, 57, 32, 29), (2, 1), 3, "band"),
    # Non-contiguous and broadcast (stride 0) dense sources.
    ((8, 8), (2, 2), None, "transpose"),
    ((16, 8), (2, 2), None, "offset"),
    ((16, 8), (2, 2), None, "step"),
    ((4, 8, 16, 32), (4, 8), None, "channels_last"),
    ((4, 8, 8), (2, 2), None, "expand"),
]

_QUICK_ROWS = [((2, 19, 7), (1, 1), None, "band")]

_GRID_CASES = tu.selected_cases(_ROWS, quick=_QUICK_ROWS)

# The dense-to-BSR conversion relocates the stored payload, so its gradient is
# exactly the upstream restricted to the stored blocks. The three forms below
# are the ones the native backward supports; a dense-sum composite silently
# skips the batched and dense-tail forms, so the direct form is used instead.
_BACKWARD_CASES = [
    ((512, 256), (4, 4), None),
    ((4, 8, 8), (2, 2), None),
    ((2, 8, 8, 3), (2, 2), 1),
]

# Default-only gradient fixture: the stored blocks of the band pattern keep a
# single anchor each, so every other element inside a stored block is zero. An
# implementation whose gradient is weighted by the elementwise `self != 0` mask
# instead of the stored-block mask therefore loses the upstream mass sitting on
# those interior zeros.
_INTERIOR_ZERO_CASE = ((16, 16), (4, 4), None)

# (name, shape, call arguments, accepted exception classes). Only genuinely
# invalid calls are listed: a bool, int or tuple `blocksize` is accepted by the
# native operator (measured), so it is not a negative case. Zero-batch shapes are
# native boundaries ("to_sparse_bsr: Expected product of batch dimensions to be
# non-zero."). A missing or fractional argument reaches the candidate as a
# Python call error before any kernel runs.
_NEGATIVE_CASES = [
    ("blocksize_omitted", (8, 8), (), (TypeError, RuntimeError)),
    ("blocksize_arity", (8, 8), ([2],), RuntimeError),
    ("blocksize_zero", (8, 8), ([0, 2],), RuntimeError),
    ("blocksize_negative", (8, 8), ([-2, 2],), RuntimeError),
    ("blocksize_fractional", (8, 8), ([2.5, 2.0],), (TypeError, RuntimeError)),
    ("blocksize_not_dividing", (8, 8), ([3, 3],), RuntimeError),
    ("blocksize_not_dividing_batched", (2, 4, 8, 8), ([3, 3],), RuntimeError),
    ("dense_dim_negative", (2, 4, 8, 8), ([2, 2], -1), RuntimeError),
    ("dense_dim_beyond_rank", (2, 4, 8, 8), ([2, 2], 3), IndexError),
    ("dense_dim_leaves_no_matrix", (8, 8), ([2, 2], 1), IndexError),
    ("rank_one", (8,), ([2, 2],), IndexError),
    ("rank_zero", (), ([2, 2],), IndexError),
    ("zero_batch", (0, 8, 8), ([2, 2],), RuntimeError),
    ("zero_batch_lead", (2, 0, 8, 8), ([2, 2],), RuntimeError),
]


def _dense_dim(dense_dim):
    return 0 if dense_dim is None else dense_dim


def _matrix_dims(shape, dense_dim):
    dense_dim = _dense_dim(dense_dim)
    return tuple(shape[len(shape) - 2 - dense_dim : len(shape) - dense_dim])


def _block_counts(shape, blocksize, dense_dim):
    rows, cols = _matrix_dims(shape, dense_dim)
    return rows // blocksize[0], cols // blocksize[1]


def _batch_size(shape, dense_dim):
    size = 1
    for extent in shape[: len(shape) - 2 - _dense_dim(dense_dim)]:
        size *= extent
    return size


def _clamped_range(dtype, value_range):
    # The bounds the requested range can actually represent in this dtype;
    # make_input clamps the same way (a negative bound of an unsigned dtype
    # becomes the dtype minimum).
    if dtype == torch.bool:
        return 0, 1
    low = tu.resolve_bound(value_range[0], dtype)
    high = tu.resolve_bound(value_range[1], dtype)
    if dtype.is_floating_point:
        finfo = torch.finfo(dtype)
        low, high = max(low, finfo.min), min(high, finfo.max)
    else:
        dtype_min, dtype_max = tu.dtype_bounds(dtype)
        low, high = max(int(low), int(dtype_min)), min(int(high), int(dtype_max))
    return low, max(low, high)


def _stored_value(dtype, value_range):
    # The anchor written into every stored block: a value the requested range
    # actually contains, or None when the range holds only zero (an unsigned
    # dtype whose negative bound was clamped away). A None anchor keeps the
    # constant fill, so the fixture stores no block and no out-of-range value is
    # injected.
    if dtype == torch.bool:
        return True
    low, high = _clamped_range(dtype, value_range)
    for candidate in (1, -1):
        if low <= candidate <= high:
            return candidate
    return None


def _block_pattern(batch, n_row_blocks, n_col_blocks, pattern):
    # Boolean (batch, block row, block column) occupancy. All index arithmetic is
    # int32 so the fixture adds no auxiliary int64 requirement, and no block is
    # selected probabilistically. Every auxiliary tensor is built on the
    # candidate device, so no mask can meet a tensor from another device.
    device = flag_gems.device
    blocks = torch.zeros(
        (batch, n_row_blocks, n_col_blocks), dtype=torch.bool, device=device
    )
    if batch == 0 or n_row_blocks == 0 or n_col_blocks == 0 or pattern == "zero":
        return blocks
    rows = torch.arange(n_row_blocks, dtype=torch.int32, device=device).view(
        1, n_row_blocks, 1
    )
    cols = torch.arange(n_col_blocks, dtype=torch.int32, device=device).view(
        1, 1, n_col_blocks
    )
    batches = torch.arange(batch, dtype=torch.int32, device=device).view(batch, 1, 1)
    if pattern == "interior":
        if n_row_blocks < 3 or n_col_blocks < 3:
            return blocks
        slots = n_col_blocks - 2
        interior = (rows >= 1) & (rows <= n_row_blocks - 2)
        picked = ((cols - 1 - (rows + batches) % slots) % slots) == 0
        return interior & picked & (cols >= 1) & (cols <= n_col_blocks - 2)
    if pattern == "edges":
        return ((rows == 0) | (rows == n_row_blocks - 1)) & (
            (cols == 0) | (cols == n_col_blocks - 1)
        )
    diagonal = cols == (rows + batches) % n_col_blocks
    if n_col_blocks == 1:
        return diagonal
    return diagonal | (cols == (rows + batches + 1) % n_col_blocks)


def _block_mask(shape, blocksize, dense_dim, pattern):
    # Boolean mask over the stored blocks: the block grid is first expanded to
    # the rank-matched block layout `lead + (n_row_blocks, blocksize[0],
    # n_col_blocks, blocksize[1]) + tail` and only then folded back to the dense
    # shape, so the batch axes in front and the dense tail behind line up.
    dense_dim = _dense_dim(dense_dim)
    n_row_blocks, n_col_blocks = _block_counts(shape, blocksize, dense_dim)
    batch = _batch_size(shape, dense_dim)
    blocks = _block_pattern(batch, n_row_blocks, n_col_blocks, pattern)
    lead = tuple(shape[: len(shape) - 2 - dense_dim])
    tail = tuple(shape[len(shape) - dense_dim :])
    full = lead + (n_row_blocks, blocksize[0], n_col_blocks, blocksize[1]) + tail
    return (
        blocks.reshape(lead + (n_row_blocks, 1, n_col_blocks, 1) + (1,) * dense_dim)
        .expand(full)
        .reshape(shape)
    )


def _anchor_payload(shape, blocksize, dense_dim, pattern, anchor, dtype):
    # Mask of the first position of every stored block plus the anchor written
    # there, both in the same rank-matched block layout as `_block_mask`. The mask
    # is built by expansion alone, so the fixture needs no auxiliary index tensor
    # at all.
    dense_dim = _dense_dim(dense_dim)
    n_row_blocks, n_col_blocks = _block_counts(shape, blocksize, dense_dim)
    batch = _batch_size(shape, dense_dim)
    blocks = _block_pattern(batch, n_row_blocks, n_col_blocks, pattern)
    if not bool(blocks.any()):
        return None, None
    lead = tuple(shape[: len(shape) - 2 - dense_dim])
    tail = tuple(shape[len(shape) - dense_dim :])
    full = lead + (n_row_blocks, blocksize[0], n_col_blocks, blocksize[1]) + tail
    interior = blocks.reshape(
        lead + (n_row_blocks, 1, n_col_blocks, 1) + (1,) * dense_dim
    ).expand(full)
    unit_shape = (
        (1,) * len(lead) + (1, blocksize[0], 1, blocksize[1]) + (1,) * dense_dim
    )
    corner_unit = torch.zeros(unit_shape, dtype=torch.bool, device=flag_gems.device)
    corner_unit[(0,) * len(unit_shape)] = True
    return (
        interior & corner_unit,
        torch.full((), anchor, dtype=dtype, device=flag_gems.device),
    )


def _write_stored_anchors(work, shape, blocksize, dense_dim, pattern, anchor):
    if anchor is None:
        return work
    corner, anchor_value = _anchor_payload(
        shape, blocksize, dense_dim, pattern, anchor, work.dtype
    )
    if corner is None:
        return work
    # The mask is built in block layout, so it is folded back to the dense shape
    # before the elementwise write (a block-shaped mask would not line up with
    # the flat dense tensor at all).
    return torch.where(corner.reshape(shape), anchor_value, work)


def _view_input(shape, blocksize, dtype, value_range, dense_dim, pattern):
    # Non-contiguous and broadcast (stride 0) dense sources: a plain blocked
    # matrix is built first and turned into a view afterwards, so the stored
    # anchors stay in place and the conversion has to read the block matrix
    # through the source's strides. The real parent allocation is returned next
    # to the logical input so the caller can prove the conversion did not write
    # to it; `channels_last` materializes a copy instead of a view.
    if pattern == "transpose":
        base = _make_dense_input(
            tuple(shape[:-2]) + (shape[-1], shape[-2]),
            blocksize[::-1],
            dtype,
            value_range,
            dense_dim,
            "band",
        )
        return base.transpose(-1, -2), base
    if pattern == "offset":
        base = _make_dense_input(
            tuple(shape[:-2])
            + (shape[-2] + 2 * blocksize[0], shape[-1] + 2 * blocksize[1]),
            blocksize,
            dtype,
            value_range,
            dense_dim,
            "band",
        )
        view = base[
            ...,
            blocksize[0] : blocksize[0] + shape[-2],
            blocksize[1] : blocksize[1] + shape[-1],
        ]
        return view, base
    if pattern == "step":
        base = _make_dense_input(
            tuple(shape[:-2]) + (2 * shape[-2], shape[-1]),
            blocksize,
            dtype,
            value_range,
            dense_dim,
            "band",
        )
        return base[..., ::2, :], base
    if pattern == "channels_last":
        base = _make_dense_input(
            shape, blocksize, dtype, value_range, dense_dim, "band"
        )
        # `to(memory_format=...)` copies, so the returned tensor owns its storage
        # and has no parent allocation behind it to compare against.
        return base.to(memory_format=torch.channels_last), None
    if pattern == "expand":
        base = _make_dense_input(
            (1,) + tuple(shape[1:]), blocksize, dtype, value_range, dense_dim, "band"
        )
        return base.expand(shape), base
    raise ValueError(f"unknown fixture pattern: {pattern}")


def _make_dense_input(shape, blocksize, dtype, value_range, dense_dim, pattern):
    if pattern in _VIEW_PATTERNS:
        # Only the logical tensor is used here; the view fixture's own callers go
        # through `_dense_fixture` to also get the parent allocation.
        return _view_input(shape, blocksize, dtype, value_range, dense_dim, pattern)[0]
    dense_dim = _dense_dim(dense_dim)
    inp = tu.make_input(dtype, shape, value_range)
    n_row_blocks, n_col_blocks = _block_counts(shape, blocksize, dense_dim)
    if inp.numel() == 0 or n_row_blocks == 0 or n_col_blocks == 0:
        return inp
    if pattern == "zero":
        return torch.zeros_like(inp)
    anchor = _stored_value(dtype, value_range)
    mask = _block_mask(shape, blocksize, dense_dim, pattern)
    staged = dtype in _FP8_DTYPES
    if staged:
        # Float8 has no elementwise multiply on this backend: the whole fixture
        # is staged in float32 and cast back once, after every write, so the
        # tested tensor keeps the requested dtype and value range.
        masked = inp.to(torch.float32) * mask
    elif dtype == torch.bool:
        masked = inp & mask
    else:
        masked = inp * mask
    filled = _write_stored_anchors(masked, shape, blocksize, dense_dim, pattern, anchor)
    return filled.to(dtype) if staged else filled


def _dense_fixture(shape, blocksize, dtype, value_range, dense_dim, pattern):
    # (logical input, real parent allocation or None). The block arithmetic stays
    # on the source device; only the independent reference snapshots in the tests
    # are moved to the configured reference device.
    if pattern in _VIEW_PATTERNS:
        return _view_input(shape, blocksize, dtype, value_range, dense_dim, pattern)
    return (
        _make_dense_input(shape, blocksize, dtype, value_range, dense_dim, pattern),
        None,
    )


def _interior_zero_input(shape, blocksize, dense_dim, dtype):
    # Every stored block holds exactly one nonzero anchor and is otherwise zero;
    # the anchor keeps the block in the result, so the stored structure is the
    # same as a fully populated band.
    anchor = _stored_value(dtype, ["0", "1"])
    zeros = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    return _write_stored_anchors(zeros, shape, blocksize, dense_dim, "band", anchor)


def _interior_zero_upstream(shape, blocksize, dense_dim, dtype):
    # Strictly nonzero and nonuniform over the stored blocks (values in
    # [0.5, 1.5) in every supported dtype), including the positions where the
    # input is zero, so an elementwise `self != 0` gradient mask is detectable.
    mask = _block_mask(shape, blocksize, dense_dim, "band")
    staged = torch.rand(shape, dtype=torch.float32, device=flag_gems.device) + 0.5
    zeros = torch.zeros_like(staged)
    return torch.where(mask, staged, zeros).to(dtype)


def _assert_bsr_structure(out, shape, blocksize, dense_dim):
    # Candidate-side block geometry that the shared sparse-aware value assertion
    # does not cover: the layout, the compressed shape, the sparse dimensions,
    # the row-block pointer length and the block geometry carried by the values
    # shape. Stored block indices and payload are compared by
    # tu.assert_result_equal, so they are not re-checked here.
    dense_dim = _dense_dim(dense_dim)
    n_row_blocks, _ = _block_counts(shape, blocksize, dense_dim)
    lead = tuple(shape[: len(shape) - 2 - dense_dim])
    assert out.layout == torch.sparse_bsr, out.layout
    assert tuple(out.shape) == tuple(shape), (out.shape, shape)
    assert out.sparse_dim() == 2, out.sparse_dim()
    assert out.dense_dim() == dense_dim, (out.dense_dim(), dense_dim)
    assert tuple(out.crow_indices().shape) == lead + (n_row_blocks + 1,), (
        out.crow_indices().shape,
    )
    values = out.values()
    block_tail = tuple(blocksize) + tuple(shape[len(shape) - dense_dim :])
    assert tuple(values.shape[len(values.shape) - 2 - dense_dim :]) == block_tail, (
        values.shape,
    )


@pytest.mark.to_sparse_bsr
@pytest.mark.parametrize("case", _GRID_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_to_sparse_bsr(case, dtype, value_range):
    shape, blocksize, dense_dim, pattern = case
    inp, parent = _dense_fixture(
        shape, blocksize, dtype, value_range, dense_dim, pattern
    )
    # Independent snapshots (not live aliases) on the configured reference
    # device: the logical input, plus the real parent allocation the view is
    # taken from.
    inp_before = tu.to_reference(inp)
    parent_before = None if parent is None else tu.to_reference(parent)
    ref_inp = tu.to_reference(inp)
    args = [list(blocksize)] if dense_dim is None else [list(blocksize), dense_dim]

    ref_out = torch.ops.aten.to_sparse_bsr(ref_inp, *args)
    res_out = flag_gems.to_sparse_bsr(inp, *args)

    _assert_bsr_structure(res_out, shape, blocksize, dense_dim)
    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    # The conversion is out of place: neither the dense input nor the storage it
    # views may be written through.
    tu.assert_result_equal(inp, inp_before)
    if parent_before is not None:
        tu.assert_result_equal(parent, parent_before)


@pytest.mark.to_sparse_bsr
@pytest.mark.parametrize("case", tu.selected_cases(_BACKWARD_CASES, quick=[]))
@pytest.mark.parametrize("dtype", tu.selected_cases(_BACKWARD_DTYPES, quick=[]))
def test_to_sparse_bsr_backward(case, dtype):
    shape, blocksize, dense_dim = case
    args = [list(blocksize)] if dense_dim is None else [list(blocksize), dense_dim]
    inp = _make_dense_input(shape, blocksize, dtype, ["-1", "1"], dense_dim, "band")
    inp_before = tu.to_reference(inp)
    # A nonuniform upstream storing exactly the same blocks as the forward
    # result, built on the reference device: the gradient has to relocate and
    # mask it, so it is compared exactly rather than with a tolerance.
    upstream = _make_dense_input(shape, blocksize, dtype, ["0", "1"], dense_dim, "band")
    ref_inp = tu.to_reference(inp).requires_grad_(True)
    ref_upstream = tu.to_reference(upstream)
    inp.requires_grad_(True)

    ref_out = torch.ops.aten.to_sparse_bsr(ref_inp, *args)
    res_out = flag_gems.to_sparse_bsr(inp, *args)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)

    (ref_grad,) = torch.autograd.grad(
        ref_out,
        ref_inp,
        grad_outputs=torch.ops.aten.to_sparse_bsr(ref_upstream, *args),
    )
    (res_grad,) = torch.autograd.grad(
        res_out,
        inp,
        grad_outputs=torch.ops.aten.to_sparse_bsr(upstream, *args),
    )

    assert res_grad.device == inp.device
    tu.assert_result_equal(res_grad, ref_grad)
    tu.assert_result_equal(inp, inp_before)


@pytest.mark.to_sparse_bsr
@pytest.mark.parametrize("dtype", tu.selected_cases(_BACKWARD_DTYPES, quick=[]))
def test_to_sparse_bsr_backward_interior_zero(dtype):
    shape, blocksize, dense_dim = _INTERIOR_ZERO_CASE
    args = [list(blocksize)] if dense_dim is None else [list(blocksize), dense_dim]
    inp = _interior_zero_input(shape, blocksize, dense_dim, dtype)
    inp_before = tu.to_reference(inp)
    upstream = _interior_zero_upstream(shape, blocksize, dense_dim, dtype)
    ref_inp = tu.to_reference(inp).requires_grad_(True)
    ref_upstream = tu.to_reference(upstream)
    inp.requires_grad_(True)

    ref_out = torch.ops.aten.to_sparse_bsr(ref_inp, *args)
    res_out = flag_gems.to_sparse_bsr(inp, *args)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)

    (ref_grad,) = torch.autograd.grad(
        ref_out,
        ref_inp,
        grad_outputs=torch.ops.aten.to_sparse_bsr(ref_upstream, *args),
    )
    (res_grad,) = torch.autograd.grad(
        res_out,
        inp,
        grad_outputs=torch.ops.aten.to_sparse_bsr(upstream, *args),
    )

    assert res_grad.device == inp.device
    # The native gradient relocates the upstream into the stored blocks without
    # further arithmetic, so it is compared exactly: a gradient weighted by
    # `inp != 0` would drop every upstream value sitting on an interior zero.
    tu.assert_result_equal(res_grad, ref_grad)
    tu.assert_result_equal(inp, inp_before)


@pytest.mark.to_sparse_bsr
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_SPECIAL_DTYPES), quick=[]),
)
def test_to_sparse_bsr_special_values(dtype, scenario):
    shape, blocksize = (8, 8), (2, 2)
    payload = tu.make_special_input(dtype, scenario)
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    inp.view(-1)[: payload.numel()] = payload
    inp_before = tu.to_reference(inp)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse_bsr(ref_inp, list(blocksize))
    res_out = flag_gems.to_sparse_bsr(inp, list(blocksize))

    _assert_bsr_structure(res_out, shape, blocksize, None)
    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, inp_before)


@pytest.mark.to_sparse_bsr
@pytest.mark.parametrize("name,shape,call_args,expected", _NEGATIVE_CASES)
def test_to_sparse_bsr_invalid_arguments(name, shape, call_args, expected):
    del name
    inp = torch.ones(shape, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(expected):
        flag_gems.to_sparse_bsr(inp, *call_args)


@pytest.mark.to_sparse_bsr
@pytest.mark.parametrize(
    "source_layout",
    tu.selected_cases(
        [
            torch.sparse_coo,
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        ]
        if utils.int64_is_supported
        else [],
        quick=[],
    ),
)
def test_to_sparse_bsr_sparse_source(source_layout):
    dense = torch.arange(-32, 32, dtype=torch.float32, device=flag_gems.device).reshape(
        8, 8
    )
    dense[:4, :4] = 0
    kwargs = {"layout": source_layout}
    if source_layout in (torch.sparse_bsr, torch.sparse_bsc):
        kwargs["blocksize"] = (2, 2)
    inp = dense.to_sparse(**kwargs)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse_bsr(ref_inp, (2, 2))
    res_out = flag_gems.to_sparse_bsr(inp, (2, 2))

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    assert res_out.device == inp.device
    if source_layout == torch.sparse_bsr:
        assert res_out is inp
