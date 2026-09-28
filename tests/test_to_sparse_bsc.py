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

# Shape semantics, probed against the native operator on the active backend:
#   * rank < 2 is rejected (0-D and 1-D both raise IndexError), so only rank >= 2
#     spec shapes appear here and each row carries a blocksize pair dividing its
#     own sparse dims;
#   * the two dims in front of the dense_dim trailing dims are the sparse dims,
#     and any leading dims become batch axes;
#   * a batched shape must store the same number of blocks in every batch, so the
#     batched rows build one tile repeated along their batch dims;
#   * a zero-sized axis is a valid input (only a zero-sized batch axis is
#     rejected), which is why the empty geometries below are legal workloads.
_MATRIX_CASES = tu.selected_cases(
    [
        ((1024, 1024), (2, 2), 0),
        ((20, 320, 15), (2, 2), 1),
        ((16, 128, 64, 60), (4, 4), 2),
        ((16, 7, 57, 32, 29), (1, 1), 3),
        ((0, 8), (2, 2), 0),
        ((8, 0), (2, 2), 0),
        ((0, 0), (2, 2), 0),
        ((2, 0, 8), (2, 2), 0),
        ((8, 8, 0), (2, 2), 1),
        ((2, 8, 8, 0), (2, 2), 1),
    ],
    quick=[((2, 19, 7), (1, 1), 1)],
)

# Extra native-supported types, all probed on this backend. complex64 is two
# float32 components, so only float64 and complex128 depend on 64-bit floats.
_DTYPE_FLAGS = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float64: utils.fp64_is_supported,
    torch.complex128: utils.fp64_is_supported,
}
_BSC_DTYPES = [
    dtype
    for dtype in tu.REQUIRED_DTYPES
    + [torch.int16, torch.bool, torch.complex64, torch.float64, torch.complex128]
    if _DTYPE_FLAGS.get(dtype, True)
]
_BACKWARD_DTYPES = [
    dtype
    for dtype in (torch.float16, torch.float32, torch.bfloat16)
    if _DTYPE_FLAGS.get(dtype, True)
]


def _sparse_row_axis(tensor, dense_dim):
    """Axis of the sparse-matrix row dimension for this dense_dim."""
    return tensor.dim() - dense_dim - 2


def _assert_bsc_equal(res, ref, inp):
    """Candidate metadata/device plus the shared exact comparison of the payload.

    This build exposes no blocksize() accessor, so the block geometry is observed
    through the stored structure, which the shared sparse-aware assertion compares
    exactly (cc/row indices, values and stored count). It checks the original
    dtype before widening FP8, so no dense materialisation is needed -- FP8 has no
    dense scatter kernel on this backend.
    """
    assert res.layout == torch.sparse_bsc
    # the result belongs on the device of the candidate's own input, which is not
    # necessarily the device the reference ran on
    assert res.device == inp.device
    tu.assert_result_equal(res, ref)


def _apply_block_pattern(inp, pattern, blocksize, dense_dim):
    """Zero one block-aligned region of the sparse axis, in place.

    ``block_zeroed`` removes the first block entirely (it is then not stored);
    ``interior_zero`` zeroes the first row of every block row, so the stored
    blocks keep an interior zero row, and pins the next row to a representable
    nonzero so every block stays stored no matter what the payload generator
    produced.
    """
    row_axis = _sparse_row_axis(inp, dense_dim)
    if pattern == "block_zeroed":
        inp.narrow(row_axis, 0, blocksize[0]).narrow(
            row_axis + 1, 0, blocksize[1]
        ).zero_()
        return
    for index in range(0, inp.size(row_axis), blocksize[0]):
        inp.select(row_axis, index).zero_()
        if blocksize[0] > 1:
            inp.select(row_axis, index + 1).fill_(1)


def _zero_block_rows(inp, blocksize, dense_dim):
    """Zero whole block rows of the declared sparse axis, block-aligned."""
    row_axis = _sparse_row_axis(inp, dense_dim)
    start = (inp.size(row_axis) // blocksize[0] // 2) * blocksize[0]
    inp.narrow(row_axis, start, inp.size(row_axis) - start).zero_()


def _structure_input(pattern, dtype, shape, blocksize, dense_dim):
    """Build an input whose all-zero-block pattern exercises one prune path."""
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    row_axis = _sparse_row_axis(inp, dense_dim)
    if pattern == "zeros":
        inp.zero_()
    elif pattern == "single_block":
        inp.zero_()
        inp.narrow(row_axis, 0, blocksize[0]).narrow(
            row_axis + 1, 0, blocksize[1]
        ).fill_(1)
    elif pattern == "interior_zero":
        _apply_block_pattern(inp, pattern, blocksize, dense_dim)
    elif pattern == "block_rows_pruned":
        _zero_block_rows(inp, blocksize, dense_dim)
    elif pattern == "distinct_batches":
        inp.zero_()
        # each batch keeps one block row at a different offset, so the stored
        # block counts stay equal (required by the batched form) while the
        # per-batch patterns differ
        for batch in range(inp.size(0)):
            view = inp.select(0, batch)
            axis = _sparse_row_axis(view, dense_dim)
            start = (batch % (view.size(axis) // blocksize[0])) * blocksize[0]
            view.narrow(axis, start, blocksize[0]).fill_(1 if batch % 2 == 0 else -1)
    else:
        raise AssertionError(pattern)
    return inp


# Non-contiguous and offset inputs: transposed, sliced, copied, stepped and
# expanded views plus a conjugated complex view, which must materialise exactly.
# flip_copy is flip(0).flip(1), which copies; it is not a negative-stride view.
_LAYOUT_CASES = tu.selected_cases(
    [
        ("transpose", torch.float16),
        ("transpose", torch.int32),
        ("offset", torch.float32),
        ("offset", torch.int8),
        ("flip_copy", torch.float32),
        ("stepped", torch.float32),
        ("expanded", torch.float32),
        ("conj", torch.complex64),
    ],
    quick=[],
)

# Batched rows: one tile repeated along the batch dims, dense_dim omitted so the
# two trailing dims stay sparse (the schema default for a dense input).
_BATCHED_CASES = tu.selected_cases(
    [
        ((4,), (8, 8), (2, 2)),
        ((2, 4), (16, 16), (4, 4)),
    ],
    quick=[],
)

# blocksize coverage: unit blocks, repeated blocks along one axis, the whole
# matrix as a single block, and two hybrid dense_dim forms.
_BLOCKSIZE_CASES = tu.selected_cases(
    [
        ((1024, 1024), (1, 1), 0),
        ((1024, 1024), (2, 2), 0),
        ((1024, 1024), (2, 8), 0),
        ((1024, 1024), (8, 8), 0),
        ((1024, 1024), (1024, 1024), 0),
        ((20, 320, 15), (5, 5), 1),
        ((16, 128, 64, 60), (4, 8), 2),
    ],
    quick=[],
)

# dense_dim is optional, so the call that omits it must work; the native default
# for a dense input keeps the last two dims sparse.
_OPTIONAL_DENSE_DIM_CASES = tu.selected_cases(
    [
        ("omitted", (1024, 1024), (4, 4)),
        ("none", (1024, 1024), (4, 4)),
        ("omitted", (8, 16, 16), (4, 4)),
        ("none", (8, 16, 16), (4, 4)),
    ],
    quick=[],
)

# Stored-block structure: an empty result, a fully stored matrix, blocks that keep
# interior zeros, half-pruned block rows (also in hybrid dense_dim form) and two
# batches whose patterns differ while their stored-block counts stay equal.
_STRUCTURE_CASES = tu.selected_cases(
    [
        ("zeros", (8, 8), (2, 2), 0),
        ("single_block", (8, 8), (2, 2), 0),
        ("interior_zero", (8, 8), (2, 2), 0),
        ("block_rows_pruned", (8, 8), (2, 2), 0),
        ("interior_zero", (4, 8, 16), (2, 4), 1),
        ("block_rows_pruned", (4, 8, 16), (2, 4), 1),
        ("distinct_batches", (2, 8, 8), (2, 2), 0),
    ],
    quick=[],
)

# nan/inf scenarios for every supported floating dtype plus the complex pair,
# which reuses the same payload as exact real components. The shared generator
# emits nan, inf and nan+inf together and already omits the inf scenarios for
# float8_e4m3fn, which cannot represent infinity.
_COMPLEX_SPECIAL_DTYPES = [torch.complex64] + (
    [torch.complex128] if utils.fp64_is_supported else []
)
_SPECIAL_DTYPES = [
    dtype
    for dtype in (
        torch.float16,
        torch.float32,
        torch.bfloat16,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if _DTYPE_FLAGS.get(dtype, True)
]
_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases(_SPECIAL_DTYPES)
    + [
        (dtype, scenario)
        for dtype in _COMPLEX_SPECIAL_DTYPES
        for scenario in ("nan", "inf", "mixed")
    ],
    quick=[],
)
_SPECIAL_LAYOUTS = tu.selected_cases(
    [((1, 5), (1, 1), None), ((2, 5), (2, 5), None), ((2, 5, 5), (1, 1), 0)],
    quick=[],
)

# Backward is default-only. Probed natively, the relayout gradient equals the
# nonuniform upstream itself at every position, including across a pruned block,
# so the fixture keeps an upstream that stays nonzero even where the forward
# input has zero rows; the interior_zero row adds the case where a stored block
# keeps a zero row.
_BACKWARD_CASES = tu.selected_cases(
    [
        ((64, 64), (2, 2), 0, "block_zeroed"),
        ((4, 64, 64), (2, 2), 1, "block_zeroed"),
        ((2, 8, 32, 32), (2, 2), 2, "block_zeroed"),
        ((4, 64, 64), (2, 2), 1, "interior_zero"),
    ],
    quick=[],
)

# Exceptions measured with the native operator on the active backend: rank and
# dense_dim-too-large failures surface as IndexError, invalid blocksize and
# out-of-range dense_dim values as RuntimeError. Every probed dtype is supported,
# so there is no unsupported-dtype row, and a bool blocksize pair is accepted
# natively and is therefore not a negative.
_INVALID_CASES = [
    ((), (1, 1), 0, IndexError),
    ((0,), (1, 1), 0, IndexError),
    ((8,), (1, 1), 0, IndexError),
    ((8, 8), (2,), 0, RuntimeError),
    ((8, 8), (2, 2, 2), 0, RuntimeError),
    ((8, 8), (0, 0), 0, RuntimeError),
    ((8, 8), (-2, -2), 0, RuntimeError),
    ((8, 8), (3, 3), 0, RuntimeError),
    ((8, 8), (2, 2), -1, RuntimeError),
    ((8, 8), (2, 2), 1, IndexError),
    ((1, 3, 4, 5, 6), (1, 1), 4, IndexError),
    ((1, 3, 4, 5, 6), (1, 1), -5, RuntimeError),
]


def _layout_input(base, layout):
    if layout == "transpose":
        return base.t()
    if layout == "offset":
        return base[2:18, 3:19]
    if layout == "flip_copy":
        return base.flip(0).flip(1)
    if layout == "stepped":
        return base[::2, ::2]
    if layout == "expanded":
        return base[:1].expand(32, 32)
    return base.conj()


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("dtype", _BSC_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape,blocksize,dense_dim", _MATRIX_CASES)
def test_to_sparse_bsc(shape, blocksize, dense_dim, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    before = inp.clone()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse_bsc(ref_inp, blocksize, dense_dim)
    res_out = flag_gems.to_sparse_bsc(inp, blocksize, dense_dim)

    # the relayout only reads its input
    tu.assert_result_equal(inp, before)
    _assert_bsc_equal(res_out, ref_out, inp)


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("dtype", _BSC_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("batch_shape,shape,blocksize", _BATCHED_CASES)
def test_to_sparse_bsc_batched(batch_shape, shape, blocksize, value_range, dtype):
    tile = tu.make_input(dtype, shape, value_range)
    inp = tile.repeat(*batch_shape, *([1] * len(shape)))
    before = inp.clone()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse_bsc(ref_inp, blocksize)
    res_out = flag_gems.to_sparse_bsc(inp, blocksize)

    # the relayout only reads its input
    tu.assert_result_equal(inp, before)
    _assert_bsc_equal(res_out, ref_out, inp)


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("dtype", [torch.float16, torch.int32])
@pytest.mark.parametrize("shape,blocksize,dense_dim", _BLOCKSIZE_CASES)
def test_to_sparse_bsc_blocksize(shape, blocksize, dense_dim, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    before = inp.clone()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse_bsc(ref_inp, blocksize, dense_dim)
    res_out = flag_gems.to_sparse_bsc(inp, blocksize, dense_dim)

    # the relayout only reads its input
    tu.assert_result_equal(inp, before)
    _assert_bsc_equal(res_out, ref_out, inp)


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("dtype", [torch.float32, torch.int16])
@pytest.mark.parametrize("argument,shape,blocksize", _OPTIONAL_DENSE_DIM_CASES)
def test_to_sparse_bsc_optional_dense_dim(argument, shape, blocksize, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    before = inp.clone()
    ref_inp = tu.to_reference(inp)

    if argument == "omitted":
        ref_out = torch.ops.aten.to_sparse_bsc(ref_inp, blocksize)
        res_out = flag_gems.to_sparse_bsc(inp, blocksize)
    else:
        ref_out = torch.ops.aten.to_sparse_bsc(ref_inp, blocksize, None)
        res_out = flag_gems.to_sparse_bsc(inp, blocksize, None)

    # the relayout only reads its input
    tu.assert_result_equal(inp, before)
    _assert_bsc_equal(res_out, ref_out, inp)


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("pattern,shape,blocksize,dense_dim", _STRUCTURE_CASES)
def test_to_sparse_bsc_payload_structure(pattern, shape, blocksize, dense_dim, dtype):
    inp = _structure_input(pattern, dtype, shape, blocksize, dense_dim)
    before = inp.clone()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse_bsc(ref_inp, blocksize, dense_dim)
    res_out = flag_gems.to_sparse_bsc(inp, blocksize, dense_dim)

    # the relayout only reads its input
    tu.assert_result_equal(inp, before)
    _assert_bsc_equal(res_out, ref_out, inp)


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("shape,blocksize,dense_dim", _SPECIAL_LAYOUTS)
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_to_sparse_bsc_special_values(dtype, scenario, shape, blocksize, dense_dim):
    payload = tu.make_special_input(dtype, scenario)
    count = 1
    for size in shape:
        count *= size
    inp = payload.repeat(count // payload.numel()).view(*shape)
    before = inp.clone()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse_bsc(ref_inp, blocksize, dense_dim)
    res_out = flag_gems.to_sparse_bsc(inp, blocksize, dense_dim)

    # the relayout only reads its input, and nan must compare equal here
    tu.assert_result_equal(inp, before)
    _assert_bsc_equal(res_out, ref_out, inp)


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("layout,dtype", _LAYOUT_CASES)
def test_to_sparse_bsc_input_layout(layout, dtype):
    base = tu.make_input(dtype, (32, 32), ["-1", "1"])
    before_base = base.clone()
    inp = _layout_input(base, layout)
    # flip_copy is a real copy, so the candidate operand needs its own snapshot
    before_inp = inp.clone()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse_bsc(ref_inp, (2, 2), 0)
    res_out = flag_gems.to_sparse_bsc(inp, (2, 2), 0)

    # the relayout only reads its operand: compare the operand itself and, for
    # the genuine views, the whole storage so a write outside the viewed
    # slice/offset/expansion is caught as well
    tu.assert_result_equal(inp, before_inp)
    tu.assert_result_equal(base, before_base)
    _assert_bsc_equal(res_out, ref_out, inp)


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
@pytest.mark.parametrize("shape,blocksize,dense_dim,pattern", _BACKWARD_CASES)
def test_to_sparse_bsc_backward(shape, blocksize, dense_dim, pattern, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    # the input is kept free of accidental zeros so only the applied pattern
    # creates zero blocks and interior zero rows
    inp = torch.where(inp == 0, torch.ones_like(inp), inp)
    # the upstream stays nonuniform and nonzero everywhere, including inside the
    # pruned block and at the input's interior zero rows, so the fixture pins the
    # gradient at the exact positions a candidate could wrongly drop
    upstream = tu.make_input(dtype, shape, ["-1", "1"])
    upstream = torch.where(upstream == 0, torch.ones_like(upstream), upstream)
    _apply_block_pattern(inp, pattern, blocksize, dense_dim)
    inp.requires_grad_(True)
    before = inp.detach().clone()
    ref_inp = tu.to_reference(inp)
    ref_upstream = tu.to_reference(upstream)

    ref_out = torch.ops.aten.to_sparse_bsc(ref_inp, blocksize, dense_dim)
    res_out = flag_gems.to_sparse_bsc(inp, blocksize, dense_dim)

    # the relayout only reads its input
    tu.assert_result_equal(inp, before)
    _assert_bsc_equal(res_out, ref_out, inp)

    # the sparse grad_output is a fixture built with the native operator on both
    # sides, so the candidate is only exercised on its forward output and the
    # backward that consumes this grad_output
    res_upstream = torch.ops.aten.to_sparse_bsc(upstream, blocksize, dense_dim)
    ref_sparse_upstream = torch.ops.aten.to_sparse_bsc(
        ref_upstream, blocksize, dense_dim
    )
    (res_grad,) = torch.autograd.grad(res_out, inp, grad_outputs=res_upstream)
    (ref_grad,) = torch.autograd.grad(
        ref_out, ref_inp, grad_outputs=ref_sparse_upstream
    )

    # the relayout performs no arithmetic reduction, so both gradients carry the
    # upstream values through unchanged and are compared exactly
    tu.assert_result_equal(res_grad, ref_grad)
    tu.assert_result_equal(inp, before)


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("shape,blocksize,dense_dim,raises", _INVALID_CASES)
def test_to_sparse_bsc_invalid_arguments(shape, blocksize, dense_dim, raises):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    with pytest.raises(raises):
        flag_gems.to_sparse_bsc(inp, blocksize, dense_dim)


@pytest.mark.to_sparse_bsc
def test_to_sparse_bsc_unequal_batch_blocks():
    """Batches with different stored-block counts are rejected."""
    tile = tu.make_input(torch.float32, (4, 4), ["-1", "1"])
    inp = tile.repeat(2, 1, 1)
    inp.select(0, 1).zero_()

    with pytest.raises(RuntimeError):
        flag_gems.to_sparse_bsc(inp, (2, 2))
