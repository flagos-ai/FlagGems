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

"""Correctness tests for ``aten::_sparse_sparse_matmul`` (2-D COO @ 2-D COO).

Both operands must be rank-2 sparse COO matrices, so the operator has no scalar
form and does not broadcast: the spec's dense shape ladder is expressed as
(m, k, n) operand descriptors and rank 0/1/3/4/5 operands are covered as
negatives.  Floating and complex kernels exist; the integer/bool/fp8 family is a
measured kernel-availability limit of this backend and stays a separate negative
family.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

_UNIT_RANGE = ["-1", "1"]
_NNZ_DIVISOR = 16
# Number of small inner indices the coordinate anchor always stores.
_ANCHOR_INNER = 3

# fp16/fp32/complex64 are the kernels measured on this backend; bf16/fp64/
# complex128 depend on the device capability flags (all three are supported on
# the measured NVIDIA CUDA backend).  These are measured target capabilities:
# serving the same dtypes on host does not establish a device kernel.
_BASE_DTYPES = [torch.float16, torch.float32, torch.complex64]
_MATMUL_DTYPES = (
    _BASE_DTYPES
    + ([torch.bfloat16] if utils.bf16_is_supported else [])
    + ([torch.float64, torch.complex128] if utils.fp64_is_supported else [])
)

# Kernel-availability limit measured on the NVIDIA CUDA backend torch 2.8.0a0:
# ``"sparse_matmul" not implemented for 'Char'/'Byte'/'Int'/'Long'/'Bool'/
# 'Float8_e4m3fn'/'Float8_e5m2'``.  The rows are selected at collection time
# from the actual vendor and the existing dtype flags, so this is a limit of one
# measured backend rather than a portable operator contract, and no other
# backend is either required to reject those dtypes or skipped at runtime.
_UNSUPPORTED_DTYPES = [torch.int8, torch.uint8, torch.int32, torch.bool]
if utils.int64_is_supported:
    _UNSUPPORTED_DTYPES.append(torch.int64)
if utils.fp8_is_supported:
    _UNSUPPORTED_DTYPES.extend([torch.float8_e4m3fn, torch.float8_e5m2])
_MEASURED_KERNEL_LIMITS = (
    _UNSUPPORTED_DTYPES if flag_gems.vendor_name in ("nvidia",) else []
)
_UNSUPPORTED_DTYPE_ROWS = tu.selected_cases(
    _MEASURED_KERNEL_LIMITS, quick=_MEASURED_KERNEL_LIMITS
)

# The auxiliary dimensions (empty results, .out, COO layouts, stored zeros) use
# a representative real dtype pair; the main grid carries the full dtype set.
_AUX_DTYPES = [torch.float16, torch.float32]

# (m, k, n) descriptors.  The dense ladder itself cannot be used because only
# rank-2 COO operands are accepted: the first seven rows map the ladder's ranks
# and element scales, and the last three keep its large element counts with a
# small shared dimension.  The stored entries of an operand are the requested
# draws (m*k//16 resp. k*n//16, uncapped) deduplicated, plus the anchor grid: the
# stored NNZ is the deduplicated draw count plus the anchor entries, so for small
# shapes the anchors make it exceed the requested draw count.
_MKN_CASES = [
    (1, 1, 1),
    (7, 3, 5),
    (256, 128, 64),
    (20, 320, 15),
    (16, 128, 64),
    (16, 7, 57),
    (1024, 1024, 1024),
]
_LARGE_MKN_CASES = [
    (1, 6400, 15),
    (1, 131072, 60),
    (1, 204288, 29),
]

# Each row carries its own value range; the large descriptors are extra shape
# coverage beyond the 5 x 7 grid and run the full range set too.
_MAIN_ROWS = tu.selected_cases(
    [
        (m, k, n, value_range)
        for (m, k, n) in _MKN_CASES + _LARGE_MKN_CASES
        for value_range in tu.selected_ranges()
    ],
    quick=[(2, 19, 7, _UNIT_RANGE)],
)

# ((m, k, n), mat1 has no stored entries, mat2 has none): the operand without
# stored positions, plus the zero-extent descriptors the native operator accepts.
_EMPTY_ROWS = tu.selected_cases(
    [
        pytest.param((8, 8, 8), True, False, id="mat1-empty"),
        pytest.param((8, 8, 8), False, True, id="mat2-empty"),
        pytest.param((8, 8, 8), True, True, id="both-empty"),
        pytest.param((3, 0, 4), True, True, id="k-zero"),
        pytest.param((0, 4, 4), True, True, id="m-zero"),
        pytest.param((4, 4, 0), True, True, id="n-zero"),
    ],
    quick=[],
)

# Products written through the .out overload; the buffer is an empty COO tensor
# of the result shape.
_OUT_ROWS = tu.selected_cases([(6, 6, 6), (3, 0, 4)], quick=[])

# (mat1 stores duplicated unsorted coordinates, mat2 does, mat1 values are a
# strided view with a nonzero offset, mat2 values are).  Every combination of
# coalesced / uncoalesced operands is covered, on either side and on both.
_LAYOUT_ROWS = tu.selected_cases(
    [
        pytest.param(False, False, False, False, id="both-coalesced"),
        pytest.param(True, False, False, False, id="mat1-uncoalesced"),
        pytest.param(False, True, False, False, id="mat2-uncoalesced"),
        pytest.param(True, True, False, False, id="both-uncoalesced"),
        pytest.param(True, False, True, False, id="mat1-strided-values"),
        pytest.param(False, True, False, True, id="mat2-strided-values"),
    ],
    quick=[],
)

# Rows differ only in whether mat1's zero-valued positions are stored or left
# structurally missing; both must contribute nothing to the product.
_STORED_ZERO_ROWS = tu.selected_cases(
    [pytest.param(True, id="stored"), pytest.param(False, id="missing")],
    quick=[],
)

_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_MATMUL_DTYPES), quick=[])

# (m, k, n, both operands and the upstream gradient store every position).  The
# fully stored rows exercise dense-valued gradient structure; the sparse rows
# keep structurally missing entries, with the anchor grid keeping the small
# inner indices shared on both sides.
_BACKWARD_ROWS = tu.selected_cases(
    [
        pytest.param(4, 4, 4, True, id="full-4x4x4"),
        pytest.param(8, 6, 5, True, id="full-8x6x5"),
        pytest.param(6, 5, 4, False, id="sparse-6x5x4"),
        pytest.param(7, 3, 6, False, id="sparse-7x3x6"),
    ],
    quick=[],
)
_BACKWARD_DTYPES = tu.selected_cases(
    [torch.float32, torch.float16, torch.complex64]
    + ([torch.float64, torch.complex128] if utils.fp64_is_supported else []),
    quick=[],
)

_NON_2D_RANKS = [0, 1, 3, 4, 5]
_OPERAND_SIDES = ["mat1", "mat2"]


def _nnz(size):
    """Requested random coordinates for an operand with ``size`` positions.  The
    stored entries are these draws deduplicated plus the anchor grid, so the
    stored NNZ is not exactly this number."""
    if size <= 0:
        return 0
    return max(1, size // _NNZ_DIVISOR)


def _stored_codes(shape, count, seed):
    """Distinct linearised positions for one operand: ``count`` seeded random
    draws plus the fixed anchor grid ``(0, 0..K-1)`` and ``(0..K-1, 0)``, all
    deduplicated.  Whichever axis is the shared inner one, the anchor leaves the
    small inner indices stored, so two operands share stored inner indices and
    the product is never structurally vacant -- a vacant product would make the
    value comparison vacuous, because two empty results compare equal.  The
    drawn positions are randomised, so the stored count is the deduplicated draw
    count plus the anchor positions, not exactly ``count``.
    """
    rows, cols = shape
    device = flag_gems.device
    generator = torch.Generator(device=device).manual_seed(seed)
    drawn = torch.randint(
        0,
        rows * cols,
        (max(count, 1),),
        generator=generator,
        device=device,
        dtype=torch.int64,
    )
    keep = min(_ANCHOR_INNER, max(rows, cols))
    anchor = torch.cat(
        [
            torch.arange(min(cols, keep), dtype=torch.int64, device=device),
            torch.arange(min(rows, keep), dtype=torch.int64, device=device) * cols,
        ]
    )
    return torch.unique(torch.cat([drawn, anchor]))


def _coordinates(shape, count, seed):
    """Distinct stored coordinates for one operand; a device-local seeded
    generator keeps each case reproducible without touching global RNG state."""
    rows, cols = shape
    if count <= 0 or rows == 0 or cols == 0:
        return torch.empty(2, 0, dtype=torch.int64, device=flag_gems.device)
    codes = _stored_codes(shape, count, seed)
    return torch.stack([codes // cols, codes % cols])


def _operand(shape, count, value_range, dtype, seed=0):
    """COO operand whose values come from the shared value-range helper."""
    index = _coordinates(shape, count, seed)
    nnz = index.shape[1]
    values = (
        torch.empty(0, dtype=dtype, device=flag_gems.device)
        if nnz == 0
        else tu.make_input(dtype, (nnz,), value_range)
    )
    return torch.sparse_coo_tensor(index, values, shape, device=flag_gems.device)


def _constant_operand(shape, count, dtype, seed=0):
    """Operand of constant nonzero values, used where the value range is not the
    dimension under test (dtype, rank and layout negatives)."""
    index = _coordinates(shape, count, seed)
    return torch.sparse_coo_tensor(
        index,
        torch.ones(index.shape[1], dtype=dtype, device=flag_gems.device),
        shape,
        device=flag_gems.device,
    )


def _rank_operand(rank, dtype):
    """Sparse COO tensor of the requested rank; rank 0 uses an empty index
    tensor."""
    device = flag_gems.device
    if rank == 0:
        return torch.sparse_coo_tensor(
            torch.empty(0, 0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=dtype, device=device),
            (),
            device=device,
        )
    size = tuple(2 + axis for axis in range(rank))
    count = 2
    index = torch.stack(
        [
            torch.arange(count, device=device, dtype=torch.int64) % size[axis]
            for axis in range(rank)
        ]
    )
    return torch.sparse_coo_tensor(
        index, torch.ones(count, dtype=dtype, device=device), size, device=device
    )


def _pattern_operand(shape, count, dtype, *, duplicated, strided_values=False, seed=0):
    """Operand with deterministically unsorted coordinates -- every position twice
    when ``duplicated``, which the native kernel sums -- optionally holding its
    values as a strided view with a nonzero storage offset."""
    rows, cols = shape
    device = flag_gems.device
    codes = _stored_codes(shape, count, seed)
    if duplicated:
        # Concatenating the reversed coordinates on top of the original ones
        # stores each position exactly twice, in reverse order, so the operand
        # is both duplicated and unsorted.
        index = torch.stack(
            [
                torch.cat([axis, torch.flip(axis, [0])])
                for axis in (codes // cols, codes % cols)
            ]
        )
    else:
        index = torch.stack([codes // cols, codes % cols])
    values = tu.make_input(dtype, (index.shape[1],), _UNIT_RANGE)
    if strided_values:
        # Stride-2 view of a larger buffer: the stored values are neither
        # contiguous nor at offset zero.
        buffer = torch.zeros(2 * index.shape[1] + 4, dtype=dtype, device=device)
        buffer[2 : 2 + 2 * index.shape[1] : 2] = values
        values = buffer[2 : 2 + 2 * index.shape[1] : 2]
    operand = torch.sparse_coo_tensor(
        index, values, shape, device=device, is_coalesced=False
    )
    if duplicated:
        return operand
    # Without duplicates the honest representation of the coalesced case is a
    # sorted, deduplicated copy.
    return operand.coalesce()


def _partially_zero_operand(shape, dtype, *, store_zeros):
    """3 x 3 operand whose two zero-valued positions are either stored or left
    structurally missing."""
    positions = [(0, 0, 0.0), (0, 2, 2.0), (1, 1, 3.0), (2, 0, 4.0), (2, 1, 0.0)]
    kept = [entry for entry in positions if store_zeros or entry[2] != 0.0]
    return torch.sparse_coo_tensor(
        torch.tensor(
            [[row for row, _, _ in kept], [col for _, col, _ in kept]],
            dtype=torch.int64,
            device=flag_gems.device,
        ),
        torch.tensor(
            [value for _, _, value in kept],
            dtype=torch.float32,
            device=flag_gems.device,
        ).to(dtype),
        shape,
        device=flag_gems.device,
    )


def _fully_stored_operand(shape, dtype):
    """COO operand storing every position."""
    rows, cols = shape
    device = flag_gems.device
    flat = torch.arange(rows * cols, dtype=torch.int64, device=device)
    return torch.sparse_coo_tensor(
        torch.stack([flat // cols, flat % cols]),
        tu.make_input(dtype, (rows * cols,), _UNIT_RANGE),
        shape,
        device=device,
    )


def _empty_out(shape, dtype, device):
    """Allocate an empty COO result buffer for the .out overload."""
    return torch.sparse_coo_tensor(
        torch.empty(2, 0, dtype=torch.int64, device=device),
        torch.empty(0, dtype=dtype, device=device),
        shape,
        device=device,
    )


def _snapshot(operand):
    """Stored entries plus the metadata taken before a candidate call: a
    coalescing implementation can rewrite the coalesced flag or the shape in
    place without changing the entries a values-only comparison would see."""
    return (
        operand._indices().clone(),
        operand._values().clone(),
        operand.is_coalesced(),
        tuple(operand.shape),
        operand.layout,
    )


def _assert_unmutated(operand, snapshot):
    """The candidate must compute from its operands without rewriting them,
    which a coalescing implementation could do in place."""
    indices, values, coalesced, shape, layout = snapshot
    tu.assert_result_equal(operand._indices(), indices)
    tu.assert_result_equal(operand._values(), values)
    assert operand.is_coalesced() == coalesced
    assert tuple(operand.shape) == shape
    assert operand.layout == layout


@pytest.mark.sparse_sparse_matmul
@pytest.mark.parametrize("dtype", _MATMUL_DTYPES)
@pytest.mark.parametrize("m, k, n, value_range", _MAIN_ROWS)
def test__sparse_sparse_matmul_mm(m, k, n, value_range, dtype):
    inp_a = _operand((m, k), _nnz(m * k), value_range, dtype, seed=0)
    inp_b = _operand((k, n), _nnz(k * n), value_range, dtype, seed=1)
    ref_a = tu.to_reference(inp_a)
    ref_b = tu.to_reference(inp_b)
    snapshots = (_snapshot(inp_a), _snapshot(inp_b))

    ref_out = torch.ops.aten._sparse_sparse_matmul(ref_a, ref_b)
    res_out = flag_gems._sparse_sparse_matmul(inp_a, inp_b)

    tu.assert_result_close(res_out, ref_out)
    _assert_unmutated(inp_a, snapshots[0])
    _assert_unmutated(inp_b, snapshots[1])


@pytest.mark.sparse_sparse_matmul
@pytest.mark.parametrize("dtype", _AUX_DTYPES)
@pytest.mark.parametrize("shape, mat1_empty, mat2_empty", _EMPTY_ROWS)
def test__sparse_sparse_matmul_empty(shape, mat1_empty, mat2_empty, dtype):
    m, k, n = shape
    inp_a = _operand((m, k), 0 if mat1_empty else _nnz(m * k), _UNIT_RANGE, dtype, 0)
    inp_b = _operand((k, n), 0 if mat2_empty else _nnz(k * n), _UNIT_RANGE, dtype, 1)
    ref_a = tu.to_reference(inp_a)
    ref_b = tu.to_reference(inp_b)

    ref_out = torch.ops.aten._sparse_sparse_matmul(ref_a, ref_b)
    res_out = flag_gems._sparse_sparse_matmul(inp_a, inp_b)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.sparse_sparse_matmul
@pytest.mark.parametrize("dtype", _AUX_DTYPES)
@pytest.mark.parametrize("shape", _OUT_ROWS)
def test__sparse_sparse_matmul_out(shape, dtype):
    m, k, n = shape
    inp_a = _operand((m, k), _nnz(m * k), _UNIT_RANGE, dtype, seed=0)
    inp_b = _operand((k, n), _nnz(k * n), _UNIT_RANGE, dtype, seed=1)
    ref_a = tu.to_reference(inp_a)
    ref_b = tu.to_reference(inp_b)
    res_out = _empty_out((m, n), dtype, inp_a.device)
    ref_out = _empty_out((m, n), dtype, ref_a.device)
    snapshots = (_snapshot(inp_a), _snapshot(inp_b))

    ref_ret = torch.ops.aten._sparse_sparse_matmul.out(ref_a, ref_b, out=ref_out)
    res_ret = flag_gems._sparse_sparse_matmul(inp_a, inp_b, out=res_out)

    tu.assert_result_close(res_ret, ref_ret)
    # .out writes the result into the provided buffer and returns that buffer.
    assert res_ret is res_out
    _assert_unmutated(inp_a, snapshots[0])
    _assert_unmutated(inp_b, snapshots[1])


@pytest.mark.sparse_sparse_matmul
@pytest.mark.parametrize("dtype", _AUX_DTYPES)
@pytest.mark.parametrize(
    "mat1_unsorted, mat2_unsorted, mat1_strided, mat2_strided", _LAYOUT_ROWS
)
def test__sparse_sparse_matmul_coo_layout(
    mat1_unsorted, mat2_unsorted, mat1_strided, mat2_strided, dtype
):
    m, k, n = 6, 6, 6
    inp_a = _pattern_operand(
        (m, k),
        _nnz(m * k),
        dtype,
        duplicated=mat1_unsorted,
        strided_values=mat1_strided,
        seed=0,
    )
    inp_b = _pattern_operand(
        (k, n),
        _nnz(k * n),
        dtype,
        duplicated=mat2_unsorted,
        strided_values=mat2_strided,
        seed=1,
    )
    ref_a = tu.to_reference(inp_a)
    ref_b = tu.to_reference(inp_b)
    snapshots = (_snapshot(inp_a), _snapshot(inp_b))

    ref_out = torch.ops.aten._sparse_sparse_matmul(ref_a, ref_b)
    res_out = flag_gems._sparse_sparse_matmul(inp_a, inp_b)

    # The strided-values cases hand the candidate a stride-2 values view with a
    # nonzero storage offset, while tu.to_reference materializes a contiguous
    # clone: the oracle compares logical values, and the candidate's ability to
    # read the view is graded by this whole-output comparison.
    tu.assert_result_close(res_out, ref_out)
    # The native operator returns a coalesced result for uncoalesced operands.
    assert res_out.is_coalesced()
    _assert_unmutated(inp_a, snapshots[0])
    _assert_unmutated(inp_b, snapshots[1])


@pytest.mark.sparse_sparse_matmul
@pytest.mark.parametrize("dtype", _AUX_DTYPES)
@pytest.mark.parametrize("store_zeros", _STORED_ZERO_ROWS)
def test__sparse_sparse_matmul_stored_zero(store_zeros, dtype):
    shape = (3, 3)
    inp_a = _partially_zero_operand(shape, dtype, store_zeros=store_zeros)
    inp_b = torch.ones(shape, dtype=dtype, device=flag_gems.device).to_sparse()
    ref_a = tu.to_reference(inp_a)
    ref_b = tu.to_reference(inp_b)

    ref_out = torch.ops.aten._sparse_sparse_matmul(ref_a, ref_b)
    res_out = flag_gems._sparse_sparse_matmul(inp_a, inp_b)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.sparse_sparse_matmul
@pytest.mark.parametrize("dtype, scenario", _SPECIAL_CASES)
def test__sparse_sparse_matmul_special_values(dtype, scenario):
    # mat1 stores a complete special row next to a complete finite row and mat2
    # stores every position, so each stored special is multiplied by a nonzero
    # factor and reaches the accumulation instead of being skipped as a missing
    # entry.
    k, width = 5, 3
    specials = tu.make_special_input(dtype, scenario)
    finite = torch.ones(k, dtype=dtype, device=flag_gems.device)
    columns = torch.arange(k, dtype=torch.int64, device=flag_gems.device)
    rows = torch.cat(
        [
            torch.zeros(k, dtype=torch.int64, device=flag_gems.device),
            torch.ones(k, dtype=torch.int64, device=flag_gems.device),
        ]
    )
    inp_a = torch.sparse_coo_tensor(
        torch.stack([rows, columns.repeat(2)]),
        torch.cat([specials, finite]),
        (2, k),
        device=flag_gems.device,
    )
    inp_b = torch.ones(k, width, dtype=dtype, device=flag_gems.device).to_sparse()
    ref_a = tu.to_reference(inp_a)
    ref_b = tu.to_reference(inp_b)

    ref_out = torch.ops.aten._sparse_sparse_matmul(ref_a, ref_b)
    res_out = flag_gems._sparse_sparse_matmul(inp_a, inp_b)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.sparse_sparse_matmul
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
@pytest.mark.parametrize("m, k, n, fully_stored", _BACKWARD_ROWS)
def test__sparse_sparse_matmul_backward(m, k, n, fully_stored, dtype):
    # The candidate is graded through torch.autograd.grad on its own output, so
    # a candidate that detaches, drops an operand or returns zeros fails here.
    # The upstream gradient must be sparse: the native backward rejects a dense
    # one ("mat2_.is_sparse() INTERNAL ASSERT FAILED" in SparseMatMul.cu).
    if fully_stored:
        inp_a = _fully_stored_operand((m, k), dtype)
        inp_b = _fully_stored_operand((k, n), dtype)
        upstream = _fully_stored_operand((m, n), dtype)
    else:
        inp_a = _operand((m, k), _nnz(m * k), _UNIT_RANGE, dtype, seed=0)
        inp_b = _operand((k, n), _nnz(k * n), _UNIT_RANGE, dtype, seed=1)
        upstream = _operand((m, n), _nnz(m * n), _UNIT_RANGE, dtype, seed=2)
    inp_a.requires_grad_(True)
    inp_b.requires_grad_(True)
    ref_a = tu.to_reference(inp_a)
    ref_b = tu.to_reference(inp_b)
    ref_upstream = tu.to_reference(upstream)

    res_out = flag_gems._sparse_sparse_matmul(inp_a, inp_b)
    ref_out = torch.ops.aten._sparse_sparse_matmul(ref_a, ref_b)
    tu.assert_result_close(res_out, ref_out)

    res_grad_a, res_grad_b = torch.autograd.grad(
        res_out, (inp_a, inp_b), grad_outputs=upstream
    )
    ref_grad_a, ref_grad_b = torch.autograd.grad(
        ref_out, (ref_a, ref_b), grad_outputs=ref_upstream
    )
    for res_grad, ref_grad in ((res_grad_a, ref_grad_a), (res_grad_b, ref_grad_b)):
        # A sparse operand's gradient is sparse; a dense gradient would drop the
        # index structure.  The candidate's own coalescing is not part of the
        # gradient contract, so raw layouts are checked first and the complete
        # sparse gradients are then compared in canonical coalesced form.
        assert res_grad.layout == ref_grad.layout
        tu.assert_result_close(res_grad.coalesce(), ref_grad.coalesce())


@pytest.mark.sparse_sparse_matmul
@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPE_ROWS)
def test__sparse_sparse_matmul_rejects_unsupported_dtype(dtype):
    inp_a = _constant_operand((4, 4), 4, dtype, seed=0)
    inp_b = _constant_operand((4, 4), 4, dtype, seed=1)
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems._sparse_sparse_matmul(inp_a, inp_b)


@pytest.mark.sparse_sparse_matmul
@pytest.mark.parametrize("side", _OPERAND_SIDES)
@pytest.mark.parametrize("rank", _NON_2D_RANKS)
def test__sparse_sparse_matmul_rejects_non_2d(rank, side):
    # The operator requires rank-2 operands ("Expected mat1_/mat2_.dim() == 2 to
    # be true"); each operand is varied independently.
    valid = _constant_operand((4, 4), 4, torch.float32, seed=0)
    invalid = _rank_operand(rank, torch.float32)
    args = (invalid, valid) if side == "mat1" else (valid, invalid)
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems._sparse_sparse_matmul(*args)


@pytest.mark.sparse_sparse_matmul
def test__sparse_sparse_matmul_rejects_shape_mismatch():
    # mat1 (4, 5) cannot be multiplied with mat2 (6, 3).
    inp_a = _constant_operand((4, 5), 6, torch.float32, seed=0)
    inp_b = _constant_operand((6, 3), 6, torch.float32, seed=1)
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems._sparse_sparse_matmul(inp_a, inp_b)


@pytest.mark.sparse_sparse_matmul
def test__sparse_sparse_matmul_rejects_dtype_mismatch():
    # Both dtypes have kernels, but the two operands must agree.
    inp_a = _constant_operand((4, 4), 4, torch.float16, seed=0)
    inp_b = _constant_operand((4, 4), 4, torch.float32, seed=1)
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems._sparse_sparse_matmul(inp_a, inp_b)


@pytest.mark.sparse_sparse_matmul
@pytest.mark.parametrize("side", _OPERAND_SIDES)
def test__sparse_sparse_matmul_rejects_dense_operand(side):
    # Each operand position is validated on its own; mat2 is not assumed to
    # inherit mat1's check.
    valid = _constant_operand((4, 4), 4, torch.float32, seed=0)
    dense = torch.ones(4, 4, device=flag_gems.device)
    args = (dense, valid) if side == "mat1" else (valid, dense)
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems._sparse_sparse_matmul(*args)


@pytest.mark.sparse_sparse_matmul
@pytest.mark.parametrize("side", _OPERAND_SIDES)
def test__sparse_sparse_matmul_rejects_csr_operand(side):
    valid = _constant_operand((4, 4), 4, torch.float32, seed=0)
    csr = _constant_operand((4, 4), 4, torch.float32, seed=1).to_sparse_csr()
    args = (csr, valid) if side == "mat1" else (valid, csr)
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems._sparse_sparse_matmul(*args)
