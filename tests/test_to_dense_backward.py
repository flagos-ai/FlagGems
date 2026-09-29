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

"""Correctness tests for aten::to_dense_backward.

to_dense_backward(grad, input, masked_grad=None) is the autograd formula of
_to_dense: it brings the dense grad into the representation of input.  The
native operator dispatches on the layout of input:

* strided: grad converted to the scalar type of input, keeping the stored
  values and the shape of grad;
* COO: grad.sparse_mask(input.coalesce()) when masked_grad is true (the
  default), grad.to_sparse(input.sparse_dim()) when it is false;
* CSR/CSC/BSR/BSC: the same two branches re-wrapped in the layout of input.

No branch combines the operands elementwise, so the operator has no broadcast
rule; the input-metadata family pins that down in place of the spec's broadcast
patterns.  The operator has no .out overload (its only registered overload is
default), so there is nothing to call there.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu


def _avail(dtype):
    """Static capability gate: reads runtime flags, never probes the operator."""
    if dtype == torch.bfloat16:
        # Both the shared alias and the device capability are consulted, per the
        # repository convention (flag_gems.runtime.device.support_bf16).
        return utils.bf16_is_supported and flag_gems.runtime.device.support_bf16
    if dtype == torch.float64:
        return utils.fp64_is_supported
    if dtype == torch.int64:
        return utils.int64_is_supported
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return utils.fp8_is_supported
    return True


_GRID_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
]
DENSE_DTYPES = [dtype for dtype in _GRID_DTYPES if _avail(dtype)]

# Supported dtypes outside the required nine.  The shared value ranges carry no
# information for bool, so bool is covered at shape level instead of inside the
# range grid.
DENSE_EXTRA_DTYPES = [dtype for dtype in (torch.float64, torch.bool) if _avail(dtype)]

# The nan / inf matrix is derived from the dtypes this operator accepts on the
# active backend, not from a generic float constant: e5m2 holds inf, e4m3fn
# cannot, so the two FP8 types keep different scenarios.
_SPECIAL_DTYPES = [
    dtype
    for dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if _avail(dtype)
]
SPECIAL_VALUE_CASES = tu.special_value_cases(_SPECIAL_DTYPES)

# Sentinel: call without the optional argument, exercising the schema default
# instead of an explicitly passed value.  It is matched by value everywhere, so
# a caller-supplied string keeps its meaning.
OMITTED = "<omitted>"

_CROSS_ROWS = [
    (torch.float32, torch.float16, (1024, 1024)),
    (torch.float32, torch.bfloat16, (20, 320, 15)),
    (torch.float32, torch.int32, (1024, 1024)),
    (torch.float32, torch.int64, (1024, 1024)),
    (torch.float32, torch.int8, (1024, 1024)),
    (torch.float32, torch.uint8, (1024, 1024)),
    (torch.float32, torch.float8_e4m3fn, (1024, 1024)),
    (torch.float32, torch.float8_e5m2, (1024, 1024)),
    (torch.float16, torch.float32, (20, 320, 15)),
    (torch.bfloat16, torch.float16, (20, 320, 15)),
    (torch.int32, torch.float32, (1024, 1024)),
    (torch.int64, torch.int8, (1024, 1024)),
]
CROSS_DTYPE_CASES = [row for row in _CROSS_ROWS if _avail(row[0]) and _avail(row[1])]
QUICK_CROSS_DTYPE_CASES = [(torch.float32, torch.float16, (2, 19, 7))]

# The ranges used above are dominated by values near zero, whose integer and
# bool conversions carry no information.  These rows use representable
# boundaries instead.  The conversions are deterministic and equal to the host
# conversion measured for these values (a negative value becomes 255 in uint8,
# 0.40625 and 1.625 truncate towards zero), so the comparison is exact rather
# than tolerant.
_CAST_VALUES = [-1.75, -0.5, 0.0, 0.40625, 1.625, 4.0, 12.0]
_CAST_ROWS = [
    ((2, 19, 7), torch.float32, torch.int8),
    ((2, 19, 7), torch.float32, torch.uint8),
    ((2, 19, 7), torch.float32, torch.int32),
    ((2, 19, 7), torch.float32, torch.int64),
    ((2, 19, 7), torch.float32, torch.bool),
    ((2, 19, 7), torch.float32, torch.float16),
    ((2, 19, 7), torch.float32, torch.bfloat16),
    ((2, 19, 7), torch.float32, torch.float8_e4m3fn),
    ((16, 32), torch.float16, torch.float32),
    ((16, 32), torch.int32, torch.bool),
    ((16, 32), torch.int8, torch.bool),
    ((1024, 1024), torch.float32, torch.int64),
    ((20, 320, 15), torch.float64, torch.int32),
]
CAST_CASES = [row for row in _CAST_ROWS if _avail(row[1]) and _avail(row[2])]
QUICK_CAST_CASES = [
    ((2, 19, 7), torch.float32, torch.int8),
    ((2, 19, 7), torch.float32, torch.bool),
]

# Optional bool parameter: schema default, explicit None, True and False.
MASKED_GRAD_CASES = [OMITTED, None, True, False]
QUICK_MASKED_GRAD_CASES = [OMITTED]

# The dense masked_grad sweep carries its own (shape, dtype) rows so the bf16
# capability gate stays visible to collection instead of being hidden inside
# the test body.
_MASKED_GRAD_DENSE_ROWS = [
    ((20, 320, 15), torch.bfloat16),
    ((20, 320, 15), torch.float16),
    ((1024, 1024), torch.int32),
]
MASKED_GRAD_DENSE_CASES = [row for row in _MASKED_GRAD_DENSE_ROWS if _avail(row[1])]
QUICK_MASKED_GRAD_DENSE_CASES = [((2, 19, 7), torch.float16)]

# Optional[bool] also accepts integers, where 0 is the only false value.  The
# float32 payload is the original coverage; the int32 row keeps an integer
# tensor dtype in the family as well.
_MASKED_GRAD_INT_ROWS = [
    ((20, 320, 15), torch.float32, 0),
    ((20, 320, 15), torch.float32, 1),
    ((20, 320, 15), torch.float32, 2),
    ((1024, 1024), torch.int32, 1),
]
QUICK_MASKED_GRAD_INT_CASES = [((2, 19, 7), torch.float32, 1)]

# (grad shape, input shape, non-contiguous view flag).  The mismatched shapes
# prove the absence of any elementwise or broadcast rule, because only the
# scalar type and the layout of the input are read; the last row feeds a
# non-contiguous input view.
INPUT_METADATA_CASES = [
    ((1024, 1024), (1,), False),
    ((20, 320, 15), (5,), False),
    ((16, 128, 64, 60), (), False),
    ((16, 7, 57, 32, 29), (32, 29), False),
    ((1024, 1024), (1024, 2048), True),
]

# (base shape, size, stride, offset): strided, storage-offset and expanded
# gradients.  The dense path reads only the stored values of the view, so a
# candidate that assumes a contiguous operand is wrong here.
GRAD_LAYOUT_CASES = [
    ((16, 32), (16, 16), (32, 2), 0),
    ((16, 32), (8, 16), (32, 1), 8),
    ((16, 32), (8, 32), (0, 1), 0),
    ((4, 8, 16), (4, 8, 8), (128, 16, 2), 0),
]

# (layout, shape, nnz, dtype, value_range, masked_grad, dense_dim, block).  nnz
# is None for the compressed layouts, whose structure comes from the mask
# geometry, and block is the BSR/BSC block shape.  FP8 is absent from this
# table because the masked branch of the FP8 sparse path fails on the active
# backend (see the fp8-mask test below), while the dense grid and the
# special-value matrix keep FP8 covered.
_COO_ROWS = [
    (torch.sparse_coo, (2, 19, 7), 11, torch.float32, ["-1", "1"], OMITTED, 0, None),
    (torch.sparse_coo, (2, 19, 7), 11, torch.float32, ["-1", "1"], None, 0, None),
    (torch.sparse_coo, (2, 19, 7), 11, torch.float32, ["-1", "1"], True, 0, None),
    (torch.sparse_coo, (2, 19, 7), 11, torch.float32, ["-1", "1"], False, 0, None),
    (torch.sparse_coo, (20, 320, 15), 64, torch.float16, ["0", "1"], True, 0, None),
    (torch.sparse_coo, (20, 320, 15), 64, torch.bfloat16, ["-1", "0"], True, 0, None),
    (torch.sparse_coo, (1024, 1024), 256, torch.float32, ["0", "max"], True, 0, None),
    (torch.sparse_coo, (1024, 1024), 256, torch.int32, ["min", "0"], True, 0, None),
    (torch.sparse_coo, (1024, 1024), 256, torch.int64, ["-1", "1"], True, 0, None),
    (torch.sparse_coo, (1024, 1024), 256, torch.int8, ["-1", "1"], True, 0, None),
    (torch.sparse_coo, (1024, 1024), 256, torch.uint8, ["0", "1"], True, 0, None),
    (torch.sparse_coo, (1024, 1024), 256, torch.float32, ["-1", "1"], False, 0, None),
    (torch.sparse_coo, (5, 5), 25, torch.float32, ["-1", "1"], OMITTED, 0, None),
    (torch.sparse_coo, (5, 5), 25, torch.float32, ["-1", "1"], False, 0, None),
    (torch.sparse_coo, (5, 5), 0, torch.float32, ["-1", "1"], OMITTED, 0, None),
    (torch.sparse_coo, (5, 5), 0, torch.float32, ["-1", "1"], False, 0, None),
    (
        torch.sparse_coo,
        (20, 320, 15, 4),
        64,
        torch.float32,
        ["-1", "1"],
        OMITTED,
        1,
        None,
    ),
    (
        torch.sparse_coo,
        (20, 320, 15, 4),
        64,
        torch.float32,
        ["-1", "1"],
        False,
        1,
        None,
    ),
]
_COMPRESSED_ROWS = [
    (torch.sparse_csr, (6, 6), None, torch.float32, ["-1", "1"], OMITTED, 0, None),
    (torch.sparse_csr, (6, 6), None, torch.float32, ["-1", "1"], True, 0, None),
    (torch.sparse_csr, (6, 6), None, torch.float32, ["-1", "1"], False, 0, None),
    (torch.sparse_csc, (6, 6), None, torch.float32, ["-1", "1"], OMITTED, 0, None),
    (torch.sparse_csc, (6, 6), None, torch.float32, ["-1", "1"], True, 0, None),
    (torch.sparse_csc, (6, 6), None, torch.float32, ["-1", "1"], False, 0, None),
    (torch.sparse_bsr, (6, 6), None, torch.float32, ["-1", "1"], OMITTED, 0, (2, 2)),
    (torch.sparse_bsr, (6, 6), None, torch.float32, ["-1", "1"], True, 0, (2, 2)),
    (torch.sparse_bsr, (6, 6), None, torch.float32, ["-1", "1"], False, 0, (2, 2)),
    (torch.sparse_bsc, (6, 6), None, torch.float32, ["-1", "1"], OMITTED, 0, (2, 2)),
    (torch.sparse_bsc, (6, 6), None, torch.float32, ["-1", "1"], True, 0, (2, 2)),
    (torch.sparse_bsc, (6, 6), None, torch.float32, ["-1", "1"], False, 0, (2, 2)),
]

# Hybrid operands (sparse_dim 2, dense_dim 1) reach every masked_grad branch on
# all four compressed layouts, which was measured for the (4, 4, 3) geometry
# (native results: csr/csc masked values 30, bsr/bsc masked values 48, output
# shape (4, 4, 3), sparse_dim 2, dense_dim 1).
_HYBRID_ROWS = [
    (layout, (4, 4, 3), None, torch.float32, ["-1", "1"], masked_grad, 1, block)
    for layout, block in (
        (torch.sparse_csr, None),
        (torch.sparse_csc, None),
        (torch.sparse_bsr, (2, 2)),
        (torch.sparse_bsc, (2, 2)),
    )
    for masked_grad in MASKED_GRAD_CASES
]

# Non-square, non-trivial block, batched and zero-extent geometry.  Every row
# below was run against the native operator before being added, and every block
# divides its sparse size: a non-dividing block raises dense_to_sparse_bsr:
# tensor sparse size (h, w) must be divisible by given blocksize (bh, bw).  A
# rank-3 compressed operand only reaches the unmasked branch: its masked branch
# raises sparse_coo_to_sparse: conversion from Sparse to SparseCsr for input
# tensors with sparse_dim()!=2 is not supported (measured for CSR and CSC), so
# those two rows pass masked_grad=False.  A batched compressed tensor built
# from a dense mask is rejected earlier with Expect the same number of
# specified elements per batch, which is why _batched_compressed builds the
# equal-NNZ structure explicitly.
_BLOCK_ROWS = [
    (torch.sparse_csr, (8, 9), None, torch.float32, ["-1", "1"], OMITTED, 0, None),
    (torch.sparse_csc, (9, 8), None, torch.float32, ["-1", "1"], False, 0, None),
    (torch.sparse_bsr, (8, 9), None, torch.float32, ["-1", "1"], OMITTED, 0, (2, 3)),
    (torch.sparse_bsr, (12, 8), None, torch.float32, ["-1", "1"], OMITTED, 0, (3, 2)),
    (torch.sparse_bsr, (12, 8), None, torch.float32, ["-1", "1"], False, 0, (2, 4)),
    (torch.sparse_bsc, (8, 9), None, torch.float32, ["-1", "1"], True, 0, (2, 3)),
    (torch.sparse_csr, (0, 4), None, torch.float32, ["-1", "1"], OMITTED, 0, None),
    (torch.sparse_bsc, (4, 0), None, torch.float32, ["-1", "1"], OMITTED, 0, (2, 2)),
    (torch.sparse_csr, (2, 4, 4), None, torch.float32, ["-1", "1"], False, 0, None),
    (torch.sparse_csc, (2, 4, 4), None, torch.float32, ["-1", "1"], False, 0, None),
]

# Every sparse fixture in this file is structurally INT64: torch requires INT64
# COO coordinates, the compressed index and pointer arrays are INT64, and the
# mixed-radix arithmetic that builds the coordinates runs in INT64 because a
# legal domain can push a flat position past the INT32 range (70000 x 70000 with
# 4 entries already needs 3675000000 for its last position).  There is no INT32
# construction to fall back on, so each sparse row below is collected only when
# its payload dtype and INT64 are both supported.  The dense rows keep their own
# operand-dtype gate and stay available without INT64.
STRUCTURAL_INT64 = _avail(torch.int64)

_SPARSE_ROWS = _COO_ROWS + _COMPRESSED_ROWS + _HYBRID_ROWS + _BLOCK_ROWS
SPARSE_CASES = [row for row in _SPARSE_ROWS if STRUCTURAL_INT64 and _avail(row[3])]
QUICK_SPARSE_CASES = (
    [
        (
            torch.sparse_coo,
            (2, 19, 7),
            11,
            torch.float32,
            ["-1", "1"],
            OMITTED,
            0,
            None,
        ),
        (torch.sparse_coo, (2, 19, 7), 11, torch.float32, ["-1", "1"], False, 0, None),
    ]
    if STRUCTURAL_INT64
    else []
)

SPARSE_EMPTY_DOMAIN_SHAPES = [(0, 5), (4, 0), (0,)] if STRUCTURAL_INT64 else []

# A coordinate list holding one duplicated position: the native result is
# coalesced and the duplicated values are merged, which a candidate that keeps
# the input coordinates would fail.
SPARSE_UNCOALESCED_CASES = (
    [
        ((4, 5), [[1, 0, 0], [2, 1, 1]], [2.0, 1.0, 3.0], OMITTED),
        ((4, 5), [[1, 0, 0], [2, 1, 1]], [2.0, 1.0, 3.0], False),
    ]
    if STRUCTURAL_INT64
    else []
)

# FP8 masks: only the unmasked branch succeeds on the active NVIDIA backend for
# this COO form.  Its masked branch was measured to fail with RuntimeError:
# 'mul_cuda' not implemented for 'Float8_e4m3fn', and an FP8 grad fails on both
# branches with 'mul_cuda' / 'nonzero_cuda' not implemented, which is why FP8
# appears here only together with a non-FP8 grad on the unmasked branch.  The
# limitation is scoped to that backend/dtype/form and does not extend to the
# dense path.
_FP8_MASK_ROWS = [
    (torch.float8_e4m3fn, torch.float32, False),
    (torch.float8_e5m2, torch.float32, False),
]
FP8_MASK_CASES = [row for row in _FP8_MASK_ROWS if STRUCTURAL_INT64 and _avail(row[0])]

# (shape, grad dtype, input dtype).  The derivative exists for every supported
# pair because the dense path is linear in grad; its dtype follows grad, which
# the backward test asserts.
_BACKWARD_ROWS = [
    ((1024, 1024), torch.float32, torch.float32),
    ((1024, 1024), torch.float32, torch.float16),
    ((1024, 1024), torch.float32, torch.float8_e4m3fn),
    ((20, 320, 15), torch.float16, torch.float32),
    ((20, 320, 15), torch.bfloat16, torch.bfloat16),
    ((20, 320, 15), torch.bfloat16, torch.float16),
    ((256,), torch.float64, torch.float64),
]
BACKWARD_CASES = [row for row in _BACKWARD_ROWS if _avail(row[1]) and _avail(row[2])]
SPARSE_BACKWARD_CASES = [OMITTED, True, False] if STRUCTURAL_INT64 else []
SPARSE_BACKWARD_SHAPE = (16, 32)
SPARSE_BACKWARD_NNZ = 24


def _count(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


def _cast_grad(shape, dtype):
    """Deterministic grad holding the probed conversion-boundary values."""
    total = _count(shape)
    values = torch.tensor(_CAST_VALUES, dtype=torch.float32, device=flag_gems.device)
    flat = values.repeat(total // values.numel() + 1)[:total]
    return flat.reshape(shape).to(dtype)


def _ramp(dtype, size):
    """Non-uniform deterministic values: a ramp with an alternating sign."""
    ramp = torch.linspace(-1.0, 1.0, size, device=flag_gems.device)
    sign = torch.where(
        torch.arange(size, dtype=torch.int32, device=flag_gems.device) % 2 == 0,
        torch.full((size,), 1.0, device=flag_gems.device),
        torch.full((size,), -0.25, device=flag_gems.device),
    )
    return (ramp * sign).to(dtype)


def _coo_indices(shape, nnz, device):
    """Deterministic sorted coordinates that hit nnz exactly.

    The mixed-radix decomposition of an increasing flat index list keeps the
    coordinates in lexicographic order, so the tensor can be marked coalesced
    truthfully, and the count is exact instead of an overdraw that is truncated
    after deduplication.  nnz equal to zero yields an empty coordinate list, so
    a zero-extent domain never needs an invalid draw.  The structural
    arithmetic therefore runs in INT64: the flat position reaches the total
    element count, which a legal domain can push past the INT32 range (a
    70000 x 70000 domain with 4 entries already needs 3675000000 for its last
    position), and torch requires INT64 COO indices in any case.  Only the
    unrelated bounded auxiliary arrays elsewhere in this file stay INT32.
    """
    total = _count(shape)
    if nnz == 0:
        return torch.empty((len(shape), 0), dtype=torch.int64, device=device)
    step = max(1, total // nnz)
    flat = torch.arange(nnz, dtype=torch.int64, device=device) * step
    coordinates = []
    remainder = flat
    for dim in reversed(shape):
        coordinates.append(remainder % dim)
        remainder = remainder // dim
    return torch.stack(list(reversed(coordinates))).to(torch.int64)


def _masked_values(shape, dtype):
    """Deterministic dense payload whose nonzeros define a sparse structure."""
    flat = torch.arange(_count(shape), dtype=torch.int32, device=flag_gems.device)
    return (flat.reshape(shape) % 5 == 0).to(dtype)


def _batched_compressed(layout, shape, dtype, device):
    """Rank-3 compressed operand whose batches have equal NNZ.

    Converting a batched dense tensor to a compressed layout needs the same
    number of specified elements in every batch, so the structure is described
    explicitly instead: every column of every row is present in every batch.
    """
    batch, rows, cols = shape
    values = torch.linspace(-1.0, 1.0, batch * rows * cols, device=device)
    values = values.reshape(batch, rows * cols).to(dtype)
    if layout == torch.sparse_csr:
        inner = torch.arange(cols, dtype=torch.int64, device=device).repeat(batch, rows)
        outer = torch.arange(0, rows * cols + 1, cols, dtype=torch.int64, device=device)
        return torch.sparse_csr_tensor(
            outer.repeat(batch, 1), inner, values, size=tuple(shape), device=device
        )
    inner = (
        torch.arange(rows, dtype=torch.int64, device=device)
        .repeat(cols)
        .unsqueeze(0)
        .repeat(batch, 1)
    )
    outer = torch.arange(0, rows * cols + 1, rows, dtype=torch.int64, device=device)
    return torch.sparse_csc_tensor(
        outer.repeat(batch, 1), inner, values, size=tuple(shape), device=device
    )


def _sparse_operand(layout, shape, nnz, dtype, dense_dim, block):
    """Sparse input operand, built on the active device."""
    device = flag_gems.device
    if layout == torch.sparse_coo:
        sparse_dim = len(shape) - dense_dim
        indices = _coo_indices(shape[:sparse_dim], nnz, device)
        values = torch.ones(
            (nnz,) + tuple(shape[sparse_dim:]), dtype=dtype, device=device
        )
        return torch.sparse_coo_tensor(
            indices, values, tuple(shape), device=device, is_coalesced=True
        )
    if dense_dim == 0 and len(shape) == 3:
        return _batched_compressed(layout, shape, dtype, device)
    hybrid = _masked_values(shape, dtype).to_sparse(sparse_dim=len(shape) - dense_dim)
    if layout == torch.sparse_csr:
        return hybrid.to_sparse_csr()
    if layout == torch.sparse_csc:
        return hybrid.to_sparse_csc()
    if layout == torch.sparse_bsr:
        return hybrid.to_sparse_bsr(block)
    return hybrid.to_sparse_bsc(block)


def _sparse_parts(tensor):
    """Stored components of a sparse tensor in its own layout.

    A dense-value comparison would not cover the layout contract, and
    re-coalescing both sides would hide NNZ or index-order differences, so the
    components the output is defined by are compared directly.  An uncoalesced
    COO operand still owns its raw components, but the public .indices() /
    .values() accessors refuse to run in that state, so its raw accessors are
    used instead of coercing it.
    """
    if tensor.layout == torch.sparse_coo:
        if tensor.is_coalesced():
            return (tensor.indices(), tensor.values())
        return (tensor._indices(), tensor._values())
    if tensor.layout in (torch.sparse_csr, torch.sparse_bsr):
        return (tensor.crow_indices(), tensor.col_indices(), tensor.values())
    return (tensor.ccol_indices(), tensor.row_indices(), tensor.values())


def _sparse_meta(tensor):
    """Layout, shape and dimensionality, plus coalescing for COO."""
    meta = (
        tensor.layout,
        tuple(tensor.shape),
        tensor.sparse_dim(),
        tensor.dense_dim(),
    )
    if tensor.layout == torch.sparse_coo:
        meta = meta + (tensor.is_coalesced(),)
    return meta


def _snapshot_sparse(tensor):
    """Metadata and stored components of a sparse operand before the call."""
    return (_sparse_meta(tensor), [part.clone() for part in _sparse_parts(tensor)])


def _assert_sparse_unchanged(tensor, snapshot):
    """The candidate must leave its sparse operand exactly as it found it."""
    meta, parts = snapshot
    assert _sparse_meta(tensor) == meta
    for part, before in zip(_sparse_parts(tensor), parts):
        tu.assert_result_equal(part, before)


def _assert_sparse_result(res_out, ref_out, layout, shape, dtype):
    assert res_out.layout == ref_out.layout == layout
    assert res_out.shape == ref_out.shape == shape
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.sparse_dim() == ref_out.sparse_dim()
    assert res_out.dense_dim() == ref_out.dense_dim()
    if layout == torch.sparse_coo:
        assert res_out.is_coalesced() == ref_out.is_coalesced()
    for got, want in zip(_sparse_parts(res_out), _sparse_parts(ref_out)):
        tu.assert_result_equal(got, want)


def _backward_upstream(shape, nnz, masked_grad):
    """Upstream gradient matching the structure of the forward result.

    The masked branches keep only the input-mask positions, so their upstream
    carries those coordinates.  The unmasked branch keeps every grad entry, so
    its upstream also carries coordinates outside the input mask: a candidate
    that wrongly masks that branch then scatters only the mask coordinates and
    fails instead of quietly returning the same gradient.
    """
    device = flag_gems.device
    count = _count(shape) if masked_grad is False else nnz
    return torch.sparse_coo_tensor(
        _coo_indices(shape, count, device),
        _ramp(torch.float32, count),
        shape,
        device=device,
        is_coalesced=True,
    )


@pytest.mark.to_dense_backward
@pytest.mark.parametrize("dtype", DENSE_DTYPES)
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_to_dense_backward_dense_layout(shape, value_range, dtype):
    """Strided path: the result is grad in the scalar type of input.

    Both operands share the dtype here, so the operator is a value-preserving
    copy and the comparison is exact.
    """
    grad = tu.make_input(dtype, shape, value_range)
    inp = tu.make_input(dtype, shape, value_range)
    ref_grad = tu.to_reference(grad)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_dense_backward(ref_grad, ref_inp)
    res_out = flag_gems.to_dense_backward(grad, inp)

    assert res_out.dtype == dtype
    assert res_out.shape == grad.shape
    assert res_out.layout == torch.strided
    assert res_out.device == grad.device
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(grad, ref_grad)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_dense_backward
@pytest.mark.parametrize("dtype", DENSE_EXTRA_DTYPES)
@pytest.mark.parametrize("shape", tu.selected_cases(tu.REQUIRED_SHAPES, quick=[]))
def test_to_dense_backward_dense_layout_extra_dtypes(shape, dtype):
    """Dense path for supported dtypes outside the required nine."""
    grad = tu.make_input(dtype, shape, ["-1", "1"])
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_grad = tu.to_reference(grad)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_dense_backward(ref_grad, ref_inp)
    res_out = flag_gems.to_dense_backward(grad, inp)

    assert res_out.dtype == dtype
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(grad, ref_grad)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_dense_backward
@pytest.mark.parametrize(
    "grad_dtype,input_dtype,shape",
    tu.selected_cases(CROSS_DTYPE_CASES, quick=QUICK_CROSS_DTYPE_CASES),
)
def test_to_dense_backward_cross_dtype_cast(grad_dtype, input_dtype, shape):
    """Differing dtypes: the result is grad converted to the input dtype."""
    grad = tu.make_input(grad_dtype, shape, ["-1", "1"])
    inp = tu.make_input(input_dtype, shape, ["-1", "1"])
    ref_grad = tu.to_reference(grad)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_dense_backward(ref_grad, ref_inp)
    res_out = flag_gems.to_dense_backward(grad, inp)

    assert res_out.dtype == input_dtype
    assert res_out.shape == grad.shape
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(grad, ref_grad)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_dense_backward
@pytest.mark.parametrize(
    "shape,grad_dtype,input_dtype",
    tu.selected_cases(CAST_CASES, quick=QUICK_CAST_CASES),
)
def test_to_dense_backward_cast_boundaries(shape, grad_dtype, input_dtype):
    """Sign, magnitude and truncation boundaries of the conversion."""
    grad = _cast_grad(shape, grad_dtype)
    inp = tu.make_input(input_dtype, shape, ["-1", "1"])
    ref_grad = tu.to_reference(grad)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_dense_backward(ref_grad, ref_inp)
    res_out = flag_gems.to_dense_backward(grad, inp)

    assert res_out.dtype == input_dtype
    assert res_out.shape == grad.shape
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(grad, ref_grad)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_dense_backward
@pytest.mark.parametrize(
    "base_shape,size,stride,offset",
    tu.selected_cases(GRAD_LAYOUT_CASES, quick=[]),
)
def test_to_dense_backward_grad_view(base_shape, size, stride, offset):
    """Strided, offset and expanded grads are copied by stored value.

    A zero stride entry is an expanded view and a nonzero offset starts the
    window inside the base storage; both are valid operands for the native
    operator, which reads only the values of the view.
    """
    base = tu.make_input(torch.float32, base_shape, ["-1", "1"])
    base_before = base.clone()
    grad = base.as_strided(size, stride, offset)
    grad_before = grad.clone()
    inp = tu.make_input(torch.float32, base_shape, ["-1", "1"])
    ref_grad = tu.to_reference(grad)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_dense_backward(ref_grad, ref_inp)
    res_out = flag_gems.to_dense_backward(grad, inp)

    assert res_out.shape == tuple(size)
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(grad, grad_before)
    # The view shares storage with base, so the whole parent is compared as
    # well: a candidate writing outside the view's window would otherwise go
    # unnoticed.
    tu.assert_result_equal(base, base_before)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_dense_backward
@pytest.mark.parametrize(
    "grad_shape,input_shape,non_contiguous",
    tu.selected_cases(INPUT_METADATA_CASES, quick=[]),
)
def test_to_dense_backward_ignores_input_metadata(
    grad_shape, input_shape, non_contiguous
):
    """The dense path reads only the scalar type and layout of the input.

    The input is filled from a different range and given an unrelated shape and,
    in the last row, a non-contiguous view; a candidate that combines the
    operands elementwise, or that requires matching shapes, cannot reproduce the
    native result.
    """
    grad = tu.make_input(torch.float32, grad_shape, ["-1", "1"])
    base = tu.make_input(torch.float32, input_shape, ["0", "1"])
    base_before = base.clone()
    inp = base[:, ::2] if non_contiguous else base
    ref_grad = tu.to_reference(grad)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_dense_backward(ref_grad, ref_inp)
    res_out = flag_gems.to_dense_backward(grad, inp)

    assert res_out.shape == grad.shape
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(grad, ref_grad)
    tu.assert_result_equal(inp, ref_inp)
    # The non-contiguous row is a slice of base, so the part of the parent
    # storage outside the slice is checked too.
    tu.assert_result_equal(base, base_before)


@pytest.mark.to_dense_backward
@pytest.mark.parametrize(
    "masked_grad",
    tu.selected_cases(MASKED_GRAD_CASES, quick=QUICK_MASKED_GRAD_CASES),
)
@pytest.mark.parametrize(
    "shape,dtype",
    tu.selected_cases(MASKED_GRAD_DENSE_CASES, quick=QUICK_MASKED_GRAD_DENSE_CASES),
)
def test_to_dense_backward_masked_grad_dense_input(masked_grad, shape, dtype):
    """masked_grad (default, None, True, False) changes no dense result."""
    grad = tu.make_input(dtype, shape, ["-1", "1"])
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_grad = tu.to_reference(grad)
    ref_inp = tu.to_reference(inp)
    extra = () if masked_grad == OMITTED else (masked_grad,)

    ref_out = torch.ops.aten.to_dense_backward(ref_grad, ref_inp, *extra)
    res_out = flag_gems.to_dense_backward(grad, inp, *extra)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(grad, ref_grad)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_dense_backward
@pytest.mark.parametrize(
    "shape,dtype,masked_grad",
    tu.selected_cases(_MASKED_GRAD_INT_ROWS, quick=QUICK_MASKED_GRAD_INT_CASES),
)
def test_to_dense_backward_masked_grad_integer(shape, dtype, masked_grad):
    """Optional[bool] accepts integers, where 0 is the only false value."""
    grad = tu.make_input(dtype, shape, ["-1", "1"])
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_grad = tu.to_reference(grad)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_dense_backward(ref_grad, ref_inp, masked_grad)
    res_out = flag_gems.to_dense_backward(grad, inp, masked_grad)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(grad, ref_grad)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_dense_backward
@pytest.mark.parametrize(
    "layout,shape,nnz,dtype,value_range,masked_grad,dense_dim,block",
    tu.selected_cases(SPARSE_CASES, quick=QUICK_SPARSE_CASES),
)
def test_to_dense_backward_sparse_input(
    layout, shape, nnz, dtype, value_range, masked_grad, dense_dim, block
):
    """Sparse input in each supported layout: the result follows its layout.

    The masked branch (default, None, True) selects grad at the mask positions,
    the unmasked branch converts every grad element to sparse.  The operand is
    snapshotted as a whole (metadata plus stored components), so a candidate
    that rewrites or re-coalesces its input is caught.
    """
    grad = tu.make_input(dtype, shape, value_range)
    inp = _sparse_operand(layout, shape, nnz, dtype, dense_dim, block)
    snapshot = _snapshot_sparse(inp)
    ref_grad = tu.to_reference(grad)
    ref_inp = tu.to_reference(inp)
    extra = () if masked_grad == OMITTED else (masked_grad,)

    ref_out = torch.ops.aten.to_dense_backward(ref_grad, ref_inp, *extra)
    res_out = flag_gems.to_dense_backward(grad, inp, *extra)

    assert res_out.device == grad.device
    _assert_sparse_result(res_out, ref_out, layout, shape, dtype)
    _assert_sparse_unchanged(inp, snapshot)
    tu.assert_result_equal(grad, ref_grad)


@pytest.mark.to_dense_backward
@pytest.mark.parametrize("shape", SPARSE_EMPTY_DOMAIN_SHAPES)
@pytest.mark.parametrize("masked_grad", tu.selected_cases([OMITTED, False], quick=[]))
def test_to_dense_backward_sparse_empty_domain(shape, masked_grad):
    """Zero-extent domains: an empty coordinate list, no invalid draw."""
    inp = torch.sparse_coo_tensor(
        torch.empty((len(shape), 0), dtype=torch.int64, device=flag_gems.device),
        torch.empty(0, device=flag_gems.device),
        tuple(shape),
        device=flag_gems.device,
        is_coalesced=True,
    )
    grad = tu.make_input(torch.float32, shape, ["-1", "1"])
    snapshot = _snapshot_sparse(inp)
    ref_grad = tu.to_reference(grad)
    ref_inp = tu.to_reference(inp)
    extra = () if masked_grad == OMITTED else (masked_grad,)

    ref_out = torch.ops.aten.to_dense_backward(ref_grad, ref_inp, *extra)
    res_out = flag_gems.to_dense_backward(grad, inp, *extra)

    _assert_sparse_result(res_out, ref_out, torch.sparse_coo, shape, torch.float32)
    assert res_out._nnz() == 0
    _assert_sparse_unchanged(inp, snapshot)
    tu.assert_result_equal(grad, ref_grad)


@pytest.mark.to_dense_backward
@pytest.mark.parametrize(
    "shape,indices,values,masked_grad",
    tu.selected_cases(SPARSE_UNCOALESCED_CASES, quick=[]),
)
def test_to_dense_backward_sparse_uncoalesced(shape, indices, values, masked_grad):
    """An input with duplicate coordinates is merged into the native result.

    The operand is not coalesced, so its components are read through the raw
    accessors and the result is compared without re-coalescing either side.
    """
    inp = torch.sparse_coo_tensor(
        torch.tensor(indices, dtype=torch.int64, device=flag_gems.device),
        torch.tensor(values, dtype=torch.float32, device=flag_gems.device),
        shape,
        device=flag_gems.device,
    )
    grad = tu.make_input(torch.float32, shape, ["-1", "1"])
    snapshot = _snapshot_sparse(inp)
    ref_grad = tu.to_reference(grad)
    ref_inp = tu.to_reference(inp)
    extra = () if masked_grad == OMITTED else (masked_grad,)

    ref_out = torch.ops.aten.to_dense_backward(ref_grad, ref_inp, *extra)
    res_out = flag_gems.to_dense_backward(grad, inp, *extra)

    _assert_sparse_result(res_out, ref_out, torch.sparse_coo, shape, torch.float32)
    _assert_sparse_unchanged(inp, snapshot)
    tu.assert_result_equal(grad, ref_grad)


@pytest.mark.to_dense_backward
@pytest.mark.parametrize(
    "mask_dtype,grad_dtype,masked_grad", tu.selected_cases(FP8_MASK_CASES, quick=[])
)
def test_to_dense_backward_fp8_mask_unmasked(mask_dtype, grad_dtype, masked_grad):
    """Unmasked branch with an FP8 mask: the result is grad.to_sparse().

    Only this branch is available for FP8 on the active NVIDIA backend in this
    COO form; the masked branch and an FP8 grad were measured to raise mul_cuda
    / nonzero_cuda not implemented for the FP8 type, so they are not exercised
    here while the FP8 dtype stays covered by the dense grid.
    """
    shape = (4, 4)
    nnz = 4
    inp = torch.sparse_coo_tensor(
        _coo_indices(shape, nnz, flag_gems.device),
        torch.ones(nnz, dtype=mask_dtype, device=flag_gems.device),
        shape,
        device=flag_gems.device,
        is_coalesced=True,
    )
    grad = tu.make_input(grad_dtype, shape, ["-1", "1"])
    snapshot = _snapshot_sparse(inp)
    ref_grad = tu.to_reference(grad)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_dense_backward(ref_grad, ref_inp, masked_grad)
    res_out = flag_gems.to_dense_backward(grad, inp, masked_grad)

    assert res_out.device == grad.device
    _assert_sparse_result(res_out, ref_out, torch.sparse_coo, shape, grad_dtype)
    _assert_sparse_unchanged(inp, snapshot)
    tu.assert_result_equal(grad, ref_grad)


@pytest.mark.to_dense_backward
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(SPECIAL_VALUE_CASES, quick=[])
)
def test_to_dense_backward_special_values(dtype, scenario):
    """nan, inf and mixed operands are propagated exactly on the dense path."""
    grad = tu.make_special_input(dtype, scenario)
    inp = tu.make_special_input(dtype, scenario)
    ref_grad = tu.to_reference(grad)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_dense_backward(ref_grad, ref_inp)
    res_out = flag_gems.to_dense_backward(grad, inp)

    assert res_out.dtype == dtype
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(grad, ref_grad)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_dense_backward
@pytest.mark.parametrize(
    "shape,grad_dtype,input_dtype", tu.selected_cases(BACKWARD_CASES, quick=[])
)
def test_to_dense_backward_backward(shape, grad_dtype, input_dtype):
    """The operator is linear in grad and the derivative carries the grad dtype.

    The forward result is compared as well, because the derivative alone does
    not describe the forward mapping.  The upstream gradient is built in the
    input dtype, so a wider grad keeps its own precision: an fp16 input with an
    fp32 grad produces an fp32 derivative.
    """
    grad = tu.make_input(grad_dtype, shape, ["-1", "1"]).requires_grad_(True)
    inp = tu.make_input(input_dtype, shape, ["-1", "1"])
    upstream = _ramp(input_dtype, grad.numel()).reshape(shape)
    ref_grad = tu.to_reference(grad)
    ref_inp = tu.to_reference(inp)
    ref_upstream = tu.to_reference(upstream)

    ref_out = torch.ops.aten.to_dense_backward(ref_grad, ref_inp)
    res_out = flag_gems.to_dense_backward(grad, inp)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(grad, ref_grad)
    tu.assert_result_equal(inp, ref_inp)

    ref_grad_in = torch.autograd.grad(ref_out, ref_grad, grad_outputs=ref_upstream)[0]
    res_grad_in = torch.autograd.grad(res_out, grad, grad_outputs=upstream)[0]

    assert res_grad_in.dtype == ref_grad_in.dtype == grad_dtype
    tu.assert_result_equal(res_grad_in, ref_grad_in)


@pytest.mark.to_dense_backward
@pytest.mark.parametrize(
    "masked_grad", tu.selected_cases(SPARSE_BACKWARD_CASES, quick=[])
)
def test_to_dense_backward_backward_sparse_input(masked_grad):
    """Sparse input: the derivative is an exact scatter into the mask positions.

    Both the forward structure and the derivative are compared, and both
    operands are checked for being left untouched, because the derivative alone
    does not describe the forward mapping.  The derivative of this
    non-reducing linear map is the upstream scattered at the output
    coordinates, so it is compared exactly rather than with a tolerance.
    """
    shape = SPARSE_BACKWARD_SHAPE
    nnz = SPARSE_BACKWARD_NNZ
    inp = _sparse_operand(torch.sparse_coo, shape, nnz, torch.float32, 0, None)
    grad = tu.make_input(torch.float32, shape, ["-1", "1"]).requires_grad_(True)
    upstream = _backward_upstream(shape, nnz, masked_grad)
    snapshot = _snapshot_sparse(inp)
    ref_grad = tu.to_reference(grad)
    ref_inp = tu.to_reference(inp)
    ref_upstream = tu.to_reference(upstream)
    extra = () if masked_grad == OMITTED else (masked_grad,)

    ref_out = torch.ops.aten.to_dense_backward(ref_grad, ref_inp, *extra)
    res_out = flag_gems.to_dense_backward(grad, inp, *extra)

    _assert_sparse_result(res_out, ref_out, torch.sparse_coo, shape, torch.float32)
    _assert_sparse_unchanged(inp, snapshot)
    tu.assert_result_equal(grad, ref_grad)

    ref_grad_in = torch.autograd.grad(ref_out, ref_grad, grad_outputs=ref_upstream)[0]
    res_grad_in = torch.autograd.grad(res_out, grad, grad_outputs=upstream)[0]

    assert res_grad_in.dtype == ref_grad_in.dtype == torch.float32
    tu.assert_result_equal(res_grad_in, ref_grad_in)


@pytest.mark.to_dense_backward
@pytest.mark.parametrize(
    "grad",
    [
        pytest.param([1.0, 2.0], id="list"),
        pytest.param(3.0, id="float"),
    ],
)
def test_to_dense_backward_rejects_non_tensor_grad(grad):
    """The candidate validates its schema instead of accepting a Python value."""
    inp = tu.make_input(torch.float32, (2, 3), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.to_dense_backward(grad, inp)


@pytest.mark.to_dense_backward
@pytest.mark.parametrize(
    "masked_grad",
    [
        pytest.param("not-a-bool", id="str"),
        pytest.param([True], id="list"),
    ],
)
def test_to_dense_backward_rejects_invalid_masked_grad(masked_grad):
    """masked_grad is Optional[bool]; other values are rejected."""
    grad = tu.make_input(torch.float32, (2, 3), ["-1", "1"])
    inp = tu.make_input(torch.float32, (2, 3), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.to_dense_backward(grad, inp, masked_grad)
