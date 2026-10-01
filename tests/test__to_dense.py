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

"""Correctness tests for ``aten::_to_dense`` (sparse input -> dense strided).

Distinct stored coordinates only scatter, so those rows compare with the shared
exact assertion. Uncoalesced duplicate coordinates make densification add the
colliding stored values in an order the kernel does not fix, so they use the
shared arithmetic comparison. Every sparse input is also checked for mutation
against its own pre-call reference copy, and every output/gradient for its device.
"""

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

_DEVICE = flag_gems.device

# Sparse COO stores its coordinates in an int64 tensor, so a COO fixture needs the
# int64 element capability even when the *values* are small; the compressed
# layouts (CSR/CSC/BSR/BSC) carry int32 metadata and are not gated by this.
_COO_SUPPORTED = utils.int64_is_supported


def _coo_cases(cases, *, quick=()):
    """COO parameter rows; empty when coordinates cannot be stored as int64."""
    if not _COO_SUPPORTED:
        return []
    return tu.selected_cases(cases, quick=quick)


_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.float32,
    torch.float16,
    torch.complex64,
]
if utils.bf16_is_supported:
    _DTYPES.append(torch.bfloat16)
if utils.int64_is_supported:
    _DTYPES.append(torch.int64)
if utils.fp64_is_supported:
    _DTYPES.append(torch.float64)
_DTYPES.append(torch.bool)

_FLOAT_DTYPES = [dtype for dtype in _DTYPES if dtype.is_floating_point]

# The small-integer duplicate fixture is exact for the integer dtypes and for
# complex64, whose real and imaginary parts are those same small integers; the
# name describes the exactness property rather than an integral element type.
_EXACT_SUM_DTYPES = [
    dtype for dtype in _DTYPES if not dtype.is_floating_point and dtype != torch.bool
]

# Duplicate coordinates are summed in an order the native kernel does not fix, so
# repeated native calls on the very same input can already disagree. Full-range
# fp16/bf16 duplicate accumulation therefore stays *pending*: those repeated native
# results can fail the shared arithmetic tolerance. Only fp32 and fp64 run the
# full-range duplicate rows, through ``tu.assert_result_close`` - a tolerance check,
# not bit-exactness. The fp16/bf16 duplicate coverage that is executed is the exact
# cancellation family below (four stored terms whose partial sums stay exact
# integers in any order); it is not small-integer coverage and not a substitute for
# the pending full-range rows.
_PENDING_DUPLICATE_DTYPES = [torch.float16, torch.bfloat16]
_ACCUMULATING_DTYPES = [
    dtype for dtype in _FLOAT_DTYPES if dtype not in _PENDING_DUPLICATE_DTYPES
]

# The four stored terms of one cell are +m, +m, -m, -m with m a power of two up to
# ``_CANCELLATION_TERM``: every partial sum is a small exact integer, so no
# accumulation order or rounding can change the result and the shared exact
# comparator applies.
_CANCELLATION_TERM = 8

# The rank-0 shape has its own fixture (COO indices of shape (0, nnz)), so the
# value grid uses the non-empty spec shapes.
_FLOAT_VALUE_SHAPES = (
    [shape for shape in tu.selected_shapes() if shape] if _COO_SUPPORTED else []
)

_DUPLICATE_SHAPES = [
    (256,),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]
_DUPLICATE_CASES = _coo_cases(
    [(shape, parity) for shape in _DUPLICATE_SHAPES for parity in (1, 4)],
    quick=[((256,), 1), ((256,), 4)],
)
_EXACT_CASES = _coo_cases(
    [(shape, parity) for shape in _DUPLICATE_SHAPES for parity in (2, 3)],
    quick=[((256,), 2), ((256,), 3)],
)
_CANCELLATION_CASES = _coo_cases(
    [
        (shape, dtype)
        for shape in [(256,), (1024, 1024), (20, 320, 15)]
        for dtype in [torch.float32, torch.float16]
        + ([torch.bfloat16] if utils.bf16_is_supported else [])
    ]
    + ([((1024, 1024), torch.float64)] if utils.fp64_is_supported else []),
    quick=[
        ((256,), dtype)
        for dtype in [torch.float32, torch.float16]
        + ([torch.bfloat16] if utils.bf16_is_supported else [])
    ],
)
_HYBRID_DTYPES = [torch.float32, torch.float16]
_HYBRID_CASES = (
    [
        ((8, 16, 32), 2, False),
        ((4, 8, 16, 32), 2, False),
        ((16, 7, 57), 1, False),
        ((2, 32, 32), 1, False),
        ((16, 7, 57), 1, True),
        ((8, 16, 32), 2, True),
    ]
    if _COO_SUPPORTED
    else []
)
_HYBRID_DUPLICATE_SHAPES = (
    [((8, 16), 1), ((16, 7, 57), 1), ((8, 16, 32), 2)] if _COO_SUPPORTED else []
)
_COMPRESSED_CASES = tu.selected_cases(
    [
        ("csr", (64, 64), 0.05, False),
        ("csr", (64, 64), 0.6, False),
        ("csc", (64, 64), 0.05, False),
        ("csc", (64, 64), 0.6, False),
        ("csr", (1024, 1024), 0.05, False),
        ("csr", (1024, 1024), 0.6, False),
        ("csc", (1024, 1024), 0.05, False),
        ("csc", (1024, 1024), 0.6, False),
        ("csr", (20, 320), 0.05, False),
        ("csr", (20, 320), 0.6, False),
        ("csc", (20, 320), 0.05, False),
        ("csc", (20, 320), 0.6, False),
        ("csr", (2, 32, 32), 0.3, True),
        ("csc", (2, 32, 32), 0.3, True),
        ("csr", (2, 32, 32), 0.3, False),
        ("csc", (2, 32, 32), 0.3, False),
    ],
    quick=[
        (layout, (64, 64), density, False)
        for layout in ("csr", "csc")
        for density in (0.05, 0.6)
    ]
    + [
        (layout, (2, 32, 32), 0.3, shared)
        for layout in ("csr", "csc")
        for shared in (False, True)
    ],
)
_COMPRESSED_DTYPES = _DTYPES
_BLOCK_CASES = tu.selected_cases(
    [
        ("bsr", (32, 32), 1, True),
        ("bsr", (32, 32), 2, True),
        ("bsr", (32, 32), 4, True),
        ("bsr", (64, 64), 4, True),
        ("bsr", (32, 32), 4, False),
        ("bsc", (32, 32), 1, True),
        ("bsc", (32, 32), 2, True),
        ("bsc", (32, 32), 4, True),
        ("bsc", (64, 64), 4, True),
        ("bsc", (32, 32), 4, False),
    ],
    quick=[
        (layout, (32, 32), block, True)
        for layout in ("bsr", "bsc")
        for block in (1, 2, 4)
    ]
    + [(layout, (32, 32), 4, False) for layout in ("bsr", "bsc")],
)
# The ``out`` overload keeps one quick row as the designated smoke case.
_OUT_CASES = _coo_cases(
    [((256, 256), torch.float32), ((20, 320), torch.float16)],
    quick=[((256, 256), torch.float32)],
)
_MASKED_GRAD_CASES = [None, True, False] if _COO_SUPPORTED else []
_MASKED_GRAD_DTYPES = [torch.float32, torch.float16]
_BACKWARD_SHAPES = [((64, 128), 2), ((64, 8, 16), 1)]
# Distinct coordinates densify as a plain scatter: forward and stored-value
# gradient are bitwise exact for every supported floating dtype.
_BACKWARD_UNIQUE_CASES = _coo_cases(
    [
        (shape, sparse_dim, False, dtype)
        for shape, sparse_dim in _BACKWARD_SHAPES
        for dtype in _FLOAT_DTYPES
    ]
)
# Repeated coordinates add the colliding stored values in the forward, so those
# rows keep the duplicate dimension on the dtypes whose duplicate sum is
# reproducible (fp32, fp64) and use the shared arithmetic comparison; the
# gradient itself is a single gather and is still compared exactly.
_BACKWARD_DUPLICATE_ROWS = _coo_cases(
    [
        (shape, sparse_dim, True, dtype)
        for shape, sparse_dim in _BACKWARD_SHAPES
        for dtype in _ACCUMULATING_DTYPES
    ]
)
_BACKWARD_CASES = _BACKWARD_UNIQUE_CASES + _BACKWARD_DUPLICATE_ROWS
_SPARSE_GRAD_CASES = _coo_cases([((16, 8), 2), ((16, 8, 4), 1)])
# Non-degenerate shapes only: every special-value payload is strictly smaller
# than the shape extent, so each stored value keeps its own coordinate and the
# exact comparison is not silently turned into an accumulation.
_SPECIAL_SHAPES = [
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
    (256,),
]
# One row per (dtype, scenario) pair of the shared special-value matrix, spread
# over the spec shapes.
_SPECIAL_ROWS = _coo_cases(
    [
        (dtype, scenario, _SPECIAL_SHAPES[index % len(_SPECIAL_SHAPES)])
        for index, (dtype, scenario) in enumerate(tu.special_value_cases(_FLOAT_DTYPES))
    ]
)
# Duplicate-coordinate variant: the stored special values are added.
_UNSAFE_SPECIAL_ROWS = _coo_cases(
    [(dtype, scenario) for dtype, scenario in tu.special_value_cases(_FLOAT_DTYPES)]
)
_STORED_ZERO_SHAPES = [(64, 64), (256,), (20, 320)] if _COO_SUPPORTED else []
_STORED_ZERO_DTYPES = [torch.float32, torch.float16, torch.int32]
_STRIDED_VALUE_CASES = _coo_cases(
    [
        ((256,), torch.float32),
        ((1024, 1024), torch.float32),
        ((20, 320, 15), torch.float16),
    ],
    quick=[((256,), torch.float32)],
)
_COALESCED_CASES = _coo_cases(
    [
        ((256,), torch.float32),
        ((1024, 1024), torch.float32),
        ((20, 320, 15), torch.float16),
    ],
    quick=[((256,), torch.float32)],
)
_NO_STORED_SHAPES = _coo_cases(
    [(256,), (1024, 1024), (16, 128, 64, 60)], quick=[(256,)]
)
_NO_STORED_DTYPES = [torch.float32, torch.int32]
_RANK0_CASES = [0, 1, 5] if _COO_SUPPORTED else []
_RANK0_DTYPES = [torch.float32, torch.int32]
_DTYPE_ARG_LAYOUTS = ["csr", "csc"] + (["coo"] if _COO_SUPPORTED else [])
# COO fixtures for the two argument-type negatives.
_COO_ARG_DTYPES = [torch.float32] if _COO_SUPPORTED else []
_FP8_NEGATIVE_DTYPES = (
    [torch.float8_e4m3fn, torch.float8_e5m2]
    if (_COO_SUPPORTED and utils.fp8_is_supported and flag_gems.vendor_name == "nvidia")
    else []
)


def _generator(seed):
    """Deterministic host generator; data is moved to the target device."""
    return torch.Generator().manual_seed(seed)


def _duplicate_seed(shape, parity):
    return 9173 * parity + len(shape) * 101 + sum(shape) % 997


def _extent(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


def _distinct_flat(count, extent, seed):
    """``count`` distinct flat indices inside ``[0, extent)``.

    ``offset + step * i`` is injective modulo ``extent`` once ``step`` is coprime
    with it, which keeps the coordinates distinct without materialising a
    permutation of the (tens of millions of element) logical shape.
    """
    if extent <= 1:
        return torch.zeros(count, dtype=torch.int64)
    generator = _generator(seed)
    step = int(torch.randint(1, extent, (1,), generator=generator).item())
    while math.gcd(step, extent) != 1:
        step += 1
    offset = int(torch.randint(0, extent, (1,), generator=generator).item())
    return (offset + step * torch.arange(count, dtype=torch.int64)) % extent


def _make_values(dtype, count, value_range, seed):
    """Stored values for ``count`` entries.

    Ordered real dtypes come from the shared range helper. ``bool`` has no ordered
    range, so both of its two values are drawn; ``complex64`` is composed from two
    real spans of the requested range.
    """
    if dtype == torch.bool:
        return (
            torch.randint(0, 2, (count,), generator=_generator(seed))
            .to(torch.bool)
            .to(_DEVICE)
        )
    if dtype == torch.complex64:
        real = tu.make_input(torch.float32, (count,), value_range)
        imag = tu.make_input(torch.float32, (count,), value_range)
        return torch.complex(real, imag).to(_DEVICE)
    return tu.make_input(dtype, (count,), value_range)


def _coo(
    shape, nnz, dtype, value_range, seed, unsafe_layout=False, pool=None, values=None
):
    """COO input with ``nnz`` stored entries.

    ``unsafe_layout`` draws coordinates from a small pool on purpose, so several
    stored entries share a cell and densification has to add them.
    """
    extent = _extent(shape)
    count = max(1, min(nnz, extent))
    if unsafe_layout:
        pool_size = pool if pool is not None else min(count, extent)
        flat = torch.randint(0, max(1, pool_size), (count,), generator=_generator(seed))
    else:
        flat = _distinct_flat(count, extent, seed)
    indices = torch.stack(torch.unravel_index(flat, tuple(shape))).to(_DEVICE)
    if values is None:
        values = _make_values(dtype, count, value_range, seed)
    return torch.sparse_coo_tensor(indices, values, tuple(shape), device=_DEVICE)


def _small_integer_values(count, dtype, seed):
    """Integer values of magnitude <= 3.

    The payload is drawn in int64 and only then converted: ``torch.randint`` has
    no draw path for a non-integral dtype such as complex64, so the requested
    dtype is applied to an already drawn integer payload.
    """
    low = 0 if dtype == torch.uint8 else -3
    payload = torch.randint(low, 4, (count,), generator=_generator(seed))
    return payload.to(dtype).to(_DEVICE)


def _rank0_coo(nnz, dtype, value_range):
    """Rank-0 sparse tensor: indices have shape (0, nnz) by convention."""
    values = tu.make_input(dtype, (nnz,), value_range)
    indices = torch.zeros((0, nnz), dtype=torch.int64, device=_DEVICE)
    return torch.sparse_coo_tensor(indices, values, (), device=_DEVICE)


def _hybrid_values(base, dense_shape, broadcast):
    """Stored values of the dense tail, keeping ``base``'s (possibly 0-) strides."""
    values = base.expand((base.shape[0],) + tuple(dense_shape))
    return values if broadcast else values.contiguous()


def _hybrid(
    shape, sparse_dim, dtype, value_range, seed, duplicate=False, broadcast=False
):
    """Hybrid COO plus the index tensor and the parent of its stored values.

    ``broadcast`` leaves the stored values as a stride-0 expansion along the dense
    tail instead of materialising them; both geometries are reproduced for the
    reference from its own copy of the parent.
    """
    sparse_extent = _extent(shape[:sparse_dim])
    count = max(1, sparse_extent // 2)
    flat = _distinct_flat(count, sparse_extent, seed)
    if duplicate:
        flat = flat % max(1, count // 4)
    indices = torch.stack(torch.unravel_index(flat, tuple(shape[:sparse_dim]))).to(
        _DEVICE
    )
    dense_shape = tuple(shape[sparse_dim:])
    base = tu.make_input(dtype, (count,) + (1,) * len(dense_shape), value_range)
    inp = torch.sparse_coo_tensor(
        indices,
        _hybrid_values(base, dense_shape, broadcast),
        tuple(shape),
        device=_DEVICE,
    )
    return inp, indices, base


def _cancellation_coo(shape, dtype, seed=0):
    """Four stored entries per cell summing to exactly zero in any order."""
    extent = _extent(shape)
    cells = min(extent, 64)
    flat = _distinct_flat(cells, extent, seed)
    generator = _generator(seed)
    exponent = int(math.log2(_CANCELLATION_TERM)) + 1
    magnitude = 2 ** torch.randint(0, exponent, (cells,), generator=generator)
    base = torch.stack(torch.unravel_index(flat, tuple(shape))).to(_DEVICE)
    indices = base.repeat_interleave(4, dim=1)
    values = (
        torch.stack([magnitude, magnitude, -magnitude, -magnitude], dim=1)
        .reshape(-1)
        .to(_DEVICE)
        .to(dtype)
    )
    return torch.sparse_coo_tensor(indices, values, tuple(shape), device=_DEVICE)


def _compressed(shape, density, dtype, seed, csc, share_structure=False):
    """CSR/CSC input with the same stored-entry count on every compressed line.

    Each line stores exactly ``k = round(density * inner)`` entries, so batched
    items have identical stored counts by construction rather than by luck; one
    entry per line is deliberately stored as zero and the rest are
    floating values use ``0.25 + 0.75 * u`` with a random sign. Integer, boolean
    and complex payloads use the shared fixture to retain nonzero values and
    imaginary components.
    """
    rows, cols = shape[-2], shape[-1]
    batch = tuple(shape[:-2])
    n_batch = _extent(batch) if batch else 1
    lines, inner = (cols, rows) if csc else (rows, cols)
    k = max(1, min(inner, int(round(density * inner))))
    generator = _generator(seed)

    def pattern():
        return torch.sort(torch.randperm(inner, generator=generator)[:k]).values

    shared = [pattern() for _ in range(lines)] if share_structure else None
    index = torch.stack(
        [
            shared[line] if shared is not None else pattern()
            for _ in range(n_batch)
            for line in range(lines)
        ]
    )
    count = n_batch * lines * k
    magnitude = 0.25 + 0.75 * torch.rand((count,), generator=generator)
    sign = torch.where(torch.rand((count,), generator=generator) < 0.5, -1.0, 1.0)
    values = (magnitude * sign).to(dtype)
    if not dtype.is_floating_point:
        values = _make_values(dtype, count, ("-1", "1"), seed).cpu()
    values[::k] = 0
    indptr = torch.arange(0, (lines + 1) * k, k, dtype=torch.int32)
    if n_batch > 1:
        indptr = indptr.unsqueeze(0).repeat(n_batch, 1).contiguous()
        index = index.reshape(n_batch, lines * k)
        values = values.reshape(n_batch, lines * k)
    else:
        indptr = indptr.contiguous()
        index = index.reshape(lines * k)
    # Compressed metadata is int32: the cast happens on the host so no int64
    # index tensor is ever moved to the target for these layouts.
    values = values.to(_DEVICE)
    index = index.to(torch.int32).to(_DEVICE)
    indptr = indptr.to(_DEVICE)
    if csc:
        return torch.sparse_csc_tensor(
            indptr, index, values, size=tuple(shape), device=_DEVICE
        )
    return torch.sparse_csr_tensor(
        indptr, index, values, size=tuple(shape), device=_DEVICE
    )


def _block(shape, block_size, dtype, seed, stored_zero_block, bsc):
    """BSR/BSC input with int32 metadata and an optional all-zero stored block."""
    rows, cols = shape
    block_rows, block_cols = rows // block_size, cols // block_size
    lines, inner = (block_cols, block_rows) if bsc else (block_rows, block_cols)
    per_line = 2 if stored_zero_block else 1
    index = torch.stack(
        [
            torch.sort(
                (torch.arange(per_line, dtype=torch.int32) + line) % inner
            ).values
            for line in range(lines)
        ]
    ).reshape(-1)
    indptr = torch.arange(0, (lines + 1) * per_line, per_line, dtype=torch.int32)
    generator = _generator(seed)
    values = 0.25 + torch.rand(
        (lines * per_line, block_size, block_size), generator=generator
    )
    if stored_zero_block:
        values[0] = 0.0
    values = values.to(dtype).to(_DEVICE)
    if bsc:
        return torch.sparse_bsc_tensor(
            indptr.to(_DEVICE), index.to(_DEVICE), values, size=shape, device=_DEVICE
        )
    return torch.sparse_bsr_tensor(
        indptr.to(_DEVICE), index.to(_DEVICE), values, size=shape, device=_DEVICE
    )


def _backward_inputs(shape, sparse_dim, duplicate, dtype, seed=5):
    """Candidate/reference sparse inputs sharing coordinates and values.

    ``tu.to_reference`` returns an independently allocated copy for both parts -
    it clones the strided stored values, keeps ``requires_grad``, and materialises
    the coordinate tensor - so the reference graph and its metadata are separate
    without an extra detach/clone round trip.
    """
    sparse_extent = _extent(shape[:sparse_dim])
    count = max(1, sparse_extent // 2)
    flat = _distinct_flat(count, sparse_extent, seed)
    if duplicate:
        flat = flat % max(1, count // 4)
    indices = torch.stack(torch.unravel_index(flat, tuple(shape[:sparse_dim]))).to(
        _DEVICE
    )
    dense_shape = tuple(shape[sparse_dim:])
    values = tu.make_input(dtype, (count,) + dense_shape, ("-1", "1")).requires_grad_(
        True
    )
    inp = torch.sparse_coo_tensor(indices, values, tuple(shape), device=_DEVICE)
    ref_values = tu.to_reference(values)
    ref_indices = tu.to_reference(indices)
    ref_inp = torch.sparse_coo_tensor(
        ref_indices, ref_values, tuple(shape), device=ref_values.device
    )
    return inp, values, ref_inp, ref_values


def _stored_parts(inp):
    """Raw stored tensors of a sparse input, in layout order.

    ``_indices()``/``_values()`` are used for COO because a tensor built by the
    constructor is not marked coalesced, which makes the unflagged accessors raise.
    """
    if inp.layout == torch.sparse_coo:
        return inp._indices(), inp._values()
    if inp.layout in (torch.sparse_csc, torch.sparse_bsc):
        return inp.ccol_indices(), inp.row_indices(), inp.values()
    return inp.crow_indices(), inp.col_indices(), inp.values()


def _assert_unchanged(inp, ref_inp):
    """Densification is read-only: layout and stored parts must survive the call.

    ``ref_inp`` is ``tu.to_reference(inp)``, taken before the call: the shared
    oracle clones the sparse tensor, so its raw stored tensors are independently
    owned copies of the pre-call values and serve as the mutation reference
    directly, without cloning every part a second time.
    """
    assert inp.layout == ref_inp.layout
    if inp.layout == torch.sparse_coo:
        assert inp.is_coalesced() == ref_inp.is_coalesced()
    for part, reference_part in zip(_stored_parts(inp), _stored_parts(ref_inp)):
        tu.assert_result_equal(part, reference_part)


@pytest.mark.to_dense
@pytest.mark.parametrize("shape", _FLOAT_VALUE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("parity", tu.selected_cases([1, 3], quick=[1]))
@pytest.mark.parametrize("dtype", tu.selected_cases(_DTYPES, quick=_DTYPES))
def test__to_dense_value_ranges(shape, value_range, parity, dtype):
    inp = _coo(
        shape, parity * shape[-1], dtype, value_range, seed=parity * 31 + len(shape)
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_dense(ref_inp)
    res_out = flag_gems._to_dense(inp)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    _assert_unchanged(inp, ref_inp)


@pytest.mark.to_dense
@pytest.mark.parametrize("nnz", _RANK0_CASES)
@pytest.mark.parametrize("dtype", _RANK0_DTYPES)
def test__to_dense_rank0(nnz, dtype):
    inp = _rank0_coo(nnz, dtype, ("-1", "1"))
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_dense(ref_inp)
    res_out = flag_gems._to_dense(inp)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    _assert_unchanged(inp, ref_inp)


@pytest.mark.to_dense
@pytest.mark.parametrize("shape", _STORED_ZERO_SHAPES)
@pytest.mark.parametrize("dtype", _STORED_ZERO_DTYPES)
def test__to_dense_stored_zeros(shape, dtype):
    # Two thirds of the stored entries are explicit zeros and every third is 1.5,
    # so the densified result is not uniformly zero.
    extent = _extent(shape)
    count = min(extent, 96)
    flat = _distinct_flat(count, extent, 5).to(_DEVICE)
    indices = torch.stack(torch.unravel_index(flat, tuple(shape))).to(_DEVICE)
    values = torch.zeros(count, dtype=torch.float32, device=_DEVICE)
    values[::3] = 1.5
    values = values.to(dtype)
    inp = torch.sparse_coo_tensor(indices, values, tuple(shape), device=_DEVICE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_dense(ref_inp)
    res_out = flag_gems._to_dense(inp)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    _assert_unchanged(inp, ref_inp)


@pytest.mark.to_dense
@pytest.mark.parametrize("shape,parity", _DUPLICATE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _ACCUMULATING_DTYPES)
def test__to_dense_duplicate_accumulate(shape, parity, value_range, dtype):
    # Non-integer, full-range duplicate coordinates: densification has to add the
    # colliding stored values, so this uses the shared arithmetic comparator.
    inp = _coo(
        shape,
        parity * shape[-1],
        dtype,
        value_range,
        seed=_duplicate_seed(shape, parity),
        unsafe_layout=True,
        pool=max(1, min(64, _extent(shape) // 4)),
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_dense(ref_inp)
    res_out = flag_gems._to_dense(inp)

    assert res_out.device == inp.device
    tu.assert_result_close(res_out, ref_out)
    _assert_unchanged(inp, ref_inp)


@pytest.mark.to_dense
@pytest.mark.parametrize("shape,parity", _EXACT_CASES)
@pytest.mark.parametrize("dtype", _EXACT_SUM_DTYPES)
def test__to_dense_duplicate_small_exact(shape, parity, dtype):
    # Integer counterpart: the stored values have magnitude <= 3, and the
    # coordinates are drawn from a pool of at most 64 cells while up to
    # ``4 * shape[-1]`` entries are stored, so a cell may well receive dozens of
    # them. Adding such terms is exact and order independent (integer addition
    # does not round), which is what makes the shared exact comparator valid here;
    # no bound on the number of collisions is claimed.
    count = max(1, min(parity * shape[-1], _extent(shape)))
    values = _small_integer_values(count, dtype, seed=_duplicate_seed(shape, parity))
    inp = _coo(
        shape,
        count,
        dtype,
        ("-1", "1"),
        seed=_duplicate_seed(shape, parity),
        unsafe_layout=True,
        pool=max(1, min(64, _extent(shape) // 4)),
        values=values,
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_dense(ref_inp)
    res_out = flag_gems._to_dense(inp)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    _assert_unchanged(inp, ref_inp)


@pytest.mark.to_dense
@pytest.mark.parametrize("shape,dtype", _CANCELLATION_CASES)
def test__to_dense_duplicate_cancellation(shape, dtype):
    # +m, +m, -m, -m per cell: every partial sum is an exact small integer, so the
    # sum is exactly zero in any accumulation order and the exact comparator is
    # the right check.
    inp = _cancellation_coo(shape, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_dense(ref_inp)
    res_out = flag_gems._to_dense(inp)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    _assert_unchanged(inp, ref_inp)


@pytest.mark.to_dense
@pytest.mark.parametrize("dtype,scenario,shape", _SPECIAL_ROWS)
def test__to_dense_nan_inf(dtype, scenario, shape):
    payload = tu.make_special_input(dtype, scenario)
    count = min(payload.numel(), _extent(shape))
    payload = payload[:count]
    flat = _distinct_flat(count, _extent(shape), seed=7)
    indices = torch.stack(torch.unravel_index(flat, tuple(shape))).to(_DEVICE)
    inp = torch.sparse_coo_tensor(
        indices, payload.to(_DEVICE), tuple(shape), device=_DEVICE
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_dense(ref_inp)
    res_out = flag_gems._to_dense(inp)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    _assert_unchanged(inp, ref_inp)


@pytest.mark.to_dense
@pytest.mark.parametrize("dtype,scenario", _UNSAFE_SPECIAL_ROWS)
def test__to_dense_nan_inf_duplicate(dtype, scenario):
    # Same special values on colliding coordinates: addition of special values is
    # order independent, but the comparison uses the arithmetic path.
    shape = (256, 256)
    payload = tu.make_special_input(dtype, scenario)
    count = payload.numel()
    flat = torch.randint(0, 4, (count,), generator=_generator(3))
    indices = torch.stack(torch.unravel_index(flat, shape)).to(_DEVICE)
    inp = torch.sparse_coo_tensor(indices, payload.to(_DEVICE), shape, device=_DEVICE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_dense(ref_inp)
    res_out = flag_gems._to_dense(inp)

    assert res_out.device == inp.device
    tu.assert_result_close(res_out, ref_out)
    _assert_unchanged(inp, ref_inp)


@pytest.mark.to_dense
@pytest.mark.parametrize("shape,sparse_dim,broadcast", _HYBRID_CASES)
@pytest.mark.parametrize("dtype", _HYBRID_DTYPES)
def test__to_dense_hybrid(shape, sparse_dim, broadcast, dtype):
    # Hybrid COO with distinct sparse coordinates. The reference is rebuilt from
    # its own copy of the stored-value parent, so both sides densify the same
    # (materialised or stride-0) dense tail instead of a compacted reference.
    inp, indices, base = _hybrid(
        shape, sparse_dim, dtype, ("-1", "1"), seed=13, broadcast=broadcast
    )
    dense_shape = tuple(shape[sparse_dim:])
    ref_base = tu.to_reference(base)
    ref_inp = torch.sparse_coo_tensor(
        tu.to_reference(indices),
        _hybrid_values(ref_base, dense_shape, broadcast),
        tuple(shape),
        device=ref_base.device,
    )

    ref_out = torch.ops.aten._to_dense(ref_inp)
    res_out = flag_gems._to_dense(inp)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(base, ref_base)
    if broadcast:
        # The stored values are an expanded view of the dense tail, so the
        # densification must read a stride-0 tail on both sides.
        assert inp._values().stride() == (1,) + (0,) * len(dense_shape)
    _assert_unchanged(inp, ref_inp)


@pytest.mark.to_dense
@pytest.mark.parametrize("shape,sparse_dim", _HYBRID_DUPLICATE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__to_dense_hybrid_duplicate(shape, sparse_dim, value_range):
    # Colliding hybrid coordinates add over the dense tail as well.
    inp, indices, base = _hybrid(
        shape, sparse_dim, torch.float32, value_range, seed=17, duplicate=True
    )
    dense_shape = tuple(shape[sparse_dim:])
    ref_base = tu.to_reference(base)
    ref_inp = torch.sparse_coo_tensor(
        tu.to_reference(indices),
        _hybrid_values(ref_base, dense_shape, False),
        tuple(shape),
        device=ref_base.device,
    )

    ref_out = torch.ops.aten._to_dense(ref_inp)
    res_out = flag_gems._to_dense(inp)

    assert res_out.device == inp.device
    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_equal(base, ref_base)
    _assert_unchanged(inp, ref_inp)


@pytest.mark.to_dense
@pytest.mark.parametrize("layout_name,shape,density,share_structure", _COMPRESSED_CASES)
@pytest.mark.parametrize("dtype", _COMPRESSED_DTYPES)
def test__to_dense_compressed(layout_name, shape, density, share_structure, dtype):
    # Batched compressed items must share the stored count per item; the fixture
    # guarantees it with a fixed entries-per-line construction instead of a
    # shared random mask, and one stored value per line is an explicit zero.
    inp = _compressed(
        shape,
        density,
        dtype,
        seed=17 + len(shape),
        csc=layout_name == "csc",
        share_structure=share_structure,
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_dense(ref_inp)
    res_out = flag_gems._to_dense(inp)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    _assert_unchanged(inp, ref_inp)


@pytest.mark.to_dense
@pytest.mark.parametrize("layout_name,shape,block_size,stored_zero_block", _BLOCK_CASES)
@pytest.mark.parametrize("dtype", _COMPRESSED_DTYPES)
def test__to_dense_block(layout_name, shape, block_size, stored_zero_block, dtype):
    # Both variants store the first block of the initial line; ``stored_zero_block``
    # only changes that stored block from a nonzero draw to an all-zero one, so
    # the pair checks that a stored-but-zero block is densified as stored zero.
    bsc = layout_name == "bsc"
    inp = _block(
        shape, block_size, dtype, seed=23, stored_zero_block=stored_zero_block, bsc=bsc
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_dense(ref_inp)
    res_out = flag_gems._to_dense(inp)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    _assert_unchanged(inp, ref_inp)


@pytest.mark.to_dense
@pytest.mark.parametrize("shape,dtype", _OUT_CASES)
def test__to_dense_out(shape, dtype):
    inp = _coo(shape, shape[-1], dtype, ("-1", "1"), seed=11)
    ref_inp = tu.to_reference(inp)
    out = torch.empty(shape, dtype=dtype, device=_DEVICE)
    ref_buffer = torch.empty(shape, dtype=dtype, device=ref_inp.device)

    ref_out = torch.ops.aten._to_dense(ref_inp, None, None, out=ref_buffer)
    res_out = flag_gems._to_dense(inp, None, None, out=out)

    assert res_out is out
    assert res_out.device == out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    _assert_unchanged(inp, ref_inp)


@pytest.mark.to_dense
@pytest.mark.parametrize("shape,dtype", _OUT_CASES)
def test__to_dense_out_overwrites_prefilled(shape, dtype):
    # The provided ``out`` is fully owned and starts from a NaN sentinel, so a
    # partially written result is visible: no sentinel may survive the call.
    inp = _coo(shape, shape[-1], dtype, ("-1", "1"), seed=23)
    ref_inp = tu.to_reference(inp)
    out = torch.full(shape, float("nan"), dtype=dtype, device=_DEVICE)
    ref_buffer = torch.full(shape, float("nan"), dtype=dtype, device=ref_inp.device)

    ref_out = torch.ops.aten._to_dense(ref_inp, None, None, out=ref_buffer)
    res_out = flag_gems._to_dense(inp, None, None, out=out)

    assert res_out is out
    assert res_out.device == out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    _assert_unchanged(inp, ref_inp)


@pytest.mark.to_dense
@pytest.mark.parametrize("shape,dtype", _STRIDED_VALUE_CASES)
def test__to_dense_strided_stored_values(shape, dtype):
    # The stored values are a strided view of their own parent. The reference is
    # the transferred parent's identical view, so densification must read the
    # stored values through their strides on both sides, and neither the parent
    # nor the sparse input may be rewritten.
    count = shape[-1]
    parent = tu.make_input(dtype, (2 * count,), ("-1", "1"))
    values = parent[::2]
    flat = _distinct_flat(count, _extent(shape), seed=29)
    indices = torch.stack(torch.unravel_index(flat, tuple(shape))).to(_DEVICE)
    inp = torch.sparse_coo_tensor(indices, values, tuple(shape), device=_DEVICE)
    ref_parent = tu.to_reference(parent)
    ref_inp = torch.sparse_coo_tensor(
        tu.to_reference(indices),
        ref_parent[::2],
        tuple(shape),
        device=ref_parent.device,
    )

    ref_out = torch.ops.aten._to_dense(ref_inp)
    res_out = flag_gems._to_dense(inp)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(parent, ref_parent)
    # The stored values stay a strided view of the parent after the call.
    assert inp._values().stride() == (2,)
    _assert_unchanged(inp, ref_inp)


@pytest.mark.to_dense
@pytest.mark.parametrize("shape,dtype", _COALESCED_CASES)
def test__to_dense_explicitly_coalesced(shape, dtype):
    inp = _coo(
        shape,
        shape[-1],
        dtype,
        ("-1", "1"),
        seed=31,
        unsafe_layout=True,
        pool=max(1, min(64, _extent(shape) // 4)),
    ).coalesce()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_dense(ref_inp)
    res_out = flag_gems._to_dense(inp)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    _assert_unchanged(inp, ref_inp)


@pytest.mark.to_dense
@pytest.mark.parametrize("shape", _NO_STORED_SHAPES)
@pytest.mark.parametrize("dtype", _NO_STORED_DTYPES)
def test__to_dense_no_stored_entries(shape, dtype):
    indices = torch.empty((len(shape), 0), dtype=torch.int64, device=_DEVICE)
    values = torch.empty((0,), dtype=dtype, device=_DEVICE)
    inp = torch.sparse_coo_tensor(indices, values, tuple(shape), device=_DEVICE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_dense(ref_inp)
    res_out = flag_gems._to_dense(inp)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    _assert_unchanged(inp, ref_inp)


@pytest.mark.to_dense
@pytest.mark.parametrize("masked_grad", _MASKED_GRAD_CASES)
@pytest.mark.parametrize("dtype", _MASKED_GRAD_DTYPES)
def test__to_dense_masked_grad_forward(masked_grad, dtype):
    # ``masked_grad`` is accepted by the schema and must not change the forward
    # values of a coalesced input.
    shape = (20, 320)
    inp = _coo(shape, shape[-1], dtype, ("-1", "1"), seed=37)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_dense(ref_inp, None, masked_grad)
    res_out = flag_gems._to_dense(inp, None, masked_grad)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    _assert_unchanged(inp, ref_inp)


@pytest.mark.to_dense
@pytest.mark.parametrize("shape,sparse_dim,duplicate,dtype", _BACKWARD_CASES)
def test__to_dense_backward_values(shape, sparse_dim, duplicate, dtype):
    inp, values, ref_inp, ref_values = _backward_inputs(
        shape, sparse_dim, duplicate, dtype
    )
    upstream = tu.make_input(dtype, shape, ("-1", "1"))
    ref_upstream = tu.to_reference(upstream)

    res_out = flag_gems._to_dense(inp)
    ref_out = torch.ops.aten._to_dense(ref_inp)
    (res_grad,) = torch.autograd.grad(res_out, values, grad_outputs=upstream)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_values, grad_outputs=ref_upstream)

    assert res_out.device == inp.device
    assert res_grad.device == values.device
    # Repeated coordinates make the forward add the colliding stored values in an
    # order the densify kernel does not fix, so those rows use the shared
    # arithmetic comparison; distinct coordinates scatter once.
    if duplicate:
        tu.assert_result_close(res_out, ref_out)
    else:
        tu.assert_result_equal(res_out, ref_out)
    # The stored-value gradient gathers the upstream value at each stored
    # coordinate - also for repeated coordinates, which only make the forward add
    # - so it stays bitwise identical on every row and compares exactly.
    tu.assert_result_equal(res_grad, ref_grad)
    _assert_unchanged(inp, ref_inp)


@pytest.mark.to_dense
@pytest.mark.parametrize("shape,sparse_dim", _SPARSE_GRAD_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES + [torch.complex64])
@pytest.mark.parametrize("masked_grad", [None, True, False])
def test__to_dense_backward_sparse_input(shape, sparse_dim, dtype, masked_grad):
    inp, _, _ = _hybrid(shape, sparse_dim, dtype, ("-1", "1"), 5)
    inp = inp.detach().requires_grad_()
    ref_inp = tu.to_reference(inp)
    upstream = tu.make_input(dtype, shape, ("-1", "1"))
    ref_upstream = tu.to_reference(upstream)

    res_out = flag_gems._to_dense(inp, None, masked_grad)
    ref_out = torch.ops.aten._to_dense(ref_inp, None, masked_grad)
    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]
    ref_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)[0]

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    assert res_grad.device == inp.device
    # Keep the sparse structure instead of densifying it for the comparison.
    assert res_grad.shape == ref_grad.shape == inp.shape
    assert res_grad.layout == ref_grad.layout == torch.sparse_coo
    res_coalesced = res_grad.coalesce()
    ref_coalesced = ref_grad.coalesce()
    tu.assert_result_equal(res_coalesced.indices(), ref_coalesced.indices())
    tu.assert_result_equal(res_coalesced.values(), ref_coalesced.values())
    _assert_unchanged(inp, ref_inp)


@pytest.mark.to_dense
@pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
def test__to_dense_rejects_dense_input(dtype):
    dense = tu.make_input(dtype, (16, 16), ("-1", "1"))
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._to_dense(dense)


@pytest.mark.to_dense
@pytest.mark.parametrize("layout_name", _DTYPE_ARG_LAYOUTS)
def test__to_dense_rejects_dtype_argument(layout_name):
    # Probed for this target: the COO densify kernel rejects the dtype conversion
    # ('dtype argument is not supported by sparse_to_dense') and the CSR/CSC
    # kernels reject it with their own message; neither is generalised here.
    shape = (16, 16)
    if layout_name == "coo":
        inp = _coo(shape, shape[-1], torch.float32, ("-1", "1"), seed=41)
    else:
        inp = _compressed(shape, 0.25, torch.float32, seed=43, csc=layout_name == "csc")
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._to_dense(inp, torch.float64)


@pytest.mark.to_dense
@pytest.mark.parametrize("dtype", _COO_ARG_DTYPES)
def test__to_dense_rejects_invalid_dtype_argument_type(dtype):
    inp = _coo((16, 16), 16, dtype, ("-1", "1"), seed=47)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._to_dense(inp, "float32")


@pytest.mark.to_dense
@pytest.mark.parametrize("dtype", _COO_ARG_DTYPES)
def test__to_dense_rejects_invalid_masked_grad_type(dtype):
    inp = _coo((16, 16), 16, dtype, ("-1", "1"), seed=53)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._to_dense(inp, None, "yes")


@pytest.mark.to_dense
@pytest.mark.parametrize("dtype", _FP8_NEGATIVE_DTYPES)
def test__to_dense_rejects_fp8_input(dtype):
    # NVIDIA-scoped COO case: the sparse densify path is not implemented for the
    # FP8 dtypes on this target, so FP8 is not a supported input dtype here.
    shape = (16, 16)
    flat = _distinct_flat(16, _extent(shape), 59)
    indices = torch.stack(torch.unravel_index(flat, shape)).to(_DEVICE)
    values = torch.zeros(16, dtype=dtype, device=_DEVICE)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=_DEVICE)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._to_dense(inp)
