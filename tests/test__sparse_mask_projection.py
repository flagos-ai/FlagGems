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


"""Correctness tests for ``aten::_sparse_mask_projection``.

``_sparse_mask_projection(self, mask, accumulate_matches=False)`` takes two COO
tensors that share one ``size()`` and one ``sparse_dim``, and returns a COO
tensor storing exactly the entries stored in ``self``: a stored entry whose
coordinate is also stored in ``mask`` keeps its value, every other stored entry
is zeroed, and coordinates stored only in ``mask`` are dropped. Mask *values* are
never truth conditions - only stored coordinates select entries. The result's
stored order, coalesced flag and dtype mirror ``self``, and
``accumulate_matches=True`` adds ``self``'s value once per matching stored mask
entry, which is a plain copy unless the mask repeats a coordinate.

Two dimensions of the regular-operator spec do not apply here:

* backward is exempt because the native operator has no autograd formula
  (probed: ``RuntimeError: derivative for aten::_sparse_mask_projection is not
  implemented``, for a sparse operand and for a plain values tensor alike), so no
  reference gradient exists to compare against;
* both operands must share one ``size()`` and one ``sparse_dim``, so the operator
  cannot broadcast, and its only optional argument is the boolean
  ``accumulate_matches`` flag (there is no tensor/scalar call form).

The whole family shares one structural prerequisite: torch stores every COO index
tensor as int64 (the constructor normalizes it) and both the operator and the
shared comparison read those stored indices, so no workload here can be built on
a backend without int64 support. That gate is static and applies to the case
lists themselves, never to a runtime probe.
"""

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

_SPARSE_STRUCTURE_SUPPORTED = bool(utils.int64_is_supported)


def _payload_dtypes(*candidates):
    """Statically selected payload dtypes under the COO structure gate."""
    if not _SPARSE_STRUCTURE_SUPPORTED:
        return []
    return [dtype for dtype, available in candidates if available]


_DTYPES = _payload_dtypes(
    (torch.int8, True),
    (torch.uint8, True),
    (torch.int16, True),
    (torch.int32, True),
    (torch.bool, True),
    (torch.float16, True),
    (torch.float32, True),
    (torch.bfloat16, utils.bf16_is_supported),
    (torch.int64, utils.int64_is_supported),
    (torch.float64, utils.fp64_is_supported),
)

# Floating payloads of the nan/inf matrix. FP8 stays negative-only on this
# backend: the native operator has no FP8 sparse kernel (probed: RuntimeError
# "binary_op_intersection_cuda" not implemented for 'Float8_e4m3fn' and for
# 'Float8_e5m2'), so no positive FP8 combination - the nan-only e4m3fn case
# included - can be validated against a reference.
_FLOAT_DTYPES = _payload_dtypes(
    (torch.float16, True),
    (torch.float32, True),
    (torch.bfloat16, utils.bf16_is_supported),
    (torch.float64, utils.fp64_is_supported),
)
_FP8_DTYPES = _payload_dtypes(
    (torch.float8_e4m3fn, utils.fp8_is_supported),
    (torch.float8_e5m2, utils.fp8_is_supported),
)

# Every negative case constructs COO operands, so it shares the family's
# structural prerequisite. Parametrizing the payload dtype over the same gated
# list keeps all six negatives collected in both the quick and the default suite
# whenever the prerequisite holds, and collects none of them on a backend that
# cannot build the inputs at all.
_NEGATIVE_DTYPES = tu.selected_cases(
    _DTYPES, quick=[dtype for dtype in _DTYPES if dtype is torch.float32]
)

# Parameter coverage of the boolean flag: both values plus the omitted argument
# (the schema default), with mask coordinates repeated 2 and 3 times so
# accumulate_matches really sums. Default-only, like every supplemental family.
_ACCUMULATE_SHAPES = tu.selected_cases(
    [(1024, 1024), (20, 320, 15), (16, 7, 57, 32, 29)], quick=[]
)
_DUPLICATE_COUNTS = tu.selected_cases([2, 3], quick=[])
_ACCUMULATE_VALUES = tu.selected_cases([None, False, True], quick=[])
_ACCUMULATE_DTYPES = _payload_dtypes(
    (torch.float16, True),
    (torch.bfloat16, utils.bf16_is_supported),
    (torch.int64, utils.int64_is_supported),
)

_LAYOUT_DTYPES = _payload_dtypes(
    (torch.float32, True), (torch.int64, utils.int64_is_supported)
)
_STRIDED_DTYPES = tu.selected_cases(_LAYOUT_DTYPES, quick=[])
_OUT_LAYOUT_DTYPES = tu.selected_cases(_LAYOUT_DTYPES, quick=[])

_OUT_DTYPES = _payload_dtypes(
    (torch.float16, True),
    (torch.float32, True),
    (torch.bfloat16, utils.bf16_is_supported),
)

_OUT_CASES = tu.selected_cases(
    [(shape, acc) for shape in tu.REQUIRED_SHAPES for acc in (False, True)],
    quick=[((2, 19, 7), False)],
)

_SPECIAL_CASES = tu.selected_cases(
    [
        (dtype, scenario, shape)
        for dtype, scenario in tu.special_value_cases(_FLOAT_DTYPES)
        for shape in ((256,), (2, 4, 5))
    ],
    quick=[],
)

# A stored value outside every tested input range, so an ``out`` buffer that was
# left untouched cannot pass.
_SENTINEL = -99


def _stored_positions(shape, modulus, residue):
    """Boolean map of the flat positions where ``position % modulus == residue``.

    Built directly in bool, which avoids an auxiliary int64 ``arange`` for the
    multi-million-element shapes. ``self`` and ``mask`` use different residues of
    different moduli, so their stored layouts overlap only partially.
    """
    flat = torch.zeros(math.prod(shape), dtype=torch.bool, device=flag_gems.device)
    flat[residue::modulus] = True
    return flat.reshape(shape)


def _coo_operand(values, positions):
    """Fully sparse COO storing ``positions`` with ``values[positions]``."""
    index = positions.nonzero().transpose(0, 1)
    return torch.sparse_coo_tensor(index, values[positions], values.shape)


def _scalar_operand(value):
    """Rank-0 COO storing one unaddressed entry.

    A rank-0 operand has ``sparse_dim == 0``: its index tensor carries no
    coordinate row (shape ``(0, 1)``) because the single stored entry has no
    address. ``Tensor.to_sparse()`` would drop a zero scalar instead of storing
    it, so the entry is written explicitly.
    """
    index = torch.zeros((0, 1), dtype=torch.int64, device=flag_gems.device)
    return torch.sparse_coo_tensor(index, value.reshape(1), ())


def _positional_operand(shape, sparse_dim, positions, values):
    """COO storing the flat positions ``positions`` of ``shape[:sparse_dim]``.

    Every caller passes ``sparse_dim >= 1``; the rank-0 layout has
    ``sparse_dim == 0`` and is built by ``_scalar_operand`` instead.
    """
    lead = tuple(shape[:sparse_dim])
    index = torch.tensor(positions, dtype=torch.int64, device=flag_gems.device).reshape(
        1, -1
    )
    if len(lead) > 1:
        index = torch.stack(torch.unravel_index(index.reshape(-1), torch.Size(lead)))
    return torch.sparse_coo_tensor(index, values, torch.Size(shape))


def _integer_values(n, tail, dtype):
    """Small exact stored values, so duplicated accumulation stays representable."""
    return (
        torch.arange(
            1, n * math.prod(tail) + 1, dtype=torch.float32, device=flag_gems.device
        )
        .reshape((n,) + tuple(tail))
        .to(dtype)
    )


def _projection_operands(dtype, shape, value_range):
    """Two COO operands with independent values and partially shared layout."""
    values = tu.make_input(dtype, shape, value_range)
    mask_values = tu.make_input(dtype, shape, value_range)
    if values.dim() == 0:
        return _scalar_operand(values), _scalar_operand(mask_values)
    return (
        _coo_operand(values, _stored_positions(shape, 2, 0)),
        _coo_operand(mask_values, _stored_positions(shape, 3, 1)),
    )


def _raw_snapshot(operand):
    """Independent copies of a COO's stored components plus the metadata a
    write-through would change (coalesced flag, logical shape, sparse_dim)."""
    return (
        tu.to_reference(operand._indices()),
        tu.to_reference(operand._values()),
        operand.is_coalesced(),
        operand.size(),
        operand.sparse_dim(),
    )


def _assert_untouched(operand, snapshot):
    indices, values, coalesced, size, sparse_dim = snapshot
    tu.assert_result_equal(operand._indices(), indices)
    tu.assert_result_equal(operand._values(), values)
    assert operand.is_coalesced() == coalesced
    assert operand.size() == size
    assert operand.sparse_dim() == sparse_dim


def _assert_projection(res_out, ref_out, operand, *, arithmetic=False):
    """Compare a candidate result with the native reference.

    The shared assertion compares the stored indices and values positionally with
    zero tolerance, so stored order and duplicate counts are observed, but it does
    not compare the coalesced flag - that part of the contract (the result mirrors
    ``self``) is checked here, together with the output device. ``arithmetic``
    selects the shared tolerance and is used only for the floating payloads whose
    ``accumulate_matches`` result is a genuine repeated sum; the integer payloads
    and every unique-coordinate fixture are exactly representable and stay on the
    exact comparison.
    """
    if arithmetic:
        tu.assert_result_close(res_out, ref_out)
    else:
        tu.assert_result_equal(res_out, ref_out)
    assert res_out.is_coalesced() == ref_out.is_coalesced()
    assert res_out.device == operand.device


@pytest.mark.sparse_mask_projection
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", tu.selected_shapes())
def test_sparse_mask_projection_value_range(dtype, shape, value_range):
    inp, mask = _projection_operands(dtype, shape, value_range)
    inp_before, mask_before = _raw_snapshot(inp), _raw_snapshot(mask)
    ref_inp, ref_mask = tu.to_reference(inp), tu.to_reference(mask)

    ref_out = torch.ops.aten._sparse_mask_projection(ref_inp, ref_mask, False)
    res_out = flag_gems._sparse_mask_projection(inp, mask, False)

    _assert_projection(res_out, ref_out, inp)
    _assert_untouched(inp, inp_before)
    _assert_untouched(mask, mask_before)


def _duplicate_operands(dtype, shape, duplicate_count):
    """``self`` stores one entry per selected position; ``mask`` repeats them."""
    values = tu.make_input(dtype, shape, ["-1", "1"])
    mask_values = tu.make_input(dtype, shape, ["-1", "1"])
    stored = _stored_positions(shape, 64, 0)
    index = stored.nonzero().transpose(0, 1)
    extra = _stored_positions(shape, 64, 1)
    mask_index = torch.cat([index.repeat(1, duplicate_count), extra.nonzero().T], dim=1)
    mask_stored = torch.cat(
        [mask_values[stored].repeat(duplicate_count), mask_values[extra]]
    )
    return (
        torch.sparse_coo_tensor(index, values[stored], values.shape),
        torch.sparse_coo_tensor(mask_index, mask_stored, values.shape),
    )


@pytest.mark.sparse_mask_projection
@pytest.mark.parametrize("dtype", _ACCUMULATE_DTYPES)
@pytest.mark.parametrize("duplicate_count", _DUPLICATE_COUNTS)
@pytest.mark.parametrize("accumulate_matches", _ACCUMULATE_VALUES)
@pytest.mark.parametrize("shape", _ACCUMULATE_SHAPES)
def test_sparse_mask_projection_accumulate_matches(
    shape, accumulate_matches, duplicate_count, dtype
):
    inp, mask = _duplicate_operands(dtype, shape, duplicate_count)
    inp_before, mask_before = _raw_snapshot(inp), _raw_snapshot(mask)
    ref_inp, ref_mask = tu.to_reference(inp), tu.to_reference(mask)
    # An omitted argument exercises the schema default, which is False.
    kwargs = (
        {} if accumulate_matches is None else {"accumulate_matches": accumulate_matches}
    )

    ref_out = torch.ops.aten._sparse_mask_projection(ref_inp, ref_mask, **kwargs)
    res_out = flag_gems._sparse_mask_projection(inp, mask, **kwargs)

    # Only a floating repeated sum can round; an int64 accumulation is exact.
    _assert_projection(
        res_out,
        ref_out,
        inp,
        arithmetic=bool(accumulate_matches) and dtype.is_floating_point,
    )
    _assert_untouched(inp, inp_before)
    _assert_untouched(mask, mask_before)


# (case, shape, self_index, self_values, mask_index, coalesced). The stored values
# are small integers, so even duplicated accumulation is exact and these rows keep
# the zero-tolerance comparison.
_LAYOUT_ROWS = tu.selected_cases(
    [
        ("overlapping", (5,), [[0, 2, 3]], [1, 2, 3], [[0, 2]], True),
        ("mask_only_extra", (5,), [[0, 2]], [1, 2], [[0, 2, 4]], True),
        ("duplicate_self", (5,), [[0, 0, 2]], [1, 2, 7], [[0, 2]], False),
        ("duplicate_mask", (5,), [[0, 2]], [3, 4], [[2, 2, 2, 0]], False),
        ("duplicate_both", (5,), [[0, 0, 2]], [1, 2, 7], [[0, 0, 0, 2]], False),
        ("reordered_self", (5,), [[3, 0, 2]], [5, 1, 2], [[0, 3]], False),
        ("reordered_mask", (5,), [[0, 3]], [1, 5], [[3, 0]], False),
        ("disjoint", (5,), [[0, 1]], [1, 2], [[3, 4]], True),
        ("empty_self", (5,), [[]], [], [[0, 2]], True),
        ("empty_mask", (5,), [[0, 2]], [1, 2], [[]], True),
        (
            "diagonal_2d",
            (3, 3),
            [[0, 1, 2], [0, 1, 2]],
            [1, 2, 3],
            [[0, 1], [0, 1]],
            True,
        ),
    ],
    quick=[],
)


def _layout_operand(shape, index_rows, values, coalesced, dtype):
    index = torch.tensor(index_rows, dtype=torch.int64, device=flag_gems.device)
    index = index.reshape(len(shape), -1)
    stored = (
        torch.full((index.shape[1],), 7, dtype=dtype, device=flag_gems.device)
        if values is None
        else torch.tensor(values, dtype=dtype, device=flag_gems.device)
    )
    operand = torch.sparse_coo_tensor(index, stored, torch.Size(shape))
    return operand.coalesce() if coalesced else operand


@pytest.mark.sparse_mask_projection
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
@pytest.mark.parametrize("accumulate_matches", [False, True])
@pytest.mark.parametrize("row", _LAYOUT_ROWS, ids=[row[0] for row in _LAYOUT_ROWS])
def test_sparse_mask_projection_stored_layout(row, accumulate_matches, dtype):
    _, shape, self_index, self_values, mask_index, coalesced = row
    inp = _layout_operand(shape, self_index, self_values, coalesced, dtype)
    mask = _layout_operand(shape, mask_index, None, coalesced, dtype)
    inp_before, mask_before = _raw_snapshot(inp), _raw_snapshot(mask)
    ref_inp, ref_mask = tu.to_reference(inp), tu.to_reference(mask)

    ref_out = torch.ops.aten._sparse_mask_projection(
        ref_inp, ref_mask, accumulate_matches
    )
    res_out = flag_gems._sparse_mask_projection(inp, mask, accumulate_matches)

    _assert_projection(res_out, ref_out, inp)
    _assert_untouched(inp, inp_before)
    _assert_untouched(mask, mask_before)


# (case, shape, self positions, mask positions, mask value kind, self payload
# dtype). A hybrid COO has ``sparse_dim == rank - 1`` plus one dense trailing
# dimension; the mask values are all False / negative / NaN in three rows, which
# must not change the selection. Every row is built from int64 index tensors, so
# the list collapses to none on a backend without the structural prerequisite.
_HYBRID_ROWS = (
    [
        ("matched", (6, 3), [0, 2, 4], [0, 2, 4], "int", None),
        ("partial", (6, 3), [0, 2, 4], [2, 4], "int", None),
        ("disjoint", (6, 3), [0, 2], [1, 3], "int", None),
        ("mask_bool_values", (6, 3), [0, 2], [0, 2], "bool", None),
        ("mask_negative_values", (6, 3), [0, 2], [0, 2], "negative", None),
        ("mask_nan_values", (6, 3), [0, 2], [0, 2], "nan", None),
        ("sparse_dim_two", (2, 4, 3), [0, 3], [0, 3], "int", None),
    ]
    if _SPARSE_STRUCTURE_SUPPORTED
    else []
)
if _SPARSE_STRUCTURE_SUPPORTED:
    _HYBRID_ROWS.append(("int_operands", (5, 2), [0, 3], [0, 3], "int", torch.int64))
_HYBRID_ROWS = tu.selected_cases(_HYBRID_ROWS, quick=[])


def _hybrid_operands(row):
    _, shape, self_positions, mask_positions, mask_kind, self_dtype = row
    sparse_dim = len(shape) - 1
    tail = tuple(shape[sparse_dim:])
    self_dtype = self_dtype or torch.float32
    self_values = _integer_values(len(self_positions), tail, self_dtype)
    if mask_kind == "bool":
        mask_values = torch.zeros(
            (len(mask_positions),) + tail, dtype=torch.bool, device=flag_gems.device
        )
    elif mask_kind == "negative":
        mask_values = torch.full(
            (len(mask_positions),) + tail, -5.0, device=flag_gems.device
        )
    elif mask_kind == "nan":
        mask_values = torch.full(
            (len(mask_positions),) + tail, float("nan"), device=flag_gems.device
        )
    else:
        mask_values = torch.full(
            (len(mask_positions),) + tail, 7, dtype=self_dtype, device=flag_gems.device
        )
    return (
        _positional_operand(shape, sparse_dim, self_positions, self_values),
        _positional_operand(shape, sparse_dim, mask_positions, mask_values),
    )


@pytest.mark.sparse_mask_projection
@pytest.mark.parametrize("row", _HYBRID_ROWS, ids=[row[0] for row in _HYBRID_ROWS])
def test_sparse_mask_projection_hybrid_coo(row):
    inp, mask = _hybrid_operands(row)
    inp_before, mask_before = _raw_snapshot(inp), _raw_snapshot(mask)
    ref_inp, ref_mask = tu.to_reference(inp), tu.to_reference(mask)

    ref_out = torch.ops.aten._sparse_mask_projection(ref_inp, ref_mask)
    res_out = flag_gems._sparse_mask_projection(inp, mask)

    _assert_projection(res_out, ref_out, inp)
    _assert_untouched(inp, inp_before)
    _assert_untouched(mask, mask_before)


# Each row lists, for all ``size`` stored coordinates of ``self``, whether the
# mask also stores that coordinate. The first row is the original all-match
# fixture; the partial row is the one that keeps copied values and zeroed entries
# inside the same strided tensor, and the last row keeps the all-unmatched case.
_STRIDED_ROWS = tu.selected_cases(
    [
        ("all_coordinates_match", [True, True, True, True]),
        ("partial_match", [True, False, True, False]),
        ("no_coordinate_matches", [False, False, False, False]),
    ],
    quick=[],
)


@pytest.mark.sparse_mask_projection
@pytest.mark.parametrize(
    "case,mask_flags", _STRIDED_ROWS, ids=[row[0] for row in _STRIDED_ROWS]
)
@pytest.mark.parametrize("dtype", _STRIDED_DTYPES)
def test_sparse_mask_projection_strided_storage(case, mask_flags, dtype):
    del case
    # Both stored components are non-contiguous views into larger buffers: the
    # values slab has storage offset 2 and stride 4, the index slab stride 2. The
    # surrounding elements are padding that no candidate may write through.
    size = len(mask_flags)
    value_buffer = tu.make_input(dtype, (4 * size,), ["-1", "1"])
    values = value_buffer[2::4]
    index_buffer = torch.zeros(2 * size, dtype=torch.int64, device=flag_gems.device)
    index_buffer[0::2] = torch.arange(size, device=flag_gems.device)
    index = index_buffer.reshape(1, -1)[:, ::2]
    inp = torch.sparse_coo_tensor(index, values, (size,))
    matched = torch.tensor(mask_flags, dtype=torch.bool, device=flag_gems.device)
    mask = _coo_operand(tu.make_input(dtype, (size,), ["-1", "1"]), matched)
    inp_before, mask_before = _raw_snapshot(inp), _raw_snapshot(mask)
    value_buffer_before = tu.to_reference(value_buffer)
    index_buffer_before = tu.to_reference(index_buffer)
    ref_inp, ref_mask = tu.to_reference(inp), tu.to_reference(mask)

    ref_out = torch.ops.aten._sparse_mask_projection(ref_inp, ref_mask)
    res_out = flag_gems._sparse_mask_projection(inp, mask)

    _assert_projection(res_out, ref_out, inp)
    # Matched entries keep the value behind the strided view, unmatched ones become
    # zero, and the padding around both slabs must stay untouched.
    expected = torch.where(matched, values, torch.zeros_like(values))
    tu.assert_result_equal(res_out._values(), tu.to_reference(expected))
    _assert_untouched(inp, inp_before)
    _assert_untouched(mask, mask_before)
    tu.assert_result_equal(value_buffer, value_buffer_before)
    tu.assert_result_equal(index_buffer, index_buffer_before)


@pytest.mark.sparse_mask_projection
@pytest.mark.parametrize("dtype,scenario,shape", _SPECIAL_CASES)
def test_sparse_mask_projection_special_values(dtype, scenario, shape):
    payload = tu.make_special_input(dtype, scenario)
    values = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    values.reshape(-1)[: payload.numel()] = payload
    positions = torch.ones(shape, dtype=torch.bool, device=flag_gems.device)
    inp = _coo_operand(values, positions)
    # The mask stores the same coordinates with finite values, so the projected
    # values pass through unchanged and keep their nan/inf classification.
    mask = _coo_operand(tu.make_input(dtype, shape, ["-1", "1"]), positions)
    inp_before, mask_before = _raw_snapshot(inp), _raw_snapshot(mask)
    ref_inp, ref_mask = tu.to_reference(inp), tu.to_reference(mask)

    ref_out = torch.ops.aten._sparse_mask_projection(ref_inp, ref_mask)
    res_out = flag_gems._sparse_mask_projection(inp, mask)

    _assert_projection(res_out, ref_out, inp)
    _assert_untouched(inp, inp_before)
    _assert_untouched(mask, mask_before)


def _sentinel_coo(shape, dtype):
    """``out`` buffer whose stored values are wrong at ``self``'s coordinates."""
    sentinel = torch.full(shape, _SENTINEL, dtype=dtype, device=flag_gems.device)
    if not shape:
        return _scalar_operand(sentinel)
    return _coo_operand(sentinel, _stored_positions(shape, 2, 0))


@pytest.mark.sparse_mask_projection
@pytest.mark.parametrize("dtype", _OUT_DTYPES)
@pytest.mark.parametrize("shape,accumulate_matches", _OUT_CASES)
def test_sparse_mask_projection_out(shape, accumulate_matches, dtype):
    inp, mask = _projection_operands(dtype, shape, ["-1", "1"])
    inp_before, mask_before = _raw_snapshot(inp), _raw_snapshot(mask)
    ref_inp, ref_mask = tu.to_reference(inp), tu.to_reference(mask)
    out = _sentinel_coo(shape, dtype)
    # A COO ``to_reference`` copy keeps the raw sparse storage and the reference
    # placement, so the reference writes into its own independent buffer.
    ref_out = tu.to_reference(out)

    ref_ret = torch.ops.aten._sparse_mask_projection(
        ref_inp, ref_mask, accumulate_matches, out=ref_out
    )
    res_ret = flag_gems._sparse_mask_projection(inp, mask, accumulate_matches, out=out)

    # ``self`` and ``mask`` each store every coordinate at most once, so
    # accumulate_matches relocates each value exactly once and the result is an
    # exact copy. The sentinel coordinates overlap the projected entries, so an
    # unwritten buffer would still hold _SENTINEL and fail the comparison.
    assert res_ret is out
    _assert_projection(res_ret, ref_ret, inp)
    _assert_untouched(inp, inp_before)
    _assert_untouched(mask, mask_before)


# (case, shape, sparse_dim, self positions, mask positions, accumulate_matches).
# ``self`` stores repeated coordinates out of order, so the result has to
# reproduce the duplicates in the stored order while the mask repeats entries.
# The payloads are small integers, so accumulation is exact and these rows keep
# the zero-tolerance comparison.
_OUT_LAYOUT_ROWS = tu.selected_cases(
    [
        ("duplicated_reordered", (5,), 1, [4, 0, 0, 2], [4, 0, 0], False),
        ("duplicated_reordered_accumulate", (5,), 1, [4, 0, 0, 2], [4, 0, 0], True),
        ("hybrid_duplicated", (4, 3), 1, [3, 0, 0, 2], [0, 0, 3], False),
        ("hybrid_reordered_accumulate", (4, 3), 1, [2, 0, 3], [3, 2], True),
    ],
    quick=[],
)


@pytest.mark.sparse_mask_projection
@pytest.mark.parametrize("dtype", _OUT_LAYOUT_DTYPES)
@pytest.mark.parametrize(
    "case,shape,sparse_dim,self_positions,mask_positions,accumulate_matches",
    _OUT_LAYOUT_ROWS,
    ids=[row[0] for row in _OUT_LAYOUT_ROWS],
)
def test_sparse_mask_projection_out_stored_layout(
    case, shape, sparse_dim, self_positions, mask_positions, accumulate_matches, dtype
):
    del case
    tail = tuple(shape[sparse_dim:])
    inp = _positional_operand(
        shape,
        sparse_dim,
        self_positions,
        _integer_values(len(self_positions), tail, dtype),
    )
    mask = _positional_operand(
        shape,
        sparse_dim,
        mask_positions,
        _integer_values(len(mask_positions), tail, dtype),
    )
    sentinel = torch.full(
        (len(self_positions),) + tail, _SENTINEL, dtype=dtype, device=flag_gems.device
    )
    out = _positional_operand(shape, sparse_dim, self_positions, sentinel)
    ref_out = tu.to_reference(out)
    inp_before, mask_before = _raw_snapshot(inp), _raw_snapshot(mask)
    ref_inp, ref_mask = tu.to_reference(inp), tu.to_reference(mask)

    ref_ret = torch.ops.aten._sparse_mask_projection(
        ref_inp, ref_mask, accumulate_matches, out=ref_out
    )
    res_ret = flag_gems._sparse_mask_projection(inp, mask, accumulate_matches, out=out)

    assert res_ret is out
    _assert_projection(res_ret, ref_ret, inp)
    _assert_untouched(inp, inp_before)
    _assert_untouched(mask, mask_before)


def _valid_pair(dtype, shape=(8,)):
    index = torch.arange(shape[-1], device=flag_gems.device).reshape(1, -1)
    values = tu.make_input(dtype, shape, ["-1", "1"])
    operand = torch.sparse_coo_tensor(index, values, shape)
    return operand, operand


@pytest.mark.sparse_mask_projection
@pytest.mark.parametrize("dtype", _NEGATIVE_DTYPES)
def test_sparse_mask_projection_rejects_dense_self(dtype):
    _, mask = _valid_pair(dtype)
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._sparse_mask_projection(
            torch.zeros(8, dtype=dtype, device=flag_gems.device), mask
        )


@pytest.mark.sparse_mask_projection
@pytest.mark.parametrize("dtype", _NEGATIVE_DTYPES)
def test_sparse_mask_projection_rejects_dense_mask(dtype):
    operand, _ = _valid_pair(dtype)
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._sparse_mask_projection(
            operand, torch.zeros(8, dtype=dtype, device=flag_gems.device)
        )


@pytest.mark.sparse_mask_projection
@pytest.mark.parametrize("dtype", _NEGATIVE_DTYPES)
def test_sparse_mask_projection_rejects_size_mismatch(dtype):
    operand, _ = _valid_pair(dtype)
    other, _ = _valid_pair(dtype, shape=(9,))
    with pytest.raises(RuntimeError):
        flag_gems._sparse_mask_projection(operand, other)


@pytest.mark.sparse_mask_projection
@pytest.mark.parametrize("dtype", _NEGATIVE_DTYPES)
def test_sparse_mask_projection_rejects_sparse_dim_mismatch(dtype):
    # Same size and same nnz, so the failure can only come from the sparse_dim
    # difference: self is fully sparse (sparse_dim == rank) while the mask splits
    # off a dense tail (sparse_dim == rank - 1).
    coordinates = torch.tensor([[0, 3], [1, 2]], device=flag_gems.device)
    self_operand = torch.sparse_coo_tensor(
        coordinates, torch.ones(2, dtype=dtype, device=flag_gems.device), (4, 4)
    )
    mask = torch.sparse_coo_tensor(
        coordinates[:1],
        torch.ones((2, 4), dtype=dtype, device=flag_gems.device),
        (4, 4),
    )
    with pytest.raises(RuntimeError):
        flag_gems._sparse_mask_projection(self_operand, mask)


@pytest.mark.sparse_mask_projection
@pytest.mark.parametrize("dtype", _NEGATIVE_DTYPES)
def test_sparse_mask_projection_rejects_invalid_accumulate(dtype):
    operand, mask = _valid_pair(dtype)
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._sparse_mask_projection(operand, mask, "yes")


@pytest.mark.sparse_mask_projection
@pytest.mark.parametrize("dtype", _NEGATIVE_DTYPES)
def test_sparse_mask_projection_rejects_dense_out(dtype):
    operand, mask = _valid_pair(dtype)
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._sparse_mask_projection(
            operand, mask, out=torch.zeros(8, dtype=dtype, device=flag_gems.device)
        )


# An allocatable dtype that the operator itself rejects: this backend has no FP8
# sparse kernel, so the uncoalesced call raises RuntimeError
# ("binary_op_intersection_cuda" not implemented for 'Float8_e4m3fn' and for
# 'Float8_e5m2'). The fixtures build without error, so the failure comes from the
# operator, not from input construction; the coalesced form is not part of this
# test because torch's own "coalesce_sparse_cuda" is unimplemented for FP8. The
# parametrization is statically empty when the backend cannot allocate FP8.
@pytest.mark.sparse_mask_projection
@pytest.mark.parametrize("dtype", _FP8_DTYPES)
def test_sparse_mask_projection_rejects_fp8(dtype):
    shape = (4,)
    index = torch.arange(4, device=flag_gems.device).reshape(1, -1)
    operand = torch.sparse_coo_tensor(
        index, torch.ones(4, dtype=dtype, device=flag_gems.device), shape
    )
    mask = torch.sparse_coo_tensor(
        index, torch.ones(4, dtype=dtype, device=flag_gems.device), shape
    )
    with pytest.raises(RuntimeError):
        flag_gems._sparse_mask_projection(operand, mask)
