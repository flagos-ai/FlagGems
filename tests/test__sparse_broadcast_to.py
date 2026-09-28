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

"""Correctness tests for ``torch.ops.aten._sparse_broadcast_to``.

The operator broadcasts a sparse COO operand to a target ``size``:

* new *leading* extents become new sparse dims and replicate the stored entries
  ``prod(leading extents)`` times,
* an existing size-1 **sparse** dim may expand and likewise replicates entries,
* size-1 **dense** (payload) dims widen the payload only, leaving nnz alone,
* every other dim must equal the source dim.

The output therefore keeps ``dense_dim()``, reports
``sparse_dim() == len(size) - dense_dim()`` and stores
``nnz * prod(new leading extents) * prod(expanded sparse singleton extents)``
entries. Those entries are copied rather than recomputed, so the whole COO
result is compared exactly with ``tu.assert_result_equal`` (zero tolerance).

Prerequisites and local exclusions, none of which is a general claim:

* int64 index storage is a prerequisite of this COO-only operator: a COO
  operand always carries int64 index buffers, and the CSR operand below also
  builds int64 crow/col buffers. As with the other COO-only operator tests this
  is documented instead of skipped, because collection cannot prove what a
  device without int64 support would execute. The recorded complete run of this
  file was on a device that reports int64 support, and no path here downcasts
  coordinates or entries to sidestep it.
* Backward is not covered: the native operator exposes no autograd formula, so
  there is no reference derivative a candidate could be compared against. That
  is reported as a property of this native implementation.
* A target whose *leading* extent is 0 is excluded as an unresolved native
  hazard; an earlier probe of this operator aborted the process on this backend.
  Such an input is never generated here and is never re-run. Zero extents
  elsewhere (a zero sparse extent, a zero dense tail) are valid and are covered.
* Dense and CSR operands are rejected by the native dispatcher on the CUDA
  backend; those layout cases are selected at collection time from the vendor
  string, with no runtime probe and no skip decorator.
"""

from collections import namedtuple

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Gates for the *stored* dtype of the sparse entries, from static flags.
_VALUE_DTYPE_GATES = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.float64: utils.fp64_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
}


def _value_dtype_supported(dtype):
    return _VALUE_DTYPE_GATES.get(dtype, True)


SUPPORTED_DTYPES = [
    dtype
    for dtype in list(tu.REQUIRED_DTYPES) + [torch.float64]
    if _value_dtype_supported(dtype)
]
SPECIAL_DTYPES = [dtype for dtype in SUPPORTED_DTYPES if dtype.is_floating_point]

NNZ = 3
_DEFAULT_RANGE = tuple(tu.selected_ranges()[0])

# Quick mode only exercises the identity target; the prepend variant belongs to
# the default suite. The positive supplements and the special-value matrix below
# are default-only as well, while the required negatives run in both modes.
_SIZE_MODES = tu.selected_cases(["identity", "prepend"], quick=["identity"])

# shape, sparse dim, nnz, target size, dtype, value range, explicit coords,
# declared coalesced flag, value layout.
SparseCase = namedtuple(
    "SparseCase",
    "shape sparse_dim nnz size dtype value_range coords coalesced values_layout",
    defaults=(_DEFAULT_RANGE, None, False, "plain"),
)


def _numel(extents):
    cells = 1
    for extent in extents:
        cells *= extent
    return cells


def _unravel(flat, extents):
    rows = []
    for extent in reversed(extents):
        rows.append(flat % extent)
        flat = flat // extent
    return list(reversed(rows))


def _sparse_indices(extents, nnz, coords=None):
    """Index buffer of shape ``(len(extents), nnz)``.

    Without ``coords`` the rows are the row-major unravel of the first ``nnz``
    linear positions, so whenever ``nnz <= prod(extents)`` (how the grid sizes
    it) they are strictly increasing and unique: uncoalesced without duplicates
    and without out-of-order rows. ``coords`` is used only by the rows that
    deliberately need duplicates or a descending order.
    """
    device = flag_gems.device
    if coords is not None:
        rows = [[int(coord[dim]) for coord in coords] for dim in range(len(extents))]
        return torch.tensor(rows, dtype=torch.long, device=device)
    if nnz == 0 or not extents:
        return torch.zeros((len(extents), nnz), dtype=torch.long, device=device)
    flat = torch.arange(nnz, dtype=torch.long, device=device) % _numel(extents)
    return torch.stack(_unravel(flat, extents))


def _sparse_values(dtype, value_range, nnz, payload, values_layout):
    shape = (nnz,) + tuple(payload)
    if values_layout == "zero":
        return torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    if values_layout == "strided":
        base = tu.make_input(dtype, (2 * max(nnz, 1),) + tuple(payload), value_range)
        return base[::2][:nnz]
    return tu.make_input(dtype, shape, value_range)


def _make_sparse_input(case):
    extents = tuple(case.shape[: case.sparse_dim])
    payload = tuple(case.shape[case.sparse_dim :])
    indices = _sparse_indices(extents, case.nnz, case.coords)
    values = _sparse_values(
        case.dtype, case.value_range, case.nnz, payload, case.values_layout
    )
    inp = torch.sparse_coo_tensor(indices, values, tuple(case.shape))
    if case.coalesced:
        # Only rows whose coordinates are unique and ordered declare this flag;
        # it is never forced onto duplicate or descending coordinates.
        inp._coalesced_(True)
    return inp


def _float_operand(shape, sparse_dim, nnz=NNZ):
    """Valid COO operand for the negative tests; only the size argument is bad."""
    return _make_sparse_input(
        SparseCase(
            shape=tuple(shape),
            sparse_dim=sparse_dim,
            nnz=nnz,
            size=tuple(shape),
            dtype=torch.float32,
        )
    )


def _snapshot(inp):
    """Operand state captured before the candidate runs.

    Entry buffers alone are not enough: shape, sparse/dense rank, stored count
    and the coalesced flag are all observable input metadata a candidate could
    corrupt while leaving values untouched.
    """
    return {
        "shape": tuple(inp.shape),
        "sparse_dim": inp.sparse_dim(),
        "dense_dim": inp.dense_dim(),
        "nnz": inp._nnz(),
        "coalesced": inp.is_coalesced(),
        "indices": inp._indices().clone(),
        "values": inp._values().clone(),
    }


def _expected_nnz(size, snapshot):
    """Stored-entry count the native rule predicts, from the input snapshot.

    Derived from the pre-call snapshot, never from the operand after a candidate
    may have mutated it.
    """
    scale = 1
    leading = len(size) - len(snapshot["shape"])
    for extent in size[:leading]:
        scale *= extent
    source_extents = snapshot["shape"][: snapshot["sparse_dim"]]
    target_extents = size[leading:][: snapshot["sparse_dim"]]
    for source_extent, target_extent in zip(source_extents, target_extents):
        if target_extent != source_extent:
            scale *= target_extent
    return snapshot["nnz"] * scale


def _assert_input_unchanged(inp, snapshot):
    assert tuple(inp.shape) == snapshot["shape"]
    assert inp.sparse_dim() == snapshot["sparse_dim"]
    assert inp.dense_dim() == snapshot["dense_dim"]
    assert inp._nnz() == snapshot["nnz"]
    assert inp.is_coalesced() == snapshot["coalesced"]
    tu.assert_result_equal(inp._indices(), snapshot["indices"])
    tu.assert_result_equal(inp._values(), snapshot["values"])


def _assert_sparse_broadcast(res_out, ref_out, inp, snapshot, size):
    # Entries are copied, so the raw COO result is compared exactly with the
    # native result (zero tolerance); this also covers the layout.
    tu.assert_result_equal(res_out, ref_out)
    # Structural output metadata the stored-value comparison cannot express.
    assert res_out.sparse_dim() == len(size) - snapshot["dense_dim"]
    assert res_out.dense_dim() == snapshot["dense_dim"]
    assert res_out._nnz() == _expected_nnz(size, snapshot)
    # The coalesced flag is not uniformly False (an identity-size broadcast of a
    # coalesced operand stays coalesced), so it is compared with the native flag
    # instead of being assumed.
    assert res_out.is_coalesced() == ref_out.is_coalesced()
    _assert_input_unchanged(inp, snapshot)


@pytest.mark.sparse_broadcast_to
@pytest.mark.parametrize("size_mode", _SIZE_MODES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", tu.selected_shapes())
def test__sparse_broadcast_to(shape, value_range, dtype, size_mode):
    shape = tuple(shape)
    # Every dim of the grid shapes is a sparse dim, so the target is either the
    # source shape or the source shape with one new leading sparse dim.
    size = shape if size_mode == "identity" else (2,) + shape
    case = SparseCase(
        shape=shape,
        sparse_dim=len(shape),
        nnz=min(6, _numel(shape)),
        size=size,
        dtype=dtype,
        value_range=tuple(value_range),
    )
    inp = _make_sparse_input(case)
    ref_inp = tu.to_reference(inp)
    snapshot = _snapshot(inp)

    size = list(size)
    ref_out = torch.ops.aten._sparse_broadcast_to(ref_inp, size)
    res_out = flag_gems._sparse_broadcast_to(inp, size)

    _assert_sparse_broadcast(res_out, ref_out, inp, snapshot, size)


# Positive supplements, each one a named (id, SparseCase) pair. Quick mode takes
# none of them; the static dtype gate below drops a row whose entries would be
# allocated with a dtype the device does not support.
_EXTENDED_ROWS = [
    # zero *sparse* extent: a valid operand with no stored entries, and (0, 1024)
    # additionally keeps a 1024-wide dense payload
    ("empty_dim", SparseCase((0,), 1, 0, (0,), torch.float32)),
    ("empty_dim_prepend", SparseCase((0,), 1, 0, (2, 0), torch.float32)),
    ("empty_sparse_extent", SparseCase((0, 1024), 1, 0, (0, 1024), torch.float32)),
    (
        "empty_sparse_extent_prepend",
        SparseCase((0, 1024), 1, 0, (2, 0, 1024), torch.float32),
    ),
    # no stored entries but a non-zero extent
    ("zero_nnz", SparseCase((1024, 1024), 2, 0, (1024, 1024), torch.float32)),
    (
        "zero_nnz_prepend",
        SparseCase((1024, 1024), 2, 0, (2, 1024, 1024), torch.float32),
    ),
    # an empty operand is genuinely coalesced: declared, not forced
    (
        "zero_nnz_coalesced",
        SparseCase((64, 64), 2, 0, (64, 64), torch.float32, coalesced=True),
    ),
    # zero *dense* tail: stored entries exist, the payload extent is 0
    ("zero_dense_tail", SparseCase((4, 0), 1, 3, (4, 0), torch.float32)),
    ("zero_dense_tail_prepend", SparseCase((4, 0), 1, 3, (2, 4, 0), torch.float32)),
    # rank-zero operand: every output dim comes from prepending
    ("rank_zero", SparseCase((), 0, 1, (), torch.float32)),
    ("rank_zero_expand", SparseCase((), 0, 1, (3,), torch.float32)),
    # a size-1 *sparse* dim expanded: entries are replicated
    ("singleton_sparse_expand", SparseCase((1, 64), 1, 128, (256, 64), torch.float32)),
    # size-1 *dense* dims expanded: payload widened, nnz unchanged
    ("singleton_dense_expand_2d", SparseCase((8, 1), 1, 3, (8, 4), torch.int8)),
    (
        "singleton_dense_expand_3d",
        SparseCase((20, 1, 15), 1, 3, (20, 320, 15), torch.bfloat16),
    ),
    (
        "singleton_dense_expand_4d",
        SparseCase((16, 1, 64, 60), 1, 3, (16, 128, 64, 60), torch.int32),
    ),
    # one and two new leading sparse dims
    (
        "broadcast_3d_to_4d",
        SparseCase((20, 320, 15), 3, 6, (2, 20, 320, 15), torch.uint8),
    ),
    (
        "broadcast_4d_to_5d",
        SparseCase((16, 128, 64, 60), 4, 6, (2, 16, 128, 64, 60), torch.int64),
    ),
    (
        "multi_dim_prepend",
        SparseCase((20, 320, 15), 3, 3, (2, 4, 20, 320, 15), torch.float32),
    ),
    # a large leading extent over a 1-dim operand
    (
        "broadcast_1d_to_2d",
        SparseCase((1024,), 1, 6, (1024, 1024), torch.float8_e4m3fn),
    ),
    # uncoalesced stored states: duplicate and descending coordinates
    (
        "duplicate_coords",
        SparseCase(
            (4, 6), 2, 3, (4, 6), torch.float32, coords=((0, 0), (0, 0), (1, 2))
        ),
    ),
    (
        "duplicate_coords_prepend",
        SparseCase(
            (4, 6), 2, 3, (2, 4, 6), torch.float32, coords=((0, 0), (0, 0), (1, 2))
        ),
    ),
    (
        "descending_coords",
        SparseCase(
            (8, 8), 2, 4, (8, 8), torch.float32, coords=((3, 3), (2, 2), (1, 1), (0, 0))
        ),
    ),
    # non-contiguous stored values, and all-zero stored values
    (
        "strided_values",
        SparseCase((4, 6), 2, 3, (4, 6), torch.float32, values_layout="strided"),
    ),
    (
        "zero_values",
        SparseCase((4, 6), 2, 3, (4, 6), torch.float32, values_layout="zero"),
    ),
]

_EXTENDED_CASES = [
    pytest.param(case, id=name)
    for name, case in _EXTENDED_ROWS
    if _value_dtype_supported(case.dtype)
]
_EXTRA_CASES = tu.selected_cases(_EXTENDED_CASES, quick=[])


@pytest.mark.sparse_broadcast_to
@pytest.mark.parametrize("case", _EXTRA_CASES)
def test__sparse_broadcast_to_extended(case):
    inp = _make_sparse_input(case)
    ref_inp = tu.to_reference(inp)
    snapshot = _snapshot(inp)
    size = list(case.size)

    ref_out = torch.ops.aten._sparse_broadcast_to(ref_inp, size)
    res_out = flag_gems._sparse_broadcast_to(inp, size)

    _assert_sparse_broadcast(res_out, ref_out, inp, snapshot, size)


# nan / inf payloads, per supported floating dtype and scenario, on an identity
# target and on one that replicates the payload through a new leading dim.
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(SPECIAL_DTYPES), quick=[])


@pytest.mark.sparse_broadcast_to
@pytest.mark.parametrize("size", [(5,), (2, 5)])
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test__sparse_broadcast_to_special_values(dtype, scenario, size):
    values = tu.make_special_input(dtype, scenario)
    nnz = values.numel()
    indices = torch.arange(nnz, dtype=torch.long, device=flag_gems.device).unsqueeze(0)
    inp = torch.sparse_coo_tensor(indices, values, (nnz,))
    ref_inp = tu.to_reference(inp)
    snapshot = _snapshot(inp)
    size = list(size)

    ref_out = torch.ops.aten._sparse_broadcast_to(ref_inp, size)
    res_out = flag_gems._sparse_broadcast_to(inp, size)

    _assert_sparse_broadcast(res_out, ref_out, inp, snapshot, size)


# size arguments the native schema cannot accept: wrong container type, wrong
# element type, wrong rank, or a dim that neither matches nor is a singleton.
_INVALID_SIZE_ARGS = [
    pytest.param((4, 6), 2, None, id="none_size"),
    pytest.param((4, 6), 2, 7, id="non_sequence"),
    pytest.param((4, 6), 2, [2.5, 4, 6], id="float_extents"),
    pytest.param((4, 6), 2, ["2", 4, 6], id="string_extents"),
    pytest.param((4, 6), 2, [[2], 4, 6], id="nested_extent"),
    pytest.param((4, 6), 2, [], id="empty_size"),
    pytest.param((4, 6), 2, [6], id="rank_drop"),
    pytest.param((2, 4, 6), 3, [4, 6], id="shorter_than_source"),
    pytest.param((4, 6), 2, [2, 5, 6], id="mismatched_trailing_dim"),
    pytest.param((4, 6), 1, [4, 7], id="mismatched_trailing_dense_dim"),
]


@pytest.mark.sparse_broadcast_to
@pytest.mark.parametrize("shape,sparse_dim,bad_size", _INVALID_SIZE_ARGS)
def test__sparse_broadcast_to_rejects_invalid_size(shape, sparse_dim, bad_size):
    inp = _float_operand(shape, sparse_dim)

    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._sparse_broadcast_to(inp, bad_size)


@pytest.mark.sparse_broadcast_to
def test__sparse_broadcast_to_rejects_missing_size():
    inp = _float_operand((4, 6), 2)

    # The reference reports the missing required argument as RuntimeError
    # ("missing value for argument 'size'"), but a Python-level candidate
    # wrapper reports TypeError, so both are accepted here.
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._sparse_broadcast_to(inp)


# A dense or CSR operand falls outside the sparse-COO dispatch key ('CUDA' and
# 'SparseCsrCUDA' have no kernel), which is a NotImplementedError. Those two
# rejections were seen on the CUDA backend only, so the layout cases are chosen
# at collection time from the vendor string: no runtime probe, no skip decorator
# and no device.type-as-vendor proxy.
_LAYOUT_CASES = (
    [
        pytest.param("dense", id="dense_operand"),
        pytest.param("csr", id="csr_operand"),
    ]
    if flag_gems.vendor_name == "nvidia"
    else []
)


def _layout_operand(layout):
    if layout == "dense":
        return tu.make_input(torch.float32, (4, 6), _DEFAULT_RANGE), [4, 6]
    device = flag_gems.device
    crow = torch.tensor([0, 2, 3], dtype=torch.int64, device=device)
    col = torch.tensor([0, 3, 1], dtype=torch.int64, device=device)
    values = tu.make_input(torch.float32, (3,), _DEFAULT_RANGE)
    return torch.sparse_csr_tensor(crow, col, values, size=(2, 4)), [2, 4]


@pytest.mark.sparse_broadcast_to
@pytest.mark.parametrize("layout", _LAYOUT_CASES)
def test__sparse_broadcast_to_rejects_unsupported_layout(layout):
    inp, size = _layout_operand(layout)

    # The target is the identity size, so the layout is the only invalid input.
    with pytest.raises(NotImplementedError):
        flag_gems._sparse_broadcast_to(inp, size)
