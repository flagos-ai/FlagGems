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

import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu


def _required_device():
    return torch.empty(0, dtype=torch.uint8, device=flag_gems.device).device


def _index_to_device(tensor, dtype):
    source = tensor.to(dtype)
    device = _required_device()
    result = source.to(device)
    # MetaX changes an ordinary contiguous rank-4 CPU index to channels-last.
    # Restore only a contiguous host index; later offset/negative views remain.
    if (
        flag_gems.vendor_name == "metax"
        and source.device.type == "cpu"
        and source.is_contiguous()
    ):
        result = result.contiguous()
    assert result.dtype == dtype and result.shape == source.shape
    assert result.device == device
    return result


def _raw_cpu(tensor):
    storage = tensor.untyped_storage()
    raw = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
        storage, 0, (storage.nbytes(),), (1,)
    )
    return raw.cpu().clone()


def _cpu_physical(tensor):
    raw = _raw_cpu(tensor)
    result = (
        torch.empty(0, dtype=torch.uint8, device="cpu")
        .view(tensor.dtype)
        .set_(
            raw.untyped_storage(),
            tensor.storage_offset(),
            tensor.shape,
            tensor.stride(),
        )
    )
    if tensor.is_conj():
        result = result.conj()
    if tensor.is_neg():
        result = torch._neg_view(result)
    return result


def _input_signature(tensor):
    storage = tensor.untyped_storage()
    return (
        type(tensor),
        tensor.device,
        tensor.dtype,
        tensor.layout,
        tuple(tensor.shape),
        tensor.stride(),
        tensor.storage_offset(),
        tensor.is_conj(),
        tensor.is_neg(),
        tensor.requires_grad,
        tensor._version,
        storage._cdata,
        tensor.data_ptr(),
        storage.data_ptr(),
        storage.nbytes(),
    )


def _call_validator(*args, required_device, **kwargs):
    # Used only by original positive callers. Native negative calls remain direct.
    try:
        tensors = [x for x in (*args, *kwargs.values()) if isinstance(x, torch.Tensor)]
        assert len(tensors) == 3
        assert all(
            x.device == required_device and x.layout == torch.strided for x in tensors
        )
        states = [
            (x, x.untyped_storage(), _input_signature(x), _raw_cpu(x)) for x in tensors
        ]
    except Exception as error:
        raise AssertionError("invalid positive validator input") from error
    try:
        return flag_gems._validate_sparse_csc_tensor_args(*args, **kwargs)
    finally:
        # Validate original ABI and held storage before observing post-call bytes.
        try:
            for tensor, storage, signature, before in states:
                assert _input_signature(tensor) == signature
                assert tensor.untyped_storage()._cdata == storage._cdata
                assert torch.equal(_raw_cpu(tensor), before)
        except Exception as error:
            raise AssertionError("validator modified a read-only input") from error


def _assert_result_equal(result, reference, required_device):
    assert isinstance(result, torch.Tensor) and isinstance(reference, torch.Tensor)
    assert result.device == required_device
    assert reference.device == (
        torch.device("cpu") if utils.TO_CPU else required_device
    )
    assert result.dtype == reference.dtype and result.shape == reference.shape
    assert result.layout == reference.layout == torch.strided
    if flag_gems.vendor_name == "kunlunxin" and (
        result.is_complex() or reference.is_complex()
    ):
        result, reference = _cpu_physical(result), _cpu_physical(reference)
    elif flag_gems.vendor_name == "ascend" and (result.is_complex() or result.ndim > 8):
        result, reference = result.cpu(), reference.cpu()
    tu.assert_result_equal(result, reference)


_ASCEND_FP8_INPUT_SKIP = pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend cannot generate nonempty FP8 test payloads",
)


def _input_marks(dtype):
    if dtype.is_complex:
        return (_ASCEND_COMPLEX_SKIP,)
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return (_ASCEND_FP8_INPUT_SKIP,)
    return ()


_ASCEND_VALUES_RANK_SKIP = pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend input generation does not support values with more than 8 dimensions",
)

_ASCEND_COMPLEX_SKIP = pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend complex input/reference support is incomplete",
)


# The operator validates the arguments of a CSC tensor and returns nothing:
#   _validate_sparse_csc_tensor_args(Tensor ccol_indices, Tensor row_indices,
#                                    Tensor values, int[] size) -> ()
# A valid workload is observable through the call being accepted, the return
# value being None and the inspected tensors staying unmodified. The host worker
# checks the argument preconditions (ranks, strides, dtypes, sizes, devices); the
# index *content* invariants are checked by a kernel on the device as well.
# Three spec dimensions drop out for that reason:
#   * broadcast: nothing is combined elementwise, so no operand pair broadcasts;
#   * backward: the schema returns no tensor and is not differentiable;
#   * tensor vs scalar operand: the only non-tensor argument is the required
#     int[] size, which has no scalar form.
# The schema declares a single default overload (probed on the active backend:
# overloads() == ["default"]), so there is no .out call form to exercise.

_RUNTIME_DEVICE = flag_gems.runtime.device

# Static backend capability flags decide which tensors this file builds; dtype
# support is never probed during collection or execution.
_DTYPE_GATES = {
    torch.bfloat16: "support_bf16",
    torch.float64: "support_fp64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e4m3fnuz: "support_fp8",
    torch.float8_e5m2: "support_fp8",
    torch.float8_e5m2fnuz: "support_fp8",
    torch.int64: "support_int64",
}


def _dtype_supported(dtype):
    # A flag that is absent is treated as available, so an unsupported dtype
    # fails at build time instead of dropping coverage silently.
    flag = _DTYPE_GATES.get(dtype)
    return flag is None or bool(getattr(_RUNTIME_DEVICE, flag, True))


# Index tensors must use a device-supported integer dtype. Where the backend has
# no int64 the same structures are built with int32 indices instead; the int64
# metadata used to compute the structure is a host temporary, so it does not
# depend on the device index dtype and is not gated.
HAS_INT64_INDEX = _dtype_supported(torch.int64)
INDEX_DTYPE_LONG = torch.int64 if HAS_INT64_INDEX else torch.int32

# The validator never dispatches on the values dtype: every spec dtype plus
# float64, bool and complex64 were accepted by the native operator on the active
# backend, so each gate-passing dtype is exercised.
VALUE_DTYPES = [
    dtype
    for dtype in list(tu.REQUIRED_DTYPES) + [torch.float64, torch.bool, torch.complex64]
    if _dtype_supported(dtype)
]

# A CSC structure is described by a size with len(size) == batch + 2 + dense, so
# the 0-dim and 1-dim entries of tu.selected_shapes() cannot describe one and are
# filtered out. Rank >= 2 is the operator's own rule.
SIZE_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) >= 2]

# Rows are (size, nnz, dense_ndim, index_dtype, layout). The plain shape grid
# cannot express empty structures, full density, int32 indices, strided/offset
# operands or an index rank beyond 8, so the geometry, the index dtype and the
# storage layout of the operands are covered here. Index ranks 8 and 9 appear as
# generic rank-boundary cases; how the implementation dispatches such ranks was
# not established here.
STRUCTURE_ROWS = [
    ((1, 1), 1, 0, torch.int64, "plain"),  # smallest non-empty structure
    ((1, 1), 0, 0, torch.int64, "plain"),  # no stored entries
    ((4, 4), 16, 0, torch.int64, "plain"),  # fully dense
    ((5, 4), 4, 0, torch.int64, "plain"),  # one entry per column
    ((4, 0), 0, 0, torch.int64, "plain"),  # no columns
    ((0, 4), 0, 0, torch.int64, "plain"),  # no rows
    ((3, 4, 5), 15, 0, torch.int64, "plain"),  # batched, fully dense
    ((3, 0, 4), 0, 0, torch.int64, "plain"),  # empty plain dimension
    ((0, 3, 4), 0, 0, torch.int64, "plain"),  # empty batch dimension
    ((2, 3, 4, 5), 20, 0, torch.int64, "plain"),  # two batch dimensions
    ((1,) * 7 + (2, 3), 3, 0, torch.int64, "plain"),  # index rank 8
    ((1,) * 8 + (2, 3), 3, 0, torch.int64, "plain"),  # index rank 9
    ((2, 3, 4, 5, 7), 20, 1, torch.int64, "plain"),  # one dense dimension
    ((2, 3, 5, 0), 0, 1, torch.int64, "plain"),  # empty dense dimension
    ((4, 4), 4, 0, torch.int64, "offset_indices"),  # index storage offset
    ((4, 4), 4, 0, torch.int64, "strided_values"),  # strided values view
    ((4, 4), 16, 0, torch.int32, "plain"),  # int32 index dtype
    ((3, 4, 5), 15, 0, torch.int32, "plain"),
    ((8, 5), 5, 0, torch.int32, "plain"),
    ((2, 3, 4, 5, 7), 20, 1, torch.int32, "plain"),
    ((16, 128, 64, 60), 60, 0, torch.int64, "plain"),  # wide batch
    ((2, 19, 7), 7, 0, torch.int64, "plain"),  # regular 3-dim structure
]

# These structural positives run in the default suite only; the main grid below
# keeps the quick smoke subset. Rows whose index dtype the backend cannot build
# are dropped, which is the int32-only fallback when there is no int64 device
# dtype; the int32 rows stay either way.
STRUCTURE_CASES = [
    row
    for row in tu.selected_cases(STRUCTURE_ROWS, quick=[])
    if _dtype_supported(row[3])
]

# One broken host-side precondition per row, isolated so each failure can be
# attributed to a single argument check of
# _validate_sparse_compressed_tensor_args_worker. The index *content* invariants
# of the same native worker (sorted and distinct column entries, ccol[-1] == nnz,
# 0 <= ccol[0], per-batch contiguity of the stored column slices) are checked
# under CUDA_KERNEL_ASSERT, which aborts the CUDA context instead of raising a
# Python exception; those malformed index contents are therefore not exercised
# here and remain an unresolved part of the native contract. Only host-side
# preconditions are negative-tested. Rows are (defect, size, nnz).
INVALID_ROWS = [
    ("size_rank", (3, 4, 5), 15),
    ("ccol_rank_zero", (3, 4, 5), 15),
    ("ccol_row_rank", (3, 4, 5), 15),
    ("values_rank", (3, 4, 5), 15),
    ("row_batch_mismatch", (3, 4, 5), 15),
    ("values_batch_mismatch", (3, 4, 5), 15),
    ("ccol_length", (3, 4, 5), 15),
    ("row_length", (3, 4, 5), 15),
    ("ccol_stride", (3, 4, 5), 15),
    ("row_stride", (3, 4, 5), 15),
    *([("index_dtype_mismatch", (3, 4, 5), 15)] if HAS_INT64_INDEX else []),
    ("index_dtype_unsupported", (3, 4, 5), 15),
    ("meta_nnz", (4, 5), 3),
]

# Invalid call schemas, separate from the metadata preconditions above.
SCHEMA_DEFECTS = [
    "size_missing",
    "ccol_not_tensor",
    "size_not_int_list",
    "size_float_entry",
]

# tu.make_special_input yields five values, which fit an (8, 5) CSC structure with
# one entry per column. The shared helper already restricts the scenarios to what
# each dtype can represent (nan only for float8_e4m3fn; nan, inf and mixed for
# float8_e5m2 and the wider float types). Positive special values run in the
# default suite only.
SPECIAL_SIZE = (8, 5)

SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(VALUE_DTYPES), quick=[])

# A row coordinate above 2**31 with a huge logical row count and tiny nnz: int64
# indices describe it and no dense tensor is allocated for the structure, while
# int32 indices cannot describe the row dimension at all (see the overflow test).
LARGE_ROW_SIZE = [2**31 + 5, 1]
LARGE_ROW_COORDINATE = 2**31 + 3


def _split_size(size, dense_ndim):
    batch = tuple(size[: len(size) - 2 - dense_ndim])
    nrows = size[len(batch)]
    ncols = size[len(batch) + 1]
    dense = tuple(size[len(batch) + 2 :])
    return batch, nrows, ncols, dense


def _column_counts(batch_count, ncols, nnz):
    # A valid column-count vector (column j holds rows 0..count_j-1, so every
    # count is at most ceil(nnz / ncols) <= nrows whenever nnz <= nrows * ncols),
    # rotated by one column per batch. Rotation keeps each batch's total and
    # per-column bound while giving the batches different column distributions
    # whenever the entries do not divide evenly, so the fixture does not repeat
    # one identical index structure for every batch. That diversity alone is not
    # evidence that a validator inspected every batch.
    counts = torch.zeros(batch_count, ncols, dtype=torch.int64)
    if ncols == 0:
        return counts
    counts += nnz // ncols
    counts[:, : nnz % ncols] += 1
    if batch_count > 1 and ncols > 1:
        columns = torch.arange(ncols, dtype=torch.int64)[None, :]
        shift = (torch.arange(batch_count, dtype=torch.int64) % ncols)[:, None]
        counts = torch.gather(counts, 1, (columns - shift) % ncols)
    return counts


def _csc_indices(size, nnz, index_dtype, dense_ndim=0):
    # Valid ccol_indices/row_indices for size: ccol[..., 0] == 0,
    # ccol[..., -1] == nnz, 0 <= diff <= nrows, and each column slice is sorted
    # and distinct whenever nnz <= nrows * ncols. The int64 metadata is built on
    # the host and moved once, independently of the device index dtype.
    batch, nrows, ncols, _ = _split_size(size, dense_ndim)
    batch_count = 1
    for dim in batch:
        batch_count *= dim
    counts = _column_counts(batch_count, ncols, nnz)
    ccol = torch.zeros(batch_count, ncols + 1, dtype=torch.int64)
    if ncols:
        ccol[:, 1:] = torch.cumsum(counts, 1)
    row = torch.zeros(batch_count, nnz, dtype=torch.int64)
    if nnz:
        # Entry i belongs to the last column whose start is <= i; its row inside
        # that column is then i - ccol[column].
        offsets = (
            torch.arange(nnz, dtype=torch.int64).expand(batch_count, nnz).contiguous()
        )
        starts = ccol[:, :-1].contiguous()
        column = torch.searchsorted(starts, offsets, right=True) - 1
        row = offsets - torch.gather(starts, 1, column)
    return (
        _index_to_device(ccol.reshape(batch + (ncols + 1,)), index_dtype),
        _index_to_device(row.reshape(batch + (nnz,)), index_dtype),
    )


def _offset_last_dim(tensor, pad):
    # Same shape and values, reached through a wider storage: storage_offset is
    # pad and stride(-1) stays 1, which the validator accepts. The parent is
    # returned so the test can check the whole storage it allocated.
    parent = torch.zeros(
        tensor.shape[:-1] + (tensor.shape[-1] + pad,),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    view = parent[..., pad:]
    view.copy_(tensor)
    return view, parent


def _strided_last_dim(tensor, stride):
    # Same shape and values, but stride(-1) == stride. The validator requires the
    # last dimension of both index tensors to be dense per batch; the defective
    # operand still holds every original value, so the negative case fails for the
    # stride and nothing else.
    parent = torch.zeros(
        tensor.shape[:-1] + (tensor.shape[-1] * stride,),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    view = parent[..., ::stride]
    view.copy_(tensor)
    return view, parent


def _make_csc_inputs(
    size, nnz, index_dtype, value_dtype, value_range, dense_ndim=0, layout="plain"
):
    # values has batch + nnz + dense axes, i.e. len(size) - 1 dimensions.
    ccol, row = _csc_indices(size, nnz, index_dtype, dense_ndim)
    batch, _, _, dense = _split_size(size, dense_ndim)
    values = tu.make_input(value_dtype, batch + (nnz,) + dense, value_range)
    parents = ()
    if layout == "offset_indices":
        ccol, ccol_parent = _offset_last_dim(ccol, 3)
        row, row_parent = _offset_last_dim(row, 3)
        parents = (ccol_parent, row_parent)
    elif layout == "strided_values":
        # Only the index tensors must be dense in their last dimension; values
        # may be a strided view.
        values, values_parent = _strided_last_dim(values, 2)
        parents = (values_parent,)
    elif layout != "plain":
        raise ValueError("unknown layout " + layout)
    return ccol, row, values, list(size), parents


def _large_row_inputs(index_dtype, row_coordinate):
    # ccol = [0, 1] with one stored entry, so the structure is valid without any
    # dense allocation for the 2**31+5 logical rows. Both index tensors are
    # allocated directly in the requested dtype.
    ccol = torch.tensor([0, 1], dtype=index_dtype, device=flag_gems.device)
    row = torch.tensor([row_coordinate], dtype=index_dtype, device=flag_gems.device)
    values = torch.ones(1, dtype=torch.float32, device=flag_gems.device)
    return ccol, row, values, list(LARGE_ROW_SIZE)


def _snapshot(*tensors):
    # Independent copies of the operands (and of the storage a view was taken
    # from) together with the view geometry the validator must not rewrite.
    return [
        (
            tensor,
            tu.to_reference(tensor),
            (tensor.shape, tensor.stride(), tensor.storage_offset()),
            tensor.device,
        )
        for tensor in tensors
    ]


def _assert_untouched(snapshot):
    for tensor, reference, geometry, required_device in snapshot:
        _assert_result_equal(tensor, reference, required_device)
        assert (tensor.shape, tensor.stride(), tensor.storage_offset()) == geometry


_MASTER_CSC_SHAPES = [
    (0, 3, (), ()),
    (3, 0, (), ()),
    (3, 5, (), ()),
    (3, 5, (2,), ()),
    (3, 5, (2, 3), ()),
    (3, 5, (0,), ()),
    (3, 5, (), (2,)),
    (3, 5, (2,), (2,)),
]


def _master_csc_inputs(shape, index_dtype, dtype, *, strided_batch=False):
    ncols, nrows, batch, dense = shape
    per_column = min(nrows, 2)
    nnz = ncols * per_column
    ccol = torch.arange(ncols + 1, dtype=torch.int64) * per_column
    row = torch.arange(per_column, dtype=torch.int64).repeat(ncols)
    ccol = _index_to_device(ccol.expand(*batch, ncols + 1).clone(), index_dtype)
    row = _index_to_device(row.expand(*batch, nnz).clone(), index_dtype)
    if strided_batch and batch:
        ccol = torch.stack((ccol, ccol), dim=-2)[..., 0, :]
        row = torch.stack((row, row), dim=-2)[..., 0, :]
    values = torch.zeros((*batch, nnz, *dense), dtype=dtype, device=ccol.device)
    return ccol, row, values, (*batch, nrows, ncols, *dense)


def _master_validator_kwargs(op):
    # On older ATen builds this schema has four arguments. All fixtures are
    # ordinary unpinned tensors on the same device, so the implicit pinning
    # policy does not change their validity.
    if any(argument.name == "check_pinning" for argument in op._schema.arguments):
        return {"check_pinning": False}
    return {}


@pytest.mark.underscore_validate_sparse_csc_tensor_args
@pytest.mark.parametrize(
    "index_dtype",
    [
        dtype
        for dtype in utils.ALL_INT_DTYPES
        if dtype in (torch.int32, torch.int64) and _dtype_supported(dtype)
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("shape", _MASTER_CSC_SHAPES)
@pytest.mark.parametrize("strided_batch", [False, True])
def test_validate_sparse_csc_tensor_args_master_grid(
    index_dtype, shape, strided_batch, dtype
):
    *inputs, size = _master_csc_inputs(
        shape, index_dtype, dtype, strided_batch=strided_batch
    )
    snapshot = _snapshot(*inputs)
    op = torch.ops.aten._validate_sparse_csc_tensor_args.default
    options = _master_validator_kwargs(op)
    reference = op(*(entry[1] for entry in snapshot), size, **options)
    result = _call_validator(
        *inputs, size, required_device=_required_device(), **options
    )
    assert result is reference is None
    flag_gems.runtime.torch_device_fn.synchronize()
    _assert_untouched(snapshot)


@pytest.mark.underscore_validate_sparse_csc_tensor_args
def test_validate_sparse_csc_tensor_args_unbatched_metadata():
    ccol, row, values, size = _master_csc_inputs(
        (2, 3, (), ()), INDEX_DTYPE_LONG, torch.float32
    )
    ccol = ccol[:-1]
    op = torch.ops.aten._validate_sparse_csc_tensor_args.default
    options = _master_validator_kwargs(op)
    ref_inputs = tuple(tu.to_reference(tensor) for tensor in (ccol, row, values))
    with pytest.raises(RuntimeError):
        op(*ref_inputs, size, **options)
    with pytest.raises(RuntimeError):
        flag_gems._validate_sparse_csc_tensor_args(ccol, row, values, size, **options)


def _invalid_csc_inputs(defect, size, nnz):
    # Build a structurally valid CSC call and then break exactly one host-side
    # precondition, named by defect.
    ccol, row, values, call_size, _ = _make_csc_inputs(
        size, nnz, INDEX_DTYPE_LONG, torch.float32, ["-1", "1"]
    )
    if defect == "size_rank":
        return ccol, row, values, call_size + [2]
    if defect == "ccol_rank_zero":
        zero_dim = torch.tensor(0, dtype=INDEX_DTYPE_LONG, device=flag_gems.device)
        return zero_dim, row, values, call_size
    if defect == "ccol_row_rank":
        return ccol, row.flatten(), values, call_size
    if defect == "values_rank":
        return ccol, row, values.flatten(), call_size
    if defect == "row_batch_mismatch":
        return ccol, row[:-1], values, call_size
    if defect == "values_batch_mismatch":
        return ccol, row, values[:-1], call_size
    if defect == "ccol_length":
        return ccol[..., :-2], row, values, call_size
    if defect == "row_length":
        return ccol, row[..., :-1], values, call_size
    if defect == "ccol_stride":
        return _strided_last_dim(ccol, 2)[0], row, values, call_size
    if defect == "row_stride":
        return ccol, _strided_last_dim(row, 2)[0], values, call_size
    if defect == "index_dtype_mismatch":
        return ccol, row.to(torch.int32), values, call_size
    if defect == "index_dtype_unsupported":
        return ccol.to(torch.float32), row.to(torch.float32), values, call_size
    if defect == "meta_nnz":
        # The meta branch is reachable only when all three tensors are on meta,
        # and it requires no stored entries: the only meta workload the native
        # contract rejects.
        return ccol.to("meta"), row.to("meta"), values.to("meta"), call_size
    raise AssertionError("unhandled defect " + defect)


def _schema_defect_args(defect, ccol, row, values, call_size):
    if defect == "size_missing":
        return (ccol, row, values)
    if defect == "ccol_not_tensor":
        return (call_size, row, values, call_size)
    if defect == "size_not_int_list":
        return (ccol, row, values, "44")
    if defect == "size_float_entry":
        return (ccol, row, values, [4, 4.0])
    raise AssertionError("unhandled defect " + defect)


@pytest.mark.underscore_validate_sparse_csc_tensor_args
@pytest.mark.parametrize(
    "dtype",
    [pytest.param(dtype, marks=_input_marks(dtype)) for dtype in VALUE_DTYPES],
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize(
    "size",
    [
        pytest.param(size, marks=_ASCEND_VALUES_RANK_SKIP if len(size) - 1 > 8 else ())
        for size in SIZE_SHAPES
    ],
)
def test__validate_sparse_csc_tensor_args(size, value_range, dtype):
    # One stored entry per column of every matrix in the batch, always within the
    # nrows * ncols capacity.
    required_device = _required_device()
    nnz = size[-1]
    ccol, row, values, call_size, parents = _make_csc_inputs(
        size, nnz, INDEX_DTYPE_LONG, dtype, value_range
    )
    snapshot = _snapshot(ccol, row, values, *parents)

    torch.ops.aten._validate_sparse_csc_tensor_args(
        snapshot[0][1], snapshot[1][1], snapshot[2][1], call_size
    )
    res_out = _call_validator(
        ccol, row, values, call_size, required_device=required_device
    )

    # The schema returns (), so the only comparable value is the returned None.
    assert res_out is None
    # A validator must not modify the tensors it inspects.
    _assert_untouched(snapshot)


@pytest.mark.underscore_validate_sparse_csc_tensor_args
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize(
    "size,nnz,dense_ndim,index_dtype,layout",
    [
        pytest.param(
            size,
            nnz,
            dense_ndim,
            index_dtype,
            layout,
            marks=_ASCEND_VALUES_RANK_SKIP if len(size) - 1 > 8 else (),
        )
        for size, nnz, dense_ndim, index_dtype, layout in STRUCTURE_CASES
    ],
)
def test__validate_sparse_csc_tensor_args_structure(
    size, nnz, dense_ndim, index_dtype, layout, value_range
):
    required_device = _required_device()
    ccol, row, values, call_size, parents = _make_csc_inputs(
        size, nnz, index_dtype, torch.float32, value_range, dense_ndim, layout
    )
    snapshot = _snapshot(ccol, row, values, *parents)

    torch.ops.aten._validate_sparse_csc_tensor_args(
        snapshot[0][1], snapshot[1][1], snapshot[2][1], call_size
    )
    res_out = _call_validator(
        ccol, row, values, call_size, required_device=required_device
    )

    assert res_out is None
    _assert_untouched(snapshot)


@pytest.mark.underscore_validate_sparse_csc_tensor_args
@pytest.mark.parametrize(
    "dtype,scenario",
    [
        pytest.param(dtype, scenario, marks=_input_marks(dtype))
        for dtype, scenario in SPECIAL_CASES
    ],
)
def test__validate_sparse_csc_tensor_args_special_values(dtype, scenario):
    required_device = _required_device()
    values = tu.make_special_input(dtype, scenario)
    ccol, row = _csc_indices(SPECIAL_SIZE, values.numel(), INDEX_DTYPE_LONG)
    snapshot = _snapshot(ccol, row, values)

    torch.ops.aten._validate_sparse_csc_tensor_args(
        snapshot[0][1], snapshot[1][1], snapshot[2][1], list(SPECIAL_SIZE)
    )
    res_out = _call_validator(
        ccol, row, values, list(SPECIAL_SIZE), required_device=required_device
    )

    assert res_out is None
    _assert_untouched(snapshot)


@pytest.mark.underscore_validate_sparse_csc_tensor_args
@pytest.mark.parametrize(
    "defect,size,nnz",
    [
        pytest.param(
            defect,
            size,
            nnz,
            marks=_ASCEND_VALUES_RANK_SKIP if len(size) - 1 > 8 else (),
        )
        for defect, size, nnz in INVALID_ROWS
    ],
)
def test__validate_sparse_csc_tensor_args_invalid(defect, size, nnz):
    ccol, row, values, call_size = _invalid_csc_inputs(defect, size, nnz)
    # Only the candidate is called: the native result is the reference contract,
    # not a second assertion.
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._validate_sparse_csc_tensor_args(ccol, row, values, call_size)


_CSC_INDEX_ERRORS = {
    "endpoint": r"ccol_indices.*== 0",
    "bounds": r"0 <= row_indices < nrows",
    "duplicate": r"sorted and distinct",
}


def _csc_index_inputs(case, device):
    ccol = torch.tensor([0, 2, 3], dtype=torch.int64, device=device)
    row = torch.tensor([0, 1, 0], dtype=torch.int64, device=device)
    if case == "endpoint":
        ccol[0] = 1
    elif case == "bounds":
        row[0] = 3
    elif case == "duplicate":
        row[1] = row[0]
    return ccol, row, torch.zeros(3, device=device), (3, 2)


def _csc_child_override(request):
    from flag_gems.cli_override import load_override_config, parse_override_spec

    options = request.config.option
    configured = getattr(options, "override_config", None)
    overrides = load_override_config(configured) if configured else {}
    for spec in getattr(options, "override", None) or ():
        name, path, symbol = parse_override_spec(spec)
        overrides[name] = (path, symbol)
    selected = overrides.get("_validate_sparse_csc_tensor_args")
    if selected is None:
        return None
    path, symbol = selected
    source = Path(path).resolve()
    return str(source), symbol, hashlib.sha256(source.read_bytes()).hexdigest()


def _csc_candidate_binding():
    candidate = inspect.unwrap(flag_gems._validate_sparse_csc_tensor_args)
    code = getattr(candidate, "__code__", None)
    return {
        "name": getattr(candidate, "__qualname__", None),
        "file": code.co_filename if code else None,
        "line": code.co_firstlineno if code else None,
        "schema": str(getattr(candidate, "_schema", "")),
    }


def _run_csc_index_child(case, override):
    # Device assertions may poison or terminate the process. Keep the malformed
    # call in a fresh interpreter and forward only this operator's override.
    child = textwrap.dedent("""
        import hashlib
        import json
        import resource
        import sys
        from pathlib import Path

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        import flag_gems
        from flag_gems.dynamic_registry import DynamicOpOverride
        from tests import conftest, accuracy_utils
        from tests.test__validate_sparse_csc_tensor_args import (
            _csc_candidate_binding, _csc_index_inputs, _required_device,
        )

        args = json.loads(sys.argv[1])
        conftest.TO_CPU = accuracy_utils.TO_CPU = args["ref_cpu"]
        key = "_validate_sparse_csc_tensor_args"
        registry = DynamicOpOverride()
        if args["override"] is not None:
            path, symbol, digest = args["override"]
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
            assert registry.override_from_file(key, path, symbol)
        assert _csc_candidate_binding() == args["binding"]
        candidate = flag_gems._validate_sparse_csc_tensor_args
        device = _required_device()
        assert candidate(*_csc_index_inputs("valid", device)) is None
        flag_gems.runtime.torch_device_fn.synchronize()

        operands = _csc_index_inputs(args["case"], device)
        try:
            candidate(*operands)
            flag_gems.runtime.torch_device_fn.synchronize()
        except RuntimeError as error:
            if args["override"] is not None:
                assert registry.call_counts().get(f"flag_gems.{key}") == 2
            print("CSC_INDEX_ERROR " + json.dumps(str(error)))
        else:
            raise AssertionError("invalid CSC index content was silently accepted")
        """)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root / "src"), *sys.path]))
    args = json.dumps(
        dict(
            case=case,
            override=override,
            ref_cpu=utils.TO_CPU,
            binding=_csc_candidate_binding(),
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", child, args],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    prefix = "CSC_INDEX_ERROR "
    reports = [
        line[len(prefix) :]
        for line in result.stdout.splitlines()
        if line.startswith(prefix)
    ]
    assert len(reports) == 1, result.stdout[-2000:]
    return json.loads(reports[0])


@pytest.mark.underscore_validate_sparse_csc_tensor_args
@pytest.mark.parametrize("case", tuple(_CSC_INDEX_ERRORS))
def test_validate_sparse_csc_tensor_args_invalid_indices(case, request):
    with pytest.raises(RuntimeError, match=_CSC_INDEX_ERRORS[case]):
        torch.ops.aten._validate_sparse_csc_tensor_args.default(
            *_csc_index_inputs(case, "cpu")
        )

    # The legal control is observable by parent dummy checks. The child below
    # separately proves that the real candidate rejects the malformed inputs.
    device = _required_device()
    result = _call_validator(
        *_csc_index_inputs("valid", device), required_device=device
    )
    assert result is None
    flag_gems.runtime.torch_device_fn.synchronize()

    override = _csc_child_override(request)
    message = _run_csc_index_child(case, override)
    pattern = _CSC_INDEX_ERRORS[case]
    if override is None:
        pattern += "|invalid sparse compressed indices"
    assert re.search(pattern, message), message


@pytest.mark.underscore_validate_sparse_csc_tensor_args
@pytest.mark.parametrize("defect", SCHEMA_DEFECTS)
def test__validate_sparse_csc_tensor_args_schema_defects(defect):
    ccol, row, values, call_size, _ = _make_csc_inputs(
        (4, 4), 4, INDEX_DTYPE_LONG, torch.float32, ["-1", "1"]
    )
    args = _schema_defect_args(defect, ccol, row, values, call_size)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._validate_sparse_csc_tensor_args(*args)


@pytest.mark.underscore_validate_sparse_csc_tensor_args
@pytest.mark.parametrize(
    "index_dtype", tu.selected_cases([torch.int64] if HAS_INT64_INDEX else [], quick=[])
)
def test__validate_sparse_csc_tensor_args_large_row_coordinate(index_dtype):
    required_device = _required_device()
    ccol, row, values, call_size = _large_row_inputs(index_dtype, LARGE_ROW_COORDINATE)
    snapshot = _snapshot(ccol, row, values)

    torch.ops.aten._validate_sparse_csc_tensor_args(
        snapshot[0][1], snapshot[1][1], snapshot[2][1], call_size
    )
    res_out = _call_validator(
        ccol, row, values, call_size, required_device=required_device
    )

    assert res_out is None
    _assert_untouched(snapshot)


@pytest.mark.underscore_validate_sparse_csc_tensor_args
def test__validate_sparse_csc_tensor_args_row_dimension_overflow():
    # int32 indices cannot describe the 2**31+5 row dimension, so the native
    # operator rejects the call on the host, before any kernel is launched.
    ccol, row, values, call_size = _large_row_inputs(torch.int32, 3)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._validate_sparse_csc_tensor_args(ccol, row, values, call_size)
