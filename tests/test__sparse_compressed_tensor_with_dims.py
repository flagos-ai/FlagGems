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

"""Correctness tests for ``aten::_sparse_compressed_tensor_with_dims``.

The operator is an allocation factory: it takes no tensor operand and returns a
sparse compressed tensor whose compressed indices and values stay uninitialized.
Its contract is the requested structure -- layout, logical shape, part
shapes/dtypes/contiguity, ``nnz`` and device -- together with the caller's
ability to write into every part, which is what these tests check. The value
range, broadcast, backward and positive NaN/Inf grids do not apply here: no
operand carries data, and the undefined contents of a fresh allocation are never
compared.
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu

_CSR = torch.sparse_csr
_CSC = torch.sparse_csc
_BSR = torch.sparse_bsr
_BSC = torch.sparse_bsc
_ROWWISE = (_CSR, _BSR)

# The nine required dtypes plus bool and float64; the probed native operator
# accepted all of them for the value part. Complex is accepted natively too but
# is outside the requested set.
_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.float64,
]
_INDEX_DTYPES = [torch.int32, torch.int64]

# (layout, dense_dim, size, blocksize, nnz).
#
# ``size`` always carries the two sparse dims, so it must be at least rank 2:
# native rejects rank 0/1 sizes with "dimensionality must be at least
# dense_dim(=0) + sparse_dim(=2)", and the 1-D spec shape is therefore covered
# through ``(256, 4)``. ``dense_dim`` counts the extra dense dims beyond the
# compressed structure (0 for the block layouts, which take an explicit
# ``blocksize`` instead).
_ALLOCATION_ROWS = [
    (_CSR, 0, (5, 4), (), 3),
    (_CSR, 0, (256, 4), (), 3),  # 1-D spec shape
    (_CSR, 0, (1024, 1024), (), 3),
    (_CSR, 0, (20, 320, 15), (), 3),
    (_CSR, 0, (16, 128, 64, 60), (), 3),
    (_CSR, 0, (16, 7, 57, 32, 29), (), 3),
    (_CSR, 0, (2, 5, 4), (), 3),  # batched
    (_CSR, 1, (5, 4, 3), (), 3),  # one dense tail dim
    (_CSR, 2, (5, 4, 3, 2), (), 3),  # two dense tail dims
    (_CSR, 0, (0, 4), (), 0),  # empty rows
    (_CSR, 0, (5, 0), (), 0),  # empty columns
    (_CSR, 0, (1, 1), (), 1),  # singleton
    (_CSC, 0, (5, 4), (), 3),
    (_CSC, 0, (2, 4, 5), (), 3),
    (_BSR, 0, (6, 4), (2, 2), 3),
    (_BSR, 0, (6, 4), (1, 1), 3),
    (_BSR, 0, (1024, 1024), (32, 32), 3),
    (_BSR, 0, (2, 6, 4), (2, 2), 3),
    (_BSR, 0, (16, 128, 64, 60), (2, 2), 3),
    (_BSC, 0, (4, 6), (2, 3), 3),
    (_BSC, 0, (2, 4, 6), (2, 3), 3),
]
# Quick keeps a small representative of every cheap structural branch; only the
# larger shape scale and the nnz sweep stay default-only.
_QUICK_ALLOCATION_ROWS = [
    (_CSR, 0, (2, 19, 7), (), 3),
    (_CSC, 0, (2, 19, 7), (), 3),
    (_BSR, 0, (2, 18, 6), (2, 3), 3),
    (_BSC, 0, (2, 18, 6), (2, 3), 3),
    (_CSR, 0, (2, 5, 4), (), 3),
    (_CSR, 1, (2, 4, 3), (), 3),
    (_CSR, 2, (2, 4, 3, 2), (), 2),
    (_CSR, 0, (0, 4), (), 0),
    (_CSR, 0, (5, 0), (), 0),
    (_CSR, 0, (1, 1), (), 1),
]
_ALLOCATION_CASES = tu.selected_cases(
    _ALLOCATION_ROWS
    + [row for row in _QUICK_ALLOCATION_ROWS if row not in _ALLOCATION_ROWS],
    quick=_QUICK_ALLOCATION_ROWS,
)

# One row per layout; these exercise the storage roles of the three parts.
_LAYOUT_ROWS = [
    (_CSR, 0, (5, 4), (), 3),
    (_CSC, 0, (5, 4), (), 3),
    (_BSR, 0, (6, 4), (2, 2), 3),
    (_BSC, 0, (4, 6), (2, 3), 3),
]
_LAYOUT_CASES = tu.selected_cases(_LAYOUT_ROWS, quick=_LAYOUT_ROWS)

# Structures written into both allocations before densifying; block rows keep
# nnz below the block capacity so the written entries stay distinct.
_WRITE_ROWS = [
    (_CSR, 0, (5, 4), (), 3),
    (_CSR, 0, (2, 5, 4), (), 3),
    (_CSR, 1, (5, 4, 3), (), 3),
    (_CSC, 0, (5, 4), (), 3),
    (_BSR, 0, (6, 4), (2, 2), 2),
    (_BSC, 0, (4, 6), (2, 3), 2),
]
_QUICK_WRITE_ROWS = _WRITE_ROWS
_WRITE_CASES = tu.selected_cases(_WRITE_ROWS, quick=_QUICK_WRITE_ROWS)

_NNZ_VALUES = [0, 1, 7, 20]
# Both cheap pin-memory argument branches remain in quick.
_PIN_VALUES = [False, True]

_NO_LAYOUT = object()  # sentinel: omit the required ``layout`` keyword


def _invalid(
    case_id, nnz, dense_dim, size, blocksize, index_dtype, exc, layout=_CSR, **kwargs
):
    """One rejected call: positional schema args, keywords, expected error."""
    args = (nnz, dense_dim, list(size), list(blocksize), index_dtype)
    if layout is _NO_LAYOUT:
        return pytest.param(args, dict(kwargs), exc, id=case_id)
    kwargs["layout"] = layout
    return pytest.param(args, kwargs, exc, id=case_id)


# Every negative row is collected in both quick and default mode.
_INVALID_CALLS = [
    _invalid("nnz_negative", -1, 0, (5, 4), (), torch.int64, RuntimeError),
    _invalid("nnz_float", 1.5, 0, (5, 4), (), torch.int64, (TypeError, RuntimeError)),
    _invalid(
        "nnz_nan",
        float("nan"),
        0,
        (5, 4),
        (),
        torch.int64,
        (TypeError, RuntimeError),
    ),
    _invalid(
        "nnz_inf",
        float("inf"),
        0,
        (5, 4),
        (),
        torch.int64,
        (TypeError, RuntimeError),
    ),
    _invalid(
        "dense_dim_negative",
        2,
        -1,
        (5, 4),
        (),
        torch.int64,
        (ValueError, RuntimeError),
    ),
    _invalid("dense_dim_beyond_rank", 2, 3, (5, 4), (), torch.int64, RuntimeError),
    _invalid(
        "dense_dim_without_dense_dims", 2, 1, (5, 4), (), torch.int64, RuntimeError
    ),
    _invalid("size_rank_1", 2, 0, (5,), (), torch.int64, RuntimeError),
    _invalid("size_rank_0", 2, 0, (), (), torch.int64, RuntimeError),
    _invalid(
        "size_not_integer", 2, 0, (5.5, 4), (), torch.int64, (TypeError, RuntimeError)
    ),
    # ``blocksize`` is block-layout only, must have length 2 and divide both
    # sparse dims.
    _invalid("blocksize_on_csr", 2, 0, (5, 4), (1, 1), torch.int64, RuntimeError),
    _invalid(
        "blocksize_length_1",
        2,
        0,
        (6, 4),
        (2,),
        torch.int64,
        RuntimeError,
        layout=_BSR,
    ),
    _invalid(
        "blocksize_length_3",
        2,
        0,
        (6, 4),
        (2, 2, 2),
        torch.int64,
        RuntimeError,
        layout=_BSR,
    ),
    _invalid(
        "blocksize_rows_not_divisor",
        2,
        0,
        (5, 4),
        (2, 2),
        torch.int64,
        RuntimeError,
        layout=_BSR,
    ),
    _invalid(
        "blocksize_cols_not_divisor",
        2,
        0,
        (6, 5),
        (2, 2),
        torch.int64,
        RuntimeError,
        layout=_BSR,
    ),
    # Only the four compressed layouts are accepted and ``layout`` is required.
    _invalid(
        "layout_sparse_coo",
        2,
        0,
        (5, 4),
        (),
        torch.int64,
        RuntimeError,
        layout=torch.sparse_coo,
    ),
    _invalid(
        "layout_strided",
        2,
        0,
        (5, 4),
        (),
        torch.int64,
        RuntimeError,
        layout=torch.strided,
    ),
    _invalid(
        "layout_missing", 2, 0, (5, 4), (), torch.int64, RuntimeError, layout=_NO_LAYOUT
    ),
    # The index parts are Int/Long only.
    _invalid("index_dtype_int16", 2, 0, (5, 4), (), torch.int16, RuntimeError),
    _invalid("index_dtype_int8", 2, 0, (5, 4), (), torch.int8, RuntimeError),
    _invalid("index_dtype_float32", 2, 0, (5, 4), (), torch.float32, RuntimeError),
    _invalid("index_dtype_bool", 2, 0, (5, 4), (), torch.bool, RuntimeError),
]
if torch.device(flag_gems.device).type == "cuda":
    # Pinned storage is a CPU-only contract; on the probed CUDA backend a CUDA
    # device with ``pin_memory=True`` is rejected with
    # "Only dense CPU tensors can be pinned".
    _INVALID_CALLS.append(
        _invalid(
            "pin_memory_non_cpu",
            2,
            0,
            (5, 4),
            (),
            torch.int64,
            RuntimeError,
            device=flag_gems.device,
            pin_memory=True,
        )
    )


def _parts(tensor):
    """Return the ``(compressed, other, values)`` parts of any layout."""
    if tensor.layout in _ROWWISE:
        return tensor.crow_indices(), tensor.col_indices(), tensor.values()
    return tensor.ccol_indices(), tensor.row_indices(), tensor.values()


def _matches_device(actual, requested):
    """Compare an allocated tensor's device against the requested device.

    ``torch.device('cuda') != torch.device('cuda:0')``, and an allocation made
    for ``'cuda'`` lands on ``cuda:0``, so the device type is what is compared;
    the index is additionally required whenever the request names one.
    """
    expected = torch.device(requested)
    if actual.type != expected.type:
        return False
    return expected.index is None or expected.index == actual.index


def _expected_part_shapes(layout, size, blocksize, nnz, dense_dim):
    k = len(size) - dense_dim
    batch = tuple(size[: k - 2])
    nrows, ncols = size[k - 2], size[k - 1]
    dense = tuple(size[k:])
    block = tuple(blocksize)
    entries = batch + (nnz,)
    if layout == _CSR:
        return batch + (nrows + 1,), entries, entries + dense
    if layout == _CSC:
        return batch + (ncols + 1,), entries, entries + dense
    if layout == _BSR:
        return batch + (nrows // block[0] + 1,), entries, entries + block
    return batch + (ncols // block[1] + 1,), entries, entries + block


def _assert_allocation(res, ref, case, dtype, device, index_dtype=None, check_nnz=True):
    layout, dense_dim, size, blocksize, nnz = case
    expected = _expected_part_shapes(layout, size, blocksize, nnz, dense_dim)
    for tensor in (res, ref):
        comp, other, values = _parts(tensor)
        assert tensor.layout == layout
        assert tuple(tensor.shape) == tuple(size)
        assert tensor.dense_dim() == dense_dim
        if check_nnz:
            assert torch.ops.aten._nnz(tensor) == nnz
        assert (tuple(comp.shape), tuple(other.shape), tuple(values.shape)) == expected
        assert values.dtype == dtype
        assert values.is_contiguous()
        assert _matches_device(tensor.device, device)
    res_comp, res_other, _ = _parts(res)
    ref_comp, ref_other, _ = _parts(ref)
    assert res_comp.dtype == ref_comp.dtype
    assert res_other.dtype == ref_other.dtype
    if index_dtype is not None:
        assert ref_comp.dtype == index_dtype
        assert ref_other.dtype == index_dtype


def _fill_canonical(tensor):
    """Write one deterministic, index-valid structure into an allocation."""
    comp, other, values = _parts(tensor)
    nnz = other.shape[-1]
    k = len(tensor.shape) - tensor.dense_dim()
    nrows, ncols = tensor.shape[k - 2], tensor.shape[k - 1]
    # All entries live in the first compressed line, which is valid for any nnz.
    comp.zero_()
    comp[..., 1:] = nnz
    if tensor.layout == _CSR:
        limit = ncols
    elif tensor.layout == _CSC:
        limit = nrows
    else:
        block_rows, block_cols = values.shape[-2:]
        limit = ncols // block_cols if tensor.layout == _BSR else nrows // block_rows
    index = torch.arange(nnz, device=other.device) % max(limit, 1)
    other.copy_(index.expand_as(other))
    values.fill_(1)


@pytest.mark.sparse_compressed_tensor_with_dims
@pytest.mark.parametrize("case", _ALLOCATION_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_allocation_structure(case, dtype):
    layout, dense_dim, size, blocksize, nnz = case
    ref = torch.ops.aten._sparse_compressed_tensor_with_dims(
        nnz,
        dense_dim,
        list(size),
        list(blocksize),
        torch.int64,
        dtype=dtype,
        layout=layout,
        device=flag_gems.device,
    )
    res = flag_gems._sparse_compressed_tensor_with_dims(
        nnz,
        dense_dim,
        list(size),
        list(blocksize),
        torch.int64,
        dtype=dtype,
        layout=layout,
        device=flag_gems.device,
    )
    _assert_allocation(res, ref, case, dtype, flag_gems.device, index_dtype=torch.int64)


@pytest.mark.sparse_compressed_tensor_with_dims
@pytest.mark.parametrize("case", _LAYOUT_CASES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
def test_index_dtype(case, index_dtype):
    layout, dense_dim, size, blocksize, nnz = case
    ref = torch.ops.aten._sparse_compressed_tensor_with_dims(
        nnz,
        dense_dim,
        list(size),
        list(blocksize),
        index_dtype,
        dtype=torch.float32,
        layout=layout,
        device=flag_gems.device,
    )
    res = flag_gems._sparse_compressed_tensor_with_dims(
        nnz,
        dense_dim,
        list(size),
        list(blocksize),
        index_dtype,
        dtype=torch.float32,
        layout=layout,
        device=flag_gems.device,
    )
    # The native result is the oracle for the dtype the index parts get.
    _assert_allocation(res, ref, case, torch.float32, flag_gems.device)


@pytest.mark.sparse_compressed_tensor_with_dims
@pytest.mark.parametrize("nnz", _NNZ_VALUES)
def test_requested_nnz_is_allocated(nnz):
    size = (5, 4)
    ref = torch.ops.aten._sparse_compressed_tensor_with_dims(
        nnz,
        0,
        list(size),
        [],
        torch.int64,
        dtype=torch.float32,
        layout=_CSR,
        device=flag_gems.device,
    )
    res = flag_gems._sparse_compressed_tensor_with_dims(
        nnz,
        0,
        list(size),
        [],
        torch.int64,
        dtype=torch.float32,
        layout=_CSR,
        device=flag_gems.device,
    )
    _assert_allocation(
        res,
        ref,
        (_CSR, 0, size, (), nnz),
        torch.float32,
        flag_gems.device,
        index_dtype=torch.int64,
    )
    _, other, values = _parts(res)
    assert other.numel() == nnz
    assert values.numel() == nnz


@pytest.mark.sparse_compressed_tensor_with_dims
def test_optional_kwargs_default_to_schema():
    # Only ``layout`` is required; the schema defaults to CPU storage and
    # float32 values.
    ref = torch.ops.aten._sparse_compressed_tensor_with_dims(
        2, 0, [5, 4], [], torch.int64, layout=_CSR
    )
    res = flag_gems._sparse_compressed_tensor_with_dims(
        2, 0, [5, 4], [], torch.int64, layout=_CSR
    )
    _assert_allocation(
        res,
        ref,
        (_CSR, 0, (5, 4), (), 2),
        torch.float32,
        "cpu",
        index_dtype=torch.int64,
    )


@pytest.mark.sparse_compressed_tensor_with_dims
@pytest.mark.parametrize("pin_memory", _PIN_VALUES)
def test_cpu_pin_memory(pin_memory):
    # Pinning is a CPU contract, so this workload uses the CPU device form of
    # the call; the flag must be accepted in both states.
    ref = torch.ops.aten._sparse_compressed_tensor_with_dims(
        2,
        0,
        [5, 4],
        [],
        torch.int64,
        dtype=torch.float32,
        layout=_CSR,
        device="cpu",
        pin_memory=pin_memory,
    )
    res = flag_gems._sparse_compressed_tensor_with_dims(
        2,
        0,
        [5, 4],
        [],
        torch.int64,
        dtype=torch.float32,
        layout=_CSR,
        device="cpu",
        pin_memory=pin_memory,
    )
    _assert_allocation(
        res,
        ref,
        (_CSR, 0, (5, 4), (), 2),
        torch.float32,
        "cpu",
        index_dtype=torch.int64,
    )
    if pin_memory:
        assert _parts(res)[0].is_pinned()


@pytest.mark.sparse_compressed_tensor_with_dims
@pytest.mark.parametrize("case", _LAYOUT_CASES)
def test_meta_device_allocation(case):
    layout, dense_dim, size, blocksize, nnz = case
    ref = torch.ops.aten._sparse_compressed_tensor_with_dims(
        nnz,
        dense_dim,
        list(size),
        list(blocksize),
        torch.int64,
        dtype=torch.float32,
        layout=layout,
        device="meta",
    )
    res = flag_gems._sparse_compressed_tensor_with_dims(
        nnz,
        dense_dim,
        list(size),
        list(blocksize),
        torch.int64,
        dtype=torch.float32,
        layout=layout,
        device="meta",
    )
    # Meta storage holds no data, so the ``_nnz`` read is not meaningful there;
    # nnz is asserted on the data-backed devices and in the dedicated
    # nnz-boundary workload, and every other part shape is still compared.
    _assert_allocation(
        res,
        ref,
        case,
        torch.float32,
        "meta",
        index_dtype=torch.int64,
        check_nnz=False,
    )


@pytest.mark.sparse_compressed_tensor_with_dims
@pytest.mark.parametrize("case", _LAYOUT_CASES)
def test_allocation_parts_are_independent(case):
    layout, dense_dim, size, blocksize, nnz = case
    res = flag_gems._sparse_compressed_tensor_with_dims(
        nnz,
        dense_dim,
        list(size),
        list(blocksize),
        torch.int64,
        dtype=torch.float32,
        layout=layout,
        device=flag_gems.device,
    )
    comp, other, values = _parts(res)
    comp.fill_(1)
    other.fill_(2)
    values.fill_(3)
    # Writing one part must leave the other two untouched, i.e. the compressed
    # indices, the other indices and the values are three storages.
    assert torch.equal(comp, torch.full_like(comp, 1))
    assert torch.equal(other, torch.full_like(other, 2))
    assert torch.equal(values, torch.full_like(values, 3))
    assert len({comp.data_ptr(), other.data_ptr(), values.data_ptr()}) == 3


@pytest.mark.sparse_compressed_tensor_with_dims
@pytest.mark.parametrize("case", _WRITE_CASES)
def test_written_allocation_densifies_like_reference(case):
    layout, dense_dim, size, blocksize, nnz = case
    ref = torch.ops.aten._sparse_compressed_tensor_with_dims(
        nnz,
        dense_dim,
        list(size),
        list(blocksize),
        torch.int64,
        dtype=torch.float32,
        layout=layout,
        device=flag_gems.device,
    )
    res = flag_gems._sparse_compressed_tensor_with_dims(
        nnz,
        dense_dim,
        list(size),
        list(blocksize),
        torch.int64,
        dtype=torch.float32,
        layout=layout,
        device=flag_gems.device,
    )
    _fill_canonical(res)
    _fill_canonical(ref)
    tu.assert_result_equal(res.to_dense(), ref.to_dense())


@pytest.mark.sparse_compressed_tensor_with_dims
@pytest.mark.parametrize("args,kwargs,exc", _INVALID_CALLS)
def test_invalid_arguments_are_rejected(args, kwargs, exc):
    with pytest.raises(exc):
        flag_gems._sparse_compressed_tensor_with_dims(*args, **kwargs)
