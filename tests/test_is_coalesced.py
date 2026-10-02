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

import math
import warnings

import pytest
import torch

import flag_gems

from . import test_utils as tu

# aten::is_coalesced reports the coalesced bit recorded on a sparse COO tensor.
# It never sorts, merges or re-derives the coordinates and never writes the
# stored tensors, so a recorded True bit on out-of-order columns is still
# reported as True. Dense, CSR, CSC, BSR, BSC and non-tensor arguments raise
# RuntimeError.

_DTYPE_CAPABILITY = {
    torch.float64: "support_fp64",
    torch.int64: "support_int64",
    torch.bfloat16: "support_bf16",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name, False))


# The nine required dtypes plus float64 / int16 / bool: all of them can store
# sparse COO values, and none of them may influence a metadata query.
_DTYPES = [
    dtype
    for dtype in tu.REQUIRED_DTYPES
    + [torch.float64, torch.int16, torch.bool, torch.complex64]
    if _dtype_supported(dtype)
]

# Sparse sizes carry the shape dimension of this operator, so the spec's seven
# shapes stay as logical sizes and only the nnz stored elements are allocated
# (the dense shape grid would not change a metadata answer). Rows are
# (size, nnz, recorded bit); None lets the constructor derive the bit.
_MAIN_ROWS = [
    ((), 1, True),
    ((2, 19, 7), 19, False),
    ((2, 19, 7), 24, True),
    ((), 2, False),
    ((1,), 1, False),
    ((4,), 8, False),
    ((256,), 32, False),
    ((4, 4), 20, False),
    ((1024, 1024), 4096, False),
    ((20, 320, 15), 2000, False),
    ((16, 128, 64, 60), 4096, False),
    ((16, 7, 57, 32, 29), 1500, False),
    ((4, 4), 10, True),
    ((3, 5, 7), 60, True),
    ((16, 16), 200, True),
    ((2, 3, 4, 5), 100, True),
    ((1,), 1, True),
    ((5, 0), 0, None),
    ((0, 3), 0, None),
]

_MAIN_CASES = tu.selected_cases(
    _MAIN_ROWS,
    quick=[
        # Scalar, single-nnz, empty and zero-nnz rows, both recorded bits. The
        # dtype family is not reduced in quick.
        ((), 1, True),
        ((), 2, False),
        ((1,), 1, True),
        ((4,), 8, False),
        ((256,), 32, False),
        ((4, 4), 10, True),
        ((3, 5, 7), 60, True),
        ((2, 3, 4, 5), 100, True),
        ((1,), 1, False),
        ((5, 0), 0, None),
        ((0, 3), 0, None),
        ((2, 19, 7), 19, False),
        ((2, 19, 7), 24, True),
    ],
)

_RANGE_CASES = _MAIN_CASES

_STATE_CASES = tu.selected_cases(
    [((4, 4), 20, False), ((3, 5, 7), 60, True), ((1024, 1024), 4096, False)],
    quick=[((4, 4), 20, False), ((3, 5, 7), 60, True)],
)

_FLAG_CASES = tu.selected_cases(
    [((4, 4), 20), ((3, 5, 7), 60), ((1024, 1024), 4096)],
    quick=[((4, 4), 20), ((3, 5, 7), 60)],
)

# NaN/Inf payloads are default-only: a positive special-value smoke case is not
# part of the quick subset.
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_DTYPES), quick=[])
_SPECIAL_SIZES = [((3, 5, 7), 60, False), ((16, 16), 200, True)]


def _flat_positions(size, nnz, *, ordered):
    """Deterministic COO coordinates of shape (rank, nnz).

    ordered spaces the positions as i * numel // nnz, which is strictly
    increasing whenever nnz <= numel (the configuration every truthful True
    row uses). Otherwise the positions are drawn descending and wrapped modulo
    numel, so the columns are out of order and repeat once nnz exceeds numel.
    """
    if nnz == 0:
        return torch.empty((len(size), 0), dtype=torch.int64)
    if not size:
        # A 0-D sparse tensor has a single position, repeated for every column.
        return torch.empty((0, nnz), dtype=torch.int64)
    numel = math.prod(size)
    if ordered:
        assert nnz <= numel
        flat = torch.arange(nnz, dtype=torch.int64) * numel // nnz
    else:
        flat = torch.arange(nnz, dtype=torch.int64).flip(0) % numel
    indices = torch.empty((len(size), nnz), dtype=torch.int64)
    for dim in reversed(range(len(size))):
        indices[dim] = flat % size[dim]
        flat = flat // size[dim]
    return indices


def _make_sparse(size, nnz, dtype, *, recorded, ordered, values=None, value_range=None):
    indices = _flat_positions(size, nnz, ordered=ordered)
    if values is None:
        values = tu.make_input(dtype, (nnz,), list(value_range or ("-1", "1")))
    kwargs = {} if recorded is None else {"is_coalesced": recorded}
    return torch.sparse_coo_tensor(
        indices, values, tuple(size), device=flag_gems.device, **kwargs
    )


def _assert_flag(res_out, ref_out):
    # The schema returns a Python bool, not a 0-dim tensor: compare it directly.
    assert type(res_out) is bool, type(res_out)
    assert res_out == ref_out


@pytest.mark.is_coalesced
@pytest.mark.parametrize("size,nnz,recorded", _MAIN_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_is_coalesced(size, nnz, recorded, dtype):
    inp = _make_sparse(size, nnz, dtype, recorded=recorded, ordered=bool(recorded))
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.is_coalesced(ref_inp)
    res_out = flag_gems.is_coalesced(inp)

    _assert_flag(res_out, ref_out)
    # The recorded bit is reported and left in place.
    assert inp.is_coalesced() == ref_out


@pytest.mark.is_coalesced
@pytest.mark.parametrize("size,nnz,recorded", _RANGE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_is_coalesced_value_ranges(size, nnz, recorded, value_range, dtype):
    inp = _make_sparse(
        size, nnz, dtype, recorded=recorded, ordered=recorded, value_range=value_range
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.is_coalesced(ref_inp)
    res_out = flag_gems.is_coalesced(inp)

    _assert_flag(res_out, ref_out)
    assert inp.is_coalesced() == ref_out


@pytest.mark.is_coalesced
@pytest.mark.parametrize("size,nnz,recorded", _STATE_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_is_coalesced_keeps_input_state(size, nnz, recorded, dtype):
    inp = _make_sparse(size, nnz, dtype, recorded=recorded, ordered=recorded)
    # Snapshots sit on the configured reference device, the placement the shared
    # comparison helpers expect.
    indices_before = tu.to_reference(inp._indices())
    values_before = tu.to_reference(inp._values())
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.is_coalesced(ref_inp)
    res_out = flag_gems.is_coalesced(inp)

    _assert_flag(res_out, ref_out)
    # No reordering, deduplication, rewrap or flag rewrite is allowed.
    assert inp.layout == torch.sparse_coo
    assert inp._nnz() == nnz
    assert inp._indices().dtype == torch.int64
    tu.assert_result_equal(inp._indices(), indices_before)
    tu.assert_result_equal(inp._values(), values_before)
    assert inp.is_coalesced() == ref_out


@pytest.mark.is_coalesced
@pytest.mark.parametrize("size,nnz", _FLAG_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_is_coalesced_reports_recorded_bit(size, nnz, dtype):
    # Deliberately wrong bit: duplicated out-of-order coordinates recorded as
    # coalesced. The query must report the saved bit instead of recomputing it.
    inp = _make_sparse(size, nnz, dtype, recorded=True, ordered=False)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.is_coalesced(ref_inp)
    res_out = flag_gems.is_coalesced(inp)

    assert res_out is True
    assert inp.is_coalesced() is True
    _assert_flag(res_out, ref_out)


@pytest.mark.is_coalesced
@pytest.mark.parametrize("size,nnz,recorded", _SPECIAL_SIZES)
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_is_coalesced_special_values(size, nnz, recorded, dtype, scenario):
    payload = tu.make_special_input(dtype, scenario)
    values = payload.repeat((nnz + payload.numel() - 1) // payload.numel())[:nnz]
    inp = _make_sparse(
        size, nnz, dtype, recorded=recorded, ordered=recorded, values=values
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.is_coalesced(ref_inp)
    res_out = flag_gems.is_coalesced(inp)

    _assert_flag(res_out, ref_out)
    assert inp.is_coalesced() == ref_out


@pytest.mark.is_coalesced
@pytest.mark.parametrize("layout", ["strided", "csr", "csc", "bsr", "bsc"])
def test_is_coalesced_rejects_non_coo_layout(layout):
    dense = torch.randn((4, 4), dtype=torch.float32, device=flag_gems.device)
    with warnings.catch_warnings():
        # BSC is a beta layout and warns while it is built; the layout itself is
        # valid and is rejected by the operator.
        warnings.simplefilter("ignore")
        if layout == "strided":
            inp = dense
        elif layout == "csr":
            inp = dense.to_sparse_csr()
        elif layout == "csc":
            inp = dense.to_sparse_csc()
        elif layout == "bsr":
            inp = dense.to_sparse_bsr((2, 2))
        else:
            inp = dense.to_sparse_bsc((2, 2))

    with pytest.raises(RuntimeError):
        flag_gems.is_coalesced(inp)


@pytest.mark.is_coalesced
@pytest.mark.parametrize("bad_input", [3.14, [1, 2]])
def test_is_coalesced_rejects_non_tensor(bad_input):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_coalesced(bad_input)
