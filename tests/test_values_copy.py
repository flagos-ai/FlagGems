# Copyright 2026, The FlagOS Contributors.
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

"""Correctness tests for the sparse ``values_copy`` operator.

``values_copy`` is the non-aliasing counterpart of ``torch.values``: the result
is a freshly allocated, contiguous strided tensor holding a copy of a sparse
tensor's stored values, so every value assertion here is exact. Compressed
layouts (CSR, CSC, BSR, BSC and their batched forms) are accepted. The case
matrix mirrors the KernelGen acceptance suite so the reported speedup stays
comparable to its H20 measurement.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

# Sparse index metadata is built in the widest index dtype this backend
# declares support for, so no implicit int64 tensor is created on
# int32-only devices.
_INDEX_DTYPE = torch.int64 if utils.int64_is_supported else torch.int32

_LAYOUTS = {
    "csr": torch.sparse_csr_tensor,
    "csc": torch.sparse_csc_tensor,
    "bsr": torch.sparse_bsr_tensor,
    "bsc": torch.sparse_bsc_tensor,
}


def _capability_gated(dtypes):
    """Keep the dtypes this backend declares it supports (static device flags)."""
    gated = []
    for dtype in dtypes:
        if dtype in (torch.float64, torch.complex128) and not utils.fp64_is_supported:
            continue
        if dtype == torch.bfloat16 and not utils.bf16_is_supported:
            continue
        if dtype == torch.int64 and not utils.int64_is_supported:
            continue
        gated.append(dtype)
    return gated


_VALUE_DTYPES = _capability_gated(
    [
        torch.int8,
        torch.uint8,
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.int32,
        torch.int64,
    ]
    + [torch.int16, torch.complex64, torch.bool]
)
_RANGE_DTYPES = [dtype for dtype in _VALUE_DTYPES if dtype != torch.bool]
_SPECIAL_DTYPES = [dtype for dtype in _VALUE_DTYPES if dtype.is_floating_point]
# Complex gradients are natively unsupported for compressed sparse outputs
# (RuntimeError: "sparse_compressed_tensor does not support automatic
# differentiation for outputs with complex dtype"), so backward is float-only.
_BACKWARD_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
_OUT_BUFFER_DTYPES = [torch.float32, torch.float16]
# COO (SparseCUDA) and nested (NestedTensorCUDA) have no values_copy kernel on
# the NVIDIA dispatcher: NotImplementedError "Could not run 'aten::values_copy'
# with arguments from the 'SparseCUDA' backend" and "aten.values_copy.default".
# COO is gated on int64 support because torch stores COO indices as int64 no
# matter what dtype the constructor is given.
_REJECTION_ROWS = (
    ([("coo", torch.float32)] if utils.int64_is_supported else [])
    + [("nested", torch.float32)]
    if flag_gems.vendor_name == "nvidia"
    else []
)
# The out dtype must match the stored-values dtype (RuntimeError: "Expected out
# tensor to have dtype ..., but got ... instead").
_MISMATCHED_OUT_DTYPES = [torch.int32] + (
    [torch.float64] if utils.fp64_is_supported else []
)


def _dtype_bounds(dtype):
    """Return the (min, max) value bounds of ``dtype``."""
    if dtype == torch.bool:
        return 0, 1
    if dtype.is_complex:
        real = {
            torch.complex32: torch.float16,
            torch.complex64: torch.float32,
            torch.complex128: torch.float64,
        }[dtype]
        finfo = torch.finfo(real)
        return finfo.min, finfo.max
    if dtype.is_floating_point:
        finfo = torch.finfo(dtype)
        return finfo.min, finfo.max
    iinfo = torch.iinfo(dtype)
    return iinfo.min, iinfo.max


def _resolve_bound(symbol, dtype):
    """Resolve a range-bound symbol (-1 / 0 / 1 / max / min) for ``dtype``."""
    low, high = _dtype_bounds(dtype)
    table = {
        "-1": -1.0,
        "0": 0.0,
        "1": 1.0,
        "max": high,
        "min": low,
        "max/2": high / 2,
        "min/2": low / 2,
    }
    return table[symbol]


def _make_input(dtype, shape, value_range):
    """Build a dense tensor of ``shape`` / ``dtype`` within ``value_range``.

    Bounds are clamped to the dtype's representable range first, so unsigned
    dtypes that cannot hold ``-1`` degrade to a constant zero fill.
    """
    low = _resolve_bound(value_range[0], dtype)
    high = _resolve_bound(value_range[1], dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=flag_gems.device).bool()
    if not (dtype.is_floating_point or dtype.is_complex):
        low, high = int(low), int(high)
        dtype_min, dtype_max = _dtype_bounds(dtype)
        low = max(low, int(dtype_min))
        high = min(high, int(dtype_max))
        low = min(low, high)
    if low == high:
        return torch.full(shape, low, device=flag_gems.device, dtype=dtype)
    return torch.testing.make_tensor(
        shape, dtype=dtype, device=flag_gems.device, low=low, high=high
    )


_VALUE_RANGES = [
    ["-1", "1"],
    ["0", "1"],
    ["-1", "0"],
    ["0", "max"],
    ["min", "0"],
]


def _sparse_layout(layout, size, nnz, blocks, nbatch):
    """Validate one case row and return its index/values metadata.

    ``groups`` counts the compressed groups (rows for CSR/BSR, columns for
    CSC/BSC) and ``bound`` the exclusive plain-index bound inside a group, so
    ``groups * bound`` is the storage capacity of the layout. The values shape
    keeps the requested rank: batch axes, the nnz axis, the BSR/BSC block dims
    and the dense trailing dims in that order.
    """
    assert layout in _LAYOUTS, f"unknown sparse layout: {layout!r}"
    assert size and all(isinstance(extent, int) for extent in size), size
    assert nbatch >= 0 and len(size) >= nbatch + 2, (size, nbatch)
    batch = tuple(size[:nbatch])
    m, n = size[nbatch], size[nbatch + 1]
    dense = tuple(size[nbatch + 2 :])
    if layout in ("bsr", "bsc"):
        block_rows, block_cols = blocks
        assert m % block_rows == 0 and n % block_cols == 0, (layout, size, blocks)
        if layout == "bsr":
            groups, bound = m // block_rows, n // block_cols
        else:
            groups, bound = n // block_cols, m // block_rows
    elif layout == "csc":
        groups, bound = n, m
    else:
        groups, bound = m, n
    assert 0 <= nnz <= groups * bound, f"{layout} {size} cannot store {nnz} values"
    if groups == 0 or bound == 0:
        assert nnz == 0, f"{layout} {size} has no room for {nnz} values"
    values_shape = batch + (nnz,) + tuple(blocks or ()) + dense
    return batch, groups, bound, dense, values_shape


def _stored_shape(layout, size, nnz, blocks, nbatch, unsorted=False):
    del unsorted
    return _sparse_layout(layout, size, nnz, blocks, nbatch)[4]


def _sparse_indices(layout, size, nnz, blocks, nbatch, unsorted):
    """Build the compressed and plain index tensors of one case row on device."""
    batch, groups, bound, dense, _ = _sparse_layout(layout, size, nnz, blocks, nbatch)
    del bound, dense
    device = flag_gems.device
    index_dtype = _INDEX_DTYPE
    if groups == 0:
        # Zero-extent storage: one empty compressed-offset vector and no plain
        # index, which is the only well-formed empty structure of the layout.
        compressed = torch.zeros(1, dtype=index_dtype, device=device)
        plain = torch.empty(0, dtype=index_dtype, device=device)
    else:
        # Evenly spread the stored values over the groups, so the plain indices
        # stay distinct and inside ``bound`` even for a full-density row.
        counts = torch.full((groups,), nnz // groups, dtype=index_dtype, device=device)
        counts[: nnz % groups] += 1
        # cumsum promotes int32 to int64 by default, which would make every
        # index tensor wider than the declared dtype; pin it.
        compressed = torch.cat(
            [
                torch.zeros(1, dtype=index_dtype, device=device),
                counts.cumsum(0, dtype=index_dtype),
            ]
        )
        plain = torch.arange(nnz, device=device, dtype=index_dtype) - (
            torch.repeat_interleave(compressed[:-1], counts)
        )
        if unsorted:
            plain = torch.repeat_interleave(counts, counts) - 1 - plain
    if batch:
        compressed = compressed.expand(*batch, -1).contiguous()
        plain = plain.expand(*batch, -1).contiguous()
    return compressed, plain


def _make_sparse(*case, dtype, value_range=("-1", "1"), values=None):
    """Build the sparse input of one case row on the device of its values."""
    layout, size, nnz, blocks, nbatch, unsorted = case
    values_shape = _stored_shape(*case)
    if values is None:
        values = _make_input(dtype, values_shape, list(value_range))
    compressed, plain = _sparse_indices(layout, size, nnz, blocks, nbatch, unsorted)
    return _LAYOUTS[layout](
        compressed, plain, values, size=tuple(size), device=values.device
    )


def _assert_copy_result(res_out, ref_out, inp):
    """A fresh contiguous copy, never a view sharing the input values' storage."""
    if res_out.numel():
        assert res_out.untyped_storage().data_ptr() != (
            inp.values().untyped_storage().data_ptr()
        )
    assert res_out.is_contiguous()
    # equal_nan: the copy is bit-exact, but NaN payloads may differ between the
    # native reference copy and the Triton copy; the values still correspond.
    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


def _rejection_input(kind, dtype):
    """Build one measured-rejection input.

    A COO tensor always holds its indices as int64, whatever dtype the
    constructor receives, so this row is only selected where int64 exists.
    """
    if kind == "coo":
        values = _make_input(dtype, (2,), ["-1", "1"])
        indices = torch.tensor(
            [[0, 1], [1, 0]], dtype=torch.int64, device=flag_gems.device
        )
        return torch.sparse_coo_tensor(
            indices, values, size=(2, 2), device=flag_gems.device
        ).coalesce()
    return torch.nested.nested_tensor(
        [
            torch.zeros((2, 3), dtype=dtype, device=flag_gems.device),
            torch.zeros((2, 4), dtype=dtype, device=flag_gems.device),
        ]
    )


def _shape_to_row(shape):
    """Map one shape level onto a compressed-sparse case row.

    Stored values of a compressed layout are never 0-dim, so the 0-dim level
    uses a single stored value instead. The stored element count of each row
    equals the logical element count of the requested shape.
    """
    if len(shape) == 0:
        return ("csr", (1, 1), 1, None, 0, False)
    if len(shape) == 1:
        return ("csr", (shape[0], 1), shape[0], None, 0, False)
    if len(shape) == 2:
        return ("csr", tuple(shape), shape[0] * shape[1], None, 0, False)
    nbatch = 2 if len(shape) >= 5 else 1
    return (
        "csr",
        tuple(shape),
        shape[nbatch] * shape[nbatch + 1],
        None,
        nbatch,
        False,
    )


# Layout/rank/emptiness rows: stored values of rank 1..5, dense trailing dims,
# zero-nnz and zero-extent storage, BSR/BSC blocks, CSC, multiple batch axes
# and one row whose stored indices are unsorted.
_STORAGE_CASES = [
    ("csr", (5, 4), 6, None, 0, False),
    ("csr", (4, 1), 3, None, 0, False),
    ("csr", (1, 5), 2, None, 0, False),
    ("csr", (8, 8), 16, None, 0, False),
    ("csr", (256, 1), 256, None, 0, False),
    ("csr", (3, 3), 0, None, 0, False),
    ("csr", (16, 16, 64), 256, None, 1, False),
    ("csr", (32, 32, 4, 4), 128, None, 1, False),
    ("csr", (3, 3), 4, None, 0, True),
    ("csr", (2, 6, 8), 12, None, 1, False),
    ("csr", (2, 19, 1, 7), 19, None, 1, False),
    ("csr", (2, 3, 4, 6, 7), 24, None, 2, False),
    ("csc", (5, 4), 6, None, 0, False),
    ("csc", (2, 6, 8), 12, None, 1, False),
    ("bsr", (4, 6), 4, (2, 2), 0, False),
    ("bsr", (6, 6), 6, (3, 2), 0, False),
    ("bsr", (4, 6), 0, (2, 2), 0, False),
    ("bsr", (12, 12), 12, (3, 4), 0, False),
    ("bsr", (2, 8, 12), 6, (4, 4), 1, False),
    ("bsr", (4, 20, 10, 3), 5, (5, 5), 1, False),
    ("bsc", (6, 4), 6, (2, 2), 0, False),
    ("bsc", (6, 4), 6, (2, 2), 0, True),
    ("bsc", (2, 6, 4), 6, (2, 2), 1, False),
    ("bsc", (4, 20, 10, 3), 8, (5, 5), 1, False),
    ("bsc", (2, 3, 6, 4), 6, (2, 2), 2, False),
    ("csr", (0, 5), 0, None, 0, False),
    ("csr", (0, 5, 4), 0, None, 0, False),
    ("csr", (2, 0, 5), 0, None, 1, False),
    ("csc", (5, 0), 0, None, 0, False),
    ("csc", (0, 5), 0, None, 0, False),
    ("bsr", (0, 6), 0, (2, 2), 0, False),
    ("bsr", (4, 0), 0, (2, 2), 0, False),
    ("bsc", (6, 0), 0, (2, 2), 0, False),
    ("bsc", (4, 0), 0, (2, 2), 0, False),
]

_VALUE_RANGES = [
    ["-1", "1"],
    ["0", "1"],
    ["-1", "0"],
    ["0", "max"],
    ["min", "0"],
]

# One workload per row: the shape levels x the five value ranges over the
# dtypes that carry a range, then the storage rows over every supported dtype.
# The spec shape levels stand in for the KernelGen 0~5-dim sweep.
_COPY_ROWS = [
    (row, dtype, list(value_range))
    for row in [
        _shape_to_row(shape) for shape in [(1,), (256,), (1024, 1024), (20, 320, 15)]
    ]
    for dtype in _RANGE_DTYPES
    for value_range in _VALUE_RANGES
] + [(row, dtype, ["-1", "1"]) for row in _STORAGE_CASES for dtype in _VALUE_DTYPES]

_SPECIAL_PAYLOADS = {
    "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
    "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
    "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
}


def _special_value_cases(dtypes):
    """Representable special scenarios per floating dtype."""
    cases = []
    for dtype in dtypes:
        if not dtype.is_floating_point:
            continue
        cases.append((dtype, "nan"))
        cases.extend((dtype, kind) for kind in ("inf", "mixed"))
    return cases


_SPECIAL_ROWS = _special_value_cases(_SPECIAL_DTYPES)

# The copy is the identity on the stored values, so the gradient must be
# exactly the upstream gradient; both an element and a block layout are covered.
_BACKWARD_CASES = [
    ("csr", (8, 8), 16, None, 0, False),
    ("bsr", (4, 6), 4, (2, 2), 0, False),
]

# Stored values may live in a strided or offset view of a larger buffer.
_VIEW_CASES = [("strided", dtype) for dtype in _VALUE_DTYPES] + [
    ("offset", dtype) for dtype in _VALUE_DTYPES
]

# Out buffers at a nonzero storage offset (contiguous slice, and a stride-2
# view), each with sentinel neighbours and an empty-storage row.
_OFFSET_OUT_CASES = [
    ("csr", (2, 3), 4, None, 0, False),
    ("csr", (2, 3), 0, None, 0, False),
    ("bsr", (4, 6), 4, (2, 2), 0, False),
    ("bsr", (4, 6), 0, (2, 2), 0, False),
]
_STRIDED_OUT_CASES = [
    ("csr", (2, 3), 4, None, 0, False),
    ("bsr", (4, 6), 4, (2, 2), 0, False),
]


@pytest.mark.values_copy
@pytest.mark.parametrize("case,dtype,value_range", _COPY_ROWS)
def test_values_copy(case, dtype, value_range):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = _make_sparse(*case, dtype=dtype, value_range=value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.values_copy(ref_inp)
    res_out = flag_gems.values_copy(inp)

    _assert_copy_result(res_out, ref_out, inp)
    utils.gems_assert_equal(inp, ref_inp, equal_nan=True)


@pytest.mark.values_copy_out
@pytest.mark.parametrize("case,dtype,value_range", _COPY_ROWS)
def test_values_copy_out(case, dtype, value_range):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = _make_sparse(*case, dtype=dtype, value_range=value_range)
    ref_inp = utils.to_reference(inp)

    shape = _stored_shape(*case)
    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_out = torch.empty(shape, dtype=dtype, device=ref_inp.device)
    torch.ops.aten.values_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.values_copy(inp, out=out)

    assert res_ret is out
    utils.gems_assert_equal(out, ref_out, equal_nan=True)
    utils.gems_assert_equal(inp, ref_inp, equal_nan=True)


@pytest.mark.values_copy
@pytest.mark.parametrize("kind,dtype", _VIEW_CASES)
def test_values_copy_values_view(kind, dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    """A strided or offset stored-values view is copied faithfully, unmodified."""
    nnz = 4
    base = _make_input(dtype, (4 * nnz + 8,), ["-1", "1"])
    values = base[::2][:nnz] if kind == "strided" else base[3 : 3 + nnz]
    inp = _make_sparse("csr", (nnz, 1), nnz, None, 0, False, dtype=dtype, values=values)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.values_copy(ref_inp)
    res_out = flag_gems.values_copy(inp)

    _assert_copy_result(res_out, ref_out, inp)
    utils.gems_assert_equal(inp, ref_inp, equal_nan=True)


@pytest.mark.values_copy
@pytest.mark.parametrize("dtype", _OUT_BUFFER_DTYPES)
@pytest.mark.parametrize("case", _OFFSET_OUT_CASES)
def test_values_copy_out_into_offset_buffer(case, dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    """A contiguous out slice at a nonzero storage offset keeps its neighbours."""
    inp = _make_sparse(*case, dtype=dtype)
    ref_inp = utils.to_reference(inp)

    sentinel = 3.5
    values_shape = inp.values().shape
    numel = inp.values().numel()
    buffer = torch.full((numel + 8,), sentinel, dtype=dtype, device=flag_gems.device)
    ref_buffer = torch.full_like(buffer, sentinel, device=ref_inp.device)
    # The slice is a contiguous view at offset 3; reshaping it to the native
    # values shape keeps that offset, so the write cannot be accepted through
    # the deprecated implicit resize of the out tensor.
    out = buffer[3 : 3 + numel].reshape(values_shape)
    ref_out = ref_buffer[3 : 3 + numel].reshape(values_shape)
    torch.ops.aten.values_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.values_copy(inp, out=out)

    assert res_ret is out
    assert out.storage_offset() == 3 and out.is_contiguous()
    assert out.shape == values_shape
    # The whole buffer, so the sentinel neighbours must survive the write.
    utils.gems_assert_equal(buffer, ref_buffer, equal_nan=True)
    utils.gems_assert_equal(inp, ref_inp, equal_nan=True)


@pytest.mark.values_copy
@pytest.mark.parametrize("dtype", _OUT_BUFFER_DTYPES)
@pytest.mark.parametrize("case", _STRIDED_OUT_CASES)
def test_values_copy_out_into_strided_view(case, dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    """A genuinely strided out view is filled in place, never touching neighbours."""
    inp = _make_sparse(*case, dtype=dtype)
    ref_inp = utils.to_reference(inp)

    sentinel = 3.5
    shape = _stored_shape(*case)
    parent = torch.full(
        (2 * shape[0],) + tuple(shape[1:]),
        sentinel,
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_parent = torch.full_like(parent, sentinel, device=ref_inp.device)
    out, ref_out = parent[1::2], ref_parent[1::2]
    torch.ops.aten.values_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.values_copy(inp, out=out)

    assert res_ret is out
    # Same parent storage and the native stored-values shape: identical data
    # alone cannot reject a return that was reshaped instead of filled in place.
    assert tuple(res_ret.shape) == tuple(ref_out.shape) == tuple(inp.values().shape)
    assert out.stride() == ref_out.stride()
    # Exact whole-parent comparison: written slice and sentinel neighbours.
    utils.gems_assert_equal(parent, ref_parent, equal_nan=True)
    utils.gems_assert_equal(inp, ref_inp, equal_nan=True)


@pytest.mark.values_copy
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_ROWS)
def test_values_copy_special_values(dtype, scenario):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    values = torch.tensor(
        _SPECIAL_PAYLOADS[scenario], device=flag_gems.device, dtype=torch.float32
    ).to(dtype)
    inp = _make_sparse(
        "csr", (2, 3), values.numel(), None, 0, False, dtype=dtype, values=values
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.values_copy(ref_inp)
    res_out = flag_gems.values_copy(inp)

    _assert_copy_result(res_out, ref_out, inp)
    utils.gems_assert_equal(inp, ref_inp, equal_nan=True)


@pytest.mark.values_copy
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_ROWS)
def test_values_copy_out_special_values(dtype, scenario):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    values = torch.tensor(
        _SPECIAL_PAYLOADS[scenario], device=flag_gems.device, dtype=torch.float32
    ).to(dtype)
    inp = _make_sparse(
        "csr", (2, 3), values.numel(), None, 0, False, dtype=dtype, values=values
    )
    ref_inp = utils.to_reference(inp)

    out = torch.empty_like(values)
    ref_out = torch.empty_like(out, device=ref_inp.device)
    torch.ops.aten.values_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.values_copy(inp, out=out)

    assert res_ret is out
    utils.gems_assert_equal(out, ref_out, equal_nan=True)
    utils.gems_assert_equal(inp, ref_inp, equal_nan=True)


@pytest.mark.values_copy
@pytest.mark.parametrize("case", _BACKWARD_CASES)
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
def test_values_copy_backward(case, dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    """The copy is the identity, so the stored values receive the exact upstream."""
    values = _make_input(dtype, _stored_shape(*case), ["-1", "1"])
    values = values.requires_grad_(True)
    inp = _make_sparse(*case, dtype=dtype, values=values)

    ref_values = utils.to_reference(values).detach().requires_grad_(True)
    ref_inp = _make_sparse(*case, dtype=dtype, values=ref_values)

    ref_out = torch.ops.aten.values_copy(ref_inp)
    res_out = flag_gems.values_copy(inp)

    _assert_copy_result(res_out, ref_out, inp)

    # Nonuniform upstream, so a swapped or partially reduced gradient cannot pass.
    upstream = _make_input(dtype, res_out.shape, ["-1", "1"]) * 3.0 + 1.0
    ref_upstream = utils.to_reference(upstream).detach()
    ref_grad = torch.autograd.grad(ref_out, ref_values, grad_outputs=ref_upstream)[0]
    res_grad = torch.autograd.grad(res_out, values, grad_outputs=upstream)[0]

    utils.gems_assert_equal(res_grad, ref_grad)


@pytest.mark.values_copy
def test_values_copy_rejects_dense_input():
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    # native: RuntimeError: values expected sparse tensor layout but got Strided
    inp = torch.randn(4, 5, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems.values_copy(inp)


@pytest.mark.values_copy
@pytest.mark.parametrize("kind,dtype", _REJECTION_ROWS)
def test_values_copy_rejects_vendor_unsupported_layout(kind, dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = _rejection_input(kind, dtype)

    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems.values_copy(inp)


@pytest.mark.values_copy
def test_values_copy_rejects_non_tensor_input():
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    # native: RuntimeError: failed to match any schema with overloads None
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems.values_copy(3.14)


@pytest.mark.values_copy
@pytest.mark.parametrize("mismatched", _MISMATCHED_OUT_DTYPES)
def test_values_copy_out_rejects_mismatched_dtype(mismatched):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    # native: RuntimeError: Expected out tensor to have dtype float, but got ...
    inp = _make_sparse("csr", (4, 5), 4, None, 0, False, dtype=torch.float32)
    out = torch.empty(
        _stored_shape("csr", (4, 5), 4, None, 0, False),
        dtype=mismatched,
        device=flag_gems.device,
    )

    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems.values_copy(inp, out=out)


# Sparse COO shapes for the values copy: (n_sparse_dims, nnz, dense_tail).
# The op copies the (nnz,) + dense_tail values tensor; nnz drives the workload
# and a non-empty dense tail exercises the nd kernel.
SPARSE_VALUES_SHAPES = (
    [
        (1, 8, (16,), ()),
        (2, 4, (2, 2), ()),
        (2, 8, (4, 4, 2), (2,)),
        (2, 16, (8, 8), ()),
        (3, 32, (5, 7, 9, 3), (3,)),
    ]
    if utils.QUICK_MODE
    else [
        (1, 8, (16,), ()),
        (1, 64, (64, 2), (2,)),
        (2, 4, (2, 2), ()),
        (2, 8, (4, 4, 2), (2,)),
        (2, 64, (16, 16), ()),
        (2, 1024, (32, 32, 2, 2), (2, 2)),
        (3, 32, (5, 7, 9, 3), (3,)),
        (3, 128, (16, 16, 16), ()),
        (4, 64, (8, 8, 8, 8, 2), (2,)),
    ]
)


def _make_coo(nnd, nnz, size, dense_tail, dtype, device, seed=0):
    """Build a sparse COO tensor with distinct, sorted coordinates."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    numel = 1
    for s in size[:nnd]:
        numel *= s
    flat = torch.randperm(numel, generator=gen)[:nnz]
    indices = torch.empty((nnd, nnz), dtype=torch.int64)
    work = flat.clone()
    for d in reversed(range(nnd)):
        indices[d] = work % size[d]
        work = work // size[d]
    perm = torch.arange(nnz)
    for d in reversed(range(nnd)):
        perm = perm[torch.argsort(indices[d][perm], stable=True)]
    indices = indices[:, perm].contiguous()
    values = torch.randn((nnz,) + tuple(dense_tail), dtype=dtype, generator=gen)
    return torch.sparse_coo_tensor(indices.to(device), values.to(device), size)


@pytest.mark.underscore_values_copy
@pytest.mark.parametrize("nnd, nnz, size, dense_tail", SPARSE_VALUES_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__values_copy(nnd, nnz, size, dense_tail, dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    # The native CUDA op has no sparse-COO kernel, so the reference is the
    # plain values accessor cloned on device.
    res_inp = _make_coo(nnd, nnz, size, dense_tail, dtype, flag_gems.device, seed=3)
    ref_values = torch.clone(res_inp._values())

    res_out = flag_gems._values_copy(res_inp)

    assert res_out.shape == (nnz,) + tuple(dense_tail)
    # The copy must not alias the input storage.
    assert res_out.data_ptr() != res_inp._values().data_ptr()
    utils.gems_assert_close(res_out, ref_values, dtype)


@pytest.mark.underscore_values_copy
def test__values_copy_empty():
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    res_inp = torch.sparse_coo_tensor(
        torch.zeros((2, 0), dtype=torch.int64, device=flag_gems.device),
        torch.zeros((0,), device=flag_gems.device),
        (4, 4),
    )
    res_out = flag_gems._values_copy(res_inp)

    assert res_out.shape == (0,)


@pytest.mark.underscore_values_copy_out
def test__values_copy_out():
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    res_inp = _make_coo(2, 16, (8, 8, 2), (2,), torch.float32, flag_gems.device, seed=5)
    vals = res_inp._values()
    res_out = torch.empty_like(vals)

    res_ret = flag_gems._values_copy_out(res_inp, out=res_out)

    assert res_ret is res_out
    utils.gems_assert_close(res_out, vals, torch.float32)
