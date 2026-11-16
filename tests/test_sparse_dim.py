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


def _device(device):
    expected = torch.device(device)
    if expected.index is None and expected.type not in ("cpu", "meta"):
        expected = torch.device(
            expected.type, flag_gems.runtime.torch_device_fn.current_device()
        )
    return expected


def _view_from_storage(inp, storage, device):
    out = torch.empty(0, dtype=inp.dtype, device=device).set_(
        storage, inp.storage_offset(), inp.size(), inp.stride()
    )
    if inp.is_conj():
        out = out.conj()
    if inp.is_neg():
        out = torch._neg_view(out)
    assert out.dtype == inp.dtype and out.device == _device(device)
    assert out.shape == inp.shape and out.stride() == inp.stride()
    assert out.storage_offset() == inp.storage_offset()
    assert (out.is_conj(), out.is_neg()) == (inp.is_conj(), inp.is_neg())
    return out.requires_grad_(inp.requires_grad)


def _copy_to(inp, device):
    # Copy physical bytes before restoring the view and lazy flags. A typed
    # cross-device copy can resolve conjugation or require unavailable kernels.
    nbytes = inp.untyped_storage().nbytes()
    raw = torch.empty(0, dtype=torch.uint8, device=inp.device).set_(
        inp.untyped_storage(), 0, (nbytes,), (1,)
    )
    raw = raw.to(device).clone()
    return _view_from_storage(inp, raw.untyped_storage(), raw.device)


def _check_input(inp, dtype, shape):
    assert inp.dtype == dtype and inp.shape == torch.Size(shape)
    assert inp.device == _device(flag_gems.device)
    return inp


def _cpu_input(dtype, shape, value_range):
    # The original factory recipe, with the same bounds and degenerate fills.
    low, high = (tu.resolve_bound(v, dtype) for v in value_range)
    if not dtype.is_floating_point:
        low, high = int(low), int(high)
        dtype_min, dtype_max = tu.dtype_bounds(dtype)
        low = max(low, int(dtype_min))
        high = min(high, int(dtype_max))
        low = min(low, high)
    if low == high:
        return torch.full(shape, low, dtype=dtype, device="cpu")
    return torch.testing.make_tensor(
        shape, dtype=dtype, device="cpu", low=low, high=high
    )


def _make_input(dtype, shape, value_range):
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        inp = _copy_to(_cpu_input(dtype, shape, value_range), flag_gems.device)
    else:
        inp = tu.make_input(dtype, shape, value_range)
    return _check_input(inp, dtype, shape)


def _make_special_input(dtype, scenario):
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        payloads = {
            "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
            "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
            "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
        }
        value = torch.tensor(payloads[scenario], dtype=torch.float32, device="cpu")
        inp = _copy_to(value.to(dtype), flag_gems.device)
    else:
        inp = tu.make_special_input(dtype, scenario)
    return _check_input(inp, dtype, (5,))


def _repeat(inp, *sizes):
    if (
        flag_gems.vendor_name in ("ascend", "kunlunxin")
        and inp.dtype
        in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        )
        and inp.device.type != "cpu"
    ):
        return _copy_to(_copy_to(inp, "cpu").repeat(*sizes), inp.device)
    return inp.repeat(*sizes)


# sparse_dim counts sparse dimensions: zero for strided tensors, the
# leading sparse rank for COO, and two for the CSR layouts below.
_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool]
)

# Dense shapes, including scalars and higher ranks.
_DENSE_SHAPES = tu.selected_cases(
    [
        (),
        (5,),
        (3, 4),
        (8, 8, 8),
        (3, 4, 2, 5),
        (3, 4, 5, 4, 5),
        (3, 6, 4, 4, 6, 5, 4),
        (7, 3, 12, 4, 2, 15, 2, 2),
    ],
    quick=[(2, 19, 7)],
)

# COO: (sparse_shape, dense_shape, nnz).
_COO_CASES = tu.selected_cases(
    [
        ((4, 4), (), 8),
        ((8, 8, 8), (), 64),
        ((4, 4), (3,), 8),
        ((2, 3, 4), (5,), 12),
        ((16, 16), (7, 13), 40),
        ((2, 3, 4), (5, 6), 12),
        ((3,), (4, 5, 6), 2),
        ((12, 9, 3, 6), (4,), 9),
        ((3, 4, 2, 5, 3), (4, 2), 11),
    ],
    quick=[((2, 19, 7), (), 8)],
)

# COO value ranges: all-sparse and hybrid layouts.
_COO_RANGE_CASES = tu.selected_cases(
    [
        ((3, 4), (), 7),
        ((3, 4), (3,), 8),
        ((12, 9, 3, 6), (4,), 9),
    ],
    quick=[((2, 19, 7), (), 8)],
)

# CSR: (shape, nnz), including batched tensors.
_CSR_CASES = tu.selected_cases(
    [
        ((4, 4), 3),
        ((2, 4, 4), 5),
        ((3, 5, 7), 3),
        ((3, 4, 4), 4),
    ],
    quick=[((2, 19, 7), 3)],
)

_SPEC_NNZ = 6  # Keep duplicates possible even in small index spaces.


def _make_coo(sparse_shape, dense_shape, nnz, dtype, value_range, seed=0):
    # Seeded CPU indices allow duplicates; values are created on the test device.
    gen = torch.Generator("cpu").manual_seed(seed)
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    values = _make_input(dtype, (nnz,) + tuple(dense_shape), value_range)
    size = tuple(sparse_shape) + tuple(dense_shape)
    return torch.sparse_coo_tensor(indices, values, size, device=flag_gems.device)


def _make_csr(shape, nnz, dtype, value_range):
    if len(shape) == 2:
        rows, cols = shape
    else:
        _, rows, cols = shape
    assert 0 <= nnz <= rows * cols
    counts = torch.full((rows,), nnz // rows, dtype=torch.long)
    counts[: nnz % rows] += 1
    crow_indices = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    col_indices = torch.arange(nnz) - torch.repeat_interleave(crow_indices[:-1], counts)
    if len(shape) == 3:
        # Batched CSR: every batch stores the same nnz entries (shared
        # crow/col pattern), so the layout stays 2-D sparse for every batch.
        crow_indices = crow_indices.expand(shape[0], -1).contiguous()
        col_indices = col_indices.expand(shape[0], -1).contiguous()
        values = _make_input(dtype, (shape[0], nnz), value_range)
    else:
        values = _make_input(dtype, (nnz,), value_range)
    return torch.sparse_csr_tensor(
        crow_indices, col_indices, values, shape, device=flag_gems.device
    )


def _make_csr_with_dense_dims(dtype, value_range):
    """CSR layout carrying a dense dim: shape (rows, cols, dense)."""
    rows, cols, dense, nnz = 4, 4, 3, 5
    # crow segments: row0 -> 1, row1 -> 1, row2 -> 2, row3 -> 1 stored block.
    crow = torch.tensor([0, 1, 2, 4, 5], dtype=torch.long, device=flag_gems.device)
    col = torch.tensor([0, 1, 0, 1, 2], dtype=torch.long, device=flag_gems.device)
    values = _make_input(dtype, (nnz, dense), value_range)
    return torch.sparse_csr_tensor(
        crow, col, values, (rows, cols, dense), device=flag_gems.device
    )


def _make_empty_csr(shape, dtype):
    """Build an empty plain or batched CSR tensor."""
    if len(shape) == 2:
        rows, _ = shape
        crow_indices = torch.zeros(rows + 1, dtype=torch.long, device=flag_gems.device)
        col_indices = torch.empty(0, dtype=torch.long, device=flag_gems.device)
        values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    else:
        _, rows, _ = shape
        crow_indices = torch.zeros(
            shape[0], rows + 1, dtype=torch.long, device=flag_gems.device
        )
        col_indices = torch.empty(
            shape[0], 0, dtype=torch.long, device=flag_gems.device
        )
        values = torch.empty(shape[0], 0, dtype=dtype, device=flag_gems.device)
    return torch.sparse_csr_tensor(
        crow_indices, col_indices, values, shape, device=flag_gems.device
    )


def _assert_result(res_out, ref_out):
    assert type(res_out) is int
    utils.gems_assert_equal(res_out, ref_out)


_INPUT_DTYPE_CASES = [
    pytest.param(
        dtype,
    )
    for dtype in _DTYPES
]


@pytest.mark.sparse_dim
@pytest.mark.parametrize("shape", _DENSE_SHAPES)
@pytest.mark.parametrize("dtype", _INPUT_DTYPE_CASES)
def test_sparse_dim_dense_layouts(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("shape", [(0,), (0, 5), (2, 0, 3)])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_empty_dense(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"])
    assert inp.numel() == 0
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


# Sparse dimensions depend on the tensor layout, not stored values.
# One ordinary value range per dtype for the fixed-structure queries.
_DTYPE_RANGE_CASES = [
    pytest.param(
        dtype,
        value_range,
    )
    for dtype in _DTYPES
    for value_range in tu.selected_ranges()[:1]
]


# Explicit default/quick pairs; dtype lists retain device capability filtering.
_SHAPE_DTYPE_CASES = tu.selected_cases(
    [
        ((), torch.float32),
        ((1,), torch.float32),
        ((256,), torch.float32),
        ((1024, 1024), torch.float32),
        ((16, 128, 64, 60), torch.float32),
        ((16, 7, 57, 32, 29), torch.float32),
    ]
    + [((20, 320, 15), dtype) for dtype in _DTYPES],
    quick=[((2, 19, 7), dtype) for dtype in _DTYPES],
)


@pytest.mark.sparse_dim
@pytest.mark.parametrize(
    "shape, dtype",
    [
        pytest.param(
            shape,
            dtype,
        )
        for shape, dtype in _SHAPE_DTYPE_CASES
    ],
)
@pytest.mark.parametrize("value_range", tu.selected_ranges()[:1])
def test_sparse_dim_dense_spec_shapes_value_ranges(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("case", _COO_CASES)
@pytest.mark.parametrize("dtype", _INPUT_DTYPE_CASES)
def test_sparse_dim_coo_layouts(case, dtype):
    sparse_shape, dense_shape, nnz = case
    inp = _make_coo(sparse_shape, dense_shape, nnz, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)
    # Pure metadata query: the input layout is untouched.
    assert inp.sparse_dim() == len(sparse_shape)
    assert inp.dense_dim() == len(dense_shape)
    assert inp._nnz() == nnz


# Explicit default/quick pairs; dtype lists retain device capability filtering.
_COO_SPEC_DTYPE_CASES = tu.selected_cases(
    [
        ((1,), torch.float32),
        ((256,), torch.float32),
        ((1024, 1024), torch.float32),
        ((16, 128, 64, 60), torch.float32),
        ((16, 7, 57, 32, 29), torch.float32),
    ]
    + [((20, 320, 15), dtype) for dtype in _DTYPES],
    quick=[((2, 19, 7), dtype) for dtype in _DTYPES],
)


@pytest.mark.sparse_dim
@pytest.mark.parametrize(
    "shape, dtype",
    [
        pytest.param(
            shape,
            dtype,
        )
        for shape, dtype in _COO_SPEC_DTYPE_CASES
    ],
)
@pytest.mark.parametrize("value_range", tu.selected_ranges()[:1])
def test_sparse_dim_coo_spec_shapes_value_ranges(shape, value_range, dtype):
    inp = _make_coo(tuple(shape), (), _SPEC_NNZ, dtype, value_range, seed=0)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


# Explicit default/quick pairs; dtype lists retain device capability filtering.
_COO_DTYPE_CASES = tu.selected_cases(
    [
        (((3, 4), (), 7), torch.float32),
        (((3, 4), (3,), 8), torch.float32),
    ]
    + [(((12, 9, 3, 6), (4,), 9), dtype) for dtype in _DTYPES],
    quick=[(((2, 19, 7), (), 8), dtype) for dtype in _DTYPES],
)


@pytest.mark.sparse_dim
@pytest.mark.parametrize(
    "case, dtype",
    [
        pytest.param(
            case,
            dtype,
        )
        for case, dtype in _COO_DTYPE_CASES
    ],
)
@pytest.mark.parametrize("value_range", tu.selected_ranges()[:1])
def test_sparse_dim_coo_value_ranges(case, value_range, dtype):
    sparse_shape, dense_shape, nnz = case
    inp = _make_coo(sparse_shape, dense_shape, nnz, dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_empty_coo(dtype):
    sparse_shape, dense_shape = (3, 4), (5, 6)
    indices = torch.empty(
        len(sparse_shape), 0, dtype=torch.long, device=flag_gems.device
    )
    values = torch.empty(
        (0,) + tuple(dense_shape), dtype=dtype, device=flag_gems.device
    )
    inp = torch.sparse_coo_tensor(
        indices, values, sparse_shape + dense_shape, device=flag_gems.device
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("dtype", _INPUT_DTYPE_CASES)
def test_sparse_dim_uncoalesced_coo(dtype):
    sparse_shape, dense_shape = (2, 2), (3,)
    indices = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 0, 1, 0]], dtype=torch.long)
    values = _make_input(dtype, (5,) + tuple(dense_shape), ["-1", "1"])
    inp = torch.sparse_coo_tensor(
        indices, values, sparse_shape + dense_shape, device=flag_gems.device
    )
    assert not inp.is_coalesced()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("case", _CSR_CASES)
@pytest.mark.parametrize("dtype", _INPUT_DTYPE_CASES)
def test_sparse_dim_csr_layouts(case, dtype):
    shape, nnz = case
    inp = _make_csr(shape, nnz, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)
    # Pure metadata query: the input layout is untouched.
    assert inp.sparse_dim() == 2
    assert inp.dense_dim() == 0


@pytest.mark.sparse_dim
@pytest.mark.parametrize(
    "dtype, value_range",
    _DTYPE_RANGE_CASES,
)
def test_sparse_dim_csr_value_ranges(value_range, dtype):
    shape, nnz = (4, 4), 5
    inp = _make_csr(shape, nnz, dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize(
    "dtype, value_range",
    _DTYPE_RANGE_CASES,
)
def test_sparse_dim_csr_dense_dims(value_range, dtype):
    inp = _make_csr_with_dense_dims(dtype, value_range)
    assert inp.sparse_dim() == 2
    assert inp.dense_dim() == 1
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("shape", [(4, 4), (3, 4, 4)])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_empty_csr(shape, dtype):
    inp = _make_empty_csr(shape, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize(
    "shape", [shape for shape in tu.selected_shapes() if 2 <= len(shape) <= 3]
)
@pytest.mark.parametrize("dtype", _INPUT_DTYPE_CASES)
def test_sparse_dim_csr_spec_shapes(shape, dtype):
    nnz = 5 if len(shape) == 2 else 3
    inp = _make_csr(shape, nnz, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize(
    "dtype,scenario",
    [
        pytest.param(
            dtype,
            scenario,
        )
        for dtype, scenario in tu.selected_cases(tu.special_value_cases(_DTYPES))
    ],
)
def test_sparse_dim_nan_inf_dense(dtype, scenario):
    inp = _repeat(_make_special_input(dtype, scenario), 2)[:6].reshape(2, 3)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize(
    "dtype,scenario",
    [
        pytest.param(
            dtype,
            scenario,
        )
        for dtype, scenario in tu.selected_cases(tu.special_value_cases(_DTYPES))
    ],
)
def test_sparse_dim_nan_inf_coo(dtype, scenario):
    values = _repeat(_make_special_input(dtype, scenario), 2)[:6]
    indices = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    inp = torch.sparse_coo_tensor(indices, values, (6,), device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize(
    "dtype,scenario",
    [
        pytest.param(
            dtype,
            scenario,
        )
        for dtype, scenario in tu.selected_cases(tu.special_value_cases(_DTYPES))
    ],
)
def test_sparse_dim_nan_inf_csr(dtype, scenario):
    values = _repeat(_make_special_input(dtype, scenario), 2)[:6]
    crow_indices = torch.tensor([0, 3, 4, 6], dtype=torch.long)
    col_indices = torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long)
    inp = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, (3, 4), device=flag_gems.device
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
def test_sparse_dim_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_dim(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError, NotImplementedError)):
        flag_gems.sparse_dim(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError, NotImplementedError)):
        flag_gems.sparse_dim(None)


def _sparse_dim_dispatch_keys():
    device_type = torch.device(flag_gems.device).type
    backend = torch._C._dispatch_key_for_device(device_type)
    return backend, "Sparse" + backend, "SparseCsr" + backend


# sparse_dim is a layout-metadata query returning a plain int; the result is
# independent of dtype and stored data. Coverage is driven by layout instead:
# strided tensors of several ranks, sparse COO (plain, 1-D, hybrid with dense
# trailing dims, zero-nnz, uncoalesced) and sparse compressed (CSR/CSC).
if utils.QUICK_MODE:
    _MASTER_STRIDED_SHAPES = [(2, 3)]
else:
    _MASTER_STRIDED_SHAPES = [(), (5,), (2, 3)]


def _master_make_coo(nnd, nnz, size, seed=0):
    gen = torch.Generator().manual_seed(seed)
    numel = 1
    for s in size[:nnd]:
        numel *= s
    flat = torch.randperm(numel, generator=gen)[:nnz]
    indices = torch.empty((nnd, nnz), dtype=torch.int64)
    for d in reversed(range(nnd)):
        indices[d] = flat % size[d]
        flat = flat // size[d]
    # Deliberately unsorted: uncoalesced COO must return the same metadata.
    values = torch.randn(nnz, generator=gen)
    return torch.sparse_coo_tensor(indices, values, size)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("shape", _MASTER_STRIDED_SHAPES)
def test_sparse_dim_strided(shape):
    inp = torch.randn(shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = ref_inp.sparse_dim()
    res_out = flag_gems.sparse_dim(inp)

    assert isinstance(res_out, int)
    assert res_out == ref_out
    assert res_out == 0


@pytest.mark.sparse_dim
def test_sparse_dim_sparse_coo():
    for name, sparse, expected in [
        ("plain", _master_make_coo(2, 3, (2, 3), seed=1), 2),
        ("1d", _master_make_coo(1, 3, (3,), seed=2), 1),
        (
            "hybrid",
            torch.sparse_coo_tensor(torch.tensor([[0, 1]]), torch.randn(2, 3), (2, 3)),
            1,
        ),
        (
            "zero-nnz",
            torch.sparse_coo_tensor(
                torch.zeros(2, 0, dtype=torch.int64),
                torch.zeros(0),
                (2, 3),
            ),
            2,
        ),
        ("uncoalesced", _master_make_coo(2, 4, (2, 3), seed=3), 2),
    ]:
        sparse = sparse.to(flag_gems.device)
        ref_sparse = utils.to_reference(sparse)
        ref_out = ref_sparse.sparse_dim()
        res_out = flag_gems.sparse_dim(sparse)
        assert res_out == ref_out, f"{name}: {res_out} != {ref_out}"
        assert res_out == expected, f"{name}: {res_out} != {expected}"


@pytest.mark.sparse_dim
def test_sparse_dim_sparse_compressed():
    c = torch.tensor([0, 2, 4], dtype=torch.int64, device=flag_gems.device)
    cc = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device=flag_gems.device)
    val = torch.randn(4, device=flag_gems.device)

    csr = torch.sparse_csr_tensor(c, cc, val, (2, 3))
    csc = torch.sparse_csc_tensor(cc, c, val, (2, 3))
    for name, t in [("csr", csr), ("csc", csc)]:
        res_out = flag_gems.sparse_dim(t)
        # ATen contract: compressed layouts always have sparse_dim == 2,
        # independent of the logical shape.
        assert res_out == 2, f"{name}: {res_out} != 2"
        assert res_out == utils.to_reference(t).sparse_dim()


@pytest.mark.sparse_dim
def test_sparse_dim_matches_reference_table():
    # Reference values captured in a process that never imported flag_gems
    # (un-substituted environment), pinned here as literals.
    table = {
        "strided()": 0,
        "strided(5,)": 0,
        "strided(2,3)": 0,
        "coo-1d": 1,
        "coo-2d": 2,
        "coo-hybrid": 1,
        "coo-0nnz": 2,
        "coo-uncoalesced": 2,
        "csr": 2,
        "csc": 2,
    }
    dev = flag_gems.device
    assert flag_gems.sparse_dim(torch.randn((), device=dev)) == table["strided()"]
    assert flag_gems.sparse_dim(torch.randn(5, device=dev)) == table["strided(5,)"]
    assert flag_gems.sparse_dim(torch.randn(2, 3, device=dev)) == table["strided(2,3)"]
    assert (
        flag_gems.sparse_dim(_master_make_coo(1, 3, (3,), seed=6).to(dev)) == table["coo-1d"]
    )
    assert (
        flag_gems.sparse_dim(_master_make_coo(2, 3, (2, 3), seed=5).to(dev)) == table["coo-2d"]
    )
    h = torch.sparse_coo_tensor(torch.tensor([[0, 1]]), torch.randn(2, 3), (2, 3)).to(
        dev
    )
    assert flag_gems.sparse_dim(h) == table["coo-hybrid"]
    z = torch.sparse_coo_tensor(
        torch.zeros(2, 0, dtype=torch.int64, device=dev),
        torch.zeros(0, device=dev),
        (2, 3),
    )
    assert flag_gems.sparse_dim(z) == table["coo-0nnz"]
    assert (
        flag_gems.sparse_dim(_master_make_coo(2, 4, (2, 3), seed=3).to(dev))
        == table["coo-uncoalesced"]
    )


@pytest.mark.sparse_dim
def test_sparse_dim_disjoint_from_native_neighbors():
    # The implementation must not re-dispatch the operator it implements
    # (registering on the CUDA/sparse keys makes an internal
    # `self.sparse_dim()` call recurse into this very kernel), and it must
    # not depend on related accessors being overridden: `_indices` is left
    # native here, and a plain metadata read on every layout must keep
    # working with unrelated ops (e.g. `_nnz`) dispatched around it.
    t = torch.randn(3, 4, device=flag_gems.device)
    sparse = _master_make_coo(2, 5, (2, 3), seed=7).to(flag_gems.device)
    csr = sparse.to_sparse_csr()

    # Exercise the dispatcher path repeatedly: a recursion bug shows up here.
    for _ in range(3):
        assert flag_gems.sparse_dim(t) == 0
        assert flag_gems.sparse_dim(sparse) == 2
        assert flag_gems.sparse_dim(csr) == 2

    # Cross-op sanity: dispatching a related metadata op afterwards must not
    # corrupt or re-route sparse_dim (no shared mutable state in the impl).
    nnz = torch.ops.aten._nnz(sparse)
    assert nnz == 5
    assert flag_gems.sparse_dim(sparse) == 2
    assert flag_gems.sparse_dim(t) == 0


@pytest.mark.sparse_dim
def test_sparse_dim_dispatch_sentinel():
    # Registration-path check: a sentinel wrapper on the three dispatch keys
    # (CUDA for strided, SparseCUDA for COO, SparseCsrCUDA for compressed)
    # proves every layout actually routes into the registered implementation
    # instead of falling through to the native kernel.
    hits = []

    def sentinel(self):
        hits.append(str(self.layout))
        return flag_gems.sparse_dim(self)

    lib = torch.library.Library("aten", "IMPL")
    for key in _sparse_dim_dispatch_keys():
        lib.impl("sparse_dim", sentinel, key)

    dev = flag_gems.device
    t = torch.randn(2, 3, device=dev)
    coo = _master_make_coo(2, 3, (2, 3), seed=13).to(dev)
    row = torch.tensor([0, 2, 4], dtype=torch.int64, device=dev)
    col = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device=dev)
    csr = torch.sparse_csr_tensor(row, col, torch.randn(4, device=dev), (2, 3))

    assert torch.ops.aten.sparse_dim(t) == 0
    assert torch.ops.aten.sparse_dim(coo) == 2
    assert torch.ops.aten.sparse_dim(csr) == 2

    assert set(hits) == {"torch.strided", "torch.sparse_coo", "torch.sparse_csr"}
    assert len(hits) == 3


def _resolve_dense_dim():
    # dense_dim ships in a sibling branch/worktree (PR #639) and is absent
    # when this branch is tested on its own, so fall back to an equivalent
    # local implementation of the same ATen semantics. The coexistence test
    # only needs a second metadata operator to register on the same keys.
    try:
        dense_dim = flag_gems.dense_dim

        return dense_dim
    except AttributeError:

        def dense_dim(self):
            if self.layout == torch.strided:
                return self.dim()
            if self.layout == torch.sparse_coo:
                return self.dim() - self._indices().shape[0]
            return self.dim() - 2

        return dense_dim


@pytest.mark.sparse_dim
def test_sparse_dim_coexistence():
    # Combination test: sparse_dim and dense_dim registered side by side on
    # the same three dispatch keys (CUDA + SparseCUDA + SparseCsrCUDA) must
    # both route correctly and never recurse into each other.
    dense_dim = _resolve_dense_dim()

    lib = torch.library.Library("aten", "IMPL")
    for key in _sparse_dim_dispatch_keys():
        lib.impl("sparse_dim", flag_gems.sparse_dim, key)
        lib.impl("dense_dim", dense_dim, key)

    t = torch.randn(3, 4, device=flag_gems.device)
    sparse = _master_make_coo(2, 5, (2, 3), seed=11).to(flag_gems.device)
    csr = sparse.to_sparse_csr()
    hybrid = torch.sparse_coo_tensor(
        torch.tensor([[0, 1]]), torch.randn(2, 3), (2, 3)
    ).to(flag_gems.device)

    # Repeated dispatcher-routed calls: a recursion bug surfaces here.
    for _ in range(3):
        assert torch.ops.aten.sparse_dim(t) == 0
        assert torch.ops.aten.sparse_dim(sparse) == 2
        assert torch.ops.aten.sparse_dim(csr) == 2
        assert torch.ops.aten.sparse_dim(hybrid) == 1

        assert torch.ops.aten.dense_dim(t) == 2
        assert torch.ops.aten.dense_dim(sparse) == 0
        assert torch.ops.aten.dense_dim(csr) == 0
        assert torch.ops.aten.dense_dim(hybrid) == 1

        # Cross-op dispatch in between must not re-route either operator.
        assert torch.ops.aten._nnz(sparse) == 5
        assert torch.ops.aten.sparse_dim(sparse) == 2

    # Bound accessors on the tensors themselves must also reach the
    # registered implementations (this is the path that used to recurse).
    assert sparse.sparse_dim() == 2
    assert sparse.dense_dim() == 0
    assert t.sparse_dim() == 0
    assert flag_gems.sparse_dim(hybrid) == 1


@pytest.mark.sparse_dim
def test_sparse_dim_bsr_bsc_coverage():
    # BSR and BSC compressed layouts: native reports sparse_dim == 2 for
    # every compressed layout (independent of the logical tensor shape).
    # The registration claims support for all compressed sparse layouts,
    # so the test exercises all four.
    for make in (
        lambda: torch.randn(1, 4, 6, device=flag_gems.device).to_sparse_bsr((2, 2)),
        lambda: torch.randn(1, 4, 6, device=flag_gems.device).to_sparse_bsc((2, 2)),
        lambda: torch.randn(4, 6, device=flag_gems.device).to_sparse_csr(),
        lambda: torch.randn(4, 6, device=flag_gems.device).to_sparse_csc(),
    ):
        t = make()
        ref = t.sparse_dim()
        res = flag_gems.sparse_dim(t)
        assert res == ref == 2, f"layout={t.layout}: ours={res} native={ref}"


@pytest.mark.sparse_dim
def test_sparse_dim_unsupported_layout_parity():
    # Nested tensors are the reachable unsupported case: a non-jagged nested
    # tensor reports layout == torch.strided (and is_nested == True), so
    # without the is_nested guard it would take the strided branch and
    # return 0. Native ATen instead rejects it with a NotImplementedError
    # naming the NestedTensorCUDA dispatch key; the implementation must
    # reproduce that.
    nested = torch.nested.nested_tensor(
        [
            torch.randn(2, 3, device=flag_gems.device),
            torch.randn(4, 5, device=flag_gems.device),
        ]
    )
    assert nested.layout == torch.strided
    assert nested.is_nested

    with pytest.raises(NotImplementedError) as ref_err:
        torch.ops.aten.sparse_dim(nested)
    with pytest.raises(NotImplementedError) as res_err:
        flag_gems.sparse_dim(nested)

    native_msg = str(ref_err.value)
    res_msg = str(res_err.value)
    backend = "NestedTensor" + nested.device.type.upper()
    # The dispatcher's opening sentence (which names the backend key) is the
    # build-independent part of the native message; match it exactly. The
    # trailing registered-backend list varies per build and is not repeated.
    if flag_gems.vendor_name == "mthreads":
        native_msg = native_msg.replace("NestedTensorPrivateUse1", backend)
        res_msg = res_msg.replace("NestedTensorPrivateUse1", backend)
    assert f"'{backend}' backend" in native_msg
    assert f"'{backend}' backend" in res_msg
    assert res_msg.split(". ")[0] == native_msg.split(". ")[0]
