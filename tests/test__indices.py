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


def _check_values(inp, dtype, shape):
    assert inp.dtype == dtype and inp.shape == torch.Size(shape)
    assert inp.device == _device(flag_gems.device)
    return inp


def _cpu_values(dtype, shape, value_range):
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


def _make_values(dtype, shape, value_range):
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        inp = _copy_to(_cpu_values(dtype, shape, value_range), flag_gems.device)
    else:
        inp = tu.make_input(dtype, shape, value_range)
    return _check_values(inp, dtype, shape)


def _make_special_values(dtype, scenario):
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
    return _check_values(inp, dtype, (5,))


def _repeat_values(inp, *sizes):
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


def _assert_values_equal(actual, expected, *, equal_nan=False):
    if flag_gems.vendor_name in ("ascend", "kunlunxin", "mthreads") and actual.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        # Retain the original tensor contract before observing FP8 values on
        # CPU, where decoding avoids the missing device comparison kernels.
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape and actual.layout == expected.layout
        assert actual.device == _device(flag_gems.device)
        assert expected.device == (
            torch.device("cpu") if utils.TO_CPU else actual.device
        )
        actual = _copy_to(actual, "cpu").float()
        expected = _copy_to(expected, "cpu").float()
    utils.gems_assert_equal(actual, expected, equal_nan=equal_nan)


# _indices returns a view of the COO index storage, including duplicate entries.
# Dense and compressed sparse inputs are rejected.
_INDICES_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool]
)

# COO: (shape, sparse_dim, nnz).
_INDICES_COO_CASES = tu.selected_cases(
    [
        ((5,), 1, 4),
        ((3, 4), 2, 7),
        ((3, 4), 1, 16),
        ((8, 8, 8), 3, 32),
        ((3, 4, 2), 2, 12),
        ((4, 3, 4, 5), 1, 24),
        ((3, 4, 5, 4, 5), 3, 40),
        ((12, 9, 3, 6), 4, 9),
        ((3, 6, 4, 4, 6, 5), 4, 11),
        ((7, 3, 12, 4, 2, 15), 5, 10),
        ((3, 4, 2, 5, 3, 4, 2), 3, 13),
    ],
    quick=[((2, 19, 7), 2, 8)],
)

# Value-range layouts: all-sparse and hybrid COO.
_INDICES_RANGE_CASES = tu.selected_cases(
    [
        ((3, 4), 2, 7),
        ((3, 4, 2), 2, 12),
        ((12, 9, 3, 6), 4, 9),
    ],
    quick=[((2, 19, 7), 2, 8)],
)

_INDICES_SPEC_NNZ = 6  # Allow repeated coordinates in small index spaces.


def _make_coo_input(shape, sparse_dim, nnz, dtype, value_range, seed=0):
    # Seeded CPU indices may repeat; values are created on the test device.
    gen = torch.Generator("cpu").manual_seed(seed)
    sparse_shape = shape[:sparse_dim]
    dense_shape = shape[sparse_dim:]
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    values = _make_values(dtype, (nnz,) + dense_shape, value_range)
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


def _assert_result(res_out, ref_out, inp, ref_inp):
    assert inp.dtype == ref_inp.dtype and inp.shape == ref_inp.shape
    assert inp.layout == ref_inp.layout and inp.device == _device(flag_gems.device)
    assert ref_inp.device == (torch.device("cpu") if utils.TO_CPU else inp.device)
    assert res_out.dtype == ref_out.dtype and res_out.shape == ref_out.shape
    assert res_out.layout == ref_out.layout == torch.strided
    assert res_out.device == inp.device
    # Check exact output, storage aliasing and unchanged input metadata/values.
    tu.assert_result_equal(res_out, ref_out)
    assert res_out.data_ptr() == inp._indices().data_ptr()
    _assert_values_equal(inp._indices(), ref_inp._indices())
    if inp.dtype.is_floating_point:
        _assert_values_equal(inp._values(), ref_inp._values(), equal_nan=True)
    else:
        _assert_values_equal(inp._values(), ref_inp._values())


@pytest.mark.underscore_indices
@pytest.mark.parametrize("case", _INDICES_COO_CASES)
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_layouts(case, dtype):
    shape, sparse_dim, nnz = case
    inp = _make_coo_input(shape, sparse_dim, nnz, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = flag_gems._indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.underscore_indices
@pytest.mark.parametrize(
    "shape", [shape for shape in tu.selected_shapes() if len(shape) >= 1]
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_spec_shapes_value_ranges(shape, value_range, dtype):
    nnz = _INDICES_SPEC_NNZ
    inp = _make_coo_input(shape, len(shape), nnz, dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = flag_gems._indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.underscore_indices
@pytest.mark.parametrize("case", _INDICES_RANGE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_value_ranges(case, value_range, dtype):
    shape, sparse_dim, nnz = case
    inp = _make_coo_input(shape, sparse_dim, nnz, dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = flag_gems._indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.underscore_indices
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_empty(dtype):
    shape, sparse_dim = (3, 4), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = flag_gems._indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.underscore_indices
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_empty_hybrid(dtype):
    shape, sparse_dim = (4, 5, 6), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, 6, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = flag_gems._indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.underscore_indices
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_uncoalesced(dtype):
    shape = (3, 4)
    indices = torch.tensor([[0, 0, 1, 2, 0], [1, 1, 2, 3, 1]], dtype=torch.long)
    values = _make_values(dtype, (5,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = flag_gems._indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.underscore_indices
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_explicit_zeros(dtype):
    shape = (3, 3)
    indices = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    if dtype == torch.bool:
        values = torch.tensor([False, False, False], dtype=dtype)
    else:
        values = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = flag_gems._indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)
    assert res_out.shape == (2, 3)


@pytest.mark.underscore_indices
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_full_storage(dtype):
    shape = (2, 3)
    nnz = shape[0] * shape[1]
    indices = torch.stack(
        torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing="ij")
    ).reshape(2, nnz)
    values = _make_values(dtype, (nnz,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert inp._nnz() == nnz
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = flag_gems._indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.underscore_indices
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_INDICES_DTYPES))
)
def test__indices_nan_inf_values_ignored(dtype, scenario):
    indices = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    values = _repeat_values(_make_special_values(dtype, scenario), 2)[:6]
    inp = torch.sparse_coo_tensor(indices, values, (6,), device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = flag_gems._indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.underscore_indices
def test__indices_dense_raises():
    inp = _make_values(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises(NotImplementedError):
        torch.ops.aten._indices(tu.to_reference(inp))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError, ValueError)):
        flag_gems._indices(inp)


@pytest.mark.underscore_indices
def test__indices_csr_raises():
    crow_indices = torch.tensor([0, 2, 4], dtype=torch.long, device=flag_gems.device)
    col_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=flag_gems.device)
    values = _make_values(torch.float32, (4,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, (2, 4), device=flag_gems.device
    )
    with pytest.raises(NotImplementedError):
        torch.ops.aten._indices(tu.to_reference(inp))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError, ValueError)):
        flag_gems._indices(inp)


@pytest.mark.underscore_indices
def test__indices_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten._indices(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError, NotImplementedError)):
        flag_gems._indices(3.14)


_MASTER_COO_SHAPES = [
    (1, 8, (16,)),
    (2, 4, (2, 2)),
    (2, 8, (4, 4)),
    (2, 64, (16, 16)),
    (3, 32, (5, 7, 9)),
]



def _make_master_coo(nnd, nnz, size, seed=0):
    """Build a coalesced sparse COO tensor with distinct, sorted coordinates."""
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
    values = torch.randn(nnz, generator=gen)
    # The public accessor requires a coalesced input; construction alone does
    # not set the coalesced flag even when the coordinates are already unique
    # and sorted.
    return torch.sparse_coo_tensor(
        indices.to(flag_gems.device), values.to(flag_gems.device), size
    ).coalesce()



@pytest.mark.underscore_indices
@pytest.mark.parametrize("nnd, nnz, size", _MASTER_COO_SHAPES)
def test__indices_master(nnd, nnz, size):
    res_inp = _make_master_coo(nnd, nnz, size, seed=3)
    ref_inp = utils.to_reference(res_inp)

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = flag_gems._indices(res_inp)

    assert res_out.dtype == torch.int64
    assert res_out.shape == (nnd, nnz)
    utils.gems_assert_equal(res_out, ref_inp._indices())
    utils.gems_assert_equal(res_out, ref_out)



@pytest.mark.underscore_indices
def test__indices_master_uncoalesced_keeps_coordinates():
    # The public indices() requires coalescence; _indices does not, and both
    # native and this implementation return the stored coordinates as-is.
    # The shared ``_make_coo`` helper coalesces its result for the public
    # accessor, so build the raw (uncoalesced) tensor directly here.
    gen = torch.Generator(device="cpu").manual_seed(5)
    flat = torch.randperm(16, generator=gen)[:8]
    raw = torch.stack([flat % 4, flat // 4]).contiguous()
    res_inp = torch.sparse_coo_tensor(
        raw.to(flag_gems.device),
        torch.randn(8, generator=gen).to(flag_gems.device),
        (4, 4),
    )
    assert not res_inp.is_coalesced()
    ref_out = torch.ops.aten._indices(utils.to_reference(res_inp))
    res_out = flag_gems._indices(res_inp)
    utils.gems_assert_equal(res_out, ref_out)



@pytest.mark.underscore_indices
def test__indices_master_empty():
    res_inp = torch.sparse_coo_tensor(
        torch.zeros((2, 0), dtype=torch.int64, device=flag_gems.device),
        torch.zeros((0,), device=flag_gems.device),
        (4, 4),
    )
    res_out = flag_gems._indices(res_inp)
    assert res_out.shape == (2, 0)



@pytest.mark.underscore_indices
def test__indices_master_hybrid():
    # A hybrid COO tensor keeps its dense dimensions in values; the coordinate
    # table stays (sparse_dim, nnz).
    res_inp = torch.sparse_coo_tensor(
        torch.tensor([[1, 0], [0, 1]], device=flag_gems.device),
        torch.randn(2, 3, 4, device=flag_gems.device),
        (2, 2, 3, 4),
    )
    res_out = flag_gems._indices(res_inp)
    ref_out = torch.ops.aten._indices(utils.to_reference(res_inp))
    assert res_out.shape == (2, 2)
    utils.gems_assert_equal(res_out, ref_out)



@pytest.mark.underscore_indices
def test__indices_master_alias_and_mutation():
    # The schema ``Tensor(a) self -> Tensor(a)`` makes the result an aliasing
    # view: two reads share storage, and mutating the result mutates the
    # sparse tensor's stored indices. A copying implementation breaks both.
    res_inp = _make_master_coo(2, 8, (4, 4), seed=7)

    first = flag_gems._indices(res_inp)
    second = flag_gems._indices(res_inp)
    assert (
        first.data_ptr() == second.data_ptr()
    ), "_indices must return an aliasing view, not a fresh copy"
    assert first.data_ptr() == res_inp._indices().data_ptr()

    first[0, 0] = 77
    assert (
        res_inp._indices()[0, 0].item() == 77
    ), "mutating the returned view must be visible in the sparse tensor"



@pytest.mark.underscore_indices
@pytest.mark.parametrize("layout", ["csr", "csc"], ids=["csr", "csc"])
def test__indices_master_unsupported_layout_raises(layout):
    # Native ATen has no _indices kernel outside sparse COO; the rejection is
    # a NotImplementedError from the dispatcher, mirrored here.
    indptr = torch.tensor([0, 2, 4], dtype=torch.int64, device=flag_gems.device)
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=flag_gems.device)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], device=flag_gems.device)
    if layout == "csr":
        inp = torch.sparse_csr_tensor(indptr, indices, values, size=(2, 4))
    else:
        inp = torch.sparse_csc_tensor(indptr, indices, values, size=(2, 4))
    with pytest.raises(NotImplementedError):
        flag_gems._indices(inp)



@pytest.mark.underscore_indices
def test__indices_master_dense_rejected():
    inp = torch.randn(2, 3, device=flag_gems.device)
    with pytest.raises(NotImplementedError):
        flag_gems._indices(inp)
