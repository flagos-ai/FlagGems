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

# aten::_sparse_compressed_tensor_unsafe(compressed_indices, plain_indices, values,
#     int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None,
#     bool? pin_memory=None) -> Tensor
#
# Unchecked CSR/CSC/BSR/BSC constructor: it skips
# `_validate_sparse_compressed_tensor_args`, so the caller's components and `size`
# are stored verbatim, including strides, storage offsets and storage sharing.
#
# One workload is a (layout, batch, base, dense, blocks, nnz) description from which
# the compressed/plain indices, the values shape and the logical `size` are derived.
# tu.selected_shapes() cannot describe a compressed layout on its own (()/(1,)/(256,)
# have no compressed + plain pair), so the rank >= 2 spec shapes appear as extents.
# The components are structural and never combined: no broadcast, no scalar operand,
# and no backward (the result carries no autograd history).

_ROW_MAJOR_LAYOUTS = (torch.sparse_csr, torch.sparse_bsr)
_BLOCKED_LAYOUTS = (torch.sparse_bsr, torch.sparse_bsc)

# Static capability flags of the active backend; the values component is an ordinary
# strided tensor, so only dtypes the backend cannot materialise are dropped.
_DTYPE_GATES = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.float64: utils.fp64_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
}


def _supported(dtypes):
    return [dtype for dtype in dtypes if _DTYPE_GATES.get(dtype, True)]


_VALUES_DTYPES = _supported(tu.REQUIRED_DTYPES + [torch.float64, torch.bool])
_INDEX_DTYPES = _supported([torch.int32, torch.int64])
_MAIN_INDEX_DTYPE = torch.int64 if utils.int64_is_supported else torch.int32
# flag_gems.device is a plain device string in this checkout, not a torch.device.
_IS_CPU_BACKEND = torch.device(flag_gems.device).type == "cpu"


def _sparse_extents(layout, base, blocks):
    block_rows, block_cols = blocks
    rows, cols = base
    if layout in _ROW_MAJOR_LAYOUTS:
        return rows // block_rows, cols // block_cols
    return cols // block_cols, rows // block_rows


def _num_batches(batch):
    total = 1
    for extent in batch:
        total *= extent
    return total


def _values_shape(layout, batch, nnz, blocks, dense):
    block_shape = blocks if layout in _BLOCKED_LAYOUTS else ()
    return (*batch, nnz, *block_shape, *dense)


def _check_description(layout, batch, base, dense, blocks, nnz):
    assert all(extent >= 0 for extent in (*batch, *base, *dense, nnz))
    # A zero-size block aborts the process natively (TORCH_INTERNAL_ASSERT), not raises.
    assert blocks[0] >= 1 and blocks[1] >= 1, blocks
    assert base[0] % blocks[0] == 0 and base[1] % blocks[1] == 0, (base, blocks)
    compressed_dim, plain_dim = _sparse_extents(layout, base, blocks)
    if compressed_dim == 0:
        assert nnz == 0, nnz
    else:
        quotient, remainder = divmod(nnz, compressed_dim)
        assert quotient + (1 if remainder else 0) <= plain_dim, (
            nnz,
            compressed_dim,
            plain_dim,
        )


def _build_indices(layout, batch, base, dense, blocks, nnz, index_dtype):
    """Deterministic in-range compressed/plain indices plus the logical size."""
    _check_description(layout, batch, base, dense, blocks, nnz)
    compressed_dim, plain_dim = _sparse_extents(layout, base, blocks)
    quotient, remainder = divmod(nnz, compressed_dim) if compressed_dim else (0, 0)

    compressed_flat = []
    plain_flat = []
    for batch_index in range(_num_batches(batch)):
        compressed_row = [0]
        for row in range(compressed_dim):
            count = quotient + (
                1 if ((row + batch_index) % compressed_dim) < remainder else 0
            )
            compressed_row.append(compressed_row[-1] + count)
            if count:
                start = ((batch_index + 1) * 7 + row * 3) % (plain_dim - count + 1)
                plain_flat.extend(range(start, start + count))
        compressed_flat.extend(compressed_row)

    compressed = torch.tensor(
        compressed_flat, dtype=index_dtype, device=flag_gems.device
    ).reshape(*batch, compressed_dim + 1)
    plain = torch.tensor(
        plain_flat, dtype=index_dtype, device=flag_gems.device
    ).reshape(*batch, nnz)
    size = [*batch, base[0], base[1], *dense]
    return compressed, plain, size


def _description(
    layout, batch, base, dense, blocks, nnz, index_dtype, values_dtype, value_range
):
    compressed, plain, size = _build_indices(
        layout, batch, base, dense, blocks, nnz, index_dtype
    )
    values = tu.make_input(
        values_dtype, _values_shape(layout, batch, nnz, blocks, dense), value_range
    )
    return compressed, plain, values, size


def _stored_indices(res):
    """The (compressed, plain) index tensors actually held by ``res``."""
    if res.layout in _ROW_MAJOR_LAYOUTS:
        return torch.ops.aten.crow_indices(res), torch.ops.aten.col_indices(res)
    return torch.ops.aten.ccol_indices(res), torch.ops.aten.row_indices(res)


_SMALL_CSR = (torch.sparse_csr, (), (3, 4), (), (1, 1), 5)
_SMALL_CSC = (torch.sparse_csc, (), (4, 3), (), (1, 1), 5)
_SMALL_BSR = (torch.sparse_bsr, (), (6, 8), (), (2, 2), 5)
_SMALL_BSC = (torch.sparse_bsc, (), (8, 6), (), (2, 2), 5)
_SMALL_EMPTY = (torch.sparse_csr, (), (7, 13), (), (1, 1), 0)

# Every description whose components are small stays in the quick level as well:
# all four layouts, an empty description, non-square blocks, batch, dense tail,
# batch + dense, and the zero extents.
_SMALL_DESCRIPTORS = [
    _SMALL_CSR,
    _SMALL_CSC,
    _SMALL_BSR,
    _SMALL_BSC,
    _SMALL_EMPTY,
    (torch.sparse_bsr, (), (6, 9), (), (2, 3), 4),
    (torch.sparse_bsc, (), (10, 6), (), (2, 3), 4),
    (torch.sparse_csr, (4,), (16, 128), (), (1, 1), 64),
    (torch.sparse_csr, (), (6, 6), (2, 3), (1, 1), 4),
    (torch.sparse_csr, (1,), (1, 3), (1,), (1, 1), 1),
    (torch.sparse_bsc, (2, 3), (8, 10), (), (1, 1), 9),
    (torch.sparse_csr, (0,), (3, 4), (), (1, 1), 0),
    (torch.sparse_csr, (), (3, 4), (0,), (1, 1), 0),
]

# The rank >= 2 spec shapes as base + batch/dense extents, plus the other layouts at
# the largest extent. These are the only descriptions the quick level drops.
_LARGE_DESCRIPTORS = [
    (torch.sparse_csr, (), (1024, 1024), (), (1, 1), 256),
    (torch.sparse_csr, (20,), (320, 15), (), (1, 1), 120),
    (torch.sparse_csr, (16,), (128, 64), (60,), (1, 1), 512),
    (torch.sparse_csr, (16, 7), (57, 32), (29,), (1, 1), 100),
    (torch.sparse_csc, (), (1024, 1024), (), (1, 1), 256),
    (torch.sparse_bsr, (), (1024, 1024), (), (2, 2), 256),
]

_DESCRIPTORS = tu.selected_cases(
    _SMALL_DESCRIPTORS + _LARGE_DESCRIPTORS, quick=_SMALL_DESCRIPTORS
)


@pytest.mark.sparse_compressed_tensor_unsafe
@pytest.mark.parametrize("layout,batch,base,dense,blocks,nnz", _DESCRIPTORS)
@pytest.mark.parametrize("values_dtype", _VALUES_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__sparse_compressed_tensor_unsafe(
    layout, batch, base, dense, blocks, nnz, values_dtype, value_range
):
    compressed, plain, values, size = _description(
        layout,
        batch,
        base,
        dense,
        blocks,
        nnz,
        _MAIN_INDEX_DTYPE,
        values_dtype,
        value_range,
    )
    ref_compressed = tu.to_reference(compressed)
    ref_plain = tu.to_reference(plain)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten._sparse_compressed_tensor_unsafe(
        ref_compressed,
        ref_plain,
        ref_values,
        list(size),
        dtype=values_dtype,
        layout=layout,
        device=ref_values.device,
    )
    res_out = flag_gems._sparse_compressed_tensor_unsafe(
        compressed,
        plain,
        values,
        list(size),
        dtype=values_dtype,
        layout=layout,
        device=values.device,
    )

    # The result is a sparse wrapper over the caller's components, so observe the
    # stored indices/values on both sides instead of densifying.
    assert res_out.layout == layout
    assert res_out.shape == torch.Size(size)
    assert res_out.dtype == values_dtype
    assert res_out.device == values.device
    res_compressed, res_plain = _stored_indices(res_out)
    ref_stored_compressed, ref_stored_plain = _stored_indices(ref_out)
    tu.assert_result_equal(res_compressed, ref_stored_compressed)
    tu.assert_result_equal(res_plain, ref_stored_plain)
    tu.assert_result_equal(res_out.values(), ref_out.values())
    assert torch.equal(res_compressed, compressed)
    assert torch.equal(res_plain, plain)
    assert torch.equal(res_out.values(), values)
    # nnz is carried by the plain-index tensor, not by the logical shape.
    assert torch.ops.aten._nnz(res_out) == nnz


# All three are small, so the quick level keeps every index width.
_INDEX_DTYPE_DESCRIPTORS = [
    _SMALL_CSR,
    _SMALL_BSR,
    (torch.sparse_csc, (2, 3), (8, 10), (), (1, 1), 9),
]


@pytest.mark.sparse_compressed_tensor_unsafe
@pytest.mark.parametrize("layout,batch,base,dense,blocks,nnz", _INDEX_DTYPE_DESCRIPTORS)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__sparse_compressed_tensor_unsafe_index_dtype(
    layout, batch, base, dense, blocks, nnz, index_dtype, value_range
):
    compressed, plain, values, size = _description(
        layout,
        batch,
        base,
        dense,
        blocks,
        nnz,
        index_dtype,
        torch.float32,
        value_range,
    )
    ref_out = torch.ops.aten._sparse_compressed_tensor_unsafe(
        tu.to_reference(compressed),
        tu.to_reference(plain),
        tu.to_reference(values),
        list(size),
        dtype=torch.float32,
        layout=layout,
        device=values.device,
    )
    res_out = flag_gems._sparse_compressed_tensor_unsafe(
        compressed,
        plain,
        values,
        list(size),
        dtype=torch.float32,
        layout=layout,
        device=values.device,
    )

    res_compressed, res_plain = _stored_indices(res_out)
    ref_compressed, ref_plain = _stored_indices(ref_out)
    assert res_out.layout == layout
    tu.assert_result_equal(res_compressed, ref_compressed)
    tu.assert_result_equal(res_plain, ref_plain)
    tu.assert_result_equal(res_out.values(), ref_out.values())
    # The index width is independent of the values dtype and is stored as passed.
    assert res_compressed.dtype == index_dtype
    assert res_plain.dtype == index_dtype


# Descriptions the checked constructor rejects but this one stores verbatim: the
# stored indices and `size` are the caller's, and nnz still comes from the plain
# index tensor. Rows are (id, compressed, plain, num_values, size) and every row is
# small, so the quick level keeps them all.
_UNCHECKED_ROWS = [
    ("empty_size", [0, 2, 4, 5], [0, 1, 2, 0, 1], 5, []),
    ("rank1_size", [0, 2, 4, 5], [0, 1, 2, 0, 1], 5, [4]),
    ("rank1_size_bigger", [0, 2, 4, 5], [0, 1, 2, 0, 1], 5, [5]),
    ("size_wider_than_base", [0, 2, 4, 5], [0, 1, 2, 0, 1], 5, [99, 4]),
    ("unsorted_plain", [0, 2, 4, 5], [3, 0, 3, 0, 1], 5, [3, 4]),
    ("duplicate_plain", [0, 2, 4, 5], [1, 1, 1, 1, 1], 5, [3, 4]),
    ("out_of_range_plain", [0, 2, 4, 5], [9, 9, 9, 9, 9], 5, [3, 4]),
    ("negative_plain", [0, 2, 4, 5], [-1, -2, 2, 0, 1], 5, [3, 4]),
    ("short_compressed", [0, 2, 4], [0, 1, 2, 0, 1], 5, [3, 4]),
    ("long_compressed", [0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0], 5, [3, 4]),
    ("short_plain", [0, 2, 4, 5], [0, 1, 2], 5, [3, 4]),
    ("long_plain", [0, 2, 4, 5], [0, 1, 2, 0, 1, 2, 3], 5, [3, 4]),
    ("compressed_stores_nothing", [0, 0, 0, 0], [0, 1, 2, 0, 1], 5, [3, 4]),
]


@pytest.mark.sparse_compressed_tensor_unsafe
@pytest.mark.parametrize(
    "case_id,compressed_data,plain_data,num_values,size",
    _UNCHECKED_ROWS,
    ids=[row[0] for row in _UNCHECKED_ROWS],
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__sparse_compressed_tensor_unsafe_accepts_unchecked_descriptions(
    case_id, compressed_data, plain_data, num_values, size, value_range
):
    del case_id
    compressed = torch.tensor(
        compressed_data, dtype=_MAIN_INDEX_DTYPE, device=flag_gems.device
    )
    plain = torch.tensor(plain_data, dtype=_MAIN_INDEX_DTYPE, device=flag_gems.device)
    values = tu.make_input(torch.float32, (num_values,), value_range)

    ref_out = torch.ops.aten._sparse_compressed_tensor_unsafe(
        tu.to_reference(compressed),
        tu.to_reference(plain),
        tu.to_reference(values),
        list(size),
        dtype=torch.float32,
        layout=torch.sparse_csr,
        device=flag_gems.device,
    )
    res_out = flag_gems._sparse_compressed_tensor_unsafe(
        compressed,
        plain,
        values,
        list(size),
        dtype=torch.float32,
        layout=torch.sparse_csr,
        device=flag_gems.device,
    )

    # An inconsistent description cannot go through a whole-tensor comparison, so
    # observe the same stored components on both sides.
    assert res_out.layout == torch.sparse_csr
    assert res_out.shape == torch.Size(size)
    res_compressed, res_plain = _stored_indices(res_out)
    ref_compressed, ref_plain = _stored_indices(ref_out)
    tu.assert_result_equal(res_compressed, ref_compressed)
    tu.assert_result_equal(res_plain, ref_plain)
    tu.assert_result_equal(res_out.values(), ref_out.values())
    assert torch.equal(res_compressed, compressed)
    assert torch.equal(res_plain, plain)
    assert torch.equal(res_out.values(), values)
    assert torch.ops.aten._nnz(res_out) == plain.numel()


# Both rows are small, so the quick level keeps the whole alias contract.
_ALIAS_DESCRIPTORS = [
    (torch.sparse_csr, (3, 4), (1, 1), 5),
    (torch.sparse_bsr, (6, 8), (2, 2), 5),
]


@pytest.mark.sparse_compressed_tensor_unsafe
@pytest.mark.parametrize("layout,base,blocks,nnz", _ALIAS_DESCRIPTORS)
def test__sparse_compressed_tensor_unsafe_preserves_component_metadata(
    layout, base, blocks, nnz
):
    compressed, plain, size = _build_indices(
        layout, (), base, (), blocks, nnz, _MAIN_INDEX_DTYPE
    )
    # Offset and non-contiguous components: the constructor must keep the caller's
    # strides and storage offsets instead of compacting the components.
    compressed = torch.cat(
        [torch.zeros(2, dtype=_MAIN_INDEX_DTYPE, device=flag_gems.device), compressed]
    )[2:]
    plain = torch.cat(
        [torch.zeros(1, dtype=_MAIN_INDEX_DTYPE, device=flag_gems.device), plain]
    )[1:]
    block_shape = blocks if layout in _BLOCKED_LAYOUTS else ()
    buffer = torch.arange(
        2 * nnz * blocks[0] * blocks[1], dtype=torch.float32, device=flag_gems.device
    )
    values = buffer.reshape(2 * nnz, *block_shape)[1::2]

    ref_out = torch.ops.aten._sparse_compressed_tensor_unsafe(
        tu.to_reference(compressed),
        tu.to_reference(plain),
        tu.to_reference(values),
        list(size),
        dtype=torch.float32,
        layout=layout,
        device=flag_gems.device,
    )
    res_out = flag_gems._sparse_compressed_tensor_unsafe(
        compressed,
        plain,
        values,
        list(size),
        dtype=torch.float32,
        layout=layout,
        device=flag_gems.device,
    )

    res_compressed, res_plain = _stored_indices(res_out)
    ref_compressed, ref_plain = _stored_indices(ref_out)
    tu.assert_result_equal(res_compressed, ref_compressed)
    tu.assert_result_equal(res_plain, ref_plain)
    tu.assert_result_equal(res_out.values(), ref_out.values())
    assert res_out.values().stride() == values.stride()
    assert res_out.values().storage_offset() == values.storage_offset()
    assert res_compressed.stride() == compressed.stride()
    assert res_compressed.storage_offset() == compressed.storage_offset()
    assert res_plain.storage_offset() == plain.storage_offset()
    # New tensor objects over the caller's storage: identity differs, storage does not.
    assert res_out.values() is not values
    assert (
        res_out.values().untyped_storage().data_ptr()
        == values.untyped_storage().data_ptr()
    )
    # Mutating the result must be visible through the caller's tensors (no copy).
    res_out.values().fill_(7.0)
    assert torch.all(values == 7.0)


# dtype and pin_memory have schema defaults, so the omitted form is exercised too.
# The schema default for dtype is the global default dtype (float32) rather than the
# values dtype, so the omitted-dtype row uses float32 values.
_OPTIONAL_ARGUMENT_CASES = [
    ("dtype_and_pin_memory_omitted", None, None),
    ("dtype_explicit", torch.float32, None),
    ("pin_memory_false", torch.float32, False),
]


@pytest.mark.sparse_compressed_tensor_unsafe
@pytest.mark.parametrize(
    "case_id,dtype_arg,pin_memory_arg",
    _OPTIONAL_ARGUMENT_CASES,
    ids=[row[0] for row in _OPTIONAL_ARGUMENT_CASES],
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__sparse_compressed_tensor_unsafe_optional_arguments(
    case_id, dtype_arg, pin_memory_arg, value_range
):
    del case_id
    compressed, plain, values, size = _description(
        torch.sparse_csr,
        (),
        (3, 4),
        (),
        (1, 1),
        5,
        _MAIN_INDEX_DTYPE,
        torch.float32,
        value_range,
    )
    kwargs = {"layout": torch.sparse_csr, "device": flag_gems.device}
    if dtype_arg is not None:
        kwargs["dtype"] = dtype_arg
    if pin_memory_arg is not None:
        kwargs["pin_memory"] = pin_memory_arg

    ref_out = torch.ops.aten._sparse_compressed_tensor_unsafe(
        tu.to_reference(compressed),
        tu.to_reference(plain),
        tu.to_reference(values),
        list(size),
        **kwargs,
    )
    res_out = flag_gems._sparse_compressed_tensor_unsafe(
        compressed, plain, values, list(size), **kwargs
    )

    assert res_out.layout == torch.sparse_csr
    assert res_out.dtype == torch.float32
    res_compressed, res_plain = _stored_indices(res_out)
    tu.assert_result_equal(res_compressed, _stored_indices(ref_out)[0])
    tu.assert_result_equal(res_plain, _stored_indices(ref_out)[1])
    tu.assert_result_equal(res_out.values(), ref_out.values())


# The constructor is device-generic, so dense CPU components are part of its
# contract; only dense CPU tensors are pinnable, which is where pin_memory=False can
# be exercised as a positive form.
_CPU_CASES = [
    ("device_omitted", {}),
    ("pin_memory_false", {"dtype": torch.float32, "pin_memory": False}),
]


@pytest.mark.sparse_compressed_tensor_unsafe
@pytest.mark.parametrize(
    "case_id,kwargs", _CPU_CASES, ids=[row[0] for row in _CPU_CASES]
)
def test__sparse_compressed_tensor_unsafe_cpu_components(case_id, kwargs):
    del case_id
    compressed = torch.tensor([0, 2, 4, 5], dtype=_MAIN_INDEX_DTYPE, device="cpu")
    plain = torch.tensor([0, 1, 2, 0, 1], dtype=_MAIN_INDEX_DTYPE, device="cpu")
    values = tu.make_input(torch.float32, (5,), ["-1", "1"]).cpu()
    call_kwargs = {"layout": torch.sparse_csr}
    call_kwargs.update(kwargs)

    ref_out = torch.ops.aten._sparse_compressed_tensor_unsafe(
        compressed.clone(), plain.clone(), values.clone(), [3, 4], **call_kwargs
    )
    res_out = flag_gems._sparse_compressed_tensor_unsafe(
        compressed, plain, values, [3, 4], **call_kwargs
    )

    res_compressed, res_plain = _stored_indices(res_out)
    assert res_out.device == values.device
    tu.assert_result_equal(res_compressed, _stored_indices(ref_out)[0])
    tu.assert_result_equal(res_plain, _stored_indices(ref_out)[1])
    tu.assert_result_equal(res_out.values(), ref_out.values())


_VIOLATIONS = [
    "layout_sparse_coo",
    "layout_strided",
    "layout_omitted",
    "dtype_mismatch",
    "negative_size",
    "compressed_not_tensor",
]
if not _IS_CPU_BACKEND:
    # Accelerator tensors are not pinnable; dense CPU tensors are, so this violation
    # only exists on a non-CPU backend.
    _VIOLATIONS.append("pin_memory_true")


@pytest.mark.sparse_compressed_tensor_unsafe
@pytest.mark.parametrize("violation", _VIOLATIONS)
def test__sparse_compressed_tensor_unsafe_invalid_arguments(violation):
    compressed = torch.tensor(
        [0, 2, 4, 5], dtype=_MAIN_INDEX_DTYPE, device=flag_gems.device
    )
    plain = torch.tensor(
        [0, 1, 2, 0, 1], dtype=_MAIN_INDEX_DTYPE, device=flag_gems.device
    )
    values = tu.make_input(torch.float32, (5,), ["-1", "1"])
    size = [3, 4]
    kwargs = {
        "dtype": torch.float32,
        "layout": torch.sparse_csr,
        "device": flag_gems.device,
    }

    if violation == "layout_sparse_coo":
        kwargs["layout"] = torch.sparse_coo
    elif violation == "layout_strided":
        kwargs["layout"] = torch.strided
    elif violation == "layout_omitted":
        del kwargs["layout"]
    elif violation == "dtype_mismatch":
        kwargs["dtype"] = torch.float64
    elif violation == "negative_size":
        size = [-1, 4]
    elif violation == "compressed_not_tensor":
        compressed = [0, 2, 4, 5]
    elif violation == "pin_memory_true":
        kwargs["pin_memory"] = True
    else:
        raise AssertionError(violation)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._sparse_compressed_tensor_unsafe(
            compressed, plain, values, size, **kwargs
        )


# e4m3fn has no infinity encoding, so tu.special_value_cases emits only its nan
# scenario; e5m2 and the wider floats cover nan, inf and mixed. Only the stored
# values are compared, so no FP8 sparse arithmetic is assumed.
_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases(
        _supported(
            [
                torch.float32,
                torch.bfloat16,
                torch.float16,
                torch.float64,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            ]
        )
    ),
    quick=[],
)
_SPECIAL_DESCRIPTORS = tu.selected_cases(
    [(torch.sparse_csr, (3, 4), (1, 1)), (torch.sparse_bsr, (6, 8), (2, 2))],
    quick=[],
)


@pytest.mark.sparse_compressed_tensor_unsafe
@pytest.mark.parametrize("values_dtype,scenario", _SPECIAL_CASES)
@pytest.mark.parametrize("layout,base,blocks", _SPECIAL_DESCRIPTORS)
def test__sparse_compressed_tensor_unsafe_special_values(
    values_dtype, scenario, layout, base, blocks
):
    nnz = 5
    compressed, plain, size = _build_indices(
        layout, (), base, (), blocks, nnz, _MAIN_INDEX_DTYPE
    )
    payload = tu.make_special_input(values_dtype, scenario)
    if layout in _BLOCKED_LAYOUTS:
        payload = payload.reshape(-1, 1, 1).expand(-1, *blocks).reshape(-1, *blocks)

    ref_out = torch.ops.aten._sparse_compressed_tensor_unsafe(
        tu.to_reference(compressed),
        tu.to_reference(plain),
        tu.to_reference(payload),
        list(size),
        dtype=values_dtype,
        layout=layout,
        device=flag_gems.device,
    )
    res_out = flag_gems._sparse_compressed_tensor_unsafe(
        compressed,
        plain,
        payload,
        list(size),
        dtype=values_dtype,
        layout=layout,
        device=flag_gems.device,
    )

    assert torch.ops.aten._nnz(res_out) == nnz
    res_compressed, res_plain = _stored_indices(res_out)
    tu.assert_result_equal(res_compressed, _stored_indices(ref_out)[0])
    tu.assert_result_equal(res_plain, _stored_indices(ref_out)[1])
    tu.assert_result_equal(res_out.values(), ref_out.values())
