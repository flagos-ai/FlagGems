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

from . import test_utils as tu

# aten::_sparse_bsr_tensor_unsafe(Tensor crow_indices, Tensor col_indices,
#     Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None,
#     Device? device=None, bool? pin_memory=None) -> Tensor
#
# Host-side *unsafe* BSR constructor: it stores ``size`` verbatim and wraps the
# caller's crow/col/values without validating them and without touching element
# values.  The result therefore aliases the caller's component storage, is not
# differentiable and has no second operand, so the broadcast / backward /
# in-place spec dimensions do not apply; parameter, metadata, alias and negative
# coverage replace them.  The candidate is injected under the exact operator
# name, underscore included, and only the pytest marker strips it.

SUPPORTED_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.float64,
    torch.bool,
]

# The shared helper filters scenarios per dtype: float8_e4m3fn cannot represent
# infinity, so only its nan case is collected, while float8_e5m2 keeps nan, inf
# and the mixed payload.
SPECIAL_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]
SPECIAL_SIZES = [(8, 8), (3, 6, 4)]

# (size, block): declared extents tiled exactly by a non-square or unit block.
BLOCK_CASES = [
    ((6, 4), (3, 2)),
    ((8, 3), (4, 1)),
    ((3, 8), (1, 4)),
    ((6, 6), (2, 3)),
]

INDEX_DTYPES = [torch.int32, torch.int64]

# (size, dense_shape): trailing dense dims stored once per block entry.
DENSE_DIM_CASES = [((4, 4, 3), (3,)), ((2, 4, 4, 3), (3,)), ((6, 6, 2, 4), (2, 4))]

# Declared sizes the unsafe variant accepts and stores without validation.
SIZE_VERBATIM_CASES = [[], [3], [5, 7], [4, 4, 6]]

# Keyword arguments of the native call.  The empty row exercises the documented
# defaults (dtype=float32, layout=sparse_bsr, device=cpu, pin_memory=False).
OPTIONAL_KWARG_CASES = [
    {},
    {"dtype": torch.float32},
    {"dtype": torch.float32, "device": torch.device("cpu")},
    {"layout": torch.sparse_bsr},
    {"pin_memory": False},
    {"pin_memory": True},
]


def _batch_dims(size):
    """Batch dims of a BSR tensor: everything before the trailing (rows, cols)."""
    return tuple(size[:-2]) if len(size) >= 2 else ()


def _block_for(size, block):
    """The requested block shape, or one that tiles ``size`` exactly."""
    if block is not None:
        return block
    if len(size) < 2:
        return (2, 2)
    rows, cols = size[-2], size[-1]
    # An odd declared extent is only tiled by a 1x1 block.
    return (2, 2) if rows % 2 == 0 and cols % 2 == 0 else (1, 1)


def _bsparse_components(
    size, dtype, value_range, *, block=None, index_dtype=torch.int32, dense_shape=()
):
    """Independent crow/col/values components for a BSR tensor of ``size``.

    One block is stored per row block, so ``crow`` is a plain prefix sum and
    every col index stays in range; ``values`` is sized from the declared extent
    and the block shape.
    """
    matrix_size = size[: -len(dense_shape)] if dense_shape else size
    batch = _batch_dims(matrix_size)
    block = _block_for(matrix_size, block)
    if len(matrix_size) >= 2:
        row_blocks = max(matrix_size[-2] // block[0], 1)
        col_blocks = max(matrix_size[-1] // block[1], 1)
    else:
        # A rank<2 size declares no sparse extent; a single block row keeps the
        # components minimal while the stored size stays unconstrained.
        row_blocks, col_blocks = 1, 1
    crow_1d = torch.arange(row_blocks + 1, dtype=index_dtype, device=flag_gems.device)
    col_1d = torch.arange(row_blocks, dtype=index_dtype, device=flag_gems.device)
    col_1d = col_1d % col_blocks
    if batch:
        crow = crow_1d.repeat(*batch, 1)
        col = col_1d.repeat(*batch, 1)
    else:
        crow, col = crow_1d, col_1d
    values_shape = batch + (row_blocks, block[0], block[1]) + tuple(dense_shape)
    return crow, col, tu.make_input(dtype, values_shape, value_range)


def _bsparse_special_components(size, dtype, scenario):
    """Components whose stored blocks carry the shared nan/inf payload."""
    crow, col, template = _bsparse_components(size, dtype, ["-1", "1"])
    payload = tu.make_special_input(dtype, scenario)
    count = template.numel()
    values = payload.repeat(-(-count // payload.numel()))[:count].reshape(
        template.shape
    )
    return crow, col, values


def _assert_bsr_matches(res, ref):
    """Compare two BSR constructions through their stored components."""
    assert res.layout == ref.layout == torch.sparse_bsr
    assert res.shape == ref.shape
    assert res.sparse_dim() == ref.sparse_dim()
    assert res.dense_dim() == ref.dense_dim()
    tu.assert_result_equal(res.crow_indices(), ref.crow_indices())
    tu.assert_result_equal(res.col_indices(), ref.col_indices())
    tu.assert_result_equal(res.values(), ref.values())


@pytest.mark.sparse_bsr_tensor_unsafe
@pytest.mark.parametrize("size", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_sparse_bsr_tensor_unsafe(size, value_range, dtype):
    crow, col, values = _bsparse_components(size, dtype, value_range)
    ref_crow = tu.to_reference(crow)
    ref_col = tu.to_reference(col)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten._sparse_bsr_tensor_unsafe(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_values.device
    )
    res_out = flag_gems._sparse_bsr_tensor_unsafe(
        crow, col, values, list(size), dtype=dtype, device=flag_gems.device
    )

    assert res_out.dtype == dtype
    assert res_out.device == values.device
    _assert_bsr_matches(res_out, ref_out)


@pytest.mark.sparse_bsr_tensor_unsafe
@pytest.mark.parametrize("size,block", BLOCK_CASES)
def test_sparse_bsr_tensor_unsafe_block_shapes(size, block):
    crow, col, values = _bsparse_components(
        size, torch.float32, ["-1", "1"], block=block
    )
    ref_crow = tu.to_reference(crow)
    ref_col = tu.to_reference(col)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten._sparse_bsr_tensor_unsafe(
        ref_crow,
        ref_col,
        ref_values,
        list(size),
        dtype=torch.float32,
        device=ref_values.device,
    )
    res_out = flag_gems._sparse_bsr_tensor_unsafe(
        crow, col, values, list(size), dtype=torch.float32, device=flag_gems.device
    )

    # The block shape comes from the trailing values dims, not from ``size``.
    assert tuple(res_out.values().shape[-2:]) == block
    _assert_bsr_matches(res_out, ref_out)


@pytest.mark.sparse_bsr_tensor_unsafe
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_sparse_bsr_tensor_unsafe_index_dtypes(index_dtype):
    crow, col, values = _bsparse_components(
        (8, 8), torch.float32, ["-1", "1"], index_dtype=index_dtype
    )
    ref_crow = tu.to_reference(crow)
    ref_col = tu.to_reference(col)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten._sparse_bsr_tensor_unsafe(
        ref_crow,
        ref_col,
        ref_values,
        [8, 8],
        dtype=torch.float32,
        device=ref_values.device,
    )
    res_out = flag_gems._sparse_bsr_tensor_unsafe(
        crow, col, values, [8, 8], dtype=torch.float32, device=flag_gems.device
    )

    # The caller's index dtype is stored as part of the metadata.
    assert res_out.crow_indices().dtype == index_dtype
    assert res_out.col_indices().dtype == index_dtype
    _assert_bsr_matches(res_out, ref_out)


@pytest.mark.sparse_bsr_tensor_unsafe
def test_sparse_bsr_tensor_unsafe_values_view_metadata():
    # The constructor wraps the caller's tensor as-is, so a strided values view
    # must keep its storage, offset and strides in the result.
    base = tu.make_input(torch.float32, (4, 4, 4), ["-1", "1"])
    values = base[1:3].transpose(-1, -2)
    crow = torch.tensor([0, 1, 2], dtype=torch.int32, device=flag_gems.device)
    col = torch.tensor([0, 1], dtype=torch.int32, device=flag_gems.device)
    ref_crow = tu.to_reference(crow)
    ref_col = tu.to_reference(col)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten._sparse_bsr_tensor_unsafe(
        ref_crow,
        ref_col,
        ref_values,
        [8, 8],
        dtype=torch.float32,
        device=ref_values.device,
    )
    res_out = flag_gems._sparse_bsr_tensor_unsafe(
        crow, col, values, [8, 8], dtype=torch.float32, device=flag_gems.device
    )

    # Compared against the input view rather than the reference tensor, because
    # the shared helper may relocate the reference to another device (which would
    # allocate a fresh, offset-0 storage for it).
    assert res_out.values().data_ptr() == values.data_ptr()
    assert res_out.values().storage_offset() == values.storage_offset()
    assert res_out.values().stride() == values.stride()
    _assert_bsr_matches(res_out, ref_out)


@pytest.mark.sparse_bsr_tensor_unsafe
@pytest.mark.parametrize("size,dense_shape", DENSE_DIM_CASES)
def test_sparse_bsr_tensor_unsafe_dense_dims(size, dense_shape):
    crow, col, values = _bsparse_components(
        size, torch.float32, ["-1", "1"], dense_shape=dense_shape
    )
    ref_crow = tu.to_reference(crow)
    ref_col = tu.to_reference(col)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten._sparse_bsr_tensor_unsafe(
        ref_crow,
        ref_col,
        ref_values,
        list(size),
        dtype=torch.float32,
        device=ref_values.device,
    )
    res_out = flag_gems._sparse_bsr_tensor_unsafe(
        crow, col, values, list(size), dtype=torch.float32, device=flag_gems.device
    )

    assert res_out.sparse_dim() == 2
    # dense_dim() counts the trailing dense dims of ``values``; the batch dims
    # declared by ``size`` (or the batch override) are not dense dims.
    assert res_out.dense_dim() == len(dense_shape)
    _assert_bsr_matches(res_out, ref_out)


@pytest.mark.sparse_bsr_tensor_unsafe
@pytest.mark.parametrize(
    "size",
    SIZE_VERBATIM_CASES,
    ids=["scalar", "1d", "extent-mismatch", "extra-dense"],
)
def test_sparse_bsr_tensor_unsafe_stores_size_verbatim(size):
    # The unsafe variant performs no validation: it stores ``size`` as given,
    # even when the declared extent disagrees with the stored block grid.
    crow, col, values = _bsparse_components((), torch.float32, ["-1", "1"])
    ref_crow = tu.to_reference(crow)
    ref_col = tu.to_reference(col)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten._sparse_bsr_tensor_unsafe(
        ref_crow,
        ref_col,
        ref_values,
        list(size),
        dtype=torch.float32,
        device=ref_values.device,
    )
    res_out = flag_gems._sparse_bsr_tensor_unsafe(
        crow, col, values, list(size), dtype=torch.float32, device=flag_gems.device
    )

    assert res_out.shape == torch.Size(size)
    _assert_bsr_matches(res_out, ref_out)


@pytest.mark.sparse_bsr_tensor_unsafe
def test_sparse_bsr_tensor_unsafe_zero_nnz():
    crow = torch.zeros(5, dtype=torch.int32, device=flag_gems.device)
    col = torch.zeros(0, dtype=torch.int32, device=flag_gems.device)
    values = torch.zeros((0, 2, 2), dtype=torch.float32, device=flag_gems.device)
    ref_crow = tu.to_reference(crow)
    ref_col = tu.to_reference(col)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten._sparse_bsr_tensor_unsafe(
        ref_crow,
        ref_col,
        ref_values,
        [8, 8],
        dtype=torch.float32,
        device=ref_values.device,
    )
    res_out = flag_gems._sparse_bsr_tensor_unsafe(
        crow, col, values, [8, 8], dtype=torch.float32, device=flag_gems.device
    )

    assert res_out._nnz() == 0
    _assert_bsr_matches(res_out, ref_out)


@pytest.mark.sparse_bsr_tensor_unsafe
def test_sparse_bsr_tensor_unsafe_wraps_components_without_copying():
    crow, col, values = _bsparse_components((8, 8), torch.float32, ["-1", "1"])
    crow_before = crow.clone()
    col_before = col.clone()
    values_before = values.clone()
    ref_crow = tu.to_reference(crow)
    ref_col = tu.to_reference(col)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten._sparse_bsr_tensor_unsafe(
        ref_crow,
        ref_col,
        ref_values,
        [8, 8],
        dtype=torch.float32,
        device=ref_values.device,
    )
    res_out = flag_gems._sparse_bsr_tensor_unsafe(
        crow, col, values, [8, 8], dtype=torch.float32, device=flag_gems.device
    )

    # The native constructor aliases its component storage instead of copying,
    # and the candidate must reproduce that contract.
    assert ref_out.values().data_ptr() == ref_values.data_ptr()
    assert res_out.values().data_ptr() == values.data_ptr()
    assert res_out.crow_indices().data_ptr() == crow.data_ptr()
    assert res_out.col_indices().data_ptr() == col.data_ptr()
    # Constructing is not differentiable and returns a fresh leaf, so there is
    # no original operator input to differentiate (no backward case exists).
    assert res_out.requires_grad is False
    assert res_out.grad_fn is None
    _assert_bsr_matches(res_out, ref_out)
    # The three components are inputs and stay untouched.
    tu.assert_result_equal(crow, crow_before)
    tu.assert_result_equal(col, col_before)
    tu.assert_result_equal(values, values_before)


@pytest.mark.sparse_bsr_tensor_unsafe
@pytest.mark.parametrize("kwargs", OPTIONAL_KWARG_CASES)
def test_sparse_bsr_tensor_unsafe_optional_arguments(kwargs):
    # The native default device is CPU and pin_memory only applies to dense CPU
    # components, so this probe feeds the same real CPU components to both the
    # reference and the injected candidate.
    cpu = torch.device("cpu")
    crow, col, values = _bsparse_components((4, 4), torch.float32, ["-1", "1"])
    crow, col, values = crow.to(cpu), col.to(cpu), values.to(cpu)
    ref_out = torch.ops.aten._sparse_bsr_tensor_unsafe(
        tu.to_reference(crow),
        tu.to_reference(col),
        tu.to_reference(values),
        [4, 4],
        **kwargs,
    )
    res_out = flag_gems._sparse_bsr_tensor_unsafe(crow, col, values, [4, 4], **kwargs)

    assert res_out.device == torch.device("cpu")
    assert res_out.is_pinned() == ref_out.is_pinned()
    _assert_bsr_matches(res_out, ref_out)


SPECIAL_CASES = [
    (dtype, scenario, size)
    for dtype, scenario in tu.special_value_cases(SPECIAL_DTYPES)
    for size in SPECIAL_SIZES
]


@pytest.mark.sparse_bsr_tensor_unsafe
@pytest.mark.parametrize(
    "dtype,scenario,size", tu.selected_cases(SPECIAL_CASES, quick=[])
)
def test_sparse_bsr_tensor_unsafe_special_values(dtype, scenario, size):
    crow, col, values = _bsparse_special_components(size, dtype, scenario)
    ref_crow = tu.to_reference(crow)
    ref_col = tu.to_reference(col)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten._sparse_bsr_tensor_unsafe(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_values.device
    )
    res_out = flag_gems._sparse_bsr_tensor_unsafe(
        crow, col, values, list(size), dtype=dtype, device=flag_gems.device
    )

    # Stored payloads are compared with the shared helper, which matches NaNs
    # and never narrows the tested dtype.
    _assert_bsr_matches(res_out, ref_out)


@pytest.mark.sparse_bsr_tensor_unsafe
@pytest.mark.parametrize("position", ["crow_indices", "col_indices", "values"])
def test_sparse_bsr_tensor_unsafe_rejects_non_tensor_components(position):
    crow, col, values = _bsparse_components((4, 4), torch.float32, ["-1", "1"])
    components = {"crow_indices": crow, "col_indices": col, "values": values}
    components[position] = [[0, 1, 2]]

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._sparse_bsr_tensor_unsafe(
            components["crow_indices"],
            components["col_indices"],
            components["values"],
            [4, 4],
            dtype=torch.float32,
            device=flag_gems.device,
        )


@pytest.mark.sparse_bsr_tensor_unsafe
@pytest.mark.parametrize("size", [[4.0, 4.0], ["4", "4"]], ids=["float", "str"])
def test_sparse_bsr_tensor_unsafe_rejects_non_int_size(size):
    crow, col, values = _bsparse_components((4, 4), torch.float32, ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._sparse_bsr_tensor_unsafe(
            crow, col, values, size, dtype=torch.float32, device=flag_gems.device
        )


@pytest.mark.sparse_bsr_tensor_unsafe
@pytest.mark.parametrize(
    "kwargs",
    [
        {"layout": torch.sparse_csr},
        {"layout": "sparse_bsr"},
        {"dtype": "float32"},
        {"dtype": torch.float64},
    ],
    ids=["csr-layout", "str-layout", "str-dtype", "dtype-mismatch"],
)
def test_sparse_bsr_tensor_unsafe_rejects_invalid_kwargs(kwargs):
    crow, col, values = _bsparse_components((4, 4), torch.float32, ["-1", "1"])
    call_kwargs = {"dtype": torch.float32, "device": flag_gems.device}
    call_kwargs.update(kwargs)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._sparse_bsr_tensor_unsafe(crow, col, values, [4, 4], **call_kwargs)


@pytest.mark.sparse_bsr_tensor_unsafe
def test_sparse_bsr_tensor_unsafe_requires_size():
    crow, col, values = _bsparse_components((4, 4), torch.float32, ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._sparse_bsr_tensor_unsafe(
            crow, col, values, dtype=torch.float32, device=flag_gems.device
        )


@pytest.mark.sparse_bsr_tensor_unsafe
def test_sparse_bsr_tensor_unsafe_rejects_device_mismatch():
    # The components and the requested output device must agree; "meta" is a
    # real, backend-independent second device, so this row does not depend on a
    # CPU-only build.
    crow, col, values = _bsparse_components((4, 4), torch.float32, ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._sparse_bsr_tensor_unsafe(
            crow,
            col,
            values,
            [4, 4],
            dtype=torch.float32,
            device=torch.device("meta"),
        )


@pytest.mark.sparse_bsr_tensor_unsafe
@pytest.mark.skipif(
    torch.device(flag_gems.device).type == "cpu",
    reason="pinning is only rejected when the component allocation is not on the CPU",
)
def test_sparse_bsr_tensor_unsafe_rejects_pin_memory_for_accelerator_components():
    crow, col, values = _bsparse_components((4, 4), torch.float32, ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._sparse_bsr_tensor_unsafe(
            crow,
            col,
            values,
            [4, 4],
            dtype=torch.float32,
            device=flag_gems.device,
            pin_memory=True,
        )


@pytest.mark.sparse_bsr_tensor_unsafe
@pytest.mark.parametrize(
    "dtype",
    [
        dtype
        for dtype in SUPPORTED_DTYPES
        if dtype.is_floating_point or dtype.is_complex
    ],
)
def test__sparse_bsr_tensor_unsafe_requires_grad_input(dtype):
    # Unsafe factories alias the payload but do not create an autograd edge.
    compressed = torch.tensor([0, 1, 2], dtype=torch.int64, device=flag_gems.device)
    plain = torch.tensor([0, 1], dtype=torch.int64, device=flag_gems.device)
    values_shape = (2, 2, 2)
    size = [4, 4]
    values = tu.make_input(dtype, values_shape, ["-1", "1"]).requires_grad_()
    ref_out = torch.ops.aten._sparse_bsr_tensor_unsafe(
        compressed,
        plain,
        values,
        size,
        dtype=dtype,
        layout=torch.sparse_bsr,
        device=values.device,
    )
    res_out = flag_gems._sparse_bsr_tensor_unsafe(
        compressed,
        plain,
        values,
        size,
        dtype=dtype,
        layout=torch.sparse_bsr,
        device=values.device,
    )

    assert not res_out.requires_grad
    assert res_out.requires_grad == ref_out.requires_grad
    assert res_out.grad_fn is ref_out.grad_fn is None
    assert res_out.values().data_ptr() == values.data_ptr()
    tu.assert_result_equal(res_out.values(), ref_out.values())
