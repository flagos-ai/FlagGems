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

# Correctness tests for aten::_nested_view_from_buffer. The operator returns a
# zero-copy nested view of a flat buffer's storage: component i is
# as_strided(size_i, stride_i, offsets_i) over that storage, so element
# comparisons are exact. Only the default overload exists (no .out).

import pytest
import torch

import flag_gems

from . import conftest as cfg
from . import test_utils as tu


def _metadata(sizes, strides, offsets):
    # The native path reads the metadata on the host, so these tensors stay on
    # CPU while the buffer follows flag_gems.device.
    return (
        torch.tensor(sizes, dtype=torch.int64),
        torch.tensor(strides, dtype=torch.int64),
        torch.tensor(offsets, dtype=torch.int64),
    )


def _contiguous_strides(shape):
    strides = [1] * len(shape)
    for dim in range(len(shape) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * shape[dim + 1]
    return strides


def _packed_metadata(shape, count=2):
    # count packed components of `shape`: stride 1 and no gaps between them.
    strides = _contiguous_strides(shape)
    numel = 1
    for dim in shape:
        numel *= dim
    return (
        [list(shape)] * count,
        [list(strides)] * count,
        [i * numel for i in range(count)],
    )


def _buffer_length(sizes, strides, offsets):
    # Smallest buffer the native in-storage check accepts.
    length = 1
    for size, stride, offset in zip(sizes, strides, offsets):
        span = sum((dim - 1) * step for dim, step in zip(size, stride))
        length = max(length, offset + span + 1)
    return length


# The operator only re-indexes storage, so every available buffer dtype works.
# float64 is an optional extra: keep it only where the backend can create it.
EXTRA_DTYPE_CAPABILITY = {torch.float64: "support_fp64"}


def _dtype_supported(dtype):
    capability = EXTRA_DTYPE_CAPABILITY.get(dtype)
    if capability is None:
        return True
    return bool(getattr(flag_gems.runtime.device, capability, False))


def _extra_dtypes(*dtypes):
    return [dtype for dtype in dtypes if _dtype_supported(dtype)]


GRID_DTYPES = list(tu.REQUIRED_DTYPES) + [torch.bool] + _extra_dtypes(torch.float64)

# (label, component sizes, component strides, component offsets). The native
# operator accepts gapped, scalar, empty, transposed and overlapping components.
# Every row uses at most 17 buffer elements, so all of them stay in --quick.
LAYOUTS = [
    ("packed", [[4], [6]], [[1], [1]], [0, 4]),
    ("aligned_rows", [[3, 4], [2, 4]], [[8, 1], [8, 1]], [0, 16]),
    ("single_component", [[5]], [[1]], [0]),
    ("zero_size_component", [[0], [3]], [[1], [1]], [0, 0]),
    ("scalar_components", [[], []], [[], []], [0, 1]),
    ("transposed_2d", [[4, 3], [4, 3]], [[1, 4], [1, 4]], [0, 12]),
    ("overlapping_components", [[4], [4]], [[1], [1]], [0, 0]),
]

SPECIAL_DTYPES = [
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
] + _extra_dtypes(torch.float64)
SPECIAL_CASES = tu.special_value_cases(SPECIAL_DTYPES)

# FP8 gradients are supported on the GPU reference path but not when the
# reference runs on CPU (the nested-view backward raises "create_nt_buffer not
# implemented for Float8_e4m3fn/e5m2" there), so only CPU references omit them.
BACKWARD_DTYPES = (
    [torch.float32, torch.bfloat16, torch.float16]
    + _extra_dtypes(torch.float64)
    + ([] if cfg.TO_CPU else [torch.float8_e4m3fn, torch.float8_e5m2])
)


def _assert_components(res_out, ref_out, sizes, strides):
    # A legacy nested tensor rejects .shape/.size(), so the requested layout is
    # checked per component. Shape and stride come straight from the metadata;
    # storage_offset is compared against the native reference because the
    # operator indexes the buffer's storage rather than the buffer view.
    res_components = res_out.unbind()
    ref_components = ref_out.unbind()
    assert len(res_components) == len(ref_components) == len(sizes)
    for res_component, ref_component, size, stride in zip(
        res_components, ref_components, sizes, strides
    ):
        assert list(res_component.shape) == list(size)
        assert list(res_component.stride()) == list(stride)
        assert res_component.storage_offset() == ref_component.storage_offset()
        tu.assert_result_equal(res_component, ref_component)


@pytest.mark.nested_view_from_buffer
@pytest.mark.parametrize("dtype", GRID_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", tu.selected_shapes())
def test_nested_view_from_buffer(shape, value_range, dtype):
    sizes, strides, offsets = _packed_metadata(shape)
    buffer = tu.make_input(
        dtype, (_buffer_length(sizes, strides, offsets),), value_range
    )

    ref_out = torch.ops.aten._nested_view_from_buffer(
        tu.to_reference(buffer), *_metadata(sizes, strides, offsets)
    )
    res_out = flag_gems._nested_view_from_buffer(
        buffer, *_metadata(sizes, strides, offsets)
    )

    assert res_out.is_nested
    assert res_out.dtype == dtype
    assert res_out.device == buffer.device
    assert res_out.size(0) == len(sizes)
    assert res_out.dim() == len(shape) + 1
    _assert_components(res_out, ref_out, sizes, strides)


@pytest.mark.nested_view_from_buffer
@pytest.mark.parametrize(
    "layout",
    tu.selected_cases(LAYOUTS, quick=LAYOUTS),
    ids=[row[0] for row in LAYOUTS],
)
def test_nested_view_from_buffer_layout(layout):
    # dtype coverage lives in the grid test; these rows vary the metadata to
    # component-view mapping.
    _, sizes, strides, offsets = layout
    buffer = tu.make_input(
        torch.float32, (_buffer_length(sizes, strides, offsets),), ["-1", "1"]
    )

    ref_out = torch.ops.aten._nested_view_from_buffer(
        tu.to_reference(buffer), *_metadata(sizes, strides, offsets)
    )
    res_out = flag_gems._nested_view_from_buffer(
        buffer, *_metadata(sizes, strides, offsets)
    )

    _assert_components(res_out, ref_out, sizes, strides)


@pytest.mark.nested_view_from_buffer
def test_nested_view_from_buffer_buffer_view():
    # The metadata indexes the buffer's storage: a buffer view's own
    # storage_offset and stride do not shift the components (probed natively),
    # so the candidate must reproduce the reference components exactly.
    big = tu.make_input(torch.float32, (16,), ["-1", "1"])
    buffer = big[2:12:2]
    sizes, strides, offsets = [[2], [3]], [[1], [1]], [0, 2]

    ref_out = torch.ops.aten._nested_view_from_buffer(
        tu.to_reference(buffer), *_metadata(sizes, strides, offsets)
    )
    res_out = flag_gems._nested_view_from_buffer(
        buffer, *_metadata(sizes, strides, offsets)
    )

    assert (
        res_out.values().untyped_storage().data_ptr()
        == big.untyped_storage().data_ptr()
    )
    _assert_components(res_out, ref_out, sizes, strides)


@pytest.mark.nested_view_from_buffer
@pytest.mark.parametrize("dtype,scenario", tu.selected_cases(SPECIAL_CASES, quick=[]))
def test_nested_view_from_buffer_special_values(dtype, scenario):
    # Re-indexing storage must preserve NaN/Inf payloads bit for bit; the shared
    # generator owns the representable scenario set per dtype.
    buffer = tu.make_special_input(dtype, scenario)
    sizes, strides, offsets = [[2], [3]], [[1], [1]], [0, 2]

    ref_out = torch.ops.aten._nested_view_from_buffer(
        tu.to_reference(buffer), *_metadata(sizes, strides, offsets)
    )
    res_out = flag_gems._nested_view_from_buffer(
        buffer, *_metadata(sizes, strides, offsets)
    )

    _assert_components(res_out, ref_out, sizes, strides)


@pytest.mark.nested_view_from_buffer
@pytest.mark.parametrize("dtype", tu.selected_cases(BACKWARD_DTYPES, quick=[]))
def test_nested_view_from_buffer_backward(dtype):
    # Offsets [0, 4] tile the whole buffer: with gapped metadata the native
    # backward returns a gradient sized by the covered span, not by the buffer.
    sizes, strides, offsets = [[4], [4]], [[1], [1]], [0, 4]
    inp = tu.make_input(dtype, (8,), ["-1", "1"]).requires_grad_()
    ref_input = tu.to_reference(inp)
    ref_out = torch.ops.aten._nested_view_from_buffer(
        ref_input, *_metadata(sizes, strides, offsets)
    )
    res_out = flag_gems._nested_view_from_buffer(
        inp, *_metadata(sizes, strides, offsets)
    )
    upstream = [tu.make_input(dtype, (4,), ["-1", "1"]) for _ in sizes]
    ref_grad = torch.autograd.grad(
        ref_out.unbind(), ref_input, [tu.to_reference(g) for g in upstream]
    )[0]
    res_grad = torch.autograd.grad(res_out.unbind(), inp, upstream)[0]
    tu.assert_result_equal(res_grad, ref_grad)


@pytest.mark.nested_view_from_buffer
def test_nested_view_from_buffer_aliases():
    sizes, strides, offsets = [[4], [4]], [[1], [1]], [0, 4]
    buffer = tu.make_input(torch.float32, (8,), ["-1", "1"])
    ref_sizes, ref_strides, ref_offsets = _metadata(sizes, strides, offsets)
    res_sizes, res_strides, res_offsets = _metadata(sizes, strides, offsets)

    ref_out = torch.ops.aten._nested_view_from_buffer(
        tu.to_reference(buffer), ref_sizes, ref_strides, ref_offsets
    )
    res_out = flag_gems._nested_view_from_buffer(
        buffer, res_sizes, res_strides, res_offsets
    )

    # Contiguous metadata is stored as passed in, not re-created: identity, not
    # just equal values, on both the native and the candidate path.
    assert ref_out._nested_tensor_size() is ref_sizes
    assert ref_out._nested_tensor_strides() is ref_strides
    assert ref_out._nested_tensor_storage_offsets() is ref_offsets
    assert res_out._nested_tensor_size() is res_sizes
    assert res_out._nested_tensor_strides() is res_strides
    assert res_out._nested_tensor_storage_offsets() is res_offsets

    # No payload copy: the nested view and every component keep the original
    # buffer storage, at the requested offsets.
    assert (
        res_out.values().untyped_storage().data_ptr()
        == buffer.untyped_storage().data_ptr()
    )
    assert res_out.values().numel() == buffer.numel()
    for component, offset in zip(res_out.unbind(), offsets):
        assert (
            component.untyped_storage().data_ptr()
            == buffer.untyped_storage().data_ptr()
        )
        assert component.storage_offset() == offset


@pytest.mark.nested_view_from_buffer
def test_nested_view_from_buffer_mutation():
    sizes, strides, offsets = [[4], [4]], [[1], [1]], [0, 4]
    buffer = tu.make_input(torch.float32, (8,), ["-1", "1"])
    untouched = buffer[:4].clone()

    res_out = flag_gems._nested_view_from_buffer(
        buffer, *_metadata(sizes, strides, offsets)
    )
    res_out.unbind()[1].fill_(-3.5)

    # A component write lands in the shared buffer storage and leaves the rest
    # of the buffer alone.
    tu.assert_result_equal(buffer[:4], untouched)
    tu.assert_result_equal(buffer[4:], torch.full_like(buffer[4:], -3.5))


# (label, expected candidate exception, buffer kind, metadata dtype, component
# sizes, component strides, component offsets). Every row was probed as a
# native-invalid call; the flat operands hold 8 elements.
NEGATIVE_ROWS = [
    ("buffer_not_1d", RuntimeError, "2d", torch.int64, [[2], [2]], [[1], [1]], [0, 2]),
    ("size_not_2d", RuntimeError, "1d", torch.int64, [3, 4], [[1], [1]], [0, 3]),
    ("strides_not_2d", RuntimeError, "1d", torch.int64, [[3], [4]], [1, 1], [0, 3]),
    ("size_row_mismatch", RuntimeError, "1d", torch.int64, [[3], [4]], [[1]], [0, 3]),
    (
        "offset_row_mismatch",
        RuntimeError,
        "1d",
        torch.int64,
        [[3], [4]],
        [[1], [1]],
        [0],
    ),
    (
        "size_out_of_bounds",
        RuntimeError,
        "1d",
        torch.int64,
        [[30], [4]],
        [[1], [1]],
        [0, 3],
    ),
    (
        "offset_out_of_bounds",
        RuntimeError,
        "1d",
        torch.int64,
        [[3], [4]],
        [[1], [1]],
        [0, 30],
    ),
    (
        "negative_stride",
        RuntimeError,
        "1d",
        torch.int64,
        [[3], [4]],
        [[-1], [-1]],
        [0, 3],
    ),
    (
        "metadata_not_int64",
        RuntimeError,
        "1d",
        torch.float32,
        [[3], [4]],
        [[1], [1]],
        [0, 3],
    ),
    (
        "offsets_not_tensor",
        (RuntimeError, TypeError),
        "1d",
        torch.int64,
        [[3], [4]],
        [[1], [1]],
        "0,3",
    ),
    (
        "nested_tensor_buffer",
        NotImplementedError,
        "nested",
        torch.int64,
        [[3], [4]],
        [[1], [1]],
        [0, 3],
    ),
]


def _invalid_inputs(row):
    _, _, kind, metadata_dtype, sizes, strides, offsets = row
    if kind == "nested":
        buffer = torch.nested.nested_tensor(
            [
                torch.zeros(3, device=flag_gems.device),
                torch.zeros(4, device=flag_gems.device),
            ]
        )
    elif kind == "2d":
        buffer = torch.zeros((8, 2), device=flag_gems.device)
    else:
        buffer = torch.zeros(8, device=flag_gems.device)

    def as_metadata(values):
        # The offsets_not_tensor row passes a non-tensor on purpose.
        if isinstance(values, (list, tuple)):
            return torch.tensor(values, dtype=metadata_dtype)
        return values

    return buffer, as_metadata(sizes), as_metadata(strides), as_metadata(offsets)


@pytest.mark.nested_view_from_buffer
@pytest.mark.parametrize("row", NEGATIVE_ROWS, ids=[row[0] for row in NEGATIVE_ROWS])
def test_nested_view_from_buffer_invalid_input(row):
    buffer, sizes, strides, offsets = _invalid_inputs(row)
    with pytest.raises(row[1]):
        flag_gems._nested_view_from_buffer(buffer, sizes, strides, offsets)
