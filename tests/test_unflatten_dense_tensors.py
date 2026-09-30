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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# aten::unflatten_dense_tensors(Tensor flat, Tensor[] tensors) -> Tensor[]
# splits dim 0 of `flat` into one chunk per donor and views each chunk with that
# donor's shape, so every result aliases `flat`'s storage instead of copying
# donor data. Only donor shapes are read: donor dtype, layout, device and values
# are ignored, and the result dtype comes from `flat`. `flat` is never flattened
# first, so each spec shape level is expressed by one donor of that rank inside a
# 1-D flat buffer holding that many elements.
_SUPPORTED_DTYPES = (
    [torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
    + utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [torch.bool, torch.complex64]
)

# Backward needs an add kernel these dtypes lack on the active backend:
# "ufunc_add_CUDA" not implemented for 'Float8_e4m3fn' (same for e5m2), so FP8 is
# exempt from backward coverage only; its forward coverage above stays complete.
_BACKWARD_DTYPES = [
    dtype
    for dtype in _SUPPORTED_DTYPES
    if dtype.is_complex
    or (
        dtype.is_floating_point
        and dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
    )
]


def _donor_views(flat, donor_shapes):
    """Donors carrying the requested shapes, carved out of ``flat`` itself.

    Only donor shapes are read, so slicing the flat buffer supplies the argument
    without allocating a second set of tensors.
    """
    donors, offset = [], 0
    for shape in donor_shapes:
        size = math.prod(shape)
        donors.append(flat[offset : offset + size].view(shape))
        offset += size
    return donors


def _shape_donors(shapes, dtype, device):
    """Standalone donors for cases that must not be carved out of ``flat``."""
    return [torch.zeros(shape, dtype=dtype, device=device) for shape in shapes]


def _split_flat(shape, parts=3):
    """Split ``shape`` along dim 0 into unequal donors of the same rank."""
    if not shape:
        return (1,), [()]
    chunks = [
        shape[0] // parts + (1 if i < shape[0] % parts else 0) for i in range(parts)
    ]
    donors = [tuple([size] + list(shape[1:])) for size in chunks if size > 0]
    return (sum(math.prod(d) for d in donors),), donors


# (flat shape, donor shapes). Donor element counts must total flat.numel(): the op
# narrows dim 0 of the flat buffer and cannot read past its end.
_BASE_ROWS = [
    ((266,), [(2, 19, 7)]),
    ((1,), [()]),  # single 0-dim donor
    ((3,), [(), (1,), (1,)]),  # 0-dim donor next to 1-D donors
    ((8,), [(8,)]),  # single donor
    ((24,), [(2, 3), (4,), (5,), (9,)]),  # mixed donor ranks
    ((168,), [(16, 6), (2, 2, 3), (60,)]),  # 1-D .. 3-D donors
]


def _spec_rows():
    """One row per spec shape level, skipping what the base rows already cover."""
    rows = []
    for shape in tu.selected_shapes():
        row = _split_flat(shape)
        if row not in _BASE_ROWS and row not in rows:
            rows.append(row)
    return rows


_CASE_ROWS = tu.selected_cases(
    _BASE_ROWS + _spec_rows(),
    quick=_BASE_ROWS,
)


@pytest.mark.unflatten_dense_tensors
@pytest.mark.parametrize("flat_shape, donor_shapes", _CASE_ROWS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_unflatten_dense_tensors(flat_shape, donor_shapes, value_range, dtype):
    flat = tu.make_input(dtype, flat_shape, value_range)
    ref_flat = tu.to_reference(flat)

    ref_out = torch.ops.aten.unflatten_dense_tensors(
        ref_flat, _donor_views(ref_flat, donor_shapes)
    )
    res_out = flag_gems.unflatten_dense_tensors(flat, _donor_views(flat, donor_shapes))

    assert isinstance(res_out, (list, tuple))
    assert len(res_out) == len(donor_shapes)
    for res_t, ref_t in zip(res_out, ref_out):
        assert tuple(res_t.shape) == tuple(ref_t.shape)
        assert res_t.dtype == flat.dtype
        assert tuple(res_t.stride()) == tuple(ref_t.stride())
        assert res_t.storage_offset() == ref_t.storage_offset()
        assert res_t.untyped_storage().data_ptr() == flat.untyped_storage().data_ptr()
        tu.assert_result_equal(res_t, ref_t)


def _foreign_donors(device):
    """float32 donors with non-zero values and non-contiguous layouts."""
    base = torch.full((4, 4), 5.0, dtype=torch.float32, device=device)
    return [base[::2, ::2], base[0], base[2:]]


@pytest.mark.unflatten_dense_tensors
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_unflatten_dense_tensors_reads_donor_shapes_only(dtype):
    flat = tu.make_input(dtype, (16,), ["-1", "1"])
    ref_flat = tu.to_reference(flat)
    donor_shapes = [tuple(donor.shape) for donor in _foreign_donors(flat.device)]

    ref_out = torch.ops.aten.unflatten_dense_tensors(
        ref_flat, _foreign_donors(ref_flat.device)
    )
    res_out = flag_gems.unflatten_dense_tensors(flat, _foreign_donors(flat.device))

    for res_t, ref_t, shape in zip(res_out, ref_out, donor_shapes):
        # Donor dtype, layout and values are ignored: shapes come from the donors,
        # dtype and contents from `flat`.
        assert tuple(res_t.shape) == shape
        assert res_t.dtype == flat.dtype
        assert res_t.untyped_storage().data_ptr() == flat.untyped_storage().data_ptr()
        tu.assert_result_equal(res_t, ref_t)


_NON_CONTIGUOUS_FLAT_SLICES = [slice(None, None, 2), slice(3, 19)]
_NON_CONTIGUOUS_DONOR_SHAPES = [(2, 3), (4,), (6,)]


@pytest.mark.unflatten_dense_tensors
@pytest.mark.parametrize("flat_slice", _NON_CONTIGUOUS_FLAT_SLICES)
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_unflatten_dense_tensors_non_contiguous_flat(flat_slice, dtype):
    base = tu.make_input(dtype, (32,), ["-1", "1"])
    ref_base = tu.to_reference(base)
    flat = base[flat_slice]
    ref_flat = ref_base[flat_slice]

    ref_out = torch.ops.aten.unflatten_dense_tensors(
        ref_flat, _donor_views(ref_flat, _NON_CONTIGUOUS_DONOR_SHAPES)
    )
    res_out = flag_gems.unflatten_dense_tensors(
        flat, _donor_views(flat, _NON_CONTIGUOUS_DONOR_SHAPES)
    )

    for res_t, ref_t in zip(res_out, ref_out):
        # The views keep `flat`'s leading stride and element offset; only the
        # trailing dims are added by the view.
        assert tuple(res_t.shape) == tuple(ref_t.shape)
        assert res_t.dtype == flat.dtype
        assert tuple(res_t.stride()) == tuple(ref_t.stride())
        assert res_t.storage_offset() == ref_t.storage_offset()
        assert res_t.untyped_storage().data_ptr() == flat.untyped_storage().data_ptr()
        tu.assert_result_equal(res_t, ref_t)


@pytest.mark.unflatten_dense_tensors
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_unflatten_dense_tensors_views_alias_flat(dtype):
    flat = tu.make_input(dtype, (12,), ["-1", "1"])
    res_out = flag_gems.unflatten_dense_tensors(
        flat, _donor_views(flat, [(2, 3), (6,)])
    )

    assert res_out[0].untyped_storage().data_ptr() == flat.untyped_storage().data_ptr()
    assert res_out[1].untyped_storage().data_ptr() == flat.untyped_storage().data_ptr()
    # A write through a returned view reaches the flat buffer ...
    res_out[1].fill_(1.0)
    tu.assert_result_equal(flat[6:12], flat.new_ones(6))
    # ... and a write into the flat buffer shows up through the views.
    flat[0:6] = 3.0
    tu.assert_result_equal(res_out[0], flat.new_full((2, 3), 3.0))


@pytest.mark.unflatten_dense_tensors
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_unflatten_dense_tensors_zero_size_donors(dtype):
    # Native collapses a zero-element donor to a fresh empty result of shape (0,),
    # stride (1,), offset 0: it neither keeps the requested rank nor aliases
    # `flat`, so empty components are matched against the native metadata.
    flat = tu.make_input(dtype, (24,), ["-1", "1"])
    ref_flat = tu.to_reference(flat)
    donor_shapes = [(0, 3), (24,), (0,)]

    ref_out = torch.ops.aten.unflatten_dense_tensors(
        ref_flat, _donor_views(ref_flat, donor_shapes)
    )
    res_out = flag_gems.unflatten_dense_tensors(flat, _donor_views(flat, donor_shapes))

    for res_t, ref_t in zip(res_out, ref_out):
        assert tuple(res_t.shape) == tuple(ref_t.shape)
        assert tuple(res_t.stride()) == tuple(ref_t.stride())
        assert res_t.storage_offset() == ref_t.storage_offset()
        if res_t.numel():
            assert (
                res_t.untyped_storage().data_ptr() == flat.untyped_storage().data_ptr()
            )
        tu.assert_result_equal(res_t, ref_t)


@pytest.mark.unflatten_dense_tensors
@pytest.mark.parametrize(
    "dtype, scenario",
    tu.selected_cases(tu.special_value_cases(_SUPPORTED_DTYPES), quick=[]),
)
def test_unflatten_dense_tensors_nan_inf(dtype, scenario):
    flat = tu.make_special_input(dtype, scenario)
    ref_flat = tu.to_reference(flat)
    donor_shapes = [(2,), (3,)]

    ref_out = torch.ops.aten.unflatten_dense_tensors(
        ref_flat, _donor_views(ref_flat, donor_shapes)
    )
    res_out = flag_gems.unflatten_dense_tensors(flat, _donor_views(flat, donor_shapes))

    for res_t, ref_t in zip(res_out, ref_out):
        assert res_t.dtype == flat.dtype
        tu.assert_result_equal(res_t, ref_t)


_BACKWARD_CASE_ROWS = tu.selected_cases(
    [
        ((24,), [(2, 3), (4,), (5,), (9,)]),
        ((168,), [(16, 6), (2, 2, 3), (60,)]),
    ],
    quick=[],
)


@pytest.mark.unflatten_dense_tensors
@pytest.mark.parametrize("flat_shape, donor_shapes", _BACKWARD_CASE_ROWS)
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
def test_unflatten_dense_tensors_backward(flat_shape, donor_shapes, dtype):
    # Differentiate through the original flat leaf, not through an output.
    flat = tu.make_input(dtype, flat_shape, ["-1", "1"]).requires_grad_()
    ref_flat = tu.to_reference(flat)
    upstream = [tu.make_input(dtype, shape, ["-1", "1"]) for shape in donor_shapes]
    ref_upstream = [tu.to_reference(grad) for grad in upstream]

    ref_out = torch.ops.aten.unflatten_dense_tensors(
        ref_flat, _donor_views(ref_flat, donor_shapes)
    )
    (ref_grad,) = torch.autograd.grad(ref_out, ref_flat, grad_outputs=ref_upstream)

    res_out = flag_gems.unflatten_dense_tensors(flat, _donor_views(flat, donor_shapes))
    for res_t, ref_t in zip(res_out, ref_out):
        tu.assert_result_equal(res_t, ref_t)

    assert res_out[0].requires_grad
    (res_grad,) = torch.autograd.grad(res_out, flat, grad_outputs=upstream)
    tu.assert_result_equal(res_grad, ref_grad)


@pytest.mark.unflatten_dense_tensors
def test_unflatten_dense_tensors_empty_donor_list():
    # Native returns an empty list for no donors instead of raising, so this is a
    # valid workload rather than a negative case.
    flat = tu.make_input(torch.float32, (4,), ["-1", "1"])
    ref_flat = tu.to_reference(flat)

    ref_out = torch.ops.aten.unflatten_dense_tensors(ref_flat, [])
    res_out = flag_gems.unflatten_dense_tensors(flat, [])

    assert len(ref_out) == 0
    assert len(res_out) == 0


@pytest.mark.unflatten_dense_tensors
def test_unflatten_dense_tensors_rejects_zero_dim_flat():
    donors = _shape_donors([(4,)], torch.float32, flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.unflatten_dense_tensors(
            torch.zeros((), dtype=torch.float32, device=flag_gems.device), donors
        )


@pytest.mark.unflatten_dense_tensors
def test_unflatten_dense_tensors_rejects_donors_larger_than_flat():
    flat = tu.make_input(torch.float32, (8,), ["-1", "1"])
    # 2 * 3 + 4 = 10 requested elements exceed the 8-element flat buffer.
    donors = _shape_donors([(2, 3), (4,)], torch.float32, flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.unflatten_dense_tensors(flat, donors)


@pytest.mark.unflatten_dense_tensors
def test_unflatten_dense_tensors_rejects_short_flat_dim0():
    # `flat` is not flattened first: dim 0 of a rank>=2 buffer must already cover
    # the requested element count, so a (2, 8) buffer cannot serve 16 elements.
    flat = tu.make_input(torch.float32, (2, 8), ["-1", "1"])
    donors = _shape_donors([(16,)], torch.float32, flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.unflatten_dense_tensors(flat, donors)


@pytest.mark.unflatten_dense_tensors
def test_unflatten_dense_tensors_rejects_non_tensor_donors():
    flat = tu.make_input(torch.float32, (4,), ["-1", "1"])
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.unflatten_dense_tensors(flat, 3.14)
