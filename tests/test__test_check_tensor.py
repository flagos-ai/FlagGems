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

# aten::_test_check_tensor is the bool-only self-test behind TORCH_CHECK_TENSOR_ALL:
# the operand must be a bool tensor whose visible elements are all true, and the result
# is a new all-true bool tensor. Numeric value ranges, nan/inf, broadcast,
# tensor-vs-scalar and backward do not apply (a bool tensor cannot require grad), so the
# spec's nine dtypes are covered as negative rows instead.
_LAYOUTS = (
    "contiguous",
    "window",
    "narrow",
    "stepped",
    "expanded",
    "transposed",
    "permuted",
)

# Static capability flags, read while this module is imported: no tensor is allocated
# and no operator is called at collection time. complex64 rides the 32-bit float path
# and needs no flag of its own.
_DTYPE_CAPABILITY = {
    torch.bfloat16: "support_bf16",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.int64: "support_int64",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name))


# Zero-element and single-element boundaries: the requested shape and the empty result
# still have to survive.
_EDGE_SHAPES = [(0,), (0, 3), (2, 0, 5), (2, 3, 0, 4), (1, 1), (7, 1, 5)]

# Shapes used to hide false elements in the storage a view does not expose.
_SMALL_HOLE_SHAPES = [(7,), (4, 6)]
_HOLE_SHAPES = [
    (256,),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]
_HOLE_LAYOUTS = ("stepped", "window", "narrow")

_FALSE_POSITIONS = ("first", "middle", "last", "all")
_FALSE_SHAPES = [
    (),
    (1,),
    (256,),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]

_STORED_FALSE_ROWS = [
    ((256,), "stepped"),
    ((256,), "expanded"),
    ((1024, 1024), "transposed"),
    ((20, 320, 15), "stepped"),
    ((20, 320, 15), "expanded"),
    ((16, 128, 64, 60), "transposed"),
]

_NON_BOOL_DTYPES = [
    dtype
    for dtype in tu.REQUIRED_DTYPES + [torch.float64, torch.complex64, torch.complex128]
    if _dtype_supported(dtype)
]
_DTYPE_SHAPES = [(4,), (3, 5), (2, 3, 4)]
_NON_TENSOR_KEYS = ("float", "int", "str", "none", "nested_list", "tensor_list")
_COPY_SHAPES = [(256,), (20, 320, 15), (16, 128, 64, 60), (16, 7, 57, 32, 29)]


def _rank_applies(shape, layout):
    if layout in ("window", "narrow", "stepped"):
        return len(shape) >= 1
    if layout == "transposed":
        return len(shape) >= 2
    if layout == "permuted":
        return len(shape) >= 3
    if layout == "expanded":
        # A leading extent of 0 or 1 leaves no stride-0 dimension to expand.
        return len(shape) >= 1 and shape[0] > 1
    return True


def _storage_shape(shape, layout):
    if layout == "window":
        return (shape[0] + 2,) + tuple(shape[1:])
    if layout == "narrow":
        return tuple(shape[:-1]) + (shape[-1] + 2,)
    if layout == "stepped":
        return tuple(shape[:-1]) + (shape[-1] * 2,)
    if layout == "transposed":
        return tuple(shape[:-2]) + (shape[-1], shape[-2])
    if layout == "permuted":
        return tuple(reversed(shape))
    if layout == "expanded":
        return (1,) + tuple(shape[1:])
    return tuple(shape)


def _view_of(base, shape, layout):
    if layout == "window":
        return base[2:]
    if layout == "narrow":
        return base[..., 1:-1]
    if layout == "stepped":
        return base[..., ::2]
    if layout == "transposed":
        return base.transpose(-1, -2)
    if layout == "permuted":
        return base.permute(*reversed(range(base.dim())))
    if layout == "expanded":
        return base.expand(tuple(shape))
    return base


def _bool_storage(shape, layout):
    return torch.ones(
        _storage_shape(shape, layout), dtype=torch.bool, device=flag_gems.device
    )


def _make_input(shape, layout, hole_false=False):
    # Returns the operand plus the tensor owning the storage behind it.
    base = _bool_storage(shape, layout)
    if hole_false:
        # The false elements sit outside the window the view exposes.
        if layout == "stepped":
            base[..., 1::2] = False
        elif layout == "window":
            base[:2] = False
        else:
            base[..., 0] = False
            base[..., -1] = False
    return _view_of(base, shape, layout), base


def _make_visible_false_input(shape, layout):
    base = _bool_storage(shape, layout)
    if layout == "expanded":
        base[0].view(-1)[0] = False
    else:
        base.view(-1)[0] = False
    return _view_of(base, shape, layout)


def _set_false(inp, position):
    flat = inp.view(-1)
    if position == "all":
        flat.fill_(False)
    elif position == "first":
        flat[0] = False
    elif position == "middle":
        flat[flat.numel() // 2] = False
    else:
        flat[-1] = False


def _layout_facts(out):
    return (
        tuple(out.stride()),
        out.storage_offset(),
        out.is_contiguous(),
        out._is_view(),
    )


def _assert_layout(res_out, inp, expected):
    assert res_out.device == inp.device
    assert _layout_facts(res_out) == expected
    if inp.numel():
        # Only the pointer inequality needs an element: a zero-element result owns no
        # element that could alias, while every metadata field above is still compared.
        assert res_out.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()


def _invalid_args(key):
    # Arguments rejected by the operator's Tensor schema.
    if key == "tensor_list":
        return [torch.ones((2,), dtype=torch.bool, device=flag_gems.device)]
    return {
        "float": 1.0,
        "int": 1,
        "str": "true",
        "none": None,
        "nested_list": [[True, True]],
    }[key]


def _grid_rows():
    rows = []
    for shape in tu.selected_shapes():
        shape = tuple(shape)
        for layout in _LAYOUTS:
            if _rank_applies(shape, layout):
                rows.append((shape, layout, False))
    return rows


# Every cheap boundary row and the small hidden-false rows run in both modes; only the
# specification-scale hidden-false shapes are default-only.
_EDGE_ROWS = [
    (shape, layout, False)
    for shape in _EDGE_SHAPES
    for layout in _LAYOUTS
    if _rank_applies(shape, layout)
]
_SMALL_HOLE_ROWS = [
    (shape, layout, True) for shape in _SMALL_HOLE_SHAPES for layout in _HOLE_LAYOUTS
]
_HOLE_ROWS = tu.selected_cases(
    _SMALL_HOLE_ROWS
    + [(shape, layout, True) for shape in _HOLE_SHAPES for layout in _HOLE_LAYOUTS],
    quick=_SMALL_HOLE_ROWS,
)

_TRUE_ROWS = list(
    dict.fromkeys(_grid_rows() + [((), "contiguous", False)] + _EDGE_ROWS + _HOLE_ROWS)
)

_FALSE_ROWS = [
    (shape, position) for shape in _FALSE_SHAPES for position in _FALSE_POSITIONS
]


@pytest.mark.test_check_tensor
@pytest.mark.parametrize("shape,layout,hole_false", _TRUE_ROWS)
def test__test_check_tensor(shape, layout, hole_false):
    inp, parent = _make_input(shape, layout, hole_false)
    inp_before = tu.to_reference(inp)
    # The storage behind a strided or expanded operand must survive outside the
    # visible window too, so the owner is snapshotted whenever it is a distinct object.
    parent_before = None if parent is inp else tu.to_reference(parent)

    ref_inp = tu.to_reference(inp)
    ref_out = torch.ops.aten._test_check_tensor(ref_inp)
    expected = _layout_facts(ref_out)

    res_out = flag_gems._test_check_tensor(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_layout(res_out, inp, expected)
    tu.assert_result_equal(inp, inp_before)
    if parent_before is not None:
        tu.assert_result_equal(parent, parent_before)


@pytest.mark.test_check_tensor
@pytest.mark.parametrize("shape,position", _FALSE_ROWS)
def test__test_check_tensor_rejects_false_element(shape, position):
    inp = torch.ones(shape, dtype=torch.bool, device=flag_gems.device)
    _set_false(inp, position)

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems._test_check_tensor(inp)


@pytest.mark.test_check_tensor
@pytest.mark.parametrize("shape,layout", _STORED_FALSE_ROWS)
def test__test_check_tensor_rejects_false_inside_view(shape, layout):
    inp = _make_visible_false_input(shape, layout)

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems._test_check_tensor(inp)


@pytest.mark.test_check_tensor
@pytest.mark.parametrize("shape", _DTYPE_SHAPES)
@pytest.mark.parametrize("dtype", _NON_BOOL_DTYPES)
def test__test_check_tensor_rejects_non_bool_dtype(shape, dtype):
    # The native op asserts self.scalar_type() == at::kBool.
    inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems._test_check_tensor(inp)


@pytest.mark.test_check_tensor
@pytest.mark.parametrize("key", _NON_TENSOR_KEYS)
def test__test_check_tensor_rejects_non_tensor(key):
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._test_check_tensor(_invalid_args(key))


@pytest.mark.test_check_tensor
@pytest.mark.parametrize("shape", _COPY_SHAPES)
def test__test_check_tensor_returns_independent_copy(shape):
    inp = torch.ones(shape, dtype=torch.bool, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)
    ref_out = torch.ops.aten._test_check_tensor(ref_inp)

    res_out = flag_gems._test_check_tensor(inp)

    tu.assert_result_equal(res_out, ref_out)
    assert not res_out._is_view()
    assert res_out.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()
    res_out.fill_(False)
    tu.assert_result_equal(inp, ref_inp)
