# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems

from . import test_utils as tu

pytestmark = pytest.mark.requires_grad_

# aten::requires_grad_(Tensor(a!) self, bool requires_grad=True) -> Tensor(a!)
# Host-side autograd-flag write: only the flag changes and the same tensor is
# returned. True is rejected for integral/bool dtypes, False for a non-leaf.

_REQUIRED_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
]

# Extra dtypes the operator also accepts on this backend.
_EXTRA_DTYPES = [torch.float64, torch.complex64, torch.int16, torch.bool]

# Dtypes that can never carry a gradient: only the clearing direction is valid.
_CLEAR_ONLY_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
]

# Static device capabilities select the optional dtype groups.
_DEVICE_DTYPE_FLAGS = {
    torch.float64: "support_fp64",
    torch.bfloat16: "support_bf16",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
    torch.int64: "support_int64",
}


def _supported(dtype):
    flag = _DEVICE_DTYPE_FLAGS.get(dtype)
    if flag is None:
        return True
    return getattr(flag_gems.runtime.device, flag)


def _grad_capable(dtype):
    return dtype.is_floating_point or dtype.is_complex


_GRID_DTYPES = [dt for dt in _REQUIRED_DTYPES + _EXTRA_DTYPES if _supported(dt)]
_GRAD_DTYPES = [dt for dt in _GRID_DTYPES if _grad_capable(dt)]
_GRID_ROWS = [(dt, True) for dt in _GRAD_DTYPES] + [(dt, False) for dt in _GRID_DTYPES]


def _layout(tensor):
    # Metadata that must survive the flag write; the flag itself is compared
    # separately because it is the one property the operator changes.
    return (
        tensor.dtype,
        tuple(tensor.size()),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.untyped_storage().data_ptr(),
        tensor.is_leaf,
        tensor._version,
    )


def _set_initial(inp, flag):
    # Give a clearing case a real True flag to clear; integral/bool dtypes can
    # never hold one, so for them the clearing call is an already-false no-op.
    if not flag and _grad_capable(inp.dtype):
        torch.ops.aten.requires_grad_(inp, True)


def _make_view(base, kind):
    return base.t() if kind == "transpose" else base[2:6, ::2]


def _make_non_leaf(dtype, kind):
    leaf = tu.make_input(dtype, (8, 16), ["-1", "1"])
    torch.ops.aten.requires_grad_(leaf, True)
    return leaf, (leaf * 2 if kind == "computed" else leaf.t())


@pytest.mark.requires_grad_
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype,flag", _GRID_ROWS)
def test_requires_grad_metadata_state(shape, value_range, dtype, flag):
    inp = tu.make_input(dtype, shape, value_range)
    _set_initial(inp, flag)
    ref_inp = tu.to_reference(inp)
    before = _layout(inp)

    res = flag_gems.requires_grad_(inp, flag)

    assert res is inp
    assert inp.requires_grad is flag
    assert _layout(inp) == before
    tu.assert_result_equal(res, torch.ops.aten.requires_grad_(ref_inp, flag))


_DEFAULT_FLAG_SHAPES = tu.selected_cases([(1024, 1024)], quick=[(2, 19, 7)])


@pytest.mark.parametrize("shape", _DEFAULT_FLAG_SHAPES)
@pytest.mark.parametrize("dtype", _GRAD_DTYPES)
def test_requires_grad_default_flag(shape, dtype):
    # The schema default must be reachable by omitting the argument.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    before = _layout(inp)

    res = flag_gems.requires_grad_(inp)

    assert res is inp
    assert inp.requires_grad is True
    assert _layout(inp) == before
    tu.assert_result_equal(res, torch.ops.aten.requires_grad_(ref_inp))


_STATE_ROWS = (
    [(dt, False, True) for dt in _GRAD_DTYPES]
    + [(dt, True, False) for dt in _GRAD_DTYPES]
    + [(dt, True, True) for dt in _GRAD_DTYPES]
    + [(dt, False, False) for dt in _GRID_DTYPES]
)


@pytest.mark.parametrize("dtype,initial,flag", _STATE_ROWS)
def test_requires_grad_state_transitions(dtype, initial, flag):
    inp = tu.make_input(dtype, (256,), ["-1", "1"])
    if initial:
        torch.ops.aten.requires_grad_(inp, True)
    ref_inp = tu.to_reference(inp)
    before = _layout(inp)

    res = flag_gems.requires_grad_(inp, flag)

    assert res is inp
    assert inp.requires_grad is flag
    assert _layout(inp) == before
    tu.assert_result_equal(res, torch.ops.aten.requires_grad_(ref_inp, flag))


_VIEW_ROWS = [
    (torch.float32, "transpose", True),
    (torch.float32, "transpose", False),
    (torch.float32, "slice", False),
    (torch.float8_e4m3fn, "transpose", True),
    (torch.float8_e4m3fn, "transpose", False),
    (torch.int32, "slice", False),
]
_VIEW_ROWS = [row for row in _VIEW_ROWS if _supported(row[0])]


@pytest.mark.parametrize("dtype,kind,flag", _VIEW_ROWS)
def test_requires_grad_through_view(dtype, kind, flag):
    # A view of a tensor that does not require grad is itself a leaf, so both
    # directions are valid through non-contiguous and offset layouts.
    base = tu.make_input(dtype, (8, 16), ["-1", "1"])
    view = _make_view(base, kind)
    ref_view = _make_view(tu.to_reference(base), kind)
    if not flag and _grad_capable(dtype):
        # Only gradient-capable dtypes get a real True state to clear; an
        # integral view cannot hold a gradient, so its clearing call is the
        # already-false no-op form that is the only valid one for that dtype.
        torch.ops.aten.requires_grad_(view, True)
        torch.ops.aten.requires_grad_(ref_view, True)
    before = _layout(view)

    res = flag_gems.requires_grad_(view, flag)

    assert res is view
    assert view.requires_grad is flag
    assert _layout(view) == before
    # The view must keep aliasing the base storage and leave the base flag alone.
    assert view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    assert base.requires_grad is False
    tu.assert_result_equal(res, torch.ops.aten.requires_grad_(ref_view, flag))


_NONLEAF_ROWS = [
    (torch.float32, "computed"),
    (torch.bfloat16, "computed"),
    (torch.float32, "view"),
    (torch.float8_e4m3fn, "view"),
]
_NONLEAF_ROWS = [row for row in _NONLEAF_ROWS if _supported(row[0])]


@pytest.mark.parametrize("dtype,kind", _NONLEAF_ROWS)
def test_requires_grad_true_keeps_non_leaf(dtype, kind):
    # Setting True on a non-leaf is a valid no-op that keeps the graph link.
    leaf, non_leaf = _make_non_leaf(dtype, kind)
    ref_leaf = tu.to_reference(leaf)
    ref_non_leaf = ref_leaf * 2 if kind == "computed" else ref_leaf.t()
    before = _layout(non_leaf)

    res = flag_gems.requires_grad_(non_leaf, True)

    assert res is non_leaf
    assert non_leaf.requires_grad is True
    assert _layout(non_leaf) == before
    tu.assert_result_equal(res, torch.ops.aten.requires_grad_(ref_non_leaf, True))


_EMPTY_SHAPES = [(0,), (0, 3)]
_EMPTY_ROWS = [
    (torch.float32, True),
    (torch.float32, False),
    (torch.int64, False),
    (torch.float8_e4m3fn, True),
]
_EMPTY_ROWS = [row for row in _EMPTY_ROWS if _supported(row[0])]


@pytest.mark.parametrize("shape", _EMPTY_SHAPES)
@pytest.mark.parametrize("dtype,flag", _EMPTY_ROWS)
def test_requires_grad_empty_tensor(shape, dtype, flag):
    # The shape of an empty tensor must be preserved; there is no payload.
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)
    torch.ops.aten.requires_grad_(ref_inp, flag)
    before = _layout(inp)

    res = flag_gems.requires_grad_(inp, flag)

    assert res is inp
    assert inp.requires_grad is flag
    assert _layout(inp) == before
    assert tuple(res.shape) == tuple(shape)
    tu.assert_result_equal(res, ref_inp)


_SPECIAL_ROWS = tu.special_value_cases(
    [dt for dt in _GRAD_DTYPES if dt.is_floating_point]
)


@pytest.mark.parametrize("dtype,scenario", tu.selected_cases(_SPECIAL_ROWS, quick=[]))
def test_requires_grad_special_values(dtype, scenario):
    # The flag is metadata only: nan/inf payloads stay bit-identical.
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)
    before = _layout(inp)

    res = flag_gems.requires_grad_(inp, True)

    assert res is inp
    assert inp.requires_grad is True
    assert _layout(inp) == before
    tu.assert_result_equal(res, torch.ops.aten.requires_grad_(ref_inp, True))


_BACKWARD_DTYPES = [
    dt
    for dt in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    if _supported(dt)
]
_BACKWARD_ROWS = tu.selected_cases(
    [
        (dt, shape)
        for dt in _BACKWARD_DTYPES
        for shape in [(256,), (64, 64), (20, 320, 15)]
    ],
    quick=[],
)


@pytest.mark.parametrize("dtype,shape", _BACKWARD_ROWS)
def test_requires_grad_enables_backward(dtype, shape):
    # The operator must make the original leaf differentiable; complex64 and fp8
    # are excluded, this backend has no real-scalar-loss or fp8 mul path.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    assert flag_gems.requires_grad_(inp, True) is inp
    torch.ops.aten.requires_grad_(ref_inp, True)

    res_grad = torch.autograd.grad((inp * inp).sum(), inp)[0]
    ref_grad = torch.autograd.grad((ref_inp * ref_inp).sum(), ref_inp)[0]

    tu.assert_result_close(res_grad, ref_grad)


@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _CLEAR_ONLY_DTYPES)
def test_requires_grad_rejects_non_grad_dtype(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.requires_grad_(inp, True)


@pytest.mark.parametrize("dtype,kind", _NONLEAF_ROWS)
def test_requires_grad_rejects_clearing_non_leaf(dtype, kind):
    _leaf, non_leaf = _make_non_leaf(dtype, kind)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.requires_grad_(non_leaf, False)


@pytest.mark.parametrize("value", [1.0, None])
def test_requires_grad_rejects_non_tensor_input(value):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.requires_grad_(value, True)


@pytest.mark.parametrize("flag", ["yes", "False", [True]])
def test_requires_grad_rejects_non_bool_flag(flag):
    inp = tu.make_input(torch.float32, (4, 8), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.requires_grad_(inp, flag)
