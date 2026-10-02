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

# aten::retain_grad(Tensor(a!) self) -> ()
#
# A host-side autograd-state mutation: it flips the `retains_grad` flag of the
# tensor it is handed so a later backward also fills that non-leaf's `.grad`.
# It returns None and moves no element, so the contract under test is
# state/metadata only: the flag lands on the very object passed in, it is per
# tensor, and storage, layout, version counter and values are untouched.
#
# The applicable dtypes are exactly the dtypes autograd can track, so the
# spec's int8/uint8/int32/int64/bool entries appear as negative no-grad rows.

# float16/float32/bfloat16 (+float64 where supported), complex64/complex128
# (same 64-bit gate) and both fp8 types (gated on the device fp8 flag).
_RETAIN_GRAD_DTYPES = list(utils.ALL_FLOAT_DTYPES) + [torch.complex64]
if utils.fp64_is_supported:
    _RETAIN_GRAD_DTYPES.append(torch.complex128)
if utils.fp8_is_supported:
    _RETAIN_GRAD_DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]

# Dtypes autograd refuses to track: the requires_grad=False negative rows.
_NO_GRAD_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.float16,
    torch.float32,
    torch.bfloat16,
]

# Spec shapes plus the empty / 0-extent boundaries of a state-only operator.
_STATE_SHAPES = tu.selected_cases(
    tu.REQUIRED_SHAPES + [(0,), (2, 0, 3)],
    quick=list(tu.QUICK_SHAPES) + [(0,), (2, 0, 3)],
)

# State-only rows: a plain non-leaf, a leaf (accepted as a silent no-op) and a
# repeat call on an already flagged non-leaf. Cheap enough to stay in quick mode.
_FLAG_ROWS = tu.selected_cases(
    [
        ("nonleaf", shape)
        for shape in [
            (),
            (1,),
            (256,),
            (1024, 1024),
            (20, 320, 15),
            (2, 19, 7),
            (0,),
            (2, 0, 3),
        ]
    ]
    + [("leaf", shape) for shape in [(), (256,), (2, 19, 7), (1024, 1024)]]
    + [("retained", shape) for shape in [(256,), (2, 19, 7), (1024, 1024)]],
    quick=[
        ("nonleaf", (2, 19, 7)),
        ("nonleaf", ()),
        ("nonleaf", (0,)),
        ("nonleaf", (1,)),
        ("nonleaf", (256,)),
        ("nonleaf", (2, 0, 3)),
        ("leaf", ()),
        ("leaf", (256,)),
        ("retained", (256,)),
        ("leaf", (2, 19, 7)),
        ("retained", (2, 19, 7)),
    ],
)


def _rows(shapes, quick_shapes):
    """Pair each shape with every differentiable dtype, per execution level."""
    return tu.selected_cases(
        [(shape, dtype) for shape in shapes for dtype in _RETAIN_GRAD_DTYPES],
        quick=[
            (shape, dtype) for shape in quick_shapes for dtype in _RETAIN_GRAD_DTYPES
        ],
    )


# The state groups keep a quick row; the backward groups pass an empty
# `quick_shapes`, so no positive backward case runs in quick mode.
_PAIR_STATE_ROWS = _rows([(2, 19, 7), (1024, 1024)], [(2, 19, 7)])
_VIEW_STATE_ROWS = _rows([(2, 19, 7), (20, 320, 15)], [(2, 19, 7)])
_GRADIENT_ROWS = _rows([(2, 19, 7), (256,), (1024, 1024)], [])
_PAIR_GRAD_ROWS = _rows([(2, 19, 7), (1024, 1024)], [])
_NOOP_GRAD_ROWS = _rows([(2, 19, 7), (1024, 1024)], [])
_VIEW_GRAD_ROWS = _rows([(2, 19, 7), (20, 320, 15)], [])
_SPECIAL_ROWS = tu.selected_cases(tu.special_value_cases(_RETAIN_GRAD_DTYPES), quick=[])


def _differentiable_mid(leaf):
    """Build a non-leaf tensor of the same dtype and device from `leaf`."""
    if leaf.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # CUDA has no fp8 kernel for mul/sum/neg; clone is implemented and still
        # yields a non-leaf tensor carrying a grad_fn.
        return leaf.clone()
    return leaf * 0.5


@pytest.mark.retain_grad
@pytest.mark.parametrize("dtype", _RETAIN_GRAD_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", _STATE_SHAPES)
def test_retain_grad_state(shape, value_range, dtype):
    leaf = tu.make_input(dtype, shape, value_range).requires_grad_(True)
    mid = _differentiable_mid(leaf)
    ref_mid = _differentiable_mid(tu.to_reference(leaf))

    data_ptr = mid.data_ptr()
    stride = mid.stride()
    offset = mid.storage_offset()
    version = mid._version

    assert flag_gems.retain_grad(mid) is None
    torch.ops.aten.retain_grad(ref_mid)

    # The flag belongs to the tensor that was passed in, not to a copy of it, and
    # the differentiable leaf behind it keeps its own (unretained) state.
    assert mid.retains_grad is True
    assert leaf.retains_grad is False
    assert mid.retains_grad == ref_mid.retains_grad
    # Setting the flag must not touch storage, layout, version counter or values,
    # and must not fabricate a gradient by itself.
    assert mid.data_ptr() == data_ptr
    assert mid.stride() == stride
    assert mid.storage_offset() == offset
    assert mid._version == version
    assert mid.grad is None
    tu.assert_result_equal(mid, ref_mid)


@pytest.mark.retain_grad
@pytest.mark.parametrize("dtype", _RETAIN_GRAD_DTYPES)
@pytest.mark.parametrize("kind,shape", _FLAG_ROWS)
def test_retain_grad_flag_workload(kind, shape, dtype):
    leaf = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_leaf = tu.to_reference(leaf)
    if kind == "leaf":
        # A leaf already owns its accumulator, so the call is accepted as a no-op.
        target, ref_target = leaf, ref_leaf
    else:
        target, ref_target = _differentiable_mid(leaf), _differentiable_mid(ref_leaf)
    expected = kind != "leaf"

    data_ptr = target.data_ptr()
    stride = target.stride()
    offset = target.storage_offset()
    version = target._version

    assert flag_gems.retain_grad(target) is None
    torch.ops.aten.retain_grad(ref_target)
    if kind == "retained":
        # A repeat call on an already flagged tensor is idempotent.
        assert flag_gems.retain_grad(target) is None

    assert target.retains_grad is expected
    assert target.retains_grad == ref_target.retains_grad
    assert target.data_ptr() == data_ptr
    assert target.stride() == stride
    assert target.storage_offset() == offset
    assert target._version == version
    assert target.grad is None
    tu.assert_result_equal(target, ref_target)


@pytest.mark.retain_grad
@pytest.mark.parametrize("shape,dtype", _PAIR_STATE_ROWS)
def test_retain_grad_is_per_tensor_state(shape, dtype):
    retained_leaf = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    plain_leaf = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    first = _differentiable_mid(retained_leaf)
    second = _differentiable_mid(plain_leaf)
    ref_first = _differentiable_mid(tu.to_reference(retained_leaf))
    ref_second = _differentiable_mid(tu.to_reference(plain_leaf))

    assert flag_gems.retain_grad(first) is None
    torch.ops.aten.retain_grad(ref_first)

    # Only the tensor handed to the operator is flagged; the sibling and the
    # differentiable leaf behind it stay untouched.
    assert first.retains_grad is True
    assert second.retains_grad is False
    assert retained_leaf.retains_grad is False
    assert first.retains_grad == ref_first.retains_grad
    assert second.retains_grad == ref_second.retains_grad
    tu.assert_result_equal(first, ref_first)
    tu.assert_result_equal(second, ref_second)


@pytest.mark.retain_grad
@pytest.mark.parametrize("shape,dtype", _VIEW_STATE_ROWS)
def test_retain_grad_non_contiguous_view_state(shape, dtype):
    # Non-contiguous view over a strided base: the flag must land on the view
    # object and must not materialize or copy it.
    base = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    view = base.transpose(1, 2)[:, ::2, :]
    ref_base = tu.to_reference(base)
    ref_view = ref_base.transpose(1, 2)[:, ::2, :]
    data_ptr = view.data_ptr()
    version = view._version

    assert flag_gems.retain_grad(view) is None
    torch.ops.aten.retain_grad(ref_view)

    assert view.retains_grad is True
    assert base.retains_grad is False
    assert view.retains_grad == ref_view.retains_grad
    assert base.retains_grad == ref_base.retains_grad
    # The retained view still aliases the base storage with the same layout, and
    # the flag write leaves the version counter alone.
    assert view.data_ptr() == data_ptr == base.data_ptr()
    assert view.stride() == ref_view.stride()
    assert view.storage_offset() == ref_view.storage_offset()
    assert view._version == version
    tu.assert_result_equal(base, ref_base)


@pytest.mark.retain_grad
@pytest.mark.parametrize("shape,dtype", _GRADIENT_ROWS)
def test_retain_grad_populates_grad(shape, dtype):
    leaf = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    mid = _differentiable_mid(leaf)
    ref_leaf = tu.to_reference(leaf)
    ref_mid = _differentiable_mid(ref_leaf)
    upstream = tu.make_input(dtype, shape, ["-1", "1"])
    ref_upstream = tu.to_reference(upstream)

    assert flag_gems.retain_grad(mid) is None
    torch.ops.aten.retain_grad(ref_mid)

    res_grad = torch.autograd.grad(mid, leaf, grad_outputs=upstream)[0]
    ref_grad = torch.autograd.grad(ref_mid, ref_leaf, grad_outputs=ref_upstream)[0]

    # Backward must actually fill the retained `.grad` of the non-leaf tensor,
    # and the gradient reaching the differentiable leaf must be unchanged.
    assert mid.grad is not None
    tu.assert_result_equal(mid.grad, ref_mid.grad)
    tu.assert_result_close(res_grad, ref_grad)


@pytest.mark.retain_grad
@pytest.mark.parametrize("shape,dtype", _NOOP_GRAD_ROWS)
def test_retain_grad_leaf_is_noop(shape, dtype):
    leaf = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_leaf = tu.to_reference(leaf)
    upstream = tu.make_input(dtype, shape, ["-1", "1"])
    ref_upstream = tu.to_reference(upstream)

    assert flag_gems.retain_grad(leaf) is None
    torch.ops.aten.retain_grad(ref_leaf)

    # A leaf already accumulates into `.grad`, so the call is a silent no-op.
    assert leaf.retains_grad is False
    assert leaf.retains_grad == ref_leaf.retains_grad

    torch.autograd.backward([_differentiable_mid(leaf)], [upstream])
    torch.autograd.backward([_differentiable_mid(ref_leaf)], [ref_upstream])

    tu.assert_result_close(leaf.grad, ref_leaf.grad)


@pytest.mark.retain_grad
@pytest.mark.parametrize("shape,dtype", _PAIR_GRAD_ROWS)
def test_retain_grad_is_per_tensor_backward(shape, dtype):
    retained_leaf = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    plain_leaf = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    first = _differentiable_mid(retained_leaf)
    second = _differentiable_mid(plain_leaf)
    ref_retained_leaf = tu.to_reference(retained_leaf)
    ref_plain_leaf = tu.to_reference(plain_leaf)
    ref_first = _differentiable_mid(ref_retained_leaf)
    ref_second = _differentiable_mid(ref_plain_leaf)
    upstream = tu.make_input(dtype, shape, ["-1", "1"])
    ref_upstream = tu.to_reference(upstream)

    assert flag_gems.retain_grad(first) is None
    torch.ops.aten.retain_grad(ref_first)

    # The sibling hangs off its own leaf so every accumulator takes a single
    # contribution: a second path into the same leaf would sum two gradients,
    # and the CUDA backend has no fp8 add. That limit is unrelated to the flag.
    res = torch.autograd.grad(
        [first, second],
        [retained_leaf, plain_leaf],
        grad_outputs=[upstream, upstream],
    )
    ref = torch.autograd.grad(
        [ref_first, ref_second],
        [ref_retained_leaf, ref_plain_leaf],
        grad_outputs=[ref_upstream, ref_upstream],
    )

    tu.assert_result_equal(first.grad, ref_first.grad)
    tu.assert_result_close(res[0], ref[0])
    tu.assert_result_close(res[1], ref[1])
    # Reading `.grad` on an unretained non-leaf is how the missing accumulator is
    # observed; torch warns on that read.
    assert second.grad is None


@pytest.mark.retain_grad
@pytest.mark.parametrize("shape,dtype", _VIEW_GRAD_ROWS)
def test_retain_grad_non_contiguous_view_backward(shape, dtype):
    base = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    view = base.transpose(1, 2)[:, ::2, :]
    ref_base = tu.to_reference(base)
    ref_view = ref_base.transpose(1, 2)[:, ::2, :]
    upstream = tu.make_input(dtype, tuple(view.shape), ["-1", "1"])
    ref_upstream = tu.to_reference(upstream)

    assert flag_gems.retain_grad(view) is None
    torch.ops.aten.retain_grad(ref_view)

    res_grad = torch.autograd.grad(view, base, grad_outputs=upstream)[0]
    ref_grad = torch.autograd.grad(ref_view, ref_base, grad_outputs=ref_upstream)[0]

    tu.assert_result_equal(view.grad, ref_view.grad)
    tu.assert_result_equal(res_grad, ref_grad)


@pytest.mark.retain_grad
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_ROWS)
def test_retain_grad_special_values(dtype, scenario):
    leaf = tu.make_special_input(dtype, scenario).requires_grad_(True)
    mid = _differentiable_mid(leaf)
    ref_mid = _differentiable_mid(tu.to_reference(leaf))

    assert flag_gems.retain_grad(mid) is None
    torch.ops.aten.retain_grad(ref_mid)

    assert mid.retains_grad is True
    assert mid.retains_grad == ref_mid.retains_grad
    # NaN/Inf payloads are neither sanitized nor rounded by the state change
    # (equal_nan comparison).
    tu.assert_result_equal(mid, ref_mid)


@pytest.mark.retain_grad
@pytest.mark.parametrize("dtype", _NO_GRAD_DTYPES)
def test_retain_grad_rejects_tensor_without_grad(dtype):
    inp = tu.make_input(dtype, (2, 19, 7), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.retain_grad(inp)


@pytest.mark.retain_grad
@pytest.mark.parametrize("bad_input", [1.5, [1, 2], None])
def test_retain_grad_rejects_non_tensor(bad_input):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.retain_grad(bad_input)


@pytest.mark.retain_grad
def test_retain_grad_rejects_missing_argument():
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.retain_grad()
