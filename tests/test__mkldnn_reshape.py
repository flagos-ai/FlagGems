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

# aten::_mkldnn_reshape(Tensor self, int[] shape) -> Tensor
# aten::_mkldnn_reshape.out(Tensor self, int[] shape, *, Tensor(a!) out) -> Tensor(a!)
#
# The operand is an opaque oneDNN ("mkldnn") tensor. aten::to_mkldnn is
# registered for the CPU backend only, so the reference and the injected
# candidate both receive the same CPU opaque tensor. Such tensors expose no
# storage and no data pointer, so the candidate's opaque layout, dtype, shape
# and stride are asserted first and the values are compared after the lossless
# native aten::to_dense observation on both sides.

# aten::to_mkldnn accepts float, bfloat16, half, uint8 and int8; float64,
# int32, int64, bool and fp8 are rejected by dense_to_mkldnn.
_SUPPORTED_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float32,
    torch.bfloat16,
    torch.float16,
]

_FLOAT_DTYPES = [torch.float32, torch.bfloat16, torch.float16]

# Static legality filter: the spec's 0-dim shape cannot be given a mkldnn
# operand at all (to_mkldnn raises "could not create a primitive descriptor for
# the reorder primitive"), i.e. the input cannot be constructed. This is an
# input-construction limit, not a kernel result.
GRID_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) > 0]


def _mkldnn_input(dtype, shape, value_range=("-1", "1")):
    """Shared value-range values moved into the CPU mkldnn layout under test."""
    if 0 in shape:
        return torch.zeros(shape, dtype=dtype).to_mkldnn()
    return tu.make_input(dtype, shape, list(value_range)).cpu().to_mkldnn()


def _target_shape(shape):
    """Axis-reversed reshape: a real reorder for rank >= 2, identity for rank 1."""
    return tuple(reversed(shape))


def _alias_kind(result, inp, out=None):
    """Which existing tensor a result aliases, compared with `is`."""
    if result is inp:
        return "input"
    if out is not None and result is out:
        return "out"
    return "new"


def _assert_mkldnn_result(res, ref):
    # The candidate's opaque representation first; widening/moving is never
    # allowed to hide a wrong dtype, layout or shape.
    assert res.is_mkldnn and res.layout == torch._mkldnn
    assert res.dtype == ref.dtype
    assert tuple(res.shape) == tuple(ref.shape)
    assert tuple(res.stride()) == tuple(ref.stride())
    assert res._is_view() is False and res._base is None
    tu.assert_result_equal(torch.ops.aten.to_dense(res), torch.ops.aten.to_dense(ref))


@pytest.mark.mkldnn_reshape
@pytest.mark.parametrize("shape", GRID_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test__mkldnn_reshape(shape, value_range, dtype):
    target = _target_shape(shape)
    inp = _mkldnn_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    res = flag_gems._mkldnn_reshape(inp, list(target))
    ref = torch.ops.aten._mkldnn_reshape(ref_inp, list(target))

    # Native returns the operand itself when the target shape already matches
    # it, otherwise a fresh opaque tensor; the candidate must alias exactly what
    # the oracle aliases.
    assert _alias_kind(res, inp) == _alias_kind(ref, ref_inp)
    assert tuple(res.shape) == target
    _assert_mkldnn_result(res, ref)


# ``shape`` is the operator's only parameter: an int[] needing positive, -1
# inference, zero and boundary entries, with the target rank raised to 4. Each
# row carries the resolved shape because a request containing -1 is not itself
# a shape; the zero rows use a zero-element input, the only valid use of 0.
SHAPE_ROWS = [
    ((2, 3), (3, 2), (3, 2)),
    ((2, 3), (-1,), (6,)),
    ((2, 3), (-1, 2), (3, 2)),
    ((2, 3), (3, -1), (3, 2)),
    ((2, 3), (2, 3), (2, 3)),
    ((2, 3), (1, 6), (1, 6)),
    ((2, 3), (1, 1, 6, 1), (1, 1, 6, 1)),
    ((0,), (0,), (0,)),
    ((0,), (0, 4), (0, 4)),
    ((0,), (4, 0), (4, 0)),
]


@pytest.mark.mkldnn_reshape
@pytest.mark.parametrize(
    "shape,target,resolved",
    SHAPE_ROWS,
    ids=[f"{shape}->{target}" for shape, target, _ in SHAPE_ROWS],
)
def test__mkldnn_reshape_target_shape(shape, target, resolved):
    inp = _mkldnn_input(torch.float32, shape)
    ref_inp = tu.to_reference(inp)

    res = flag_gems._mkldnn_reshape(inp, list(target))
    ref = torch.ops.aten._mkldnn_reshape(ref_inp, list(target))

    assert -1 not in res.shape
    assert tuple(res.shape) == resolved == tuple(ref.shape)
    assert _alias_kind(res, inp) == _alias_kind(ref, ref_inp)
    _assert_mkldnn_result(res, ref)


@pytest.mark.mkldnn_reshape
@pytest.mark.parametrize("shape", GRID_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
def test__mkldnn_reshape_out(shape, dtype):
    target = _target_shape(shape)
    inp = _mkldnn_input(dtype, shape)
    ref_inp = tu.to_reference(inp)
    # Zero-filled rather than uninitialized: both buffers are fully defined
    # before the call, so no undefined content is ever compared.
    out = torch.zeros(target, dtype=dtype).to_mkldnn()
    ref_out = tu.to_reference(out)

    res = flag_gems._mkldnn_reshape(inp, list(target), out=out)
    ref = torch.ops.aten._mkldnn_reshape.out(ref_inp, list(target), out=ref_out)

    assert _alias_kind(res, inp, out) == _alias_kind(ref, ref_inp, ref_out)
    _assert_mkldnn_result(res, ref)


@pytest.mark.mkldnn_reshape
@pytest.mark.parametrize(
    "shape", [s for s in GRID_SHAPES if _target_shape(s) != tuple(s)]
)
def test__mkldnn_reshape_out_returns_out(shape):
    target = _target_shape(shape)
    inp = _mkldnn_input(torch.float32, shape)
    out = torch.zeros(target, dtype=torch.float32).to_mkldnn()

    res = flag_gems._mkldnn_reshape(inp, list(target), out=out)

    # The overload hands back the existing out tensor, not a new opaque tensor;
    # identity is `is`, not an equal pointer.
    assert res is out


@pytest.mark.mkldnn_reshape
def test__mkldnn_reshape_shares_source_memory():
    """The reshaped result observes later writes to the source tensor."""
    inp = _mkldnn_input(torch.float32, (2, 3))
    ref_inp = tu.to_reference(inp)
    res = flag_gems._mkldnn_reshape(inp, [3, 2])
    ref = torch.ops.aten._mkldnn_reshape(ref_inp, [3, 2])

    replacement = torch.arange(6, 12, dtype=torch.float32).reshape(2, 3).to_mkldnn()
    inp.copy_(replacement)
    ref_inp.copy_(tu.to_reference(replacement))

    expected = torch.arange(6, 12, dtype=torch.float32).reshape(3, 2)
    tu.assert_result_equal(torch.ops.aten.to_dense(res), torch.ops.aten.to_dense(ref))
    tu.assert_result_equal(torch.ops.aten.to_dense(res), expected)


# Backward is differentiated through the original mkldnn leaf with a nonuniform
# upstream gradient. A reshape gradient is a pure relayout of that upstream, so
# it is compared exactly.
@pytest.mark.mkldnn_reshape
@pytest.mark.parametrize("shape", tu.selected_cases(GRID_SHAPES, quick=[]))
@pytest.mark.parametrize("dtype", tu.selected_cases(_FLOAT_DTYPES, quick=[]))
def test__mkldnn_reshape_backward(shape, dtype):
    target = _target_shape(shape)
    inp = _mkldnn_input(dtype, shape)
    inp.requires_grad_(True)
    ref_inp = tu.to_reference(inp)
    ref_inp.requires_grad_(True)
    # aten::to_mkldnn is CPU-only, so the upstream gradient must be a CPU
    # tensor even when flag_gems.device is a GPU.
    upstream = tu.make_input(dtype, target, ["-1", "1"]).cpu()
    ref_upstream = tu.to_reference(upstream)

    res = flag_gems._mkldnn_reshape(inp, list(target))
    ref = torch.ops.aten._mkldnn_reshape(ref_inp, list(target))

    # Forward result first, before any gradient is taken.
    assert _alias_kind(res, inp) == _alias_kind(ref, ref_inp)
    _assert_mkldnn_result(res, ref)

    (res_grad,) = torch.autograd.grad(
        torch.ops.aten.to_dense(res), inp, grad_outputs=upstream
    )
    (ref_grad,) = torch.autograd.grad(
        torch.ops.aten.to_dense(ref), ref_inp, grad_outputs=ref_upstream
    )

    assert res_grad.is_mkldnn
    dense_grad = torch.ops.aten.to_dense(res_grad)
    tu.assert_result_equal(dense_grad, torch.ops.aten.to_dense(ref_grad))
    tu.assert_result_equal(dense_grad, upstream.reshape(shape))


SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_FLOAT_DTYPES), quick=[])


@pytest.mark.mkldnn_reshape
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test__mkldnn_reshape_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario).cpu().to_mkldnn()
    ref_inp = tu.to_reference(inp)

    res = flag_gems._mkldnn_reshape(inp, [1, 5])
    ref = torch.ops.aten._mkldnn_reshape(ref_inp, [1, 5])

    assert _alias_kind(res, inp) == _alias_kind(ref, ref_inp)
    _assert_mkldnn_result(res, ref)


# Negatives, kept in both modes. A numel mismatch and an empty target shape are
# invalid for the native operator; a dense operand is outside the schema (the
# operator is registered for mkldnn tensors only); .out needs a mkldnn tensor of
# the requested shape.
NEGATIVE_SHAPE_ROWS = [
    ((2, 3), (5,)),
    ((2, 3), ()),
    ((3, 4), (2, 2)),
]


@pytest.mark.mkldnn_reshape
@pytest.mark.parametrize(
    "shape,target",
    NEGATIVE_SHAPE_ROWS,
    ids=[f"{shape}->{target}" for shape, target in NEGATIVE_SHAPE_ROWS],
)
def test__mkldnn_reshape_rejects_invalid_target_shape(shape, target):
    inp = _mkldnn_input(torch.float32, shape)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems._mkldnn_reshape(inp, list(target))


@pytest.mark.mkldnn_reshape
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test__mkldnn_reshape_rejects_dense_input(dtype):
    dense = torch.zeros((2, 3), dtype=dtype)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._mkldnn_reshape(dense, [3, 2])


@pytest.mark.mkldnn_reshape
def test__mkldnn_reshape_out_requires_mkldnn_out():
    inp = _mkldnn_input(torch.float32, (2, 3))
    dense_out = torch.zeros((3, 2), dtype=torch.float32)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._mkldnn_reshape(inp, [3, 2], out=dense_out)


@pytest.mark.mkldnn_reshape
def test__mkldnn_reshape_out_rejects_wrong_shape():
    inp = _mkldnn_input(torch.float32, (2, 3))
    out = torch.zeros(6, dtype=torch.float32).to_mkldnn()
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._mkldnn_reshape(inp, [3, 2], out=out)
