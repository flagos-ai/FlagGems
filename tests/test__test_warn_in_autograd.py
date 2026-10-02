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

"""Correctness tests for ``aten::_test_warn_in_autograd``.

The native operator materialises a fresh copy of its single tensor operand like
``clone``: same values/dtype/shape, a dense memory format preserved and anything
else compacted, lazy conjugate and negation bits resolved, and never aliasing the
input. Its backward only warns and passes the upstream gradient through
unchanged.

The operator takes exactly one tensor and has no scalar or optional parameter, so
the spec's broadcast and tensor-vs-scalar dimensions have no operand to apply to;
the ``out=`` overload, the layout matrix and the autograd contract are covered
directly instead.
"""

import warnings

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Dtypes the active backend cannot represent are dropped. These are static
# capability queries on flag_gems.runtime.device, so no tensor is allocated.
_DTYPE_FLAGS = {
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
    torch.bfloat16: utils.bf16_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float64: utils.fp64_is_supported,
}


def _gated(dtypes):
    return [dtype for dtype in dtypes if _DTYPE_FLAGS.get(dtype, True)]


_OP_DTYPES = _gated(tu.REQUIRED_DTYPES + [torch.float64, torch.bool, torch.complex64])
_OP_FLOAT_DTYPES = [dtype for dtype in _OP_DTYPES if dtype.is_floating_point]

# The native autograd key rejects complex outputs ("does not support automatic
# differentiation for outputs with complex dtype") and integer/bool tensors
# cannot require grad, so the backward grid stays on the floating dtypes.
_BACKWARD_DTYPES = _OP_FLOAT_DTYPES

# ``torch._neg_view`` needs a signed dtype; the conjugate bit needs complex.
_NEG_VIEW_DTYPES = _gated(
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float64,
        torch.int8,
        torch.int32,
        torch.int64,
    ]
)
_CONJ_DTYPES = _gated([torch.complex64])

# Zero-element tensors are valid copy inputs and reach the storage-less path the
# seven shared shapes do not; they are cheap, so the quick subset keeps them.
_CHEAP_SHAPES = [(0,), (0, 5), (3, 0, 4)]
_GRID_SHAPE_ROWS = tu.REQUIRED_SHAPES + _CHEAP_SHAPES
_GRID_SHAPES = tu.selected_cases(
    _GRID_SHAPE_ROWS, quick=tu.QUICK_SHAPES + [()] + _CHEAP_SHAPES
)

# Backward is default-only: a gradient case is not a smoke case. Every floating
# dtype is valid here, fp8 included; complex is rejected by the native autograd
# key noted above.
_BACKWARD_SHAPES = tu.selected_cases(_GRID_SHAPE_ROWS, quick=[])

# The shared shapes are all dense; these rows add non-contiguous strides, a
# nonzero storage offset, a zero-stride expansion and an unbacked view. The
# resulting stride patterns differ per row, so the native result is the oracle
# rather than a fixed expectation. Every row is cheap, so none leaves quick.
_LAYOUT_ROWS = [
    ((4, 8, 16), "transposed"),
    ((4, 8, 16), "sliced"),
    ((8, 4, 12), "offset"),
    ((2, 5), "expanded"),
    ((4, 8, 16), "as_strided"),
]

# ``out=`` buffers: a dense buffer holding a sentinel the copy must overwrite, a
# non-contiguous view inside a larger parent that only the view may be written
# through, and an empty buffer the resize path grows in place.
_OUT_ROWS = [
    ((2, 19, 7), "dense"),
    ((16, 32), "dense"),
    ((16, 32), "view"),
    ((3, 4), "resize"),
]

_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_OP_FLOAT_DTYPES), quick=[])


def _layout_input(dtype, shape, layout):
    """Build one layout row; returns ``(base, tensor)`` with ``tensor`` a view."""
    if layout == "transposed":
        base = tu.make_input(dtype, (shape[2], shape[1], shape[0]), ["-1", "1"])
        return base, base.transpose(0, 2)
    if layout == "sliced":
        base = tu.make_input(dtype, (shape[0], shape[1] * 2, shape[2]), ["-1", "1"])
        return base, base[:, ::2]
    if layout == "offset":
        base = tu.make_input(dtype, (shape[0] * 2, shape[1], shape[2]), ["-1", "1"])
        return base, base[shape[0] :]
    if layout == "expanded":
        base = tu.make_input(dtype, (1, shape[1]), ["-1", "1"])
        return base, base.expand(shape[0], shape[1])
    # as_strided: rows spread over twice the span they occupy, so the view skips
    # storage. The base covers the furthest element read
    # ((shape[0] - 1) * 2 + 1 rows) and is only a backing store.
    row = shape[1] * shape[2]
    base = tu.make_input(dtype, (row * shape[0] * 2 + 1,), ["-1", "1"])
    return base, base.as_strided(shape, (row * 2, shape[2], 1), 0)


def _assert_copy_not_view(res_out, ref_out, inp):
    """Metadata assertions the shared value comparison does not cover."""
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert not res_out._is_view()
    # A materialising copy owns fresh storage, but a zero-element tensor reports
    # a null data pointer on every allocation, so the storage check only applies
    # when the tensor actually has storage; emptiness is still covered by the
    # not-a-view check above.
    if inp.numel() > 0:
        assert res_out.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()


def _out_buffer(dtype, shape, layout, device):
    """Return ``(parent, buffer)``; ``parent`` is set for the view row only."""
    if layout == "resize":
        return None, torch.empty(0, dtype=dtype, device=device)
    fill = True if dtype == torch.bool else 7
    if layout == "dense":
        return None, torch.full(shape, fill, dtype=dtype, device=device)
    # A non-contiguous view with a nonzero offset inside a larger parent: the
    # copy may write through the view only and must leave the padding untouched.
    parent = torch.full((shape[0] + 4, shape[1] * 2), fill, dtype=dtype, device=device)
    return parent, parent[2 : 2 + shape[0], ::2]


@pytest.mark.test_warn_in_autograd
@pytest.mark.parametrize("shape", _GRID_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _OP_DTYPES)
def test__test_warn_in_autograd(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_warn_in_autograd(ref_inp)
    res_out = flag_gems._test_warn_in_autograd(inp)

    # A copy introduces no rounding, so compare exactly.
    tu.assert_result_equal(res_out, ref_out)
    _assert_copy_not_view(res_out, ref_out, inp)


@pytest.mark.test_warn_in_autograd
@pytest.mark.parametrize("shape,layout", _LAYOUT_ROWS)
@pytest.mark.parametrize("dtype", _OP_DTYPES)
def test__test_warn_in_autograd_layout(shape, layout, dtype):
    _base, inp = _layout_input(dtype, shape, layout)
    ref_inp = tu.to_reference(inp)
    snapshot = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_warn_in_autograd(ref_inp)
    res_out = flag_gems._test_warn_in_autograd(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_copy_not_view(res_out, ref_out, inp)
    # Reading a strided view must not modify it.
    tu.assert_result_equal(inp, snapshot)


@pytest.mark.test_warn_in_autograd
@pytest.mark.parametrize("shape,layout", _OUT_ROWS)
@pytest.mark.parametrize("dtype", _OP_DTYPES)
def test__test_warn_in_autograd_out(shape, layout, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    parent, buf = _out_buffer(dtype, shape, layout, flag_gems.device)
    if parent is None:
        ref_buf = tu.to_reference(buf)
    else:
        # The reference buffer has to be a view of the reference parent, so a
        # write through it is observable in the parent on both sides.
        ref_parent = tu.to_reference(parent)
        ref_buf = ref_parent[2 : 2 + shape[0], ::2]
    buf_ptr = buf.data_ptr()

    ref_out = torch.ops.aten._test_warn_in_autograd.out(ref_inp, out=ref_buf)
    res_out = flag_gems._test_warn_in_autograd(inp, out=buf)

    # The buffer is written in place and returned by identity: the candidate must
    # not rebind it to a new allocation.
    assert res_out is buf
    if layout != "resize":
        # Only the empty buffer sits in the resize path, which reallocates.
        assert buf.data_ptr() == buf_ptr
    assert buf.dtype == ref_buf.dtype
    assert buf.shape == ref_buf.shape
    assert buf.stride() == ref_buf.stride()
    tu.assert_result_equal(res_out, ref_out)
    if parent is not None:
        tu.assert_result_equal(parent, ref_parent)


@pytest.mark.test_warn_in_autograd
@pytest.mark.parametrize("dtype", _CONJ_DTYPES)
def test__test_warn_in_autograd_conj_view(dtype):
    base = tu.make_input(dtype, (4, 8), ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp, ref_inp = base.conj(), ref_base.conj()

    ref_out = torch.ops.aten._test_warn_in_autograd(ref_inp)
    res_out = flag_gems._test_warn_in_autograd(inp)

    # The lazy conjugate bit on the input is resolved by the copy.
    assert not res_out.is_conj()
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.test_warn_in_autograd
@pytest.mark.parametrize("dtype", _NEG_VIEW_DTYPES)
def test__test_warn_in_autograd_neg_view(dtype):
    base = tu.make_input(dtype, (4, 8), ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp, ref_inp = torch._neg_view(base), torch._neg_view(ref_base)

    ref_out = torch.ops.aten._test_warn_in_autograd(ref_inp)
    res_out = flag_gems._test_warn_in_autograd(inp)

    # The lazy negation bit on the input is resolved by the copy.
    assert not res_out.is_neg()
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.test_warn_in_autograd
@pytest.mark.parametrize("dtype", _OP_DTYPES)
def test__test_warn_in_autograd_result_is_independent(dtype):
    inp = tu.make_input(dtype, (8, 16), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_warn_in_autograd(ref_inp)
    res_out = flag_gems._test_warn_in_autograd(inp)
    tu.assert_result_equal(res_out, ref_out)
    assert res_out is not inp

    sentinel = torch.full_like(res_out, True if dtype == torch.bool else 3)
    res_out.copy_(sentinel)
    # The write reached the result's own storage and left the input alone.
    tu.assert_result_equal(res_out, tu.to_reference(sentinel))
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.test_warn_in_autograd
@pytest.mark.parametrize("shape", _BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
def test__test_warn_in_autograd_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_warn_in_autograd(ref_inp)
    res_out = flag_gems._test_warn_in_autograd(inp)
    assert res_out.requires_grad
    # The forward values still match before the graph is differentiated.
    tu.assert_result_equal(res_out, ref_out)

    # A non-uniform upstream gradient: the native backward passes it through
    # unchanged, so an identity pass and a wrong gather fail differently.
    grad = tu.make_input(dtype, shape, ["-1", "1"])
    ref_grad = tu.to_reference(grad)
    # The native backward warns by design; the text is an autograd artifact, not
    # part of the tensor contract, so it is not asserted here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
        ref_grad_out = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    tu.assert_result_equal(res_grad, ref_grad_out)
    # The configured reference may live on another device, so the pass-through
    # sentinel is compared against the gradient's own reference layout.
    tu.assert_result_equal(res_grad, tu.to_reference(grad))


@pytest.mark.test_warn_in_autograd
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test__test_warn_in_autograd_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_warn_in_autograd(ref_inp)
    res_out = flag_gems._test_warn_in_autograd(inp)

    # NaN and Inf are copied verbatim; matching NaNs are part of the contract.
    tu.assert_result_equal(res_out, ref_out)
    _assert_copy_not_view(res_out, ref_out, inp)


@pytest.mark.test_warn_in_autograd
def test__test_warn_in_autograd_rejects_non_tensor():
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_warn_in_autograd([1.0, 2.0, 3.0])


@pytest.mark.test_warn_in_autograd
def test__test_warn_in_autograd_out_rejects_wrong_dtype():
    inp = tu.make_input(torch.float32, (4, 8), ["-1", "1"])
    buf = torch.empty(4, 8, dtype=torch.float16, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_warn_in_autograd(inp, out=buf)


@pytest.mark.test_warn_in_autograd
def test__test_warn_in_autograd_out_rejects_non_tensor():
    inp = tu.make_input(torch.float32, (4, 8), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_warn_in_autograd(inp, out=[0.0] * 32)
