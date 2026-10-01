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

# aten::mT swaps the last two dimensions of a matrix or batch of matrices and
# returns a zero-copy view sharing storage with its input. It transposes only:
# the lazy conjugate bit of a complex input is preserved, not toggled.
#
# Rank < 2 is handled by its own tests, so the value grid keeps only the rank >= 2
# spec shapes: 0-D is a deprecated identity returning the same tensor object and
# 1-D is rejected.
#
# Values are compared exactly (a view introduces no rounding) and the view
# metadata a value comparison cannot show is checked separately: shape, strides,
# storage offset, storage sharing and the conjugate bit.

_FP64_DTYPES = [torch.float64, torch.complex128] if utils.fp64_is_supported else []
_MT_DTYPES = tu.REQUIRED_DTYPES + [torch.bool, torch.complex64] + _FP64_DTYPES

_MT_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) >= 2]


def _assert_view(res_out, ref_out, inp, ref_inp):
    assert res_out is not inp
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
    assert inp.shape == ref_inp.shape
    assert inp.stride() == ref_inp.stride()
    assert inp.storage_offset() == ref_inp.storage_offset()


def _apply_layout(base, layout, expand_shape=None):
    if layout == "asis":
        return base
    if layout == "transposed":
        return base.transpose(-1, -2)
    if layout == "column_step":
        return base[:, ::2]
    if layout == "offset_window":
        return base[2:6, 1:5]
    if layout == "both_steps":
        return base[::3, ::2]
    if layout == "expanded":
        return base.expand(tuple(expand_shape))
    raise ValueError("unsupported layout " + repr(layout))


def _layout_pair(storage_shape, layout, dtype, expand_shape=None):
    base = tu.make_input(dtype, storage_shape, ["-1", "1"])
    ref_base = tu.to_reference(base.detach())
    inp = _apply_layout(base, layout, expand_shape)
    ref_inp = _apply_layout(ref_base, layout, expand_shape)
    return inp, ref_inp, base, ref_base


@pytest.mark.mT
@pytest.mark.parametrize("shape", _MT_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _MT_DTYPES)
def test_mT(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.mT(ref_inp)
    res_out = flag_gems.mT(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view(res_out, ref_out, inp, ref_inp)


# Multi-dim / batched / degenerate-extent shapes, including empty operands whose
# stride and offset still have to match the native view.
_MT_SIZE_ROWS = [
    (1, 1),
    (1, 7),
    (7, 1),
    (0, 3),
    (3, 0),
    (0, 0),
    (2, 3, 4),
    (5, 1, 2, 3),
    (1, 2, 3, 4, 5),
]


@pytest.mark.mT
@pytest.mark.parametrize("shape", _MT_SIZE_ROWS)
def test_mT_with_size(shape):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.mT(ref_inp)
    res_out = flag_gems.mT(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view(res_out, ref_out, inp, ref_inp)


# Non-contiguous inputs: transposed, stepped, offset and stride-0 expanded
# operands. mT must swap the last two strides of whatever layout it receives.
_MT_LAYOUT_ROWS = [
    ((4, 8), "transposed", None),
    ((4, 8), "column_step", None),
    ((8, 12), "offset_window", None),
    ((6, 10), "both_steps", None),
    ((1, 9), "expanded", (4, 9)),
]

_MT_LAYOUT_DTYPES = [torch.float32, torch.complex64]


@pytest.mark.mT
@pytest.mark.parametrize("storage_shape,layout,expand_shape", _MT_LAYOUT_ROWS)
@pytest.mark.parametrize("dtype", _MT_LAYOUT_DTYPES)
def test_mT_strided_input(storage_shape, layout, expand_shape, dtype):
    inp, ref_inp, base, ref_base = _layout_pair(
        storage_shape, layout, dtype, expand_shape
    )

    ref_out = torch.ops.aten.mT(ref_inp)
    res_out = flag_gems.mT(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view(res_out, ref_out, inp, ref_inp)


@pytest.mark.mT
@pytest.mark.parametrize("shape", [(4, 6), (2, 3, 8)])
def test_mT_conjugate_bit(shape):
    # A lazy-conjugate input keeps its conj bit through the transpose view.
    base = tu.make_input(torch.complex64, shape, ["-1", "1"])
    inp = base.conj()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.mT(ref_inp)
    res_out = flag_gems.mT(inp)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.is_conj()
    _assert_view(res_out, ref_out, inp, ref_inp)


# Contiguous and strided operands: a write through the result must land in the
# input storage.
_MT_MUTATION_ROWS = [
    ((4, 6), "asis"),
    ((3, 5, 2), "asis"),
    ((6, 8), "column_step"),
    ((4, 6), "transposed"),
]


@pytest.mark.mT
@pytest.mark.parametrize("shape,layout", _MT_MUTATION_ROWS)
def test_mT_view_writes_through(shape, layout):
    inp, ref_inp, base, ref_base = _layout_pair(shape, layout, torch.float32)

    ref_out = torch.ops.aten.mT(ref_inp)
    res_out = flag_gems.mT(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view(res_out, ref_out, inp, ref_inp)

    res_out.fill_(3.0)
    ref_out.fill_(3.0)
    tu.assert_result_equal(base, ref_base)


_MT_BACKWARD_DTYPES = tu.selected_cases(
    [dtype for dtype in _MT_DTYPES if dtype.is_floating_point or dtype.is_complex],
    quick=[],
)


@pytest.mark.mT
@pytest.mark.parametrize("shape", [(4, 6), (3, 5, 7)])
@pytest.mark.parametrize("dtype", _MT_BACKWARD_DTYPES)
def test_mT_backward(shape, dtype):
    # The operator is reached through the original leaf, so the gradient is the
    # transposed upstream layout and is compared exactly.
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_inp = tu.to_reference(inp.detach()).requires_grad_(True)
    out_shape = shape[:-2] + (shape[-1], shape[-2])
    upstream = tu.make_input(dtype, out_shape, ["-1", "1"])
    ref_upstream = tu.to_reference(upstream.detach())

    ref_out = torch.ops.aten.mT(ref_inp)
    res_out = flag_gems.mT(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view(res_out, ref_out, inp, ref_inp)

    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]
    ref_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)[0]

    tu.assert_result_equal(res_grad, ref_grad)


MT_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_MT_DTYPES), quick=[])


@pytest.mark.mT
@pytest.mark.parametrize("dtype,scenario", MT_SPECIAL_CASES)
def test_mT_special_values(dtype, scenario):
    # mT only moves metadata, so NaN / Inf payloads must survive untouched.
    inp = tu.make_special_input(dtype, scenario).reshape(1, -1)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.mT(ref_inp)
    res_out = flag_gems.mT(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view(res_out, ref_out, inp, ref_inp)


@pytest.mark.mT
@pytest.mark.parametrize("dtype", _MT_DTYPES)
def test_mT_0d_identity(dtype):
    # 0-D is a deprecated identity: the input object itself is returned.
    inp = tu.make_input(dtype, (), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.mT(ref_inp)
    res_out = flag_gems.mT(inp)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out is inp


@pytest.mark.mT
@pytest.mark.parametrize("shape", [(5,), (256,), (0,)])
def test_mT_1d_rejected(shape):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems.mT(inp)


@pytest.mark.mT
@pytest.mark.parametrize("bad", [3.14, [1.0, 2.0], None])
def test_mT_non_tensor_rejected(bad):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.mT(bad)
