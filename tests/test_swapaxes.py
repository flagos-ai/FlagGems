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

_DTYPE_FLAGS = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.float64: utils.fp64_is_supported,
    torch.complex128: utils.fp64_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
}

# The native op accepts every required dtype plus bool/complex (float64 where
# the device supports it); both modes retain all supported dtypes.
_EXTRA_DTYPES = [torch.bool, torch.complex64]
if utils.fp64_is_supported:
    _EXTRA_DTYPES.append(torch.float64)
SUPPORTED_DTYPES = tu.REQUIRED_DTYPES + _EXTRA_DTYPES
SUPPORTED_DTYPES = [
    dtype for dtype in SUPPORTED_DTYPES if _DTYPE_FLAGS.get(dtype, True)
]


def _assert_view_matches(res_out, ref_out, inp, ref_inp):
    assert res_out is not inp
    assert inp.shape == ref_inp.shape
    assert inp.stride() == ref_inp.stride()
    assert inp.storage_offset() == ref_inp.storage_offset()
    assert res_out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
    # swapaxes is a zero-copy view (Tensor(a)): it permutes the size and stride
    # of the two axes, keeps the storage offset and still aliases the input.
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.data_ptr() == inp.data_ptr()
    assert res_out.device == inp.device


# (shape, axis0, axis1) rows: every accepted rank (0-D .. 5-D), the -ndim and
# ndim-1 boundaries, negative axes and equal axes. A 0-D or 1-D tensor only
# accepts the identity swap; any other index is out of range.
_AXIS_ROWS = [
    ((0, 3), 0, 1),
    ((2, 0, 4), 0, 2),
    ((), 0, 0),
    ((1,), 0, 0),
    ((256,), 0, -1),
    ((1024, 1024), 0, 1),
    ((20, 320, 15), 0, 2),
    ((16, 128, 64, 60), 1, 3),
    ((16, 7, 57, 32, 29), 0, 4),
    ((1024, 1024), -1, -2),
    ((20, 320, 15), -3, 1),
    ((16, 128, 64, 60), 3, 0),
    ((16, 128, 64, 60), -4, 2),
    ((20, 320, 15), 2, 1),
    ((16, 7, 57, 32, 29), 1, 1),
]

# Quick keeps the rank boundaries and the first/last, negative and equal-axis
# combinations on the quick shape; only the large-shape combinations are
# trimmed.
_QUICK_AXIS_ROWS = [
    ((0, 3), 0, 1),
    ((2, 0, 4), 0, 2),
    ((), 0, 0),
    ((1,), 0, 0),
    ((2, 19, 7), 0, 2),
    ((2, 19, 7), -3, -1),
    ((2, 19, 7), 1, 1),
]


@pytest.mark.swapaxes
@pytest.mark.parametrize(
    "shape,axis0,axis1", tu.selected_cases(_AXIS_ROWS, quick=_QUICK_AXIS_ROWS)
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_swapaxes(shape, axis0, axis1, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.swapaxes(ref_inp, axis0, axis1)
    res_out = flag_gems.swapaxes(inp, axis0, axis1)

    _assert_view_matches(res_out, ref_out, inp, ref_inp)
    tu.assert_result_equal(res_out, ref_out)


# Sliced inputs: non-contiguous strides and a non-zero storage offset, which the
# returned view must keep. Both rows are small, so quick keeps them.
_NON_CONTIGUOUS_ROWS = [
    pytest.param(
        (16, 64, 32),
        (slice(None, None, 2), slice(1, None), slice(None, None, 2)),
        0,
        2,
        id="strided",
    ),
    pytest.param(
        (8, 16, 32, 64),
        (slice(None), slice(3, 11), slice(None), slice(0, 30, 3)),
        1,
        3,
        id="offset",
    ),
]


@pytest.mark.swapaxes
@pytest.mark.parametrize("shape,index,axis0,axis1", _NON_CONTIGUOUS_ROWS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_swapaxes_non_contiguous(shape, index, axis0, axis1, dtype):
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base[index]
    ref_inp = ref_base[index]

    ref_out = torch.ops.aten.swapaxes(ref_inp, axis0, axis1)
    res_out = flag_gems.swapaxes(inp, axis0, axis1)

    _assert_view_matches(res_out, ref_out, inp, ref_inp)
    tu.assert_result_equal(res_out, ref_out)


# Both rows are small, so quick keeps them.
_MUTATION_ROWS = [((4, 8, 16), 0, 2), ((16, 32, 8, 4), 1, 3)]


@pytest.mark.swapaxes
@pytest.mark.parametrize("shape,axis0,axis1", _MUTATION_ROWS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_swapaxes_mutation(shape, axis0, axis1, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.swapaxes(ref_inp, axis0, axis1)
    res_out = flag_gems.swapaxes(inp, axis0, axis1)
    _assert_view_matches(res_out, ref_out, inp, ref_inp)
    tu.assert_result_equal(res_out, ref_out)

    # Writing through the returned view must reach the original input storage.
    res_out.copy_(torch.zeros_like(res_out))
    ref_out.copy_(torch.zeros_like(ref_out))

    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(res_out, ref_out)


# Equal axes still return a fresh aliasing view, never the input object.
_IDENTITY_ROWS = [
    ((20, 320, 15), 1, 1),
    ((16, 128, 64, 60), 2, -2),
    ((256,), 0, 0),
    ((), 0, 0),
]
_QUICK_IDENTITY_ROWS = [((20, 320, 15), 1, 1), ((256,), 0, 0), ((), 0, 0)]


@pytest.mark.swapaxes
@pytest.mark.parametrize(
    "shape,axis0,axis1",
    tu.selected_cases(_IDENTITY_ROWS, quick=_QUICK_IDENTITY_ROWS),
)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_swapaxes_identity_axes(shape, axis0, axis1, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.swapaxes(ref_inp, axis0, axis1)
    res_out = flag_gems.swapaxes(inp, axis0, axis1)

    assert res_out is not inp
    assert res_out._is_view() == ref_out._is_view()
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.swapaxes
@pytest.mark.parametrize(
    "shape,axis0,axis1", [((8, 16, 32), 0, 2), ((16, 32, 8, 4), 1, 3)]
)
@pytest.mark.parametrize("dtype", [torch.complex64])
def test_swapaxes_keeps_lazy_conj(shape, axis0, axis1, dtype):
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base.conj()
    ref_inp = ref_base.conj()

    ref_out = torch.ops.aten.swapaxes(ref_inp, axis0, axis1)
    res_out = flag_gems.swapaxes(inp, axis0, axis1)

    _assert_view_matches(res_out, ref_out, inp, ref_inp)
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.is_conj()
    tu.assert_result_equal(res_out, ref_out)


# The special-value matrix is derived from the supported dtypes, so e4m3fn gets
# the nan-only case while e5m2 also gets inf and mixed nan/inf.
_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases([d for d in SUPPORTED_DTYPES if d.is_floating_point]),
    quick=[],
)


@pytest.mark.swapaxes
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_swapaxes_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario).reshape(1, -1)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.swapaxes(ref_inp, 0, 1)
    res_out = flag_gems.swapaxes(inp, 0, 1)

    _assert_view_matches(res_out, ref_out, inp, ref_inp)
    tu.assert_result_equal(res_out, ref_out)


_BACKWARD_DTYPES = [
    dtype for dtype in SUPPORTED_DTYPES if dtype.is_floating_point or dtype.is_complex
]


@pytest.mark.swapaxes
@pytest.mark.parametrize(
    "shape,axis0,axis1",
    tu.selected_cases([((16, 32, 8), 0, 2), ((4, 8, 16, 32), 1, 3)], quick=[]),
)
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
def test_swapaxes_backward(shape, axis0, axis1, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.swapaxes(ref_inp, axis0, axis1)
    upstream = tu.make_input(dtype, ref_out.shape, ["-1", "1"])
    ref_upstream = tu.to_reference(upstream)
    ref_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)[0]

    res_out = flag_gems.swapaxes(inp, axis0, axis1)
    assert res_out.requires_grad
    tu.assert_result_equal(res_out, ref_out)
    _assert_view_matches(res_out, ref_out, inp, ref_inp)
    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]

    # Backward permutes the upstream gradient, so it is exact.
    assert res_grad.shape == ref_grad.shape
    tu.assert_result_equal(res_grad, ref_grad)


@pytest.mark.swapaxes
@pytest.mark.parametrize(
    "shape,axis0,axis1",
    [((2, 3, 4), 0, 3), ((2, 3, 4), -4, 0), ((256,), 0, 2), ((), 1, 0)],
)
def test_swapaxes_out_of_range_axes(shape, axis0, axis1):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])

    with pytest.raises(IndexError):
        flag_gems.swapaxes(inp, axis0, axis1)


@pytest.mark.swapaxes
@pytest.mark.parametrize("axis0,axis1", [(0.5, 1), (0, None), ("1", 0)])
def test_swapaxes_invalid_axis_type(axis0, axis1):
    inp = tu.make_input(torch.float32, (2, 3, 4), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.swapaxes(inp, axis0, axis1)


@pytest.mark.swapaxes
def test_swapaxes_rejects_non_tensor():
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.swapaxes(3.14, 0, 1)


@pytest.mark.swapaxes
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_swapaxes_expanded_input(dtype):
    base = tu.make_input(dtype, (1, 3), ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp, ref_inp = base.expand(4, 3), ref_base.expand(4, 3)

    ref_out = torch.ops.aten.swapaxes(ref_inp, 0, 1)
    res_out = flag_gems.swapaxes(inp, 0, 1)

    _assert_view_matches(res_out, ref_out, inp, ref_inp)
    tu.assert_result_equal(res_out, ref_out)
