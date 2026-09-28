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

"""Correctness tests for ``aten::_reshape_copy``.

``_reshape_copy`` returns a copy of ``self`` reshaped to ``size``: the element
count is preserved and a non-empty result never shares storage with its input.
The native op exposes a single ``default`` overload (``overloads() ==
["default"]``; ``.out`` raises ``AttributeError: The underlying op of
'aten._reshape_copy' has no overload name 'out'``), so there is no real
out-variant workload to test.

The operand is one tensor plus a shape list, so the spec's broadcast dimension
does not apply: there is no second operand whose shape could broadcast, and a
"broadcast" workload would just be another reshape workload.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Static backend capability flags, read at import time. The suite never calls
# the native operator to decide which dtypes to collect.
_DTYPE_GATES = {
    torch.float64: utils.fp64_is_supported,
    torch.bfloat16: utils.bf16_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
}


def _supported(dtypes):
    """Drop dtypes the backend's static capability flags do not cover."""
    return [dtype for dtype in dtypes if _DTYPE_GATES.get(dtype, True)]


_RESHAPE_COPY_DTYPES = _supported(
    tu.REQUIRED_DTYPES + [torch.int16, torch.bool, torch.complex64]
)


def _target_size(shape):
    """Reversed extents, which keeps the flattened element order untouched.

    Reversing the extents only changes how flat indices map onto coordinates;
    the stored values are not transposed or reordered.
    """
    return list(reversed(tuple(shape)))


def _assert_copy_semantics(res_out, ref_out, inp, ref_inp):
    """Exact values plus the independence and layout a copy must own."""
    # The native op returns a distinct tensor object even for zero-element
    # inputs, so returning the input itself is never a valid copy.
    assert res_out is not inp
    assert res_out.device == inp.device
    assert res_out.is_contiguous()
    if res_out.numel() > 0:
        # Compare underlying storage: tensors at different offsets of one
        # storage report different data_ptr() values while still sharing it, so
        # data_ptr() alone would not prove independence.
        assert res_out.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.reshape_copy
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _RESHAPE_COPY_DTYPES)
def test__reshape_copy_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    size = _target_size(shape)
    ref_out = torch.ops.aten._reshape_copy(ref_inp, size)
    res_out = flag_gems._reshape_copy(inp, size)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


# ``size`` forms: scalar to 1-D, explicit zero extents, one inferred dimension
# (-1) in different positions, and non-diagonal extent permutations. Only the
# default suite sweeps them.
_SIZE_ARG_CASES = tu.selected_cases(
    [
        ((), [1]),
        ((2, 3, 4), [4, 3, 2]),
        ((2, 3, 4), [24]),
        ((2, 3, 4), [24, 1]),
        ((2, 3, 4), [1, 24]),
        ((2, 3, 4), [3, 8]),
        ((2, 3, 4), [2, 12]),
        ((2, 3, 4), [3, -1]),
        ((2, 3, 4), [2, 3, -1]),
        ((0, 3), [3, 0]),
        ((0, 3), [3, -1]),
        ((1,), []),
    ],
    quick=[],
)


@pytest.mark.reshape_copy
@pytest.mark.parametrize("shape,size", _SIZE_ARG_CASES)
@pytest.mark.parametrize(
    "dtype", _supported([torch.bfloat16, torch.float8_e4m3fn, torch.int8])
)
def test__reshape_copy_size_forms(shape, size, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._reshape_copy(ref_inp, size)
    res_out = flag_gems._reshape_copy(inp, size)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


_KEYWORD_SIZE_CASES = tu.selected_cases(
    [
        ((2, 3, 4), [3, 8]),
        ((2, 3, 4), [24]),
    ],
    quick=[],
)


@pytest.mark.reshape_copy
@pytest.mark.parametrize("shape,size", _KEYWORD_SIZE_CASES)
def test__reshape_copy_keyword_size(shape, size):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._reshape_copy(ref_inp, size=size)
    res_out = flag_gems._reshape_copy(inp, size=size)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


_NONCONTIG_LAYOUTS = tu.selected_cases(["transpose", "slice", "offset"], quick=[])


@pytest.mark.reshape_copy
@pytest.mark.parametrize("layout", _NONCONTIG_LAYOUTS)
@pytest.mark.parametrize("dtype", _supported([torch.float32, torch.int32]))
def test__reshape_copy_non_contiguous_input(layout, dtype):
    base = tu.make_input(dtype, (4, 8, 16), ["-1", "1"])
    if layout == "transpose":
        inp = base.transpose(0, 1)
    elif layout == "slice":
        inp = base[:, ::2]
    else:
        inp = base[2:]
    ref_inp = tu.to_reference(inp)

    size = _target_size(inp.shape)
    ref_out = torch.ops.aten._reshape_copy(ref_inp, size)
    res_out = flag_gems._reshape_copy(inp, size)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


_LAZY_VIEW_CASES = tu.selected_cases(
    [(torch.complex64, "conj"), (torch.float32, "neg")]
    + ([(torch.float64, "neg")] if utils.fp64_is_supported else []),
    quick=[],
)


@pytest.mark.reshape_copy
@pytest.mark.parametrize("dtype,lazy_bit", _LAZY_VIEW_CASES)
def test__reshape_copy_drops_lazy_view_bit(dtype, lazy_bit):
    base = tu.make_input(dtype, (4, 8, 16), ["-1", "1"])
    ref_base = tu.to_reference(base)
    # ``to_reference`` copies dense storage, which cannot carry a lazy bit, so
    # the reference view is rebuilt from its own physically distinct base.
    if lazy_bit == "conj":
        inp, ref_inp = base.conj(), ref_base.conj()
    else:
        inp, ref_inp = base._neg_view(), ref_base._neg_view()
    size = _target_size(base.shape)

    ref_out = torch.ops.aten._reshape_copy(ref_inp, size)
    res_out = flag_gems._reshape_copy(inp, size)

    assert not res_out.is_conj()
    assert not res_out.is_neg()
    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


_EMPTY_CASES = tu.selected_cases(
    [
        ((0,), [0]),
        ((0, 3), [0, 3]),
        ((0, 3), [3, 0]),
        ((2, 0, 4), [8, 0]),
    ],
    quick=[],
)


@pytest.mark.reshape_copy
@pytest.mark.parametrize("shape,size", _EMPTY_CASES)
@pytest.mark.parametrize(
    "dtype", _supported([torch.float32, torch.int8, torch.float8_e4m3fn])
)
def test__reshape_copy_empty_input(shape, size, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._reshape_copy(ref_inp, size)
    res_out = flag_gems._reshape_copy(inp, size)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


# ``special_value_cases`` omits the inf-containing scenarios for dtypes that
# cannot represent infinity (float8_e4m3fn keeps only "nan") and keeps them for
# float8_e5m2. A complex input takes the same payload in both parts, so the
# imaginary part carries the special values too.
_SPECIAL_VALUE_CASES = tu.selected_cases(
    tu.special_value_cases(
        _supported(utils.ALL_FLOAT_DTYPES + [torch.float8_e4m3fn, torch.float8_e5m2])
    )
    + [(torch.complex64, scenario) for scenario in ("nan", "inf", "mixed")],
    quick=[],
)


def _special_input(dtype, scenario):
    if dtype.is_complex:
        payload = tu.make_special_input(torch.float32, scenario)
        return torch.complex(payload, payload).to(dtype)
    return tu.make_special_input(dtype, scenario)


@pytest.mark.reshape_copy
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_VALUE_CASES)
def test__reshape_copy_special_values(dtype, scenario):
    inp = _special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    size = _target_size(inp.shape)
    ref_out = torch.ops.aten._reshape_copy(ref_inp, size)
    res_out = flag_gems._reshape_copy(inp, size)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


_BACKWARD_SHAPES = tu.selected_cases([(16, 64), (7, 13, 29)], quick=[])


@pytest.mark.reshape_copy
@pytest.mark.parametrize("shape", _BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", _supported(utils.ALL_FLOAT_DTYPES))
def test__reshape_copy_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).clone().requires_grad_(True)
    ref_inp = tu.to_reference(inp).clone().requires_grad_(True)

    size = _target_size(shape)
    ref_out = torch.ops.aten._reshape_copy(ref_inp, size)
    res_out = flag_gems._reshape_copy(inp, size)

    # The upstream gradient is laid out like the result; both the candidate and
    # the reference receive the same values.
    grad_out = tu.make_input(dtype, size, ["-1", "1"])
    ref_grad_out = tu.to_reference(grad_out)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad_out)
    (res_grad,) = torch.autograd.grad(res_out, inp, grad_outputs=grad_out)

    tu.assert_result_equal(res_grad, ref_grad)


_INVALID_SIZES = tu.selected_cases(
    [
        ([0], RuntimeError),
        ([], RuntimeError),
        ([12], RuntimeError),
        ([5, 5], RuntimeError),
        ([-1, -1], RuntimeError),
        ([-2], RuntimeError),
        # ``True`` is coerced to 1 by the native schema (``[True, 24]`` is
        # accepted as shape (1, 24)), so this row is rejected for its element
        # count, not for a bool extent.
        ([True, 2], RuntimeError),
        ([1.5, 2], (TypeError, RuntimeError)),
    ],
    quick=[
        ([0], RuntimeError),
        ([5, 5], RuntimeError),
        ([1.5, 2], (TypeError, RuntimeError)),
    ],
)


@pytest.mark.reshape_copy
@pytest.mark.parametrize("size,error", _INVALID_SIZES)
def test__reshape_copy_rejects_invalid_size(size, error):
    inp = tu.make_input(torch.float32, (2, 3, 4), ["-1", "1"])
    with pytest.raises(error):
        flag_gems._reshape_copy(inp, size)


@pytest.mark.reshape_copy
def test__reshape_copy_rejects_ambiguous_empty_inference():
    # Two inferred (-1) extents stay ambiguous even when the input is empty.
    inp = tu.make_input(torch.float32, (0,), ["-1", "1"])
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._reshape_copy(inp, [-1, -1])


@pytest.mark.reshape_copy
@pytest.mark.parametrize("size", [6, 6.0])
def test__reshape_copy_rejects_non_list_size(size):
    inp = tu.make_input(torch.float32, (2, 3, 4), ["-1", "1"])
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._reshape_copy(inp, size)


@pytest.mark.reshape_copy
def test__reshape_copy_rejects_missing_size():
    inp = tu.make_input(torch.float32, (2, 3, 4), ["-1", "1"])
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._reshape_copy(inp)


@pytest.mark.reshape_copy
@pytest.mark.parametrize("source", [[1, 2, 3], 1.5])
def test__reshape_copy_rejects_non_tensor_source(source):
    # The native schema requires a Tensor for ``self``: "Expected a value of
    # type 'Tensor' for argument 'self' but instead found type 'list'".
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._reshape_copy(source, [3])
