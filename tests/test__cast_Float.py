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

# `_cast_Float` returns float32. Every dtype below was accepted by the native
# op on the target backend; the backend-dependent ones are gated by the static
# capability flags, so collection never probes a dtype. The op has no `out`
# overload (`torch.ops.aten._cast_Float.out` reports no overload named 'out'),
# so only the functional call form is tested.
_CAST_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float32,
    torch.float16,
    torch.int32,
    torch.int16,
    torch.bool,
]
if utils.fp8_is_supported:
    _CAST_DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]
if utils.bf16_is_supported:
    _CAST_DTYPES.append(torch.bfloat16)
if utils.int64_is_supported:
    _CAST_DTYPES.append(torch.int64)
if utils.fp64_is_supported:
    _CAST_DTYPES.append(torch.float64)

_VIEW_DTYPES = [torch.float16, torch.int8]
if utils.bf16_is_supported:
    _VIEW_DTYPES.append(torch.bfloat16)

_EMPTY_DTYPES = [torch.float16, torch.int8]
if utils.bf16_is_supported:
    _EMPTY_DTYPES.append(torch.bfloat16)

_PARAM_DTYPES = [torch.float16, torch.int32, torch.int8, torch.uint8]
if utils.bf16_is_supported:
    _PARAM_DTYPES.append(torch.bfloat16)
if utils.int64_is_supported:
    _PARAM_DTYPES.append(torch.int64)
if utils.fp8_is_supported:
    _PARAM_DTYPES.append(torch.float8_e4m3fn)

_BACKWARD_DTYPES = [torch.float16, torch.float32]
if utils.bf16_is_supported:
    _BACKWARD_DTYPES.append(torch.bfloat16)
if utils.fp64_is_supported:
    _BACKWARD_DTYPES.append(torch.float64)


@pytest.mark.cast_Float
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _CAST_DTYPES)
def test__cast_Float(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Float(ref_inp)
    res_out = flag_gems._cast_Float(inp)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Float
@pytest.mark.parametrize(
    "view", tu.selected_cases(["transpose", "slice", "offset"], quick=[])
)
@pytest.mark.parametrize("dtype", _VIEW_DTYPES)
def test__cast_Float_strided_input(view, dtype):
    base = tu.make_input(dtype, (4, 8, 16), ["-1", "1"])
    ref_base = tu.to_reference(base)

    if view == "transpose":
        inp, ref_inp = base.transpose(1, 2), ref_base.transpose(1, 2)
    elif view == "slice":
        inp, ref_inp = base[:, ::2, :], ref_base[:, ::2, :]
    else:
        inp, ref_inp = base[1:], ref_base[1:]

    ref_out = torch.ops.aten._cast_Float(ref_inp)
    res_out = flag_gems._cast_Float(inp)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Float
@pytest.mark.parametrize(
    "shape", tu.selected_cases([(0,), (3, 0, 4), (0, 0)], quick=[])
)
@pytest.mark.parametrize("dtype", _EMPTY_DTYPES)
def test__cast_Float_empty(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Float(ref_inp)
    res_out = flag_gems._cast_Float(inp)

    tu.assert_result_equal(res_out, ref_out)


# A float32 input is the native no-op that hands back the input object itself,
# including for 0-dim, offset and empty inputs, so the alias is asserted next
# to the values.
_IDENTITY_VIEWS = tu.selected_cases(
    [
        ("contiguous", (8, 6)),
        ("offset", (8, 6)),
        ("transpose", (8, 6)),
        ("slice", (8, 6)),
        ("scalar", ()),
        ("empty", (0,)),
    ],
    quick=[],
)


@pytest.mark.cast_Float
@pytest.mark.parametrize("view,shape", _IDENTITY_VIEWS)
def test__cast_Float_float32_identity(view, shape):
    base = tu.make_input(torch.float32, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)

    if view == "transpose":
        inp, ref_inp = base.transpose(0, 1), ref_base.transpose(0, 1)
    elif view == "slice":
        inp, ref_inp = base[:, ::2], ref_base[:, ::2]
    elif view == "offset":
        inp, ref_inp = base[1:], ref_base[1:]
    else:
        inp, ref_inp = base, ref_base

    ref_out = torch.ops.aten._cast_Float(ref_inp)
    res_out = flag_gems._cast_Float(inp)

    tu.assert_result_equal(res_out, ref_out)
    # The native op hands back the input object for a float32 input, so the
    # candidate must return that same object; a copy would pass the value
    # comparison above, and returning the input itself already fixes shape,
    # stride and storage offset.
    assert res_out is inp


# `non_blocking` is optional and its schema default is False; the argument-free
# form is already covered by test__cast_Float, which never passes it.
_NON_BLOCKING_CASES = tu.selected_cases([False, True], quick=[])


@pytest.mark.cast_Float
@pytest.mark.parametrize("non_blocking", _NON_BLOCKING_CASES)
@pytest.mark.parametrize("dtype", _PARAM_DTYPES)
def test__cast_Float_non_blocking(non_blocking, dtype):
    inp = tu.make_input(dtype, (1024, 1024), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Float(ref_inp, non_blocking=non_blocking)
    res_out = flag_gems._cast_Float(inp, non_blocking=non_blocking)

    tu.assert_result_equal(res_out, ref_out)


# Deterministic rounding probes: a random range cannot pin a rounding mode. The
# integer rows pin the first integers float32 cannot represent (ties to even)
# and the int64 extremes; the float16 row pins the subnormal and normal edges
# and the largest finite value. Literals are stored in the row dtype first, so
# the bfloat16 row already rounds at construction (257.0 stores as 256.0, and
# 1+2**-9 / 1+2**-10 store as 1.0) and only asserts that those stored values
# survive the cast. The assertions, not this comment, establish the behaviour.
_ROUNDING_ROWS = [
    (
        torch.int32,
        [16777215, 16777216, 16777217, 16777219, 2147483647, -2147483648],
    ),
    (torch.float16, [0.0, -0.0, 5.9604645e-8, 6.1e-5, 65504.0, -65504.0]),
]
if utils.int64_is_supported:
    _ROUNDING_ROWS.append(
        (torch.int64, [16777217, 16777219, 2**24 + 3, 2**62 + 1, -(2**63)])
    )
if utils.bf16_is_supported:
    _ROUNDING_ROWS.append(
        (torch.bfloat16, [0.0, -0.0, 1.0 + 2**-9, 1.0 + 2**-10, 257.0, 3.3895e38])
    )
if utils.fp64_is_supported:
    _ROUNDING_ROWS.append(
        (
            torch.float64,
            [
                0.0,
                -0.0,
                1.0 + 2**-24,
                1.0 + 3 * 2**-24,
                1.0 + 2**-23,
                1.0 + 2**-10,
                1.4012984643e-45,
                1.1754942e-38,
                5e-324,
                1e-300,
                65504.0,
                257.0,
                3.4028235e38,
                3.4028236e38,
                1e300,
            ],
        )
    )
_ROUNDING_CASES = tu.selected_cases(_ROUNDING_ROWS, quick=[])


@pytest.mark.cast_Float
@pytest.mark.parametrize("dtype,values", _ROUNDING_CASES)
def test__cast_Float_rounding_boundaries(dtype, values):
    inp = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Float(ref_inp)
    res_out = flag_gems._cast_Float(inp)

    tu.assert_result_equal(res_out, ref_out)
    # The shared equality treats +0.0 and -0.0 as equal, so the sign of zero is
    # checked with the same shared comparison applied to the signbit results.
    tu.assert_result_equal(torch.signbit(res_out), torch.signbit(ref_out))


# NaN / Inf must survive the cast per dtype; `tu.special_value_cases` drops the
# Inf scenarios float8_e4m3fn cannot represent and keeps its NaN-only case.
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_CAST_DTYPES), quick=[])


@pytest.mark.cast_Float
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test__cast_Float_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Float(ref_inp)
    res_out = flag_gems._cast_Float(inp)

    tu.assert_result_equal(res_out, ref_out)


# The backward of a cast is the upstream gradient cast back to the source
# dtype, with no accumulation, so it is compared value-exactly; a detached or
# perturbed gradient cannot pass the shared assertion.
_BACKWARD_LAYOUTS = tu.selected_cases(
    [
        ("contiguous", (20, 320, 15)),
        ("transpose", (8, 6)),
        ("slice", (8, 6)),
        ("scalar", ()),
    ],
    quick=[],
)


@pytest.mark.cast_Float
@pytest.mark.parametrize("layout,shape", _BACKWARD_LAYOUTS)
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
def test__cast_Float_backward(layout, shape, dtype):
    base = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    ref_base = tu.to_reference(base)

    if layout == "transpose":
        inp, ref_inp = base.transpose(0, 1), ref_base.transpose(0, 1)
    elif layout == "slice":
        inp, ref_inp = base[:, ::2], ref_base[:, ::2]
    else:
        inp, ref_inp = base, ref_base

    ref_out = torch.ops.aten._cast_Float(ref_inp)
    res_out = flag_gems._cast_Float(inp)

    upstream = tu.make_input(torch.float32, tuple(inp.shape), ["-1", "1"])
    ref_grad = torch.autograd.grad(
        ref_out, ref_inp, grad_outputs=tu.to_reference(upstream)
    )[0]
    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]

    tu.assert_result_equal(res_grad, ref_grad)


# Negative cases: the candidate must reject arguments the native schema
# rejects. Only its own exceptions are asserted.
@pytest.mark.cast_Float
def test__cast_Float_rejects_missing_argument():
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Float()


@pytest.mark.cast_Float
def test__cast_Float_rejects_non_tensor_input():
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Float(3.14)


@pytest.mark.cast_Float
def test__cast_Float_rejects_invalid_non_blocking():
    inp = tu.make_input(torch.float16, (2, 19, 7), ["-1", "1"])
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Float(inp, non_blocking="yes")


@pytest.mark.cast_Float
def test__cast_Float_rejects_unknown_keyword():
    inp = tu.make_input(torch.float16, (2, 19, 7), ["-1", "1"])
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Float(inp, unknown_flag=True)


@pytest.mark.cast_Float
def test__cast_Float_rejects_extra_arguments():
    inp = tu.make_input(torch.float16, (2, 19, 7), ["-1", "1"])
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Float(inp, False, False)
