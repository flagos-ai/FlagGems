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

"""Correctness tests for ``aten::ge_`` (in-place ``self >= other``)."""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

_CAPABILITY_FLAGS = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float64: utils.fp64_is_supported,
}

DTYPES = [
    dtype
    for dtype in (
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
    if _CAPABILITY_FLAGS.get(dtype, True)
]

FLOAT_DTYPES = [dtype for dtype in DTYPES if dtype.is_floating_point]

# bf16 is the spec's preferred broadcast dtype, fp16 the supported fallback.
BROADCAST_DTYPE = torch.bfloat16 if torch.bfloat16 in DTYPES else torch.float16

# (receiver shape, other shape, receiver dtype, other dtype, value range).
_GRID_CASES = [
    (shape, shape, dtype, dtype, value_range)
    for dtype in DTYPES
    for shape in tu.selected_shapes()
    for value_range in tu.selected_ranges()
]

_EXTRA_CASES = [
    # 1-dim and multi-dim broadcast; ge_ writes into its receiver, so the
    # receiver carries the full shape.
    ((20, 320, 15), (15,), BROADCAST_DTYPE, BROADCAST_DTYPE, ["-1", "1"]),
    ((16, 128, 64, 60), (16, 1, 1, 60), torch.int32, torch.int32, ["-1", "1"]),
    # Mixed dtypes compare promoted values and store 0/1 in the receiver dtype.
    ((1024, 1024), (1024, 1024), torch.int8, torch.float32, ["-1", "1"]),
    ((20, 320, 15), (20, 320, 15), torch.int32, torch.float64, ["-1", "1"]),
    ((16, 128, 64, 60), (16, 128, 64, 60), torch.float32, torch.int32, ["-1", "1"]),
    (
        (16, 7, 57, 32, 29),
        (16, 7, 57, 32, 29),
        torch.float16,
        torch.int32,
        ["-1", "1"],
    ),
]

TENSOR_CASES = _GRID_CASES + tu.selected_cases(
    [row for row in _EXTRA_CASES if row[2] in DTYPES and row[3] in DTYPES],
    quick=[],
)

# Scalar sweep: no schema default, so zero, both signs, a fractional value and
# the float nan/inf boundaries.
_FLOAT_SCALARS = (
    0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    float("inf"),
    float("-inf"),
    float("nan"),
)
_INT_SCALARS = (0, 1, -1)

SCALAR_CASES = tu.selected_cases(
    [
        (dtype, scalar)
        for dtype in DTYPES
        for scalar in (
            _FLOAT_SCALARS
            if dtype.is_floating_point
            else ((0, 1) if dtype == torch.bool else _INT_SCALARS)
        )
    ],
    quick=[],
)

BACKWARD_CASES = tu.selected_cases(
    [
        (dtype, shape, ["-1", "1"])
        for dtype in FLOAT_DTYPES
        for shape in ((256,), (20, 320, 15))
    ],
    quick=[],
)

SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(FLOAT_DTYPES), quick=[])

# The operand is the receiver as well, so every element compares with itself.
ALIAS_CASES = tu.selected_cases(
    [
        (dtype, shape, ["-1", "1"])
        for dtype in DTYPES
        if dtype in (torch.float32, torch.int32, torch.float16, torch.int8)
        for shape in ((20, 320, 15), (1024, 1024))
    ],
    quick=[],
)

# Views of a larger storage: "transposed" is non-contiguous, "offset" starts at
# storage offset 16 * 32.
_VIEW_KINDS = {
    "transposed": ((8, 16, 32), (16, 8, 32)),
    "offset": ((4, 16, 32), (3, 16, 32)),
}

VIEW_CASES = tu.selected_cases(
    [
        (dtype, kind)
        for dtype in (torch.float32, torch.bfloat16)
        if dtype in DTYPES
        for kind in _VIEW_KINDS
    ],
    quick=[],
)

# The operand neither broadcasts up to the receiver nor yields an in-place
# output of the receiver's shape, so the native op rejects it.
NEGATIVE_CASES = [
    ((2, 19, 7), (5,)),
    ((2, 19, 7), (3, 2, 19, 7)),
]

# Dtypes measured to have no native comparison kernel on the vendor's backend
# ("compare_cuda" not implemented for float8 / complex); measured once during
# generation, never probed here so collection stays metadata-only. Unmeasured
# vendors get no rejection cases.
_COMPARE_REJECTED_DTYPES = {
    "nvidia": (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        torch.complex64,
    ),
}

REJECTED_DTYPES = [
    dtype
    for dtype in _COMPARE_REJECTED_DTYPES.get(flag_gems.vendor_name, ())
    if dtype is not None
]


def _view_of(storage, kind):
    return storage.transpose(0, 1) if kind == "transposed" else storage[1:]


@pytest.mark.ge_
@pytest.mark.parametrize(
    "shape,other_shape,dtype,other_dtype,value_range", TENSOR_CASES
)
def test_ge__tensor(shape, other_shape, dtype, other_dtype, value_range):
    inp = tu.make_input(dtype, shape, value_range)
    other = tu.make_input(other_dtype, other_shape, value_range)

    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)
    res_inp = inp.clone()

    ref_out = torch.ops.aten.ge_.Tensor(ref_inp, ref_other)
    res_out = flag_gems.ge_(res_inp, other)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(res_inp, ref_inp)
    assert res_out is res_inp


@pytest.mark.ge_
@pytest.mark.parametrize("dtype,scalar", SCALAR_CASES)
def test_ge__scalar(dtype, scalar):
    inp = tu.make_input(dtype, (1024, 1024), ["-1", "1"])

    ref_inp = tu.to_reference(inp)
    res_inp = inp.clone()

    ref_out = torch.ops.aten.ge_.Scalar(ref_inp, scalar)
    res_out = flag_gems.ge_(res_inp, scalar)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(res_inp, ref_inp)
    assert res_out is res_inp


@pytest.mark.ge_
@pytest.mark.parametrize("dtype,shape,value_range", BACKWARD_CASES)
def test_ge__backward(dtype, shape, value_range):
    inp = tu.make_input(dtype, shape, value_range)
    other = tu.make_input(dtype, shape, value_range)

    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    # autograd rejects an in-place op on a grad-requiring leaf, so the receiver
    # is a non-leaf.
    ref_base = ref_inp.detach().clone().requires_grad_(True)
    ref_receiver = ref_base * 2.0
    ref_out = torch.ops.aten.ge_.Tensor(ref_receiver, ref_other)
    (ref_grad,) = torch.autograd.grad(
        ref_out, ref_base, grad_outputs=torch.ones_like(ref_out)
    )

    res_base = inp.detach().clone().requires_grad_(True)
    res_receiver = res_base * 2.0
    res_out = flag_gems.ge_(res_receiver, other)
    (res_grad,) = torch.autograd.grad(
        res_out, res_base, grad_outputs=torch.ones_like(res_out)
    )

    tu.assert_result_equal(res_out, ref_out)
    assert res_out is res_receiver
    tu.assert_result_equal(res_receiver, ref_receiver)
    tu.assert_result_close(res_grad, ref_grad)


@pytest.mark.ge_
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test_ge__special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    # The flipped operand pairs each special value with a different one, so nan
    # and inf take part on both sides of the comparison.
    other = tu.make_special_input(dtype, scenario).flip(0)

    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)
    res_inp = inp.clone()

    ref_out = torch.ops.aten.ge_.Tensor(ref_inp, ref_other)
    res_out = flag_gems.ge_(res_inp, other)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(res_inp, ref_inp)
    assert res_out is res_inp


@pytest.mark.ge_
@pytest.mark.parametrize("dtype,shape,value_range", ALIAS_CASES)
def test_ge__aliased_operand(dtype, shape, value_range):
    inp = tu.make_input(dtype, shape, value_range)

    ref_inp = tu.to_reference(inp)
    res_inp = inp.clone()

    ref_out = torch.ops.aten.ge_.Tensor(ref_inp, ref_inp)
    res_out = flag_gems.ge_(res_inp, res_inp)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(res_inp, ref_inp)
    assert res_out is res_inp


@pytest.mark.ge_
@pytest.mark.parametrize("dtype,kind", VIEW_CASES)
def test_ge__view_receiver(dtype, kind):
    storage_shape, other_shape = _VIEW_KINDS[kind]
    storage = tu.make_input(dtype, storage_shape, ["-1", "1"])
    other = tu.make_input(dtype, other_shape, ["-1", "1"])

    ref_storage = tu.to_reference(storage)
    res_storage = storage.clone()
    ref_inp = _view_of(ref_storage, kind)
    res_inp = _view_of(res_storage, kind)

    ref_out = torch.ops.aten.ge_.Tensor(ref_inp, tu.to_reference(other))
    res_out = flag_gems.ge_(res_inp, other)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(res_inp, ref_inp)
    # The whole backing storage shows the writes landed at the view's offsets.
    tu.assert_result_equal(res_storage, ref_storage)
    assert res_out is res_inp


@pytest.mark.ge_
@pytest.mark.parametrize("shape,other_shape", NEGATIVE_CASES)
def test_ge__invalid_operand(shape, other_shape):
    inp = torch.ones(shape, dtype=torch.float32, device=flag_gems.device)
    other = torch.ones(other_shape, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.ge_(inp, other)


@pytest.mark.ge_
@pytest.mark.parametrize("dtype", REJECTED_DTYPES)
def test_ge__unsupported_dtype(dtype):
    inp = torch.ones((2, 19, 7), dtype=dtype, device=flag_gems.device)
    other = torch.ones((2, 19, 7), dtype=dtype, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.ge_(inp, other)
