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

"""Correctness tests for aten::result_type.

aten::result_type is pure dtype inference: each of its four overloads
(.Tensor, .Scalar, .Scalar_Tensor, .Scalar_Scalar) returns a Python int
ScalarType code, never a tensor, and no overload reads element values. The
spec's five value ranges, broadcast and backward dimensions therefore cannot
change a result: there is no arithmetic to observe and no value to compare. The
seven-shape grid is represented instead by the only shape property this operator
looks at, the operand rank role: for two operands of different dtypes zero-dimensional and dimensioned
tensors have different promotion priority, so swapping their rank roles can
change the code, as the mixed-dtype rank-role test below
shows. The semantic dimensions covered here are the four call forms, the dtype
promotion lattice, that mixed-dtype rank role, Python scalar-class promotion,
the fp8 promotion restrictions and the rejected argument forms. Every
expectation is taken from the native operator itself, so no operand is upcast,
converted or re-created.
"""

import itertools

import pytest
import torch

import flag_gems

from . import test_utils as tu

_DTYPE_FLAGS = {
    torch.bfloat16: flag_gems.runtime.device.support_bf16,
    torch.float64: flag_gems.runtime.device.support_fp64,
    torch.complex128: flag_gems.runtime.device.support_fp64,
    torch.int64: flag_gems.runtime.device.support_int64,
    torch.float8_e4m3fn: flag_gems.runtime.device.support_fp8,
    torch.float8_e5m2: flag_gems.runtime.device.support_fp8,
}


def _dtype(name):
    dtype = getattr(torch, name, None)
    return dtype if isinstance(dtype, torch.dtype) else None


# fp8 promotes only with a second fp8 operand of the same type: every mixed
# pair raises 'Promotion for Float8 Types is not supported' (probed on the
# active backend), so fp8 dtypes stay out of the general promotion lattice and
# are covered by the dedicated positive and rejected-row tests below.
FP8_DTYPES = [d for d in (_dtype("float8_e4m3fn"), _dtype("float8_e5m2")) if d]

FP8_DTYPES = [dtype for dtype in FP8_DTYPES if _DTYPE_FLAGS.get(dtype, True)]

# Pairwise promotion is defined for these twelve types: the integer types cover
# the integer rules and complex64/complex128 the complex rules.
PROMOTION_DTYPES = [
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
]

PROMOTION_DTYPES = [
    dtype for dtype in PROMOTION_DTYPES if _DTYPE_FLAGS.get(dtype, True)
]

ALL_DTYPES = PROMOTION_DTYPES + FP8_DTYPES


def _make_tensor(dtype, shape):
    # Element values never influence result_type, so a zero fill is used: it is
    # defined for every dtype here (fp8 and complex included) and keeps the
    # tests free of value-dependent behaviour.
    return torch.zeros(shape, dtype=dtype, device=flag_gems.device)


def _assert_result(res, ref):
    # result_type returns a Python ScalarType code rather than a tensor, so
    # there is no tensor to hand to the shared value assertions; the schema type
    # is checked and the value is compared against the native oracle.
    assert type(res) is int, f"expected a python int code, got {type(res)}"
    assert res == ref, f"candidate returned {res}, native returned {ref}"


# Every row below is a pair of tiny metadata payloads, so --quick keeps all of
# them; the only rows quick drops are the positive nan/inf payloads and the
# multi-dim rank rows, where the runtime cost actually differs.
TENSOR_PAIR_ROWS = list(itertools.product(PROMOTION_DTYPES, repeat=2))
TENSOR_PAIR_ROWS += [(dtype, dtype) for dtype in FP8_DTYPES]

FP8_MIXED_ROWS = [(fp8, other) for fp8 in FP8_DTYPES for other in PROMOTION_DTYPES]
FP8_MIXED_ROWS += [(other, fp8) for fp8 in FP8_DTYPES for other in PROMOTION_DTYPES]
FP8_MIXED_ROWS += list(itertools.permutations(FP8_DTYPES, 2))

SCALAR_KINDS = [True, 1, 1.5]
COMPLEX_SCALAR = 1 + 2j

TENSOR_SCALAR_ROWS = [(dtype, kind) for dtype in ALL_DTYPES for kind in SCALAR_KINDS]
COMPLEX_SCALAR_ROWS = [(dtype, COMPLEX_SCALAR) for dtype in PROMOTION_DTYPES]
FP8_COMPLEX_SCALAR_ROWS = [(dtype, COMPLEX_SCALAR) for dtype in FP8_DTYPES]

SCALAR_SCALAR_ROWS = list(itertools.product(SCALAR_KINDS + [COMPLEX_SCALAR], repeat=2))

_RANK_SHAPES = {0: (), 1: (4,)}

# The 0-dim tensor boundary against every Python scalar kind, for every dtype.
# These payloads are tiny, so quick keeps all of them.
ZERO_DIM_ROWS = [(dtype, kind) for dtype in ALL_DTYPES for kind in (1, True)]

# Promotion is dtype driven when both operands share a dtype: a 0-dim, 1-dim,
# 3-dim, 4-dim or 5-dim operand on either side must not change the result.
RANK_ROWS = [
    ((), (), torch.float32),
    ((1,), (1,), torch.float16),
    ((256,), (1024, 1024), torch.float32),
    ((20, 320, 15), (20, 320, 15), torch.bfloat16),
    ((16, 128, 64), (16, 128, 64), torch.float32),
    ((), (2, 3, 4), torch.float32),
    ((1,), (16, 7, 57, 32, 29), torch.float16),
]

# Quick drops only the large-rank rows of this dimension; the small ones stay.
QUICK_RANK_ROWS = [
    row for row in RANK_ROWS if row not in RANK_ROWS[2:5] + RANK_ROWS[6:]
]

# The central mixed-dtype behaviour: with two different dtypes the 0-dim operand
# has a different promotion priority, so exchanging the dtype/rank
# assignments can give different codes (probed on the active backend), e.g.
# uint8[1-D] x int32[0-D] -> uint8 while uint8[0-D] x int32[1-D] -> int32.
MIXED_RANK_ROWS = [
    (torch.uint8, 0, torch.int32, 1),
    (torch.uint8, 1, torch.int32, 0),
    (torch.int8, 0, torch.int32, 1),
    (torch.int8, 1, torch.int32, 0),
    (torch.int32, 0, torch.int64, 1),
    (torch.int32, 1, torch.int64, 0),
    (torch.float16, 0, torch.bfloat16, 1),
    (torch.float16, 1, torch.bfloat16, 0),
    # Two 0-dim operands retain their dtype promotion rules.
    (torch.float16, 0, torch.bfloat16, 0),
]

# nan-only, inf-only and nan+inf payloads for every floating dtype the operator
# accepts; the scenarios the dtype cannot represent are dropped by the shared
# generator itself (e.g. e4m3fn has no infinity).
SPECIAL_VALUE_ROWS = tu.special_value_cases(
    [torch.float16, torch.bfloat16, torch.float32, torch.float64] + FP8_DTYPES
)

# Objects that match no result_type schema: neither a tensor nor a Scalar.
INVALID_OTHER = ["abc", [1, 2], (1, 2), 2**70]


@pytest.mark.result_type
@pytest.mark.parametrize("a_dtype,b_dtype", TENSOR_PAIR_ROWS)
def test_result_type_tensor_pair(a_dtype, b_dtype):
    a = _make_tensor(a_dtype, (4,))
    b = _make_tensor(b_dtype, (4,))

    ref = torch.ops.aten.result_type(a, b)
    res = flag_gems.result_type(a, b)

    _assert_result(res, ref)


@pytest.mark.result_type
@pytest.mark.parametrize("a_dtype,b_dtype", FP8_MIXED_ROWS)
def test_result_type_fp8_mixed_pair_rejected(a_dtype, b_dtype):
    a = _make_tensor(a_dtype, (4,))
    b = _make_tensor(b_dtype, (4,))

    # An fp8 operand only promotes with a second operand of the same fp8 type.
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.result_type(a, b)


@pytest.mark.result_type
@pytest.mark.parametrize("dtype,scalar", TENSOR_SCALAR_ROWS)
def test_result_type_tensor_scalar(dtype, scalar):
    inp = _make_tensor(dtype, (4,))

    ref = torch.ops.aten.result_type(inp, scalar)
    res = flag_gems.result_type(inp, scalar)

    _assert_result(res, ref)


@pytest.mark.result_type
@pytest.mark.parametrize("dtype,scalar", TENSOR_SCALAR_ROWS)
def test_result_type_scalar_tensor(dtype, scalar):
    inp = _make_tensor(dtype, (4,))

    ref = torch.ops.aten.result_type(scalar, inp)
    res = flag_gems.result_type(scalar, inp)

    _assert_result(res, ref)


@pytest.mark.result_type
@pytest.mark.parametrize("dtype,scalar", COMPLEX_SCALAR_ROWS)
def test_result_type_tensor_scalar_complex(dtype, scalar):
    inp = _make_tensor(dtype, (4,))

    ref = torch.ops.aten.result_type(inp, scalar)
    res = flag_gems.result_type(inp, scalar)

    _assert_result(res, ref)


@pytest.mark.result_type
@pytest.mark.parametrize("dtype,scalar", COMPLEX_SCALAR_ROWS)
def test_result_type_scalar_tensor_complex(dtype, scalar):
    inp = _make_tensor(dtype, (4,))

    ref = torch.ops.aten.result_type(scalar, inp)
    res = flag_gems.result_type(scalar, inp)

    _assert_result(res, ref)


@pytest.mark.result_type
@pytest.mark.parametrize("dtype,scalar", FP8_COMPLEX_SCALAR_ROWS)
def test_result_type_fp8_complex_scalar_rejected(dtype, scalar):
    inp = _make_tensor(dtype, (4,))

    # A complex Python scalar has no promotion rule for the fp8 types.
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.result_type(inp, scalar)


@pytest.mark.result_type
@pytest.mark.parametrize("first,second", SCALAR_SCALAR_ROWS)
def test_result_type_scalar_scalar(first, second):
    ref = torch.ops.aten.result_type(first, second)
    res = flag_gems.result_type(first, second)

    _assert_result(res, ref)


@pytest.mark.result_type
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_result_type_tensor_with_none(dtype):
    inp = _make_tensor(dtype, (4,))

    # None matches the optional scalar argument and simply yields the tensor
    # dtype.
    ref = torch.ops.aten.result_type(inp, None)
    res = flag_gems.result_type(inp, None)

    _assert_result(res, ref)


@pytest.mark.result_type
@pytest.mark.parametrize(
    "shape_a,shape_b,dtype", tu.selected_cases(RANK_ROWS, quick=QUICK_RANK_ROWS)
)
def test_result_type_ignores_tensor_rank(shape_a, shape_b, dtype):
    a = _make_tensor(dtype, shape_a)
    b = _make_tensor(dtype, shape_b)

    ref = torch.ops.aten.result_type(a, b)
    res = flag_gems.result_type(a, b)

    _assert_result(res, ref)
    # Two operands of the same dtype must resolve to that dtype whatever their
    # ranks or shapes are.
    assert res == torch.ops.aten.result_type(a, a)


@pytest.mark.result_type
@pytest.mark.parametrize("a_dtype,a_rank,b_dtype,b_rank", MIXED_RANK_ROWS)
def test_result_type_mixed_dtype_rank_role(a_dtype, a_rank, b_dtype, b_rank):
    a = _make_tensor(a_dtype, _RANK_SHAPES[a_rank])
    b = _make_tensor(b_dtype, _RANK_SHAPES[b_rank])

    ref = torch.ops.aten.result_type(a, b)
    res = flag_gems.result_type(a, b)

    # Swapping dtype/rank assignments can change promotion priority.
    _assert_result(res, ref)


@pytest.mark.result_type
@pytest.mark.parametrize("dtype,scalar", ZERO_DIM_ROWS)
def test_result_type_scalar_over_0d_tensor(dtype, scalar):
    inp = _make_tensor(dtype, ())

    ref = torch.ops.aten.result_type(inp, scalar)
    res = flag_gems.result_type(inp, scalar)

    _assert_result(res, ref)


@pytest.mark.result_type
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(SPECIAL_VALUE_ROWS, quick=[])
)
def test_result_type_ignores_element_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)

    ref = torch.ops.aten.result_type(inp, inp)
    res = flag_gems.result_type(inp, inp)

    _assert_result(res, ref)


@pytest.mark.result_type
@pytest.mark.parametrize("other", INVALID_OTHER)
def test_result_type_tensor_invalid_other(other):
    inp = _make_tensor(torch.float32, (4,))

    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.result_type(inp, other)


@pytest.mark.result_type
@pytest.mark.parametrize("first,second", [("a", "b"), ([1, 2], 1), (1, [1, 2])])
def test_result_type_invalid_scalar_pair(first, second):
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.result_type(first, second)


@pytest.mark.result_type
def test_result_type_missing_argument():
    inp = _make_tensor(torch.float32, (4,))

    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.result_type(inp)
