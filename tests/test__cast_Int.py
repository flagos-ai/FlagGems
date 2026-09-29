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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# ``aten::_cast_Int(Tensor self, bool non_blocking=False) -> Tensor`` returns an
# int32 tensor. Every expectation below is read from the same-device native
# reference, so the file is device portable: the reference oracle, not a
# hardcoded table, decides how truncation, int64 narrowing and out-of-int32
# values behave on the backend under test.
#
# Coverage exemptions, each established by probing the native call: no ``.out``
# overload (``torch.ops.aten._cast_Int.overloads()`` is ``["default"]`` and
# ``.out`` raises AttributeError); no gradient (int32 result, ``autograd.grad``
# raises "does not require grad"); unary with no scalar operand, so the
# broadcast and tensor-vs-scalar dimensions do not apply. Sparse COO input is
# supported natively -- coalesced and uncoalesced-with-duplicates alike -- and is
# covered by ``test__cast_Int_sparse_coo`` below.

_DTYPE_CAPABILITY = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
}

# Built from static runtime capability flags only; nothing is probed at
# collection time. bool/complex64 are valid extra inputs beyond the nine
# required dtypes.
SUPPORTED_DTYPES = [
    dtype for dtype in tu.REQUIRED_DTYPES if _DTYPE_CAPABILITY.get(dtype, True)
]
SUPPORTED_DTYPES += [torch.bool, torch.complex64]
if utils.fp64_is_supported:
    SUPPORTED_DTYPES.append(torch.float64)

# float8_e4m3fn has no infinity encoding, so its inf/mixed payloads would
# degrade to nan; ``tu.special_value_cases`` keeps its nan-only case while
# float8_e5m2 (which does encode inf) keeps all three scenarios.
SPECIAL_CASES = tu.special_value_cases(
    [dtype for dtype in SUPPORTED_DTYPES if dtype.is_floating_point]
)

_AUX_FLOAT_DTYPES = [torch.float32] + (
    [torch.bfloat16] if utils.bf16_is_supported else []
)


@pytest.mark.cast_Int
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", tu.selected_shapes())
def test__cast_Int_value_range(dtype, value_range, shape):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Int(ref_inp)
    res_out = flag_gems._cast_Int(inp)

    # int32 inputs are handed back unchanged; every other dtype is cast into a
    # fresh int32 tensor on the input's device.
    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Int
@pytest.mark.parametrize("dtype,scenario", tu.selected_cases(SPECIAL_CASES, quick=[]))
def test__cast_Int_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Int(ref_inp)
    res_out = flag_gems._cast_Int(inp)

    tu.assert_result_equal(res_out, ref_out)


def _fractional_tensor(shape, dtype=torch.float32):
    """Coordinate-varying fractional values with both signs.

    Truncation toward zero keeps a nonzero result for most elements (-2.5 -> -2,
    -0.5 -> 0), so an all-zero or floor-rounding candidate cannot pass these
    workloads even though the inputs live in a symmetric range.
    """
    numel = math.prod(shape)
    flat = torch.arange(1, numel + 1, dtype=torch.float32, device=flag_gems.device)
    return (flat * 0.5 - 3.0).reshape(shape).to(dtype)


def _layout_input(layout):
    if layout == "transpose":
        return _fractional_tensor((8, 5)).t()
    if layout == "offset_slice":
        return _fractional_tensor((6, 5))[1:5]
    if layout == "expanded":
        return _fractional_tensor((4, 5))[0:1].expand(3, 5)
    if layout == "stride_holes":
        return _fractional_tensor((12, 15))[::2, ::3]
    # The imaginary part is dropped by a real cast, so the real payload is what
    # has to vary; keep it fractional for the same reason as above.
    real = _fractional_tensor((4, 5))
    return torch.conj(torch.complex(real, -real))


_LAYOUTS = tu.selected_cases(
    ["stride_holes", "transpose", "offset_slice", "expanded", "lazy_conj"],
    quick=["stride_holes"],
)


@pytest.mark.cast_Int
@pytest.mark.parametrize("layout", _LAYOUTS)
def test__cast_Int_strided_input(layout):
    inp = _layout_input(layout)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Int(ref_inp)
    res_out = flag_gems._cast_Int(inp)

    # Native keeps the caller's strides for some views (a transposed int32 view
    # returns transposed, a float32 offset slice keeps its strides) and compacts
    # others (float32 stride-holes (24, 3) -> (4, 1)), so a candidate is only
    # equivalent if it reproduces the reference strides exactly.
    assert res_out.stride() == ref_out.stride()
    tu.assert_result_equal(res_out, ref_out)


_EMPTY_SHAPES = [(0,), (2, 0, 3), (3, 0, 2)]


@pytest.mark.cast_Int
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("shape", tu.selected_cases(_EMPTY_SHAPES, quick=[(0,)]))
def test__cast_Int_zero_extent(dtype, shape):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Int(ref_inp)
    res_out = flag_gems._cast_Int(inp)

    # Empty int32 is also an identity (same object); empty float dtypes are cast
    # into a fresh empty int32 tensor. Mirror whichever the reference does.
    assert (res_out is inp) == (ref_out is ref_inp)
    tu.assert_result_equal(res_out, ref_out)


_INT64_NARROWING = (
    2**31,
    2**32 - 1,
    -(2**31) - 1,
    2**31 - 1,
    -(2**31),
    2**40,
    -(2**40),
    0,
)
_FP_FRACTIONAL = (1.75, -1.75, 2.5, -2.5, 0.5, -0.5, 0.0, -0.0)
_FP_OUT_OF_RANGE = (
    2.0**31 - 1024.0,
    -(2.0**31) + 1024.0,
    2.0**40,
    -(2.0**40),
    1e30,
    -1e30,
)
_FP64_OUT_OF_RANGE = (
    2.0**31 + 1024.0,
    -(2.0**31) - 1024.0,
    2.0**40 + 0.5,
    -(2.0**40) - 0.5,
    1e300,
    -1e300,
)
_FP_NONFINITE = (float("inf"), float("-inf"), float("nan"))

_BOUNDARY_CASES = [("float32_fractional", torch.float32, _FP_FRACTIONAL)]
if utils.bf16_is_supported:
    _BOUNDARY_CASES.append(("bfloat16_fractional", torch.bfloat16, _FP_FRACTIONAL))
if utils.fp64_is_supported:
    _BOUNDARY_CASES.append(("float64_fractional", torch.float64, _FP_FRACTIONAL))
    _BOUNDARY_CASES.append(("float64_out_of_range", torch.float64, _FP64_OUT_OF_RANGE))
if utils.int64_is_supported:
    _BOUNDARY_CASES.append(("int64_narrowing", torch.int64, _INT64_NARROWING))

# Value-range boundaries stay distinct from the random grid: the extreme values
# here sit next to the int32 cut-over, where truncation, saturation and
# narrowing have different measured outcomes.
_POSITIVE_BOUNDARY_CASES = [
    ("float32_out_of_range", torch.float32, _FP_OUT_OF_RANGE),
    ("float32_nonfinite", torch.float32, _FP_NONFINITE),
    ("float16_fractional", torch.float16, _FP_FRACTIONAL),
    ("bool_exact", torch.bool, (True, False, True, False)),
    ("int8_exact", torch.int8, (-128, -1, 0, 1, 127)),
    ("uint8_exact", torch.uint8, (0, 1, 127, 255)),
    ("int16_exact", torch.int16, (-32768, -1, 0, 32767)),
]


@pytest.mark.cast_Int
@pytest.mark.parametrize(
    "case_name,dtype,values",
    tu.selected_cases(
        _BOUNDARY_CASES + _POSITIVE_BOUNDARY_CASES,
        quick=[("float32_fractional", torch.float32, _FP_FRACTIONAL)],
    ),
)
def test__cast_Int_boundary_values(case_name, dtype, values):
    del case_name
    inp = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Int(ref_inp)
    res_out = flag_gems._cast_Int(inp)

    tu.assert_result_equal(res_out, ref_out)


_IDENTITY_KINDS = ["contiguous", "transposed", "offset_row", "scalar", "empty"]


def _identity_input(kind):
    base = torch.arange(24, dtype=torch.int32, device=flag_gems.device).reshape(4, 6)
    if kind == "contiguous":
        return base
    if kind == "transposed":
        return base.t()
    if kind == "offset_row":
        return base[1:3]
    if kind == "scalar":
        return torch.tensor(7, dtype=torch.int32, device=flag_gems.device)
    return torch.empty((0,), dtype=torch.int32, device=flag_gems.device)


@pytest.mark.cast_Int
@pytest.mark.parametrize("kind", tu.selected_cases(_IDENTITY_KINDS, quick=[]))
def test__cast_Int_int32_identity(kind):
    inp = _identity_input(kind)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Int(ref_inp)
    res_out = flag_gems._cast_Int(inp)

    # Native returns the very same int32 object, views included, so a candidate
    # that hands back a freshly materialized copy (even one with the same
    # data_ptr, e.g. a distinct view) is not equivalent.
    assert (res_out is inp) == (ref_out is ref_inp)
    assert res_out.stride() == ref_out.stride()
    tu.assert_result_equal(res_out, ref_out)


# Sparse COO coverage. The native call keeps the COO layout, the shape and the
# caller's stored entries (including duplicate coordinates in an uncoalesced
# input, which is returned uncoalesced), so no workload here densifies,
# coalesces or deduplicates the input before calling.
#
# COO index storage is INT64 regardless of the index dtype handed in --
# ``torch.sparse_coo_tensor`` accepts, for instance, an int32 index tensor and
# normalizes the stored indices to int64 -- so these rows are collected only
# when the runtime's static INT64 capability flag is set. That is an
# index-storage requirement, not an operator-output one: nothing above this
# point is affected, and the int32 *output* coverage stays intact either way.
_COO_FLOAT_VALUES = [1.5, -2.7, 0.75, 3.5]
_COO_INT_VALUES = [5, -6, 7, -8]
# Coalesced rows: four distinct coordinates, a native-valid coalesced input.
_COO_DISTINCT_2D_INDICES = [[0, 1, 2, 0], [0, 1, 2, 2]]
# Uncoalesced rows: the last two columns share the coordinate (2, 2), so the
# entries 0.75 and 3.5 really are duplicates. The fractional float payload makes
# a numerical difference visible -- truncating the duplicates separately gives
# 0 + 3 = 3, while coalescing first gives trunc(4.25) = 4 -- whereas the int32
# duplicates 7 and -8 sum to -1 either way, so those rows instead catch the
# changed stored representation (4 stored entries vs 3, still uncoalesced) and
# the identity relation through the checks below.
_COO_DUPLICATE_2D_INDICES = [[0, 1, 2, 2], [0, 1, 2, 2]]
_COO_3D_INDICES = [[0, 1, 1], [0, 2, 2], [1, 3, 0]]
_COO_3D_VALUES = [1.5, -2.5, 0.5]

# (name, dtype, shape, indices, values, coalesce, empty)
_COO_CASES = [
    (
        "float32_coalesced",
        torch.float32,
        (3, 3),
        _COO_DISTINCT_2D_INDICES,
        _COO_FLOAT_VALUES,
        True,
        False,
    ),
    (
        "float32_uncoalesced_duplicates",
        torch.float32,
        (3, 3),
        _COO_DUPLICATE_2D_INDICES,
        _COO_FLOAT_VALUES,
        False,
        False,
    ),
    (
        "int32_coalesced",
        torch.int32,
        (3, 3),
        _COO_DISTINCT_2D_INDICES,
        _COO_INT_VALUES,
        True,
        False,
    ),
    (
        "int32_uncoalesced_duplicates",
        torch.int32,
        (3, 3),
        _COO_DUPLICATE_2D_INDICES,
        _COO_INT_VALUES,
        False,
        False,
    ),
    ("float32_empty", torch.float32, (4, 5), None, None, True, True),
    ("int32_empty", torch.int32, (4, 5), None, None, True, True),
    (
        "float32_3d_coalesced",
        torch.float32,
        (2, 3, 4),
        _COO_3D_INDICES,
        _COO_3D_VALUES,
        True,
        False,
    ),
]
if not utils.int64_is_supported:
    _COO_CASES = []


def _coo_input(dtype, shape, indices, values, coalesce, empty):
    if empty:
        return torch.sparse_coo_tensor(
            torch.empty((len(shape), 0), dtype=torch.int64, device=flag_gems.device),
            torch.empty((0,), dtype=dtype, device=flag_gems.device),
            shape,
        )
    coo = torch.sparse_coo_tensor(
        torch.tensor(indices, dtype=torch.int64, device=flag_gems.device),
        torch.tensor(values, dtype=dtype, device=flag_gems.device),
        shape,
    )
    # Duplicate-coordinate rows keep the caller's structure; only the distinct
    # rows, which were built distinct, are handed over coalesced.
    return coo.coalesce() if coalesce else coo


@pytest.mark.cast_Int
@pytest.mark.parametrize(
    "case_name,dtype,shape,indices,values,coalesce,empty",
    tu.selected_cases(_COO_CASES, quick=[]),
)
def test__cast_Int_sparse_coo(
    case_name, dtype, shape, indices, values, coalesce, empty
):
    del case_name
    inp = _coo_input(dtype, shape, indices, values, coalesce, empty)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Int(ref_inp)
    res_out = flag_gems._cast_Int(inp)

    # Layout, shape and dtype parity come from the shared full-result
    # comparison below; what native adds on top of that, and what the shared
    # helper cannot see, is the coalesced flag and the same-object identity.
    assert res_out.device == inp.device
    assert res_out.is_coalesced() == ref_out.is_coalesced()
    assert (res_out is inp) == (ref_out is ref_inp)
    # Compare the raw stored entries so preserved duplicates are proven
    # directly: a result that had been coalesced or deduplicated has a
    # different nnz and different raw indices/values. ``_indices()``/
    # ``_values()`` work on uncoalesced COO, unlike ``indices()``/``values()``.
    tu.assert_result_equal(res_out._indices(), ref_out._indices())
    tu.assert_result_equal(res_out._values(), ref_out._values())
    tu.assert_result_equal(res_out, ref_out)


# ``non_blocking`` is the only schema parameter. Both explicit bool values are
# covered in positional and keyword form; the omitted argument (schema default)
# is already exercised by the main value-range grid, which calls without it.
# These positive extras are default-only per the parameter-coverage rule.
@pytest.mark.cast_Int
@pytest.mark.parametrize("dtype", _AUX_FLOAT_DTYPES + [torch.int32])
@pytest.mark.parametrize("non_blocking", tu.selected_cases([True, False], quick=[]))
@pytest.mark.parametrize("as_keyword", tu.selected_cases([False, True], quick=[]))
def test__cast_Int_non_blocking_value(dtype, non_blocking, as_keyword):
    # Fractional, nonzero and coordinate-varying: the symmetric [-1,1) range
    # would truncate to all zeros and could not tell a real cast from a stub.
    inp = _fractional_tensor((20, 320, 15), dtype)
    ref_inp = tu.to_reference(inp)

    if as_keyword:
        ref_out = torch.ops.aten._cast_Int(ref_inp, non_blocking=non_blocking)
        res_out = flag_gems._cast_Int(inp, non_blocking=non_blocking)
    else:
        ref_out = torch.ops.aten._cast_Int(ref_inp, non_blocking)
        res_out = flag_gems._cast_Int(inp, non_blocking)

    tu.assert_result_equal(res_out, ref_out)


# Negative cases for the candidate only: a wrong-typed ``non_blocking``, a
# wrong-typed ``self`` and a missing ``self``. The native operator validates its
# schema and raises RuntimeError ("Expected a value of type 'bool' for argument
# 'non_blocking'", "Expected a value of type 'Tensor' for argument 'self'",
# "missing value for argument 'self'"); an argument-binding Python wrapper
# raises TypeError for the same calls. AttributeError is deliberately not
# accepted so a missing candidate implementation cannot pass these tests.
@pytest.mark.cast_Int
@pytest.mark.parametrize("non_blocking", ["yes", (1,)])
def test__cast_Int_rejects_invalid_non_blocking(non_blocking):
    inp = _fractional_tensor((16,))

    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Int(inp, non_blocking)


@pytest.mark.cast_Int
def test__cast_Int_rejects_non_tensor_self():
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Int(3.14)


@pytest.mark.cast_Int
def test__cast_Int_rejects_missing_self():
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Int()
