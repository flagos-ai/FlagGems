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

# aten::promote_types(ScalarType type1, ScalarType type2) -> ScalarType is a
# host-side dtype query: no tensor operands and no optional parameters, so the
# value-range / shape / broadcast / backward / nan-inf dimensions of the
# regular-operator spec have nothing to act on. The workload dimension is the
# ordered dtype pair (one pair per case). Native ATen is the oracle for both the
# returned ScalarType code and the set of rejected pairs; the family split below
# only selects cases and never computes an expected value.
#
# The lattice and every rejection row are collected in default and --quick mode
# alike: a case here carries no tensor payload, so trimming quick could only drop
# covered dtype pairs.


def _available_dtype(name):
    dtype = getattr(torch, name, None)
    return dtype if isinstance(dtype, torch.dtype) else None


_CORE_DTYPES = [
    dtype
    for dtype in (
        _available_dtype(name)
        for name in (
            "bool",
            "int8",
            "uint8",
            "int16",
            "int32",
            "int64",
            "float16",
            "bfloat16",
            "float32",
            "float64",
            "complex32",
            "complex64",
            "complex128",
        )
    )
    if dtype is not None
]

# Float8 types promote with themselves only; every mixed pair raises
# "Promotion for Float8 Types is not supported".
_FP8_DTYPES = [
    dtype
    for dtype in (
        _available_dtype(name)
        for name in (
            "float8_e4m3fn",
            "float8_e5m2",
            "float8_e4m3fnuz",
            "float8_e5m2fnuz",
        )
    )
    if dtype is not None
]

# uint16/uint32/uint64 promote with themselves and with floating types only; a
# pair with an integer, bool, complex or float8 type raises "Promotion for
# uint16, uint32, uint64 types is not supported".
_WIDE_UINT_DTYPES = [
    dtype
    for dtype in (_available_dtype(name) for name in ("uint16", "uint32", "uint64"))
    if dtype is not None
]

_FLOATING_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
_ALL_DTYPES = _CORE_DTYPES + _FP8_DTYPES + _WIDE_UINT_DTYPES


def _dtype_name(dtype):
    return str(dtype).removeprefix("torch.")


def _pair_case(type1, type2):
    return pytest.param(type1, type2, id=f"{_dtype_name(type1)}-{_dtype_name(type2)}")


def _is_promotable(type1, type2):
    if type1 in _CORE_DTYPES and type2 in _CORE_DTYPES:
        return True
    if type1 is type2:
        return True
    if type1 in _WIDE_UINT_DTYPES:
        return type2 in _FLOATING_DTYPES
    if type2 in _WIDE_UINT_DTYPES:
        return type1 in _FLOATING_DTYPES
    return False


_PROMOTE_PAIRS = [
    _pair_case(type1, type2)
    for type1 in _ALL_DTYPES
    for type2 in _ALL_DTYPES
    if _is_promotable(type1, type2)
]
_REJECTED_PAIRS = [
    _pair_case(type1, type2)
    for type1 in _ALL_DTYPES
    for type2 in _ALL_DTYPES
    if not _is_promotable(type1, type2)
]

# The native ScalarType argument is an int-valued enum, and the binding also
# accepts a raw int code or a zero-dim integer tensor holding one. Codes used
# here: uint8=0, int8=1, int32=3, int64=4, float16=5, float32=6, bool=11,
# bfloat16=15, uint16=27. Codes outside that enum are never synthesized.
_INT_CODE_PAIRS = [
    pytest.param(0, 1, id="uint8-int8"),
    pytest.param(3, 4, id="int32-int64"),
    pytest.param(5, 6, id="float16-float32"),
    pytest.param(6, 6, id="float32-float32"),
    pytest.param(11, 6, id="bool-float32"),
    pytest.param(15, 6, id="bfloat16-float32"),
    pytest.param(27, 6, id="uint16-float32"),
]

# A string, None, float, list or tuple is not an int-valued ScalarType.
_INVALID_ARG_CASES = [
    pytest.param("float32", id="str"),
    pytest.param(None, id="none"),
    pytest.param(3.14, id="float"),
    pytest.param([torch.float32], id="list"),
    pytest.param((torch.float32,), id="tuple"),
]


@pytest.mark.promote_types
@pytest.mark.parametrize("type1,type2", _PROMOTE_PAIRS)
def test_promote_types(type1, type2):
    ref_out = torch.ops.aten.promote_types(type1, type2)
    res_out = flag_gems.promote_types(type1, type2)

    # The contract is a Python int ScalarType code: compare type, then value.
    assert type(res_out) is int, f"got {type(res_out).__name__}"
    assert res_out == ref_out


@pytest.mark.promote_types
@pytest.mark.parametrize("type1,type2", _INT_CODE_PAIRS)
def test_promote_types_int_code_args(type1, type2):
    ref_out = torch.ops.aten.promote_types(type1, type2)
    res_out = flag_gems.promote_types(type1, type2)

    assert type(res_out) is int, f"got {type(res_out).__name__}"
    assert res_out == ref_out


@pytest.mark.promote_types
@pytest.mark.parametrize("type1,type2", _INT_CODE_PAIRS)
def test_promote_types_scalar_tensor_args(type1, type2):
    code1 = torch.tensor(type1, device=flag_gems.device)
    code2 = torch.tensor(type2, device=flag_gems.device)

    ref_out = torch.ops.aten.promote_types(code1, code2)
    res_out = flag_gems.promote_types(code1, code2)

    assert type(res_out) is int, f"got {type(res_out).__name__}"
    assert res_out == ref_out


@pytest.mark.promote_types
@pytest.mark.parametrize("type1,type2", _REJECTED_PAIRS)
def test_promote_types_rejects_unpromotable_pair(type1, type2):
    # The native binding raises RuntimeError; a host-side implementation may
    # signal the same invalid pair with any standard Python error type.
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.promote_types(type1, type2)


@pytest.mark.promote_types
@pytest.mark.parametrize("bad_arg", _INVALID_ARG_CASES)
def test_promote_types_rejects_non_scalartype_type1(bad_arg):
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.promote_types(bad_arg, torch.float32)


@pytest.mark.promote_types
@pytest.mark.parametrize("bad_arg", _INVALID_ARG_CASES)
def test_promote_types_rejects_non_scalartype_type2(bad_arg):
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.promote_types(torch.float32, bad_arg)
