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

"""Correctness tests for ``aten::is_signed``.

``aten::is_signed(Tensor self) -> bool`` is a dtype predicate: it reads no
stored element and returns a Python bool, so dtype is the axis that carries the
semantics and the value-range / broadcast / backward / scalar-operand grids do
not apply.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Probed against torch.ops.aten.is_signed on the active backend: True for the
# signed integer kinds and for every float/complex kind, False for bool and the
# unsigned integer kinds.
_SIGNED_DTYPES = [
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.complex32,
    torch.complex64,
    torch.complex128,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]

_UNSIGNED_DTYPES = [
    torch.bool,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
]

# Backend capability flags are the shared accuracy_utils booleans; a dtype with
# no entry is a baseline type the backend always provides.
_DTYPE_CAPABILITY = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.float64: utils.fp64_is_supported,
    torch.complex128: utils.fp64_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
}

SUPPORTED_DTYPES = [
    dtype
    for dtype in _SIGNED_DTYPES + _UNSIGNED_DTYPES
    if _DTYPE_CAPABILITY.get(dtype, True)
]

_SUPPORTED_FLOAT_DTYPES = [
    dtype for dtype in SUPPORTED_DTYPES if dtype.is_floating_point
]


# One value range: no stored element is read, so the five spec ranges would
# repeat the same (dtype, shape) workload. Content independence is proven by the
# NaN/Inf workload below instead.
@pytest.mark.is_signed
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_is_signed_dtype_and_shape(dtype, shape):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.is_signed(ref_inp)
    res_out = flag_gems.is_signed(inp)

    assert isinstance(res_out, bool), type(res_out)
    assert res_out == ref_out


# Input forms whose strides, storage offset, layout, aliasing or device must not
# change the dtype-derived answer. No broadcast, backward or tensor-vs-scalar
# workload exists: the signature is unary, the result is a Python bool with no
# autograd graph, and a non-tensor argument is rejected (negative cases below).
LAYOUT_CASES = [
    (
        "transposed_int32",
        lambda: torch.ones((4, 6), dtype=torch.int32, device=flag_gems.device).t(),
    ),
    (
        "storage_offset_int8",
        lambda: torch.ones((4, 6), dtype=torch.int8, device=flag_gems.device)[:, 1:],
    ),
    (
        "expanded_stride0_uint8",
        lambda: torch.ones((1, 6), dtype=torch.uint8, device=flag_gems.device).expand(
            4, 6
        ),
    ),
    (
        "channels_last_int32",
        lambda: torch.ones((2, 3, 4, 4), dtype=torch.int32, device=flag_gems.device).to(
            memory_format=torch.channels_last
        ),
    ),
    (
        "conj_view_complex64",
        lambda: torch.ones(6, dtype=torch.complex64, device=flag_gems.device).conj(),
    ),
    (
        "neg_view_int16",
        lambda: torch._neg_view(
            torch.ones((4, 6), dtype=torch.int16, device=flag_gems.device)
        ),
    ),
    (
        "empty_dim_int32",
        lambda: torch.empty(0, dtype=torch.int32, device=flag_gems.device),
    ),
    (
        "meta_uint8",
        lambda: torch.empty((2, 3), dtype=torch.uint8, device="meta"),
    ),
    (
        "sparse_coo_int32",
        lambda: torch.ones(
            (4, 6), dtype=torch.int32, device=flag_gems.device
        ).to_sparse(),
    ),
    (
        "sparse_coo_uint8",
        lambda: torch.ones(
            (4, 6), dtype=torch.uint8, device=flag_gems.device
        ).to_sparse(),
    ),
]


@pytest.mark.is_signed
@pytest.mark.parametrize(
    "make_case",
    [case for _, case in LAYOUT_CASES],
    ids=[name for name, _ in LAYOUT_CASES],
)
def test_is_signed_input_layout(make_case):
    inp = make_case()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.is_signed(ref_inp)
    res_out = flag_gems.is_signed(inp)

    assert isinstance(res_out, bool), type(res_out)
    assert res_out == ref_out


# Default-only: the reference ignores element contents, so NaN/Inf payloads must
# leave the bool unchanged.
SPECIAL_VALUE_CASES = tu.selected_cases(
    tu.special_value_cases(_SUPPORTED_FLOAT_DTYPES), quick=[]
)


@pytest.mark.is_signed
@pytest.mark.parametrize(
    "dtype,scenario",
    SPECIAL_VALUE_CASES,
    ids=[
        f"{str(dtype).removeprefix('torch.')}-{scenario}"
        for dtype, scenario in SPECIAL_VALUE_CASES
    ],
)
def test_is_signed_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.is_signed(ref_inp)
    res_out = flag_gems.is_signed(inp)

    assert isinstance(res_out, bool), type(res_out)
    assert res_out == ref_out


# The schema is aten::is_signed(Tensor self); a non-tensor argument is invalid.
NON_TENSOR_CASES = [
    ("int_argument", 1),
    ("float_argument", 1.0),
    ("list_argument", [1, 2]),
    ("string_argument", "x"),
]


@pytest.mark.is_signed
@pytest.mark.parametrize(
    "value",
    [value for _, value in NON_TENSOR_CASES],
    ids=[name for name, _ in NON_TENSOR_CASES],
)
def test_is_signed_rejects_non_tensor(value):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_signed(value)


@pytest.mark.is_signed
def test_is_signed_rejects_extra_argument():
    # The schema takes exactly one argument; the native operator raises
    # "expected at most 1 argument(s) but received 2".
    inp = torch.ones(4, dtype=torch.int32, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_signed(inp, inp)


# at::isSignedType has no quantized branch: the native operator raises
# "isSignedType not supported for quantized types". The fixture is the native
# CPU quantized type, so no device quantize kernel is assumed and the candidate
# receives exactly the argument type the reference rejects. quint4x2 is absent
# because torch.quantize_per_tensor cannot construct it on this build.
@pytest.mark.is_signed
@pytest.mark.parametrize(
    "qscheme_dtype", [torch.quint8, torch.qint8, torch.qint32], ids=str
)
def test_is_signed_rejects_quantized(qscheme_dtype):
    inp = torch.quantize_per_tensor(torch.ones(4, device="cpu"), 0.1, 0, qscheme_dtype)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_signed(inp)
