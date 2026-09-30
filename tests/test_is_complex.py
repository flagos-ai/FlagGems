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

# aten::is_complex(Tensor self) -> bool is answered from the dtype property
# alone: True for the complex dtypes, False for every other one. The value
# ranges below therefore only vary state the query must ignore. complex32 is
# included even though allocating it emits an experimental-dtype warning;
# pytest.ini does not turn warnings into errors.
_COMPLEX_DTYPES = [
    torch.complex64,
    torch.complex128,
    torch.complex32,
]

_NON_COMPLEX_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.float64,
    torch.int16,
    torch.bool,
]

_IS_COMPLEX_DTYPES = _COMPLEX_DTYPES + _NON_COMPLEX_DTYPES


@pytest.mark.is_complex
@pytest.mark.parametrize("dtype", _IS_COMPLEX_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", tu.selected_shapes())
def test_is_complex(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.is_complex(ref_inp)
    res_out = flag_gems.is_complex(inp)

    # The native answer is a Python bool: check the type strictly and compare
    # against the native value directly instead of wrapping it in a tensor.
    assert type(res_out) is bool
    assert res_out == ref_out


def _make_layout_tensor(kind, dtype, device):
    """Build a tensor carrying the requested dtype and storage/layout state."""
    if kind == "0-dim":
        return torch.zeros((), dtype=dtype, device=device)
    if kind == "empty":
        return torch.zeros((0,), dtype=dtype, device=device)
    if kind == "transposed":
        return torch.zeros((6, 4), dtype=dtype, device=device).t()
    if kind == "offset_slice":
        return torch.zeros((4, 8), dtype=dtype, device=device)[1:]
    if kind == "expanded":
        return torch.zeros((4, 1), dtype=dtype, device=device).expand(4, 8)
    if kind == "conj_view":
        return torch.zeros((4, 8), dtype=dtype, device=device).conj()
    if kind == "neg_view":
        return torch._neg_view(torch.zeros((4, 8), dtype=dtype, device=device))
    if kind == "sparse_coo":
        return torch.zeros((4, 8), dtype=dtype, device=device).to_sparse()
    if kind == "meta":
        return torch.zeros((4, 8), dtype=dtype, device="meta")
    if kind == "quantized":
        return torch.quantize_per_tensor(
            torch.zeros(4, dtype=torch.float32), 0.1, 0, dtype
        )
    raise AssertionError(f"unknown layout kind {kind!r}")


# Every one of these is a cheap metadata case, so they run in quick mode as well
# as in the default suite. The dtype query has to hold for all of them: 0-dim and
# empty tensors, non-contiguous strides, a nonzero storage offset, stride-0
# broadcast, a set lazy conjugate or negative bit, sparse/meta storage and
# quantized state.
_LAYOUT_CASES = [
    ("0-dim", torch.complex64),
    ("0-dim", torch.float32),
    ("empty", torch.complex64),
    ("empty", torch.float32),
    ("transposed", torch.complex64),
    ("transposed", torch.float32),
    ("offset_slice", torch.complex64),
    ("offset_slice", torch.float32),
    ("expanded", torch.complex64),
    ("expanded", torch.float32),
    ("conj_view", torch.complex64),
    ("conj_view", torch.float32),
    ("neg_view", torch.complex64),
    ("neg_view", torch.float32),
    ("sparse_coo", torch.complex64),
    ("sparse_coo", torch.float32),
    ("meta", torch.complex64),
    ("meta", torch.float32),
    ("quantized", torch.quint8),
]


@pytest.mark.is_complex
@pytest.mark.parametrize("kind,dtype", _LAYOUT_CASES)
def test_is_complex_layout(kind, dtype):
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device
    ref_inp = _make_layout_tensor(kind, dtype, ref_device)
    inp = _make_layout_tensor(kind, dtype, flag_gems.device)

    ref_out = torch.ops.aten.is_complex(ref_inp)
    res_out = flag_gems.is_complex(inp)

    assert type(res_out) is bool
    assert res_out == ref_out


# nan/inf payloads are not complex either; these are floating-dtype cases only,
# so they stay out of quick mode like the other special-value suites.
_SPECIAL_VALUE_CASES = tu.selected_cases(
    tu.special_value_cases(_IS_COMPLEX_DTYPES), quick=[]
)


@pytest.mark.is_complex
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_VALUE_CASES)
def test_is_complex_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.is_complex(ref_inp)
    res_out = flag_gems.is_complex(inp)

    assert type(res_out) is bool
    assert res_out == ref_out


@pytest.mark.is_complex
def test_is_complex_accepts_none():
    # The Tensor argument is optional at runtime: native accepts None and
    # reports False, so the candidate must accept it instead of raising.
    ref_out = torch.ops.aten.is_complex(None)
    res_out = flag_gems.is_complex(None)

    assert type(res_out) is bool
    assert res_out == ref_out


# Native accepts only a Tensor here: float/int/bool/str and a list of tensors
# raise RuntimeError, so the candidate has to reject them too.
_NON_TENSOR_KINDS = ["float", "int", "bool", "str", "tensor_list"]


def _bad_non_tensor(kind):
    # Built at call time: module import, collection and --list-cases must stay
    # free of tensor allocation and ATen calls.
    if kind == "float":
        return 1.0
    if kind == "int":
        return 1
    if kind == "bool":
        return True
    if kind == "str":
        return "1.0"
    if kind == "tensor_list":
        return [torch.zeros(2, dtype=torch.float32, device=flag_gems.device)]
    raise AssertionError(f"unknown non-tensor kind {kind!r}")


@pytest.mark.is_complex
@pytest.mark.parametrize("kind", _NON_TENSOR_KINDS)
def test_is_complex_rejects_non_tensor(kind):
    bad = _bad_non_tensor(kind)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_complex(bad)


@pytest.mark.is_complex
def test_is_complex_requires_argument():
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_complex()
