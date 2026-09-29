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

# Static device capability flags decide which optional dtypes are declared.
GER_DTYPES = [
    torch.bool,
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.float16,
    torch.float32,
]
if utils.int64_is_supported:
    GER_DTYPES.append(torch.int64)
if utils.bf16_is_supported:
    GER_DTYPES.append(torch.bfloat16)
if utils.fp64_is_supported:
    GER_DTYPES.append(torch.float64)
GER_DTYPES.append(torch.complex64)

GER_FLOAT_DTYPES = [
    dtype
    for dtype in GER_DTYPES
    if dtype in (torch.float16, torch.float32, torch.bfloat16, torch.float64)
]
GER_LAYOUT_DTYPES = [
    dtype for dtype in GER_DTYPES if dtype in (torch.float32, torch.bfloat16)
]

# ger is the rank-1 outer product: both operands must be 1-D, so the spec shape
# grid maps to (self_len, vec2_len) operand pairs and there is no broadcast
# dimension. Zero lengths are native-valid and cover the empty-output path.
GER_SHAPE_PAIRS = [
    (1, 1),
    (1, 256),
    (256, 1),
    (256, 256),
    (512, 1024),
    (1024, 512),
    (1024, 1024),
    (0, 4),
    (4, 0),
    (0, 0),
]
# Quick keeps the spec smoke shape (2, 19, 7) as operand lengths (19, 7).
VALUE_CASES = tu.selected_cases(GER_SHAPE_PAIRS, quick=[(19, 7)])


@pytest.mark.ger
@pytest.mark.parametrize("shape_pair", VALUE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", GER_DTYPES)
def test_ger_values(shape_pair, value_range, dtype):
    self_len, vec2_len = shape_pair
    inp = tu.make_input(dtype, (self_len,), value_range)
    vec2 = tu.make_input(dtype, (vec2_len,), value_range)
    ref_inp = tu.to_reference(inp)
    ref_vec2 = tu.to_reference(vec2)

    ref_out = torch.ops.aten.ger(ref_inp, ref_vec2)
    res_out = flag_gems.ger(inp, vec2)

    assert res_out.device == inp.device
    tu.assert_result_close(res_out, ref_out)


# Candidate view construction: sliced 1-D operands with stride > 1 and a
# nonzero storage offset.
STRIDED_LAYOUTS = tu.selected_cases(
    [
        ((0, 64, 2), (0, 32, 2)),
        ((3, 35, 1), (5, 37, 3)),
        ((1, 17, 5), (2, 34, 2)),
    ],
    quick=[],
)
LAYOUT_CASES = [
    (inp_sel, vec2_sel, dtype)
    for inp_sel, vec2_sel in STRIDED_LAYOUTS
    for dtype in GER_LAYOUT_DTYPES
]


@pytest.mark.ger
@pytest.mark.parametrize("inp_sel,vec2_sel,dtype", LAYOUT_CASES)
def test_ger_strided_input(inp_sel, vec2_sel, dtype):
    base = tu.make_input(dtype, (64,), ["-1", "1"])
    inp = base[inp_sel[0] : inp_sel[1] : inp_sel[2]]
    vec2 = base[vec2_sel[0] : vec2_sel[1] : vec2_sel[2]]
    ref_inp = tu.to_reference(inp)
    ref_vec2 = tu.to_reference(vec2)

    ref_out = torch.ops.aten.ger(ref_inp, ref_vec2)
    res_out = flag_gems.ger(inp, vec2)

    tu.assert_result_close(res_out, ref_out)


# Operands of different but compatible dtypes: the result takes the promoted
# common dtype, which the shared assertion checks.
MIXED_DTYPE_CASES = tu.selected_cases(
    [
        (torch.int8, torch.int32),
        (torch.uint8, torch.int8),
        (torch.int32, torch.int64),
        (torch.int64, torch.float16),
        (torch.bool, torch.float32),
        (torch.float16, torch.float32),
        (torch.float16, torch.bfloat16),
        (torch.float32, torch.float64),
        (torch.float32, torch.complex64),
    ],
    quick=[],
)
MIXED_DTYPE_CASES = [
    pair
    for pair in MIXED_DTYPE_CASES
    if all(
        (dtype is not torch.int64 or utils.int64_is_supported)
        and (dtype is not torch.bfloat16 or utils.bf16_is_supported)
        and (dtype is not torch.float64 or utils.fp64_is_supported)
        for dtype in pair
    )
]


@pytest.mark.ger
@pytest.mark.parametrize("dtype_pair", MIXED_DTYPE_CASES)
def test_ger_mixed_dtype(dtype_pair):
    self_dtype, vec2_dtype = dtype_pair
    inp = tu.make_input(self_dtype, (32,), ["-1", "1"])
    vec2 = tu.make_input(vec2_dtype, (48,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_vec2 = tu.to_reference(vec2)

    ref_out = torch.ops.aten.ger(ref_inp, ref_vec2)
    res_out = flag_gems.ger(inp, vec2)

    tu.assert_result_close(res_out, ref_out)


# d(out)/d(self) = grad_out @ vec2 and d(out)/d(vec2) = grad_out^T @ self; both
# operand orders are covered, and a nonconstant shared upstream gradient keeps
# both reductions exercised.
BACKWARD_CASES = tu.selected_cases(
    [
        (shape_pair, dtype)
        for shape_pair in ((64, 32), (32, 64))
        for dtype in GER_FLOAT_DTYPES + [torch.complex64]
    ],
    quick=[],
)


@pytest.mark.ger
@pytest.mark.parametrize("shape_pair,dtype", BACKWARD_CASES)
def test_ger_backward(shape_pair, dtype):
    self_len, vec2_len = shape_pair
    inp = tu.make_input(dtype, (self_len,), ["-1", "1"]).requires_grad_(True)
    vec2 = tu.make_input(dtype, (vec2_len,), ["-1", "1"]).requires_grad_(True)
    upstream = tu.make_input(dtype, (self_len, vec2_len), ["-1", "1"])
    ref_inp = tu.to_reference(inp).detach().requires_grad_(True)
    ref_vec2 = tu.to_reference(vec2).detach().requires_grad_(True)
    ref_upstream = tu.to_reference(upstream)

    ref_out = torch.ops.aten.ger(ref_inp, ref_vec2)
    res_out = flag_gems.ger(inp, vec2)

    tu.assert_result_close(res_out.detach(), ref_out.detach())

    ref_dself, ref_dvec2 = torch.autograd.grad(
        ref_out, (ref_inp, ref_vec2), grad_outputs=ref_upstream
    )
    res_dself, res_dvec2 = torch.autograd.grad(
        res_out, (inp, vec2), grad_outputs=upstream
    )

    tu.assert_result_close(res_dself, ref_dself)
    tu.assert_result_close(res_dvec2, ref_dvec2)


# tu.special_value_cases covers the real float dtypes; complex64 uses the same
# scenario names through the same payload helper.
SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases(GER_FLOAT_DTYPES)
    + [(torch.complex64, scenario) for scenario in ("nan", "inf", "mixed")],
    quick=[],
)


@pytest.mark.ger
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test_ger_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    vec2 = tu.make_input(dtype, (7,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_vec2 = tu.to_reference(vec2)

    ref_out = torch.ops.aten.ger(ref_inp, ref_vec2)
    res_out = flag_gems.ger(inp, vec2)

    tu.assert_result_close(res_out, ref_out)


# ger takes no scalar parameters; the rank guard is its only input validation.
INVALID_RANK_CASES = [
    ((), (4,)),
    ((2, 3), (4,)),
    ((4,), ()),
    ((4,), (2, 3)),
    ((1, 1, 4), (4,)),
]


@pytest.mark.ger
@pytest.mark.parametrize("inp_shape,vec2_shape", INVALID_RANK_CASES)
def test_ger_invalid_rank(inp_shape, vec2_shape):
    inp = tu.make_input(torch.float32, inp_shape, ["-1", "1"])
    vec2 = tu.make_input(torch.float32, vec2_shape, ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.ger(inp, vec2)


# fp8 operands need an fp8 multiply kernel, which the measured NVIDIA backend
# does not provide ("mul_cuda" is not implemented for Float8). The rejected
# cases are therefore declared for that vendor only; no test-time skip.
IS_NVIDIA_VENDOR = getattr(flag_gems, "vendor_name", "") == "nvidia"
FP8_REJECTED_DTYPES = (
    [torch.float8_e4m3fn, torch.float8_e5m2] if IS_NVIDIA_VENDOR else []
)


@pytest.mark.ger
@pytest.mark.parametrize("dtype", FP8_REJECTED_DTYPES)
def test_ger_unsupported_dtype(dtype):
    inp = tu.make_input(dtype, (4,), ["-1", "1"])
    vec2 = tu.make_input(dtype, (4,), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.ger(inp, vec2)


# aten::ger.out is a real keyword-only overload whose return value is the
# provided buffer itself, hence the identity check.
OUT_DTYPES = [torch.float32, torch.int32]
if utils.int64_is_supported:
    OUT_DTYPES.append(torch.int64)
if utils.bf16_is_supported:
    OUT_DTYPES.append(torch.bfloat16)
OUT_CASES = tu.selected_cases(
    [
        (shape_pair, dtype)
        for shape_pair in ((64, 64), (256, 256))
        for dtype in OUT_DTYPES
    ],
    quick=[],
)


@pytest.mark.ger
@pytest.mark.parametrize("shape_pair,dtype", OUT_CASES)
def test_ger_out(shape_pair, dtype):
    self_len, vec2_len = shape_pair
    inp = tu.make_input(dtype, (self_len,), ["-1", "1"])
    vec2 = tu.make_input(dtype, (vec2_len,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_vec2 = tu.to_reference(vec2)

    ref_buf = torch.empty(
        (self_len, vec2_len), dtype=ref_inp.dtype, device=ref_inp.device
    )
    torch.ops.aten.ger.out(ref_inp, ref_vec2, out=ref_buf)

    out = torch.empty((self_len, vec2_len), dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.ger(inp, vec2, out=out)

    assert res_out is out
    assert out.device == inp.device
    tu.assert_result_close(res_out, ref_buf)
