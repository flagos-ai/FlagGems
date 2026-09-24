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

"""Correctness tests for aten::special_scaled_modified_bessel_k0.

K0e(x) = exp(x) * K0(x) for x > 0. The operator is unary, so there is no
broadcast dimension. Native behaviour measured on the active CUDA-family
backend: integral and bool inputs are accepted and promote the result to
float32, float32 and float64 use a native kernel, and half/bfloat16/float8/
complex raise RuntimeError mentioning that the scaled_modified_bessel_k0_cuda
kernel is not implemented for them. At x = 0 the operator has a pole (the
result is +inf, including for the smallest subnormal) and negative inputs
return NaN. The operator registers no autograd formula -- an input created with
requires_grad=True still yields a result with requires_grad False and no
grad_fn -- so the spec's backward dimension is exempt here.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Integral/bool inputs are promoted to float32; float32 and float64 run on the
# native kernel. The remaining dtypes are covered by the negative tests below.
SUPPORTED_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.bool,
    torch.float32,
]
if utils.int64_is_supported:
    SUPPORTED_DTYPES.append(torch.int64)
if utils.fp64_is_supported:
    SUPPORTED_DTYPES.append(torch.float64)

# Floating dtypes used by the special-value and boundary families.
_FLOAT_DTYPES = [torch.float32]
if utils.fp64_is_supported:
    _FLOAT_DTYPES.append(torch.float64)

# Valid output buffers of the out= overload: the float32 result, the promoted
# int32/bool result, and a float64 sink for the float32 kernel.
_OUT_DTYPE_PAIRS = [
    (torch.float32, torch.float32),
    (torch.int32, torch.float32),
    (torch.bool, torch.float32),
]
if utils.fp64_is_supported:
    _OUT_DTYPE_PAIRS.append((torch.float32, torch.float64))

# The out= overload keeps the complete default grid but is default-only, so
# quick mode stays the smoke subset (main grid + negatives).
OUT_CASES = tu.selected_cases(
    [
        (shape, value_range, dtype, out_dtype)
        for shape in tu.REQUIRED_SHAPES
        for value_range in tu.REQUIRED_RANGES
        for dtype, out_dtype in _OUT_DTYPE_PAIRS
    ],
    quick=[],
)

# Non-contiguous and offset layouts are default-only as well. transpose2d keeps
# the original two-dimensional flipped view; transpose3d transposes the full
# three-dimensional tensor so no extent is dropped.
STRIDED_CASES = tu.selected_cases(
    [
        ("transpose2d", (20, 320)),
        ("transpose3d", (20, 320, 15)),
        ("offset", (256,)),
        ("strided_column", (1024, 1024)),
    ],
    quick=[],
)

SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_FLOAT_DTYPES), quick=[])

# Pole, dtype endpoints and the immediate neighbours of 2.0, the threshold of
# the classic rational approximations of the scaled K0.
EDGE_KINDS = tu.selected_cases(
    [
        "zero",
        "neg_zero",
        "smallest_subnormal",
        "tiny_normal",
        "one",
        "two_prev",
        "two",
        "two_next",
        "max",
        "neg_one",
        "neg_max",
    ],
    quick=[],
)

EMPTY_SHAPES = tu.selected_cases([(0,), (0, 3), (5, 0, 2)], quick=[])

# Dtypes the native kernel rejects on the measured CUDA-family backend. The
# limitation belongs to that backend's kernel, so the expectation is scoped to
# it, and every construction is gated on the shared static capability flags.
REJECTED_DTYPES = []
if flag_gems.vendor_name == "nvidia":
    REJECTED_DTYPES.append(torch.float16)
    if utils.bf16_is_supported:
        REJECTED_DTYPES.append(torch.bfloat16)
    if utils.fp8_is_supported:
        REJECTED_DTYPES.extend([torch.float8_e4m3fn, torch.float8_e5m2])
    REJECTED_DTYPES.append(torch.complex64)
    if utils.fp64_is_supported:
        REJECTED_DTYPES.append(torch.complex128)


def _strided_input(dtype, shape, kind):
    """Build a non-contiguous view over a larger parent tensor.

    transpose2d and transpose3d expose flipped strides, offset a nonzero
    storage offset, and strided_column a strided innermost dimension, so a
    candidate that assumes a packed input reads the wrong elements. Parent
    values are in [0, 1); zero is a pole of this operator, so integral inputs
    do place +inf in the compared result, which the shared comparison handles.
    """
    if kind == "transpose2d":
        return tu.make_input(dtype, shape, ["0", "1"]).t()
    if kind == "transpose3d":
        return tu.make_input(dtype, shape, ["0", "1"]).transpose(0, 1)
    if kind == "offset":
        parent = tu.make_input(dtype, (shape[0] + 3,), ["0", "1"])
        return parent[2 : 2 + shape[0]]
    parent = tu.make_input(dtype, (shape[0], shape[1] * 2), ["0", "1"])
    return parent[:, ::2]


def _edge_input(dtype, kind):
    """One-element tensor at the pole, a dtype endpoint or a 2.0 neighbour.

    Only the requested value is built, directly on the test device in the
    target dtype: no float64 intermediate (so this family also runs where
    float64 is unavailable) and no host float round trip for nextafter.
    """
    device = flag_gems.device
    finfo = torch.finfo(dtype)
    one = torch.tensor(1.0, dtype=dtype, device=device)
    two = torch.tensor(2.0, dtype=dtype, device=device)
    if kind == "zero":
        value = torch.zeros((), dtype=dtype, device=device)
    elif kind == "neg_zero":
        value = torch.tensor(-0.0, dtype=dtype, device=device)
    elif kind == "smallest_subnormal":
        value = torch.nextafter(torch.zeros((), dtype=dtype, device=device), one)
    elif kind == "tiny_normal":
        value = torch.tensor(finfo.tiny, dtype=dtype, device=device)
    elif kind == "one":
        value = one
    elif kind == "two_prev":
        value = torch.nextafter(two, one)
    elif kind == "two":
        value = two
    elif kind == "two_next":
        value = torch.nextafter(two, torch.tensor(4.0, dtype=dtype, device=device))
    elif kind == "max":
        value = torch.tensor(finfo.max, dtype=dtype, device=device)
    elif kind == "neg_one":
        value = -one
    else:
        value = torch.tensor(-finfo.max, dtype=dtype, device=device)
    return value.reshape(1)


@pytest.mark.special_scaled_modified_bessel_k0
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_special_scaled_modified_bessel_k0(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.special_scaled_modified_bessel_k0(ref_inp)
    res_out = flag_gems.special_scaled_modified_bessel_k0(inp)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.special_scaled_modified_bessel_k0
@pytest.mark.parametrize("shape,value_range,dtype,out_dtype", OUT_CASES)
def test_special_scaled_modified_bessel_k0_out(shape, value_range, dtype, out_dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_buf = torch.empty(shape, dtype=out_dtype, device=ref_inp.device)
    ref_out = torch.ops.aten.special_scaled_modified_bessel_k0.out(ref_inp, out=ref_buf)

    buf = torch.empty(shape, dtype=out_dtype, device=flag_gems.device)
    res_out = flag_gems.special_scaled_modified_bessel_k0(inp, out=buf)

    # The out= overload must write into, and return, the caller's buffer;
    # comparing pointers would also pass for a distinct alias of one storage.
    assert res_out is buf
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.special_scaled_modified_bessel_k0
@pytest.mark.parametrize("kind,shape", STRIDED_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_special_scaled_modified_bessel_k0_strided(kind, shape, dtype):
    inp = _strided_input(dtype, shape, kind)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.special_scaled_modified_bessel_k0(ref_inp)
    res_out = flag_gems.special_scaled_modified_bessel_k0(inp)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.special_scaled_modified_bessel_k0
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test_special_scaled_modified_bessel_k0_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.special_scaled_modified_bessel_k0(ref_inp)
    res_out = flag_gems.special_scaled_modified_bessel_k0(inp)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.special_scaled_modified_bessel_k0
@pytest.mark.parametrize("kind", EDGE_KINDS)
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_special_scaled_modified_bessel_k0_edge_values(kind, dtype):
    inp = _edge_input(dtype, kind)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.special_scaled_modified_bessel_k0(ref_inp)
    res_out = flag_gems.special_scaled_modified_bessel_k0(inp)

    # atol=0: the +max endpoint is a small nonzero limit value (6.79e-20 in
    # float32, 9.35e-155 in float64), so the default absolute tolerance would
    # let a candidate that flushes the tail to zero pass. rtol stays the shared
    # per-dtype resolution.
    tu.assert_result_close(res_out, ref_out, atol=0)


@pytest.mark.special_scaled_modified_bessel_k0
@pytest.mark.parametrize("shape", EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_special_scaled_modified_bessel_k0_empty(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.special_scaled_modified_bessel_k0(ref_inp)
    res_out = flag_gems.special_scaled_modified_bessel_k0(inp)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.special_scaled_modified_bessel_k0
@pytest.mark.parametrize("dtype", REJECTED_DTYPES)
def test_special_scaled_modified_bessel_k0_rejects_unsupported_dtype(dtype):
    inp = tu.make_input(dtype, (16,), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.special_scaled_modified_bessel_k0(inp)


@pytest.mark.special_scaled_modified_bessel_k0
@pytest.mark.parametrize("out_dtype", [torch.int32, torch.bool])
def test_special_scaled_modified_bessel_k0_rejects_non_float_out_buffer(out_dtype):
    # Native: RuntimeError, the float result cannot be cast to the Int/Bool
    # output type.
    inp = tu.make_input(torch.float32, (16,), ["-1", "1"])
    buf = torch.empty(16, dtype=out_dtype, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.special_scaled_modified_bessel_k0(inp, out=buf)


@pytest.mark.special_scaled_modified_bessel_k0
def test_special_scaled_modified_bessel_k0_rejects_positional_out():
    # out is keyword-only in the native schema.
    inp = tu.make_input(torch.float32, (16,), ["-1", "1"])
    buf = torch.empty(16, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.special_scaled_modified_bessel_k0(inp, buf)


@pytest.mark.special_scaled_modified_bessel_k0
def test_special_scaled_modified_bessel_k0_rejects_non_tensor_input():
    # Native: RuntimeError naming the expected Tensor argument for x.
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.special_scaled_modified_bessel_k0([1.0, 2.0])
