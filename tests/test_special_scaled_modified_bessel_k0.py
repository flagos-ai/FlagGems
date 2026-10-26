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
from .conftest import TO_CPU

# ---------------------------------------------------------------------------
# Shared value-range / shape helpers (mirrors the operator-test spec
# conventions; QUICK_MODE selects the reduced sets via pytest --quick).
# ---------------------------------------------------------------------------
QUICK_MODE = utils.QUICK_MODE


def selected_cases(cases, *, quick=()):
    """Select parameter values; an empty quick subset contributes no cases."""
    return list(quick if QUICK_MODE else cases)


# Full set: ranks 0 through 5. The operator is a two-branch rational
# approximation, so the extents keep both branches and the launch-bound regime
# covered without allocating more than 64 MiB per case.
REQUIRED_SHAPES = [
    (),  # 0-dim scalar
    (1,),  # single-element 1-dim
    (256,),  # regular 1-dim
    (1024, 1024),  # regular 2-dim
    (20, 320, 15),  # regular 3-dim
    (16, 128, 64, 60),  # 4-dim
    (16, 7, 57, 32, 29),  # 5-dim
]


def selected_shapes():
    return [(2, 19, 7)] if QUICK_MODE else REQUIRED_SHAPES


def selected_ranges():
    return [["-1", "1"]] if QUICK_MODE else REQUIRED_RANGES


REQUIRED_RANGES = [
    ["-1", "1"],
    ["0", "1"],
    ["-1", "0"],
    ["0", "max"],
    ["min", "0"],
]


def dtype_bounds(dtype):
    if dtype == torch.bool:
        return 0, 1
    if dtype.is_complex:
        real = {
            torch.complex32: torch.float16,
            torch.complex64: torch.float32,
            torch.complex128: torch.float64,
        }[dtype]
        finfo = torch.finfo(real)
        return finfo.min, finfo.max
    if dtype.is_floating_point:
        finfo = torch.finfo(dtype)
        return finfo.min, finfo.max
    iinfo = torch.iinfo(dtype)
    return iinfo.min, iinfo.max


def make_input(dtype, shape, value_range):
    """Build a tensor of ``shape`` / ``dtype`` with values in ``value_range``.

    Bounds are clamped to the dtype's representable range first, so a range a
    dtype cannot represent (e.g. uint8 and ``[-1, 0]``) degenerates to a
    constant fill instead of raising.
    """
    low, high = dtype_bounds(dtype)
    table = {
        "-1": -1.0,
        "0": 0.0,
        "1": 1.0,
        "max": high,
        "min": low,
        "max/2": high / 2,
        "min/2": low / 2,
    }
    low, high = table[value_range[0]], table[value_range[1]]

    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=flag_gems.device).bool()

    if not (dtype.is_floating_point or dtype.is_complex):
        low, high = int(low), int(high)
        info = torch.iinfo(dtype)
        low = max(low, int(info.min))
        high = min(high, int(info.max))
        low = min(low, high)

    if low == high:
        return torch.full(shape, low, device=flag_gems.device, dtype=dtype)

    return torch.testing.make_tensor(
        shape, dtype=dtype, device=flag_gems.device, low=low, high=high
    )


def make_special_input(dtype, scenario):
    payloads = {
        "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
        "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
        "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
    }
    return torch.tensor(
        payloads[scenario], device=flag_gems.device, dtype=torch.float32
    ).to(dtype)


def to_reference(inp, upcast=False):
    """Independent tensor oracle for these tests, retaining view metadata."""
    if inp is None:
        return None
    if inp.layout != torch.strided or inp.is_quantized or inp.is_nested:
        reference = inp.detach().clone()
    else:
        reference = torch.empty(0, dtype=inp.dtype, device=inp.device).set_(
            inp.untyped_storage().clone(),
            inp.storage_offset(),
            inp.size(),
            inp.stride(),
        )
        if inp.is_conj():
            reference = reference.conj()
    # In quick-cpu mode the reference runs on CPU and gems_assert_close
    # asserts the reference is already there. .to() preserves size, stride
    # and storage offset, so the view metadata survives the move.
    if TO_CPU:
        reference = reference.to("cpu")
    reference.requires_grad_(inp.requires_grad)
    return reference


def assert_result_close(result, reference, *, atol=1e-4):
    # K0e has a pole at 0 and returns NaN for negative inputs, so the
    # comparison must ignore NaNs.
    utils.gems_assert_close(
        result, reference, dtype=result.dtype, atol=atol, equal_nan=True
    )


def special_value_cases(dtypes):
    """Representable special scenarios, kept separate in collected case IDs."""
    cases = []
    for dtype in dtypes:
        if not dtype.is_floating_point:
            continue
        cases.append((dtype, "nan"))
        if dtype not in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz,
        ):
            cases.extend((dtype, kind) for kind in ("inf", "mixed"))
    return cases


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
OUT_CASES = selected_cases(
    [
        (shape, value_range, dtype, out_dtype)
        for shape in REQUIRED_SHAPES
        for value_range in REQUIRED_RANGES
        for dtype, out_dtype in _OUT_DTYPE_PAIRS
    ],
    quick=[],
)

# Non-contiguous and offset layouts are default-only as well. transpose2d keeps
# the original two-dimensional flipped view; transpose3d transposes the full
# three-dimensional tensor so no extent is dropped.
STRIDED_CASES = selected_cases(
    [
        ("transpose2d", (20, 320)),
        ("transpose3d", (20, 320, 15)),
        ("offset", (256,)),
        ("strided_column", (1024, 1024)),
    ],
    quick=[],
)

SPECIAL_CASES = selected_cases(special_value_cases(_FLOAT_DTYPES), quick=[])

# Pole, dtype endpoints and the immediate neighbours of 2.0, the threshold of
# the classic rational approximations of the scaled K0.
EDGE_KINDS = selected_cases(
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

EMPTY_SHAPES = selected_cases([(0,), (0, 3), (5, 0, 2)], quick=[])

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
        return make_input(dtype, shape, ["0", "1"]).t()
    if kind == "transpose3d":
        return make_input(dtype, shape, ["0", "1"]).transpose(0, 1)
    if kind == "offset":
        parent = make_input(dtype, (shape[0] + 3,), ["0", "1"])
        return parent[2 : 2 + shape[0]]
    parent = make_input(dtype, (shape[0], shape[1] * 2), ["0", "1"])
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
@pytest.mark.parametrize("shape", selected_shapes())
@pytest.mark.parametrize("value_range", selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_special_scaled_modified_bessel_k0(shape, value_range, dtype):
    inp = make_input(dtype, shape, value_range)
    ref_inp = to_reference(inp)

    ref_out = torch.ops.aten.special_scaled_modified_bessel_k0(ref_inp)
    res_out = flag_gems.special_scaled_modified_bessel_k0(inp)

    assert_result_close(res_out, ref_out)


@pytest.mark.special_scaled_modified_bessel_k0_out
@pytest.mark.parametrize("shape,value_range,dtype,out_dtype", OUT_CASES)
def test_special_scaled_modified_bessel_k0_out(shape, value_range, dtype, out_dtype):
    inp = make_input(dtype, shape, value_range)
    ref_inp = to_reference(inp)

    ref_buf = torch.empty(shape, dtype=out_dtype, device=ref_inp.device)
    ref_out = torch.ops.aten.special_scaled_modified_bessel_k0.out(ref_inp, out=ref_buf)

    buf = torch.empty(shape, dtype=out_dtype, device=flag_gems.device)
    res_out = flag_gems.special_scaled_modified_bessel_k0(inp, out=buf)

    # The out= overload must write into, and return, the caller's buffer;
    # comparing pointers would also pass for a distinct alias of one storage.
    assert res_out is buf
    assert_result_close(res_out, ref_out)


@pytest.mark.special_scaled_modified_bessel_k0
@pytest.mark.parametrize("kind,shape", STRIDED_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_special_scaled_modified_bessel_k0_strided(kind, shape, dtype):
    inp = _strided_input(dtype, shape, kind)
    ref_inp = to_reference(inp)

    ref_out = torch.ops.aten.special_scaled_modified_bessel_k0(ref_inp)
    res_out = flag_gems.special_scaled_modified_bessel_k0(inp)

    assert_result_close(res_out, ref_out)


@pytest.mark.special_scaled_modified_bessel_k0
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test_special_scaled_modified_bessel_k0_special_values(dtype, scenario):
    inp = make_special_input(dtype, scenario)
    ref_inp = to_reference(inp)

    ref_out = torch.ops.aten.special_scaled_modified_bessel_k0(ref_inp)
    res_out = flag_gems.special_scaled_modified_bessel_k0(inp)

    assert_result_close(res_out, ref_out)


@pytest.mark.special_scaled_modified_bessel_k0
@pytest.mark.parametrize("kind", EDGE_KINDS)
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_special_scaled_modified_bessel_k0_edge_values(kind, dtype):
    inp = _edge_input(dtype, kind)
    ref_inp = to_reference(inp)

    ref_out = torch.ops.aten.special_scaled_modified_bessel_k0(ref_inp)
    res_out = flag_gems.special_scaled_modified_bessel_k0(inp)

    # atol=0: the +max endpoint is a small nonzero limit value (6.79e-20 in
    # float32, 9.35e-155 in float64), so the default absolute tolerance would
    # let a candidate that flushes the tail to zero pass. rtol stays the shared
    # per-dtype resolution.
    assert_result_close(res_out, ref_out, atol=0)


@pytest.mark.special_scaled_modified_bessel_k0
@pytest.mark.parametrize("shape", EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_special_scaled_modified_bessel_k0_empty(shape, dtype):
    inp = make_input(dtype, shape, ["-1", "1"])
    ref_inp = to_reference(inp)

    ref_out = torch.ops.aten.special_scaled_modified_bessel_k0(ref_inp)
    res_out = flag_gems.special_scaled_modified_bessel_k0(inp)

    assert_result_close(res_out, ref_out)


@pytest.mark.special_scaled_modified_bessel_k0
@pytest.mark.parametrize("dtype", REJECTED_DTYPES)
def test_special_scaled_modified_bessel_k0_rejects_unsupported_dtype(dtype):
    inp = make_input(dtype, (16,), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.special_scaled_modified_bessel_k0(inp)


@pytest.mark.special_scaled_modified_bessel_k0_out
@pytest.mark.parametrize("out_dtype", [torch.int32, torch.bool])
def test_special_scaled_modified_bessel_k0_rejects_non_float_out_buffer(out_dtype):
    # Native: RuntimeError, the float result cannot be cast to the Int/Bool
    # output type.
    inp = make_input(torch.float32, (16,), ["-1", "1"])
    buf = torch.empty(16, dtype=out_dtype, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.special_scaled_modified_bessel_k0(inp, out=buf)


@pytest.mark.special_scaled_modified_bessel_k0_out
def test_special_scaled_modified_bessel_k0_rejects_positional_out():
    # out is keyword-only in the native schema.
    inp = make_input(torch.float32, (16,), ["-1", "1"])
    buf = torch.empty(16, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.special_scaled_modified_bessel_k0(inp, buf)


@pytest.mark.special_scaled_modified_bessel_k0
def test_special_scaled_modified_bessel_k0_rejects_non_tensor_input():
    # Native: RuntimeError naming the expected Tensor argument for x.
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.special_scaled_modified_bessel_k0([1.0, 2.0])
