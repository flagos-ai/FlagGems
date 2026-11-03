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
from .conftest import TO_CPU

# ---------------------------------------------------------------------------
# Shared value-range / shape helpers (mirrors the operator-test spec
# conventions; QUICK_MODE selects the reduced sets via pytest --quick).
# ---------------------------------------------------------------------------
QUICK_MODE = utils.QUICK_MODE


def selected_cases(cases, *, quick=()):
    """Select parameter values; an empty quick subset contributes no cases."""
    return list(quick if QUICK_MODE else cases)


# Full set: ranks 0 through 5; the extents keep the elementwise polynomial
# evaluation bandwidth-relevant while staying allocation-friendly.
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
    # psi returns NaN/inf at its poles, so the comparison must ignore NaNs.
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


_BASE_RANGE = REQUIRED_RANGES[0]  # [-1, 1]

# Native dtype behaviour with a valid input: float16/bfloat16/float32/float64
# keep their dtype, while int8/uint8/int32/int64/bool yield a float32 result.
_FLOAT_DTYPES = [torch.float16, torch.float32]
if utils.bf16_is_supported:
    _FLOAT_DTYPES.insert(1, torch.bfloat16)
if utils.fp64_is_supported:
    _FLOAT_DTYPES.append(torch.float64)

_PROMOTED_DTYPES = [torch.int8, torch.uint8, torch.int32, torch.bool]
if utils.int64_is_supported:
    _PROMOTED_DTYPES.insert(3, torch.int64)

_GRID_DTYPES = _PROMOTED_DTYPES + _FLOAT_DTYPES

_GRID_CASES = [
    (dtype, value_range) for dtype in _GRID_DTYPES for value_range in REQUIRED_RANGES
]
_QUICK_GRID_CASES = [(dtype, _BASE_RANGE) for dtype in _GRID_DTYPES]

# These dtypes have no native kernel on the measured nvidia/CUDA backend:
# RuntimeError, "digamma_cuda" not implemented for 'Float8_e4m3fn' (also
# 'Float8_e5m2', 'ComplexFloat', 'ComplexDouble'). This is a missing dtype
# kernel, not an input-value restriction: the supported dtypes deliberately
# accept psi's poles and return infinities and NaNs. Dtype and vendor identity
# are both fixed at collection time, so a backend that cannot construct a dtype
# carries no case for it.
_REJECTED_DTYPES = []
if flag_gems.runtime.device.vendor_name == "nvidia":
    if utils.fp8_is_supported:
        _REJECTED_DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]
    _REJECTED_DTYPES.append(torch.complex64)
    if utils.fp64_is_supported:
        _REJECTED_DTYPES.append(torch.complex128)

_NON_TENSOR_INPUTS = [1.5, [1.0, 2.0]]

_PLAIN_SHAPES = {
    "shape_1d": (256,),
    "shape_3d": (20, 320, 15),
    "shape_2d": (1024, 1024),
    "plain_2d": (4, 6),
}

# View kinds add non-contiguous strides, a non-zero storage offset and 0-dim /
# zero-size inputs beside the contiguous grid.
_LAYOUT_KINDS = ["transposed", "storage_offset", "strided", "empty", "scalar_view"]

# (input kind, output-buffer layout); the plain (4, 6) input carries the
# non-contiguous and offset output layouts whose writing the native out kernel
# supports in place.
_OUT_CASES = [
    ("shape_1d", "contiguous"),
    ("shape_3d", "contiguous"),
    ("shape_2d", "contiguous"),
    ("transposed", "contiguous"),
    ("storage_offset", "contiguous"),
    ("plain_2d", "transposed"),
    ("plain_2d", "strided"),
    ("plain_2d", "offset"),
    ("plain_2d", "offset_strided"),
]

_BOUNDARY_KINDS = [
    "signed_zero",
    "near_zero",
    "tiny_values",
    "negative_poles",
    "pole_neighbours",
    "positive",
    "dtype_endpoints",
]


def _case_input(dtype, kind):
    """Input tensor for the layout and out tests."""
    if kind == "empty":
        return make_input(dtype, (2, 0, 3), _BASE_RANGE)
    if kind in _PLAIN_SHAPES:
        return make_input(dtype, _PLAIN_SHAPES[kind], _BASE_RANGE)
    inp = make_input(dtype, (4, 6), _BASE_RANGE)
    if kind == "transposed":
        return inp.t()
    if kind == "storage_offset":
        return inp[:, 1:5]
    if kind == "strided":
        return inp.reshape(-1)[::2]
    return inp[2, 3]  # 0-dim view with a non-zero storage offset


def _out_buffer(shape, dtype, layout, device):
    """Empty output buffer of ``shape`` written in the requested layout."""
    if layout == "contiguous":
        return torch.empty(shape, dtype=dtype, device=device)
    if layout == "transposed":
        return torch.empty(tuple(shape)[::-1], dtype=dtype, device=device).t()
    rows, cols = shape
    if layout == "strided":
        buf = torch.empty(rows, cols * 2, dtype=dtype, device=device)
        return buf[:, ::2]
    if layout == "offset":
        buf = torch.empty(rows * cols + 8, dtype=dtype, device=device)
        return buf[8:].view(shape)
    width = 8 + cols * 2
    buf = torch.empty(rows + 2, width, dtype=dtype, device=device)
    return buf[2:, 8 : 8 + cols * 2 : 2]


def _boundary_input(dtype, kind):
    """Boundary input tensor for ``kind``, built on the target device."""
    device = flag_gems.device

    def as_tensor(values):
        return torch.tensor(values, dtype=dtype, device=device)

    if kind == "signed_zero":
        return as_tensor([0.0, -0.0])
    if kind == "near_zero":
        return as_tensor([torch.finfo(dtype).eps, 1e-3, 0.1])
    if kind == "tiny_values":
        # The smallest normal value and the smallest representable nonzero
        # (subnormal for these dtypes), both signs. Measured on nvidia/CUDA
        # the reference keeps the smallest normal finite (-16384 fp16,
        # -8.507e37 bf16/fp32, -4.494e307 fp64) and saturates the subnormal to
        # -inf/+inf, so the native result decides both.
        smallest = torch.nextafter(
            torch.zeros(1, dtype=dtype, device=device),
            torch.ones(1, dtype=dtype, device=device),
        )
        tiny = torch.finfo(dtype).tiny
        return torch.cat([as_tensor([tiny, -tiny]), smallest, -smallest])
    if kind == "negative_poles":
        return as_tensor([-1.0, -2.0, -7.0])
    if kind == "pole_neighbours":
        # Literal neighbours plus the immediate representable values on both
        # sides of each pole. Literal offsets do not consistently select adjacent
        # representable values, so immediate neighbours are derived in the target
        # dtype.
        poles = as_tensor([-1.0, -2.0, -7.0])
        below = torch.nextafter(poles, as_tensor([-float("inf")] * 3))
        above = torch.nextafter(poles, as_tensor([0.0] * 3))
        literal = as_tensor([-1.1, -0.9, -1.01, -0.99, -2.01, -1.99])
        return torch.cat([literal, below, above])
    if kind == "positive":
        return as_tensor([0.5, 1.0, 2.0, 10.0, 1e3])
    finfo = torch.finfo(dtype)
    return as_tensor([finfo.max, finfo.min])


@pytest.mark.special_psi
@pytest.mark.parametrize("shape", selected_shapes())
@pytest.mark.parametrize(
    "dtype,value_range", selected_cases(_GRID_CASES, quick=_QUICK_GRID_CASES)
)
def test_special_psi(shape, dtype, value_range):
    inp = make_input(dtype, shape, value_range)
    ref_inp = to_reference(inp)

    ref_out = torch.ops.aten.special_psi(ref_inp)
    res_out = flag_gems.special_psi(inp)

    assert_result_close(res_out, ref_out)


@pytest.mark.special_psi
@pytest.mark.parametrize("kind", selected_cases(_LAYOUT_KINDS, quick=[]))
@pytest.mark.parametrize("dtype", selected_cases(_FLOAT_DTYPES, quick=[]))
def test_special_psi_layout(kind, dtype):
    inp = _case_input(dtype, kind)
    ref_inp = to_reference(inp)

    ref_out = torch.ops.aten.special_psi(ref_inp)
    res_out = flag_gems.special_psi(inp)

    assert_result_close(res_out, ref_out)


@pytest.mark.special_psi
@pytest.mark.parametrize("kind", selected_cases(_BOUNDARY_KINDS, quick=[]))
@pytest.mark.parametrize("dtype", selected_cases(_FLOAT_DTYPES, quick=[]))
def test_special_psi_boundary_values(kind, dtype):
    inp = _boundary_input(dtype, kind)
    ref_inp = to_reference(inp)

    ref_out = torch.ops.aten.special_psi(ref_inp)
    res_out = flag_gems.special_psi(inp)

    assert_result_close(res_out, ref_out)


@pytest.mark.special_psi
@pytest.mark.parametrize(
    "dtype,scenario", selected_cases(special_value_cases(_FLOAT_DTYPES), quick=[])
)
def test_special_psi_special_values(dtype, scenario):
    inp = make_special_input(dtype, scenario)
    ref_inp = to_reference(inp)

    ref_out = torch.ops.aten.special_psi(ref_inp)
    res_out = flag_gems.special_psi(inp)

    assert_result_close(res_out, ref_out)


@pytest.mark.special_psi
@pytest.mark.parametrize("in_kind,out_layout", selected_cases(_OUT_CASES, quick=[]))
@pytest.mark.parametrize("dtype", selected_cases(_FLOAT_DTYPES, quick=[]))
def test_special_psi_out(in_kind, out_layout, dtype):
    inp = _case_input(dtype, in_kind)
    ref_inp = to_reference(inp)

    ref_out = _out_buffer(ref_inp.shape, dtype, out_layout, ref_inp.device)
    torch.ops.aten.special_psi.out(ref_inp, out=ref_out)

    res_out = _out_buffer(inp.shape, dtype, out_layout, inp.device)
    layout_before = (res_out.stride(), res_out.storage_offset())
    res_ret = flag_gems.special_psi(inp, out=res_out)

    assert res_ret is res_out
    assert (res_out.stride(), res_out.storage_offset()) == layout_before
    assert_result_close(res_out, ref_out)


@pytest.mark.special_psi
@pytest.mark.parametrize("dtype", selected_cases(_PROMOTED_DTYPES, quick=[]))
def test_special_psi_out_promoted(dtype):
    # An integral or bool input produces a float32 result, which the default
    # call form does not exercise with an explicit float32 buffer. The range
    # only describes the operand: psi(0) = -inf is part of the comparison.
    inp = make_input(dtype, (1024, 1024), ["0", "max"])
    ref_inp = to_reference(inp)

    ref_out = torch.empty(ref_inp.shape, dtype=torch.float32, device=ref_inp.device)
    torch.ops.aten.special_psi.out(ref_inp, out=ref_out)

    res_out = torch.empty(inp.shape, dtype=torch.float32, device=inp.device)
    res_ret = flag_gems.special_psi(inp, out=res_out)

    assert res_ret is res_out
    assert_result_close(res_out, ref_out)


@pytest.mark.special_psi
@pytest.mark.parametrize("dtype", selected_cases(_FLOAT_DTYPES, quick=[]))
def test_special_psi_backward(dtype):
    # Both sides draw their values from [0.5, 4.0), away from the poles at zero
    # and the negative integers, with the same non-constant upstream gradient.
    inp = torch.rand((2, 19, 7), dtype=dtype, device=flag_gems.device) * 3.5 + 0.5
    inp.requires_grad_(True)
    ref_inp = to_reference(inp)
    upstream = torch.rand((2, 19, 7), dtype=dtype, device=flag_gems.device) + 0.5
    ref_upstream = to_reference(upstream)

    ref_out = torch.ops.aten.special_psi(ref_inp)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)

    res_out = flag_gems.special_psi(inp)
    assert_result_close(res_out, ref_out)
    (res_grad,) = torch.autograd.grad(res_out, inp, grad_outputs=upstream)

    assert_result_close(res_grad, ref_grad)


@pytest.mark.special_psi
@pytest.mark.parametrize("dtype", _REJECTED_DTYPES)
def test_special_psi_rejects_unsupported_dtype(dtype):
    inp = make_input(dtype, (8,), _BASE_RANGE)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.special_psi(inp)


@pytest.mark.special_psi
@pytest.mark.parametrize("bad_input", _NON_TENSOR_INPUTS)
def test_special_psi_rejects_non_tensor_input(bad_input):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.special_psi(bad_input)


@pytest.mark.special_psi_out
def test_special_psi_out_rejects_incompatible_dtype():
    inp = make_input(torch.float32, (4,), _BASE_RANGE)
    out = torch.empty(4, dtype=torch.int8, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.special_psi(inp, out=out)
