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

"""Correctness tests for ``aten::int_repr``."""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import TO_CPU

# aten::int_repr returns the raw integer codes an affine-quantized tensor
# stores: quint8 -> uint8, qint8 -> int8, qint32 -> int32. The quantization
# parameters do not enter the result, so every comparison against the native
# result is exact (rtol=0, atol=0) and there is no gradient to check.
#
# Native behaviour of the assigned backend (CUDA / NVIDIA, torch 2.8):
#   * only quantized inputs have a kernel; every plain dtype family, complex
#     included, fails with NotImplementedError, and a non-tensor argument fails
#     the schema check. The negative rows below assert the candidate's own
#     RuntimeError.
#   * torch.quantize_per_tensor accepts float32 sources only ("Quantize only
#     works on Float Tensor"), so the spec's five value ranges are carried by a
#     float32 payload that is then quantized.
#   * per-tensor and per-channel (torch.per_channel_affine) quantization are
#     both supported; per-channel scales and zero points are float64 / int64
#     metadata with one entry per input.size(axis).
#   * rank 0 and zero-sized dimensions are valid quantized inputs on the
#     device, and int_repr.out accepts a device buffer for their empty result.
#   * transpose / offset slice / strided slice / zero-stride expand /
#     channels_last views read exactly like a compact clone.
#   * int_repr.out returns the caller's buffer, resizes a too-small buffer,
#     writes through a strided buffer, rejects a mismatched out dtype and
#     requires the out tensor on the input's device.
# Broadcasting (single operand) and backward (raw integer storage carries no
# autograd history) do not apply to this operator.

gems_device = flag_gems.runtime.device

_QUANTIZED_TO_INT = {
    torch.quint8: torch.uint8,
    torch.qint8: torch.int8,
    torch.qint32: torch.int32,
}
QUANTIZED_DTYPES = list(_QUANTIZED_TO_INT)

# Static capabilities declared for the active backend. Per-channel
# quantization stores float64 scales and int64 zero points, so its rows exist
# only where both dtypes do; the plain-dtype negative rows were measured on the
# CUDA/NVIDIA backend only. Both eligible sets are built here at import time,
# so no test carries a runtime skip.
_PER_CHANNEL_SUPPORTED = gems_device.support_fp64 and gems_device.support_int64
_NON_QUANTIZED_MEASURED = getattr(gems_device, "vendor_name", "") == "nvidia"

# Unit quantization parameters tie the stored codes to the payload: a float32
# payload in [-1, 1] yields codes in {-1, 0, 1} (unsigned {0, 1}), so the
# storage maximum is not an expected value and a buffer pre-filled with it
# exposes a candidate that returns without writing its result.
_UNIT_SCALE = 1.0
_UNIT_ZERO_POINT = 0


# ---------------------------------------------------------------------------
# Shared value-range / shape helpers (mirrors the operator-test spec
# conventions; QUICK_MODE selects the reduced sets via pytest --quick).
# ---------------------------------------------------------------------------
QUICK_MODE = utils.QUICK_MODE


def selected_cases(cases, *, quick=()):
    """Select parameter values; an empty quick subset contributes no cases."""
    return list(quick if QUICK_MODE else cases)


# Shapes: 0~5 dims, covering the ranks an elementwise quantized-storage copy
# accepts. Shapes chosen to exercise dense, transposed, sliced and expanded
# views of the quantized payload.
QUICK_SHAPES = [
    (2, 19, 7),
]
# Full set: ranks 0 through 5; the larger extents keep the copy bandwidth- and
# stride-decomposition-relevant while staying allocation-friendly.
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
    return QUICK_SHAPES if QUICK_MODE else REQUIRED_SHAPES


# Value ranges: [-1,1], [0,1], [-1,0], [0,max], [min,0]. Bounds resolve per
# dtype; here the payload is always float32, so "max"/"min" are its bounds.
QUICK_RANGES = [
    ["-1", "1"],
]
REQUIRED_RANGES = [
    ["-1", "1"],
    ["0", "1"],
    ["-1", "0"],
    ["0", "max"],
    ["min", "0"],
]


def selected_ranges():
    return QUICK_RANGES if QUICK_MODE else REQUIRED_RANGES


def _resolve_bound(symbol, dtype):
    low, high = torch.finfo(dtype).min, torch.finfo(dtype).max
    table = {
        "-1": -1.0,
        "0": 0.0,
        "1": 1.0,
        "max": high,
        "min": low,
        "max/2": high / 2,
        "min/2": low / 2,
    }
    return table[symbol]


def make_input(dtype, shape, value_range):
    """Build a float32 tensor of ``shape`` with values in ``value_range``."""
    low = _resolve_bound(value_range[0], dtype)
    high = _resolve_bound(value_range[1], dtype)
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
    """Independent tensor oracle for these tests, retaining view metadata.

    A plain clone can compact strides, reset offsets or materialize lazy
    flags, so the dense storage is copied with the original metadata instead.
    """
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
    # In quick-cpu mode the reference runs on CPU and gems_assert_equal
    # asserts the reference is already there. .to() preserves size, stride
    # and storage offset, so the view metadata survives the move.
    if TO_CPU:
        reference = reference.to("cpu")
    reference.requires_grad_(inp.requires_grad)
    return reference


def int_repr_snapshot(tensor):
    """Device-side ``aten::int_repr`` snapshot, moved to the oracle device.

    Used for the read-only checks that snapshot the input's raw codes before
    and after the candidate runs; both sides must live on the reference
    device for gems_assert_equal's TO_CPU handling.
    """
    codes = torch.ops.aten.int_repr(tensor)
    return codes.cpu() if TO_CPU else codes


def _quantized_input(
    shape, qdtype, value_range, scale=_UNIT_SCALE, zero_point=_UNIT_ZERO_POINT
):
    return _quantize(
        make_input(torch.float32, shape, value_range), qdtype, scale, zero_point
    )


def _quantize(payload, qdtype, scale, zero_point):
    # quantize_per_tensor accepts rank-0 and zero-sized payloads, so the
    # degenerate shapes below stay real quantized tensors on the device.
    return torch.quantize_per_tensor(payload, scale, zero_point, qdtype)


def _quantized_per_channel(payload, qdtype, scales, zero_points, axis):
    return torch.quantize_per_channel(
        payload,
        torch.tensor(scales, dtype=torch.float64, device=payload.device),
        torch.tensor(zero_points, dtype=torch.int64, device=payload.device),
        axis,
        qdtype,
    )


_CHANNEL_SCALES = [2.0**-10, 2.0**-6, 0.5, 1.0, 4.0, 64.0]


def _channel_qparams(channels, qdtype):
    info = torch.iinfo(_QUANTIZED_TO_INT[qdtype])
    span = info.max - info.min
    scales = [_CHANNEL_SCALES[i % len(_CHANNEL_SCALES)] for i in range(channels)]
    zero_points = [
        int(info.min + (span * i) // max(channels - 1, 1)) for i in range(channels)
    ]
    return scales, zero_points


def _apply_layout(inp, layout):
    if layout == "transposed":
        return inp.transpose(0, 1)
    if layout == "offset_slice":
        return inp[1:]
    if layout == "strided_last":
        return inp[..., ::2]
    if layout == "expanded":
        # Slicing one element keeps expand() valid whatever the first
        # dimension is; the result is a zero-stride view of shape
        # (4, *inp.shape[1:]).
        return inp[:1].expand(4, *inp.shape[1:])
    if layout == "channels_last":
        return inp.to(memory_format=torch.channels_last)
    return inp


def _sentinel(out_dtype):
    return torch.iinfo(out_dtype).max


def _sentinel_buffer(shape, out_dtype):
    return torch.full(
        shape, _sentinel(out_dtype), dtype=out_dtype, device=flag_gems.device
    )


def _unquantized_dtypes():
    """Plain dtypes with no int_repr kernel, limited by declared capabilities."""
    dtypes = [
        torch.int8,
        torch.uint8,
        torch.float32,
        torch.float16,
        torch.int32,
        torch.bool,
        torch.complex64,
    ]
    if gems_device.support_bf16:
        dtypes.append(torch.bfloat16)
    if gems_device.support_int64:
        dtypes.append(torch.int64)
    if gems_device.support_fp64:
        # complex128 is the two-float64 family and shares the fp64 capability.
        dtypes.extend([torch.float64, torch.complex128])
    if gems_device.support_fp8:
        dtypes.extend([torch.float8_e4m3fn, torch.float8_e5m2])
    return dtypes


# Rank-0 and zero-sized shapes are natively valid quantized inputs and are not
# part of the spec's seven shapes.
DEGENERATE_SHAPES = [(), (0,), (0, 3), (2, 0, 19, 7)]
DEGENERATE_ROWS = [
    (shape, qdtype) for shape in DEGENERATE_SHAPES for qdtype in QUANTIZED_DTYPES
]

# Views whose stored codes must read exactly like a compact clone.
LAYOUT_ROWS = [
    (layout, shape, qdtype)
    for shape in ((6, 7, 8), (4, 6, 8, 10))
    for layout in (
        ("transposed", "offset_slice", "strided_last", "expanded")
        + (("channels_last",) if len(shape) == 4 else ())
    )
    for qdtype in QUANTIZED_DTYPES
]

# Writing through the result must not change the quantized input.
INDEPENDENT_ROWS = [
    (qdtype, layout)
    for qdtype in QUANTIZED_DTYPES
    for layout in ("contiguous", "transposed", "offset_slice")
]

# Per-tensor parameters spanning the legal range of the storage dtype.
QPARAM_ROWS = [
    (qdtype, scale, fraction)
    for qdtype in QUANTIZED_DTYPES
    for scale, fraction in [
        (2.0**-20, 0.0),
        (2.0**20, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    ]
]

# Saturating payloads that quantize to the storage bounds.
_BOUNDARY_RULES = [
    ("saturate_high", lambda info: info.max * 1.0),
    ("saturate_low", lambda info: info.min * 1.0),
    ("zero", lambda info: 0.0),
]
BOUNDARY_ROWS = [
    (qdtype, rule) for qdtype in QUANTIZED_DTYPES for rule, _ in _BOUNDARY_RULES
]

_PER_CHANNEL_CASES = [
    (qdtype, shape, axis)
    for qdtype in QUANTIZED_DTYPES
    for shape, axis in (((4, 5), 0), ((4, 5), 1), ((3, 7, 5), 2))
]
PER_CHANNEL_ROWS = _PER_CHANNEL_CASES if _PER_CHANNEL_SUPPORTED else []

SPECIAL_ROWS = ["nan", "inf", "mixed"]

OUT_RESIZE_ROWS = QUANTIZED_DTYPES
OUT_STRIDED_ROWS = QUANTIZED_DTYPES

NON_QUANTIZED_ROWS = _unquantized_dtypes() if _NON_QUANTIZED_MEASURED else []


@pytest.mark.int_repr
@pytest.mark.parametrize("qdtype", QUANTIZED_DTYPES)
@pytest.mark.parametrize("value_range", selected_ranges())
@pytest.mark.parametrize("shape", selected_shapes())
def test_int_repr(shape, value_range, qdtype):
    inp = _quantized_input(shape, qdtype, value_range)
    ref_inp = to_reference(inp)
    before = int_repr_snapshot(inp)

    ref_out = torch.ops.aten.int_repr(ref_inp)
    res_out = flag_gems.int_repr(inp)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
    # Reading the input must not change the quantized storage.
    utils.gems_assert_equal(torch.ops.aten.int_repr(inp), before, equal_nan=True)


@pytest.mark.int_repr
@pytest.mark.parametrize("qdtype,layout", INDEPENDENT_ROWS)
def test_int_repr_result_is_independent_of_input_storage(qdtype, layout):
    inp = _apply_layout(_quantized_input((2, 19, 7), qdtype, ["-1", "1"]), layout)
    ref_inp = to_reference(inp)
    expected = torch.ops.aten.int_repr(ref_inp)

    res_out = flag_gems.int_repr(inp)
    utils.gems_assert_equal(res_out, expected, equal_nan=True)

    # Filling the result must leave the quantized input untouched.
    res_out.fill_(3)
    utils.gems_assert_equal(torch.ops.aten.int_repr(inp), expected)


@pytest.mark.int_repr_out
@pytest.mark.parametrize("qdtype", QUANTIZED_DTYPES)
@pytest.mark.parametrize("shape", selected_shapes())
def test_int_repr_out(shape, qdtype):
    inp = _quantized_input(shape, qdtype, ["-1", "1"])
    ref_inp = to_reference(inp)
    out_dtype = _QUANTIZED_TO_INT[qdtype]
    before = int_repr_snapshot(inp)

    ref_out = torch.ops.aten.int_repr.out(
        ref_inp, out=torch.empty(shape, dtype=out_dtype, device=ref_inp.device)
    )
    # The sentinel pre-fill makes the exact comparison reject a candidate that
    # returns the buffer without writing its result.
    buf = _sentinel_buffer(shape, out_dtype)
    res_out = flag_gems.int_repr(inp, out=buf)

    assert res_out is buf
    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
    utils.gems_assert_equal(torch.ops.aten.int_repr(inp), before, equal_nan=True)


@pytest.mark.int_repr_out
@pytest.mark.parametrize("qdtype", OUT_RESIZE_ROWS)
def test_int_repr_out_resizes_buffer(qdtype):
    inp = _quantized_input((2, 19, 7), qdtype, ["-1", "1"])
    ref_inp = to_reference(inp)
    out_dtype = _QUANTIZED_TO_INT[qdtype]
    before = int_repr_snapshot(inp)

    ref_out = torch.ops.aten.int_repr.out(
        ref_inp, out=torch.empty((3,), dtype=out_dtype, device=ref_inp.device)
    )
    buf = _sentinel_buffer((3,), out_dtype)
    res_out = flag_gems.int_repr(inp, out=buf)

    assert res_out is buf
    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
    # The resizing path must not disturb the quantized input's storage.
    utils.gems_assert_equal(torch.ops.aten.int_repr(inp), before, equal_nan=True)


@pytest.mark.int_repr_out
@pytest.mark.parametrize("qdtype", OUT_STRIDED_ROWS)
def test_int_repr_out_writes_through_strided_buffer(qdtype):
    inp = _quantized_input((4, 5, 6), qdtype, ["-1", "1"])
    ref_inp = to_reference(inp)
    out_dtype = _QUANTIZED_TO_INT[qdtype]
    gap = (4, 5, 12)
    sentinel = _sentinel(out_dtype)
    before = int_repr_snapshot(inp)

    ref_pad = torch.full(gap, sentinel, dtype=out_dtype, device=ref_inp.device)
    ref_out = torch.ops.aten.int_repr.out(ref_inp, out=ref_pad[:, :, ::2])
    pad = torch.full(gap, sentinel, dtype=out_dtype, device=flag_gems.device)
    buf = pad[:, :, ::2]
    res_out = flag_gems.int_repr(inp, out=buf)

    assert res_out is buf
    assert tuple(buf.stride()) == (60, 12, 2)
    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
    # Only the strided view is written; the parent's padding stays untouched.
    utils.gems_assert_equal(pad, ref_pad, equal_nan=True)
    utils.gems_assert_equal(torch.ops.aten.int_repr(inp), before, equal_nan=True)


@pytest.mark.int_repr
@pytest.mark.parametrize("layout,shape,qdtype", LAYOUT_ROWS)
def test_int_repr_reads_layout(layout, shape, qdtype):
    inp = _apply_layout(_quantized_input(shape, qdtype, ["-1", "1"]), layout)
    ref_inp = to_reference(inp)

    ref_out = torch.ops.aten.int_repr(ref_inp)
    res_out = flag_gems.int_repr(inp)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.int_repr
@pytest.mark.parametrize("qdtype,shape,axis", PER_CHANNEL_ROWS)
def test_int_repr_per_channel_quantization(qdtype, shape, axis):
    payload = make_input(torch.float32, shape, ["-1", "1"])
    scales, zero_points = _channel_qparams(shape[axis], qdtype)
    inp = _quantized_per_channel(payload, qdtype, scales, zero_points, axis)
    ref_inp = to_reference(inp)
    before = int_repr_snapshot(inp)

    ref_out = torch.ops.aten.int_repr(ref_inp)
    res_out = flag_gems.int_repr(inp)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
    utils.gems_assert_equal(torch.ops.aten.int_repr(inp), before, equal_nan=True)


@pytest.mark.int_repr
@pytest.mark.parametrize("qdtype,scale,fraction", QPARAM_ROWS)
def test_int_repr_extreme_qparams(qdtype, scale, fraction):
    info = torch.iinfo(_QUANTIZED_TO_INT[qdtype])
    zero_point = int(round(info.min + fraction * (info.max - info.min)))
    inp = _quantized_input((6, 7, 8), qdtype, ["-1", "1"], scale, zero_point)
    ref_inp = to_reference(inp)

    ref_out = torch.ops.aten.int_repr(ref_inp)
    res_out = flag_gems.int_repr(inp)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.int_repr
@pytest.mark.parametrize("qdtype,rule", BOUNDARY_ROWS)
def test_int_repr_storage_boundaries(qdtype, rule):
    info = torch.iinfo(_QUANTIZED_TO_INT[qdtype])
    value = dict(_BOUNDARY_RULES)[rule](info)
    payload = torch.full((6, 7, 8), value, dtype=torch.float32, device=flag_gems.device)
    inp = _quantize(payload, qdtype, 1.0, 0)
    ref_inp = to_reference(inp)

    ref_out = torch.ops.aten.int_repr(ref_inp)
    res_out = flag_gems.int_repr(inp)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.int_repr
@pytest.mark.parametrize("qdtype", QUANTIZED_DTYPES)
@pytest.mark.parametrize("scenario", SPECIAL_ROWS)
def test_int_repr_quantized_special_values(scenario, qdtype):
    # The affine quantizer maps the non-finite payload values onto representable
    # codes before int_repr reads them; this case exercises that preprocessing,
    # and the exact comparison uses the native raw result as the oracle for
    # every dtype.
    inp = _quantize(make_special_input(torch.float32, scenario), qdtype, 1.0, 0)
    ref_inp = to_reference(inp)

    ref_out = torch.ops.aten.int_repr(ref_inp)
    res_out = flag_gems.int_repr(inp)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.int_repr
@pytest.mark.parametrize("shape,qdtype", DEGENERATE_ROWS)
def test_int_repr_degenerate_shape(shape, qdtype):
    inp = _quantized_input(shape, qdtype, ["-1", "1"])
    ref_inp = to_reference(inp)

    ref_out = torch.ops.aten.int_repr(ref_inp)
    res_out = flag_gems.int_repr(inp)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.int_repr_out
@pytest.mark.parametrize("shape,qdtype", DEGENERATE_ROWS)
def test_int_repr_out_degenerate_shape(shape, qdtype):
    inp = _quantized_input(shape, qdtype, ["-1", "1"])
    ref_inp = to_reference(inp)
    out_dtype = _QUANTIZED_TO_INT[qdtype]
    before = int_repr_snapshot(inp)

    ref_out = torch.ops.aten.int_repr.out(
        ref_inp, out=torch.empty(shape, dtype=out_dtype, device=ref_inp.device)
    )
    buf = _sentinel_buffer(shape, out_dtype)
    res_out = flag_gems.int_repr(inp, out=buf)

    assert res_out is buf
    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
    utils.gems_assert_equal(torch.ops.aten.int_repr(inp), before, equal_nan=True)


@pytest.mark.int_repr
@pytest.mark.parametrize("plain_dtype", NON_QUANTIZED_ROWS)
def test_int_repr_rejects_non_quantized_dtype(plain_dtype):
    inp = torch.zeros((2, 19, 7), dtype=plain_dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.int_repr(inp)


@pytest.mark.int_repr
def test_int_repr_rejects_non_tensor_argument():
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.int_repr([1, 2, 3])


@pytest.mark.int_repr_out
@pytest.mark.parametrize("qdtype", QUANTIZED_DTYPES)
def test_int_repr_out_rejects_mismatched_dtype(qdtype):
    inp = _quantized_input((2, 19, 7), qdtype, ["-1", "1"])
    # int16 needs no capability flag and differs from every storage dtype
    # int_repr produces (uint8 / int8 / int32), so the out dtype check fires.
    buf = torch.empty((2, 19, 7), dtype=torch.int16, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.int_repr(inp, out=buf)
