# Copyright 2026, The FlagOS Contributors.
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

# CI derives the pytest marker from the implementation file name via
# the #6359 convention: the stripped name collides with the registry id,
# so the ``underscore_`` prefix is applied. Declare both markers so
# ``pytest -m <marker>`` selects these tests from either entry point.
pytestmark = [
    pytest.mark.neg_view_copy,
    pytest.mark.underscore_neg_view_copy,
]

# Materialize negated values in independent contiguous storage. The dtype and
# case lists mirror the KernelGen acceptance suite so the reported speedup
# stays comparable to its H20 measurement.
# Dtype coverage mirrors the KernelGen acceptance sweep; int16 is exercised
# only in the non-quick set and float64 is gated on backend support.
_NEG_VIEW_COPY_DTYPES = (
    [
        torch.int8,
        torch.uint8,
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.int32,
        torch.int64,
    ]
    # float64 is gated on backend support (see utils.fp64_is_supported).
    + ([torch.float64] if utils.fp64_is_supported else [])
    + [torch.int16]
)

_UNSUPPORTED_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2, torch.bool]

# Unary shape sweep: 1-D, 2-D and the regular 2-D core shape.
_NEG_VIEW_COPY_SHAPES = [(17,), (12, 13), (256,), (1024, 1024)]

# Non-contiguous inputs exercise the strided kernel.
_NEG_VIEW_COPY_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]

# Zero-element and zero-extent shapes.
_NEG_VIEW_COPY_EMPTY_SHAPES = [(0,), (4, 0), (2, 0, 3)]

# Shapes used for the autograd backward smoke.
_NEG_VIEW_COPY_BACKWARD_SHAPES = [(16, 64), (7, 13, 29)]

_VALUE_RANGES = [
    ["-1", "1"],
    ["0", "1"],
    ["-1", "0"],
    ["0", "max"],
    ["min", "0"],
]
_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _NEG_VIEW_COPY_DTYPES
    for value_range in _VALUE_RANGES
]


def _dtype_bounds(dtype):
    """Return the (min, max) value bounds of ``dtype``."""
    if dtype == torch.bool:
        return 0, 1
    if dtype.is_floating_point:
        finfo = torch.finfo(dtype)
        return finfo.min, finfo.max
    iinfo = torch.iinfo(dtype)
    return iinfo.min, iinfo.max


def _resolve_bound(symbol, dtype):
    """Resolve a range-bound symbol (-1 / 0 / 1 / max / min) for ``dtype``."""
    low, high = _dtype_bounds(dtype)
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


def _make_input(dtype, shape, value_range):
    """Build a dense tensor of ``shape`` / ``dtype`` within ``value_range``.

    Bounds are clamped to the dtype's representable range first, so unsigned
    dtypes that cannot hold ``-1`` degrade to a constant zero fill.
    """
    low = _resolve_bound(value_range[0], dtype)
    high = _resolve_bound(value_range[1], dtype)
    if not (dtype.is_floating_point or dtype.is_complex):
        low, high = int(low), int(high)
        dtype_min, dtype_max = _dtype_bounds(dtype)
        low = max(low, int(dtype_min))
        high = min(high, int(dtype_max))
        low = min(low, high)
    if low == high:
        return torch.full(shape, low, device=flag_gems.device, dtype=dtype)
    return torch.testing.make_tensor(
        shape, dtype=dtype, device=flag_gems.device, low=low, high=high
    )


def _assert_copy_semantics(res_out, ref_out, inp, ref_inp):
    assert res_out.is_contiguous()
    assert not res_out.is_neg()
    # Zero-element tensors carry a null data pointer on every tensor, so the
    # no-alias check is only meaningful for non-empty inputs.
    if inp.numel() > 0:
        assert res_out.data_ptr() != inp.data_ptr()
    utils.gems_assert_equal(inp, ref_inp)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test_neg_view_copy(shape, dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = _make_input(
        dtype, shape, ["0", "max"] if dtype == torch.uint8 else ["-1", "1"]
    )
    # to_reference creates an independent snapshot for the input mutation check.
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = flag_gems._neg_view_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.neg_view_copy
@pytest.mark.parametrize("shape", [(256,), (1024, 1024)])
@pytest.mark.parametrize(("dtype", "value_range"), _RANGE_CASES)
def test_neg_view_copy_value_ranges(shape, dtype, value_range):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = _make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = flag_gems._neg_view_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.neg_view_copy_out
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test_neg_view_copy_out(shape, dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = _make_input(
        dtype, shape, ["0", "max"] if dtype == torch.uint8 else ["-1", "1"]
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.full(shape, 7, dtype=ref_inp.dtype, device=ref_inp.device)
    out = torch.full(shape, 7, dtype=dtype, device=flag_gems.device)

    torch.ops.aten._neg_view_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems._neg_view_copy(inp, out=out)

    # The .out variant must write into and return the caller's buffer itself.
    assert res_ret is out
    _assert_copy_semantics(res_ret, ref_out, inp, ref_inp)


@pytest.mark.neg_view_copy
@pytest.mark.parametrize("shape", [(256,), (1024, 1024)])
@pytest.mark.parametrize(("dtype", "value_range"), _RANGE_CASES)
def test_neg_view_copy_out_value_ranges(shape, dtype, value_range):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = _make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.full(shape, 7, dtype=ref_inp.dtype, device=ref_inp.device)
    out = torch.full(shape, 7, dtype=dtype, device=flag_gems.device)

    torch.ops.aten._neg_view_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems._neg_view_copy(inp, out=out)

    assert res_ret is out
    _assert_copy_semantics(res_ret, ref_out, inp, ref_inp)


_SPECIAL_PAYLOADS = {
    "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
    "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
    "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
}


def _special_value_cases(dtypes):
    """Representable special scenarios per floating dtype."""
    cases = []
    for dtype in dtypes:
        if not dtype.is_floating_point:
            continue
        cases.append((dtype, "nan"))
        cases.extend((dtype, kind) for kind in ("inf", "mixed"))
    return cases


@pytest.mark.neg_view_copy
@pytest.mark.parametrize(
    "dtype, scenario", _special_value_cases(utils.ALL_FLOAT_DTYPES)
)
def test_neg_view_copy_special_values(dtype, scenario):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    values = torch.tensor(
        _SPECIAL_PAYLOADS[scenario], device=flag_gems.device, dtype=torch.float32
    ).to(dtype)
    ref_inp = utils.to_reference(values)

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = flag_gems._neg_view_copy(values)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
    # Exact numerical equality does not distinguish the signs of zero.
    zeros = values == 0
    utils.gems_assert_equal(
        torch.signbit(res_out[zeros]), torch.signbit(ref_out[ref_inp == 0])
    )


@pytest.mark.neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test_neg_view_copy_non_contiguous(shape, dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    base = _make_input(
        dtype, shape, ["0", "max"] if dtype == torch.uint8 else ["-1", "1"]
    )
    ref_base = utils.to_reference(base)
    inp = base.transpose(-1, -2)
    ref_inp = ref_base.transpose(-1, -2)
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = flag_gems._neg_view_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test_neg_view_copy_empty(shape, dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = _make_input(
        dtype, shape, ["0", "max"] if dtype == torch.uint8 else ["-1", "1"]
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = flag_gems._neg_view_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_neg_view_copy_backward(shape, dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = _make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    ref_grad = utils.to_reference(grad)

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    res_out = flag_gems._neg_view_copy(inp)
    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)

    assert res_out.requires_grad
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.neg_view_copy
@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)
def test_neg_view_copy_rejects_unsupported_dtypes(dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = torch.zeros(4, dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten._neg_view_copy(inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._neg_view_copy(inp)


@pytest.mark.neg_view_copy
def test_neg_view_copy_rejects_non_tensor():
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    with pytest.raises(RuntimeError):
        torch.ops.aten._neg_view_copy(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._neg_view_copy(3.14)
