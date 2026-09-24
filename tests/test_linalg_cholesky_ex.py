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


"""Correctness tests for torch.ops.aten.linalg_cholesky_ex.

The oracle is torch.ops.aten.linalg_cholesky_ex and the candidate is called
directly as flag_gems.linalg_cholesky_ex; the .L overload is exercised through
that same public name. Every case builds its own input, builds an independent
reference input, calls both, and compares the returned (L, info) tuple in full.

Coverage: the five value ranges over the spec shape levels square-adapted for
this operator, a normalized positive-definite success family, a literal extreme
boundary family, both upper paths, check_errors, non-contiguous layouts, the
callable .L out-overload, deterministic failing pivots, NaN/Inf placements,
backward and the rejection cases.

Two spec dimensions are exempted with their mechanism: broadcast, because a
Cholesky factorization takes a single square operand and therefore has no
second operand to broadcast; and tensor/scalar, because upper and check_errors
are schema flags with defaults, covered by the parameter-combination and
default-argument families instead of a scalar operand.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Probed on the active device with the real overload: the native kernel
# selection implements float32, float64, complex64 and complex128, and rejects
# float16, bfloat16, float8 and the integer types through kernel selection.
# float64 and complex128 additionally need the device FP64 capability, so they
# are gated on the static capability flag rather than probed here.
_DTYPES = [torch.float32]
if utils.fp64_is_supported:
    _DTYPES += [torch.float64, torch.complex128]
_DTYPES.append(torch.complex64)

_COMPLEX_DTYPES = [dtype for dtype in _DTYPES if dtype.is_complex]

# A Cholesky factor exists only for a square matrix, so the spec shape levels
# are adapted by rank: the rank 0 and rank 1 levels carry no square matrix and
# are replaced by the empty-square, zero-batch and n == 1 boundaries below,
# while the higher-rank levels keep their rank and scale to a square trailing
# pair.
_GRID_SHAPES = [
    (0, 0),
    (0, 3, 3),
    (1, 1),
    (2, 1, 1),
    (8, 8),
    (64, 64),
    (256, 256),
    (1024, 1024),
    (20, 320, 320),
    (16, 128, 64, 64),
    (16, 7, 57, 32, 32),
]


def _range_input(dtype, shape, value_range):
    """Value-range sample; an empty shape has no entries to generate."""
    if 0 in shape:
        return torch.empty(shape, dtype=dtype, device=flag_gems.device)
    return tu.make_input(dtype, shape, value_range)


def _pd_input(dtype, shape, value_range):
    """Definite input built from range-sampled entries.

    The range symbols supply the scale (never a runtime maximum), so extreme
    bounds cannot overflow the product. The product is positive semi-definite
    and (n + 1) * I lifts every eigenvalue, so info == 0 and the whole returned
    factor is defined. This is the normalized PD family; the raw grid keeps the
    literal ranges.
    """
    n = shape[-1]
    if n == 0:
        return torch.empty(shape, dtype=dtype, device=flag_gems.device)
    low = abs(tu.resolve_bound(value_range[0], dtype))
    high = abs(tu.resolve_bound(value_range[1], dtype))
    scale = max(low, high, 1.0)
    sampled = _range_input(dtype, shape, value_range) / scale
    gram = sampled @ sampled.conj().transpose(-2, -1)
    eye = torch.eye(n, dtype=dtype, device=flag_gems.device)
    return gram + (n + 1) * eye


def _extreme_pd_input(dtype, symbol):
    """Literal extreme-magnitude definite matrix.

    Measured on the active backend: the max-anchored diagonal
    [max, max/2, 1, 1] returns info 0 and a finite factor for float32,
    float64, complex64 and complex128. The min-anchored counterpart is not a
    definite matrix at all (measured info 1 with a non-finite factor), so this
    literal boundary family is anchored on the positive bound only.
    """
    bound = tu.resolve_bound(symbol, dtype)
    values = [bound, bound / 2, 1.0, 1.0]
    diag = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    return torch.diag(diag)


def _non_pd_input(dtype, n, kind):
    """Deterministic non-definite input.

    A definite matrix is built first and a single diagonal entry is negated,
    which fixes the failing minor by construction (1 for the first entry, n for
    the last) instead of leaving it to the sampled values.
    """
    inp = _pd_input(dtype, (n, n), ["-1", "1"])
    if kind == "first-pivot":
        inp[0, 0] = -1
    elif kind == "last-pivot":
        inp[n - 1, n - 1] = -1
    else:
        late = inp.clone()
        late[n - 1, n - 1] = -1
        early = inp.clone()
        early[0, 0] = -1
        inp = torch.stack((inp, late, early))
    return inp


def _layout_input(dtype, kind, n=16):
    """Definite input returned as a non-contiguous view of a padded buffer.

    stride gives a row stride of n + 4, offset starts one padded row into the
    buffer so the view also carries a non-zero storage offset, and batch pads
    the batch dimension so the leading stride is not the tight one.
    """
    if kind == "batch":
        padded = torch.empty(4, n, n + 4, dtype=dtype, device=flag_gems.device)
        padded[:, :, :n] = _pd_input(dtype, (4, n, n), ["-1", "1"])
        return padded[:, :, :n]
    if kind == "offset":
        padded = torch.empty(n + 2, n + 4, dtype=dtype, device=flag_gems.device)
        padded[1 : n + 1, :n] = _pd_input(dtype, (n, n), ["-1", "1"])
        return padded[1 : n + 1, :n]
    padded = torch.empty(n, n + 4, dtype=dtype, device=flag_gems.device)
    padded[:, :n] = _pd_input(dtype, (n, n), ["-1", "1"])
    return padded[:, :n]


def _payload_mask(placement, upper, n):
    """Off-diagonal cells of the requested payload region.

    Measured native geometry: the stored upper triangle (row < col) is the one
    read when upper is True and the stored lower triangle (row > col) when it
    is False; the opposite triangle is ignored. The diagonal is excluded here
    because it is always read and is covered by the diagonal placement.
    """
    row = torch.arange(n, device=flag_gems.device)[:, None]
    col = torch.arange(n, device=flag_gems.device)[None, :]
    selected = row < col if upper else row > col
    if placement == "selected-triangle":
        return selected
    return (row != col) & ~selected


def _payload_rows(dtype, scenario, n):
    """Per-row payload values: nan-only, inf-only, both together, or a finite
    asymmetric ramp."""
    if scenario == "asym":
        return 1e3 + torch.arange(n, device=flag_gems.device).to(dtype)
    payload = tu.make_special_input(dtype, scenario)
    step = 5 if scenario == "mixed" else 2
    return payload[torch.arange(n, device=flag_gems.device) % step]


def _special_input(dtype, placement, upper, scenario, n=6):
    """Input whose payload occupies one known region of the matrix.

    The placement makes the triangle geometry observable: a candidate reading
    the wrong triangle meets the payload where the native reference reads clean
    definite values (or the reverse), so its factor differs. The payload
    cardinality comes from n, never from a device reduction.
    """
    if placement == "dense":
        rows = _payload_rows(dtype, scenario, n).reshape(n, 1)
        return rows.expand(n, n).contiguous()
    if placement == "diagonal":
        return torch.diag(_payload_rows(dtype, scenario, n))
    inp = _pd_input(dtype, (n, n), ["-1", "1"])
    rows = _payload_rows(dtype, scenario, n).reshape(n, 1)
    return torch.where(_payload_mask(placement, upper, n), rows, inp)


def _buffer_metadata(buf):
    return (
        buf.data_ptr(),
        tuple(buf.shape),
        tuple(buf.stride()),
        buf.storage_offset(),
    )


def _out_buffers(shape, dtype, device, layout):
    """L / info buffers for the .L overload in three view layouts."""
    n = shape[-1]
    batch = tuple(shape[:-2])
    if layout == "plain":
        l_buf = torch.empty(shape, dtype=dtype, device=device)
    elif layout == "strided":
        padded = torch.empty(batch + (n, n + 4), dtype=dtype, device=device)
        l_buf = padded[..., :n]
    else:
        padded = torch.empty(batch + (n + 2, n), dtype=dtype, device=device)
        l_buf = padded[..., 1 : n + 1, :]
    total = 1
    for extent in batch:
        total *= extent
    if batch and layout != "plain":
        storage = torch.empty(
            2 * total if layout == "strided" else total + 1,
            dtype=torch.int32,
            device=device,
        )
        info = (storage[::2] if layout == "strided" else storage[1:]).reshape(batch)
    else:
        info = torch.empty(batch, dtype=torch.int32, device=device)
    return l_buf, info


_RAW_ROWS = [
    (shape, dtype, value_range)
    for dtype in _DTYPES
    for value_range in tu.REQUIRED_RANGES
    for shape in _GRID_SHAPES
]
# Quick mode keeps a rank-preserving square smoke shape: the spec quick shape
# (2, 19, 7) is adapted to (2, 7, 7) by squaring its trailing matrix axes while
# the batch extent stays 2.
_QUICK_ROWS = [((2, 7, 7), dtype, ["-1", "1"]) for dtype in _DTYPES]


@pytest.mark.linalg_cholesky_ex
@pytest.mark.parametrize(
    "shape,dtype,value_range", tu.selected_cases(_RAW_ROWS, quick=_QUICK_ROWS)
)
def test_linalg_cholesky_ex_raw_range(shape, dtype, value_range):
    # Range-sampled matrices are generally not definite, so the native call
    # reports a failing minor through info; the full returned tuple is compared
    # either way.
    inp = _range_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_l, ref_info = torch.ops.aten.linalg_cholesky_ex(ref_inp)
    res_l, res_info = flag_gems.linalg_cholesky_ex(inp)

    tu.assert_result_equal(res_info, ref_info)
    tu.assert_result_close(res_l, ref_l)


_PD_SHAPES = [(4, 4), (64, 64), (256, 256), (2, 16, 16)]
_PD_ROWS = [
    (shape, dtype, value_range)
    for dtype in _DTYPES
    for value_range in tu.REQUIRED_RANGES
    for shape in _PD_SHAPES
]


@pytest.mark.linalg_cholesky_ex
@pytest.mark.parametrize(
    "shape,dtype,value_range", tu.selected_cases(_PD_ROWS, quick=_QUICK_ROWS)
)
def test_linalg_cholesky_ex_pd_success(shape, dtype, value_range):
    # info == 0 by construction, so both triangles of the factor are defined.
    inp = _pd_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_l, ref_info = torch.ops.aten.linalg_cholesky_ex(ref_inp)
    res_l, res_info = flag_gems.linalg_cholesky_ex(inp)

    tu.assert_result_equal(res_info, ref_info)
    tu.assert_result_close(res_l, ref_l)


_EXTREME_ROWS = [
    (symbol, upper, dtype)
    for dtype in _DTYPES
    for upper in (False, True)
    for symbol in ("max", "max/2")
]


@pytest.mark.linalg_cholesky_ex
@pytest.mark.parametrize(
    "symbol,upper,dtype", tu.selected_cases(_EXTREME_ROWS, quick=[])
)
def test_linalg_cholesky_ex_extreme_pd(symbol, upper, dtype):
    inp = _extreme_pd_input(dtype, symbol)
    ref_inp = tu.to_reference(inp)

    ref_l, ref_info = torch.ops.aten.linalg_cholesky_ex(ref_inp, upper=upper)
    res_l, res_info = flag_gems.linalg_cholesky_ex(inp, upper=upper)

    tu.assert_result_equal(res_info, ref_info)
    tu.assert_result_close(res_l, ref_l)


_UPPER_SHAPES = [(8, 8), (64, 64), (2, 16, 16)]
_UPPER_ROWS = [
    (shape, dtype, value_range)
    for dtype in _DTYPES
    for value_range in tu.REQUIRED_RANGES
    for shape in _UPPER_SHAPES
]


@pytest.mark.linalg_cholesky_ex
@pytest.mark.parametrize(
    "shape,dtype,value_range", tu.selected_cases(_UPPER_ROWS, quick=[])
)
def test_linalg_cholesky_ex_upper(shape, dtype, value_range):
    # upper is True, so the native reads the stored upper triangle and returns U
    # with A = U^H U; the ignored triangle of the input must not reach it.
    inp = _range_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_l, ref_info = torch.ops.aten.linalg_cholesky_ex(ref_inp, upper=True)
    res_l, res_info = flag_gems.linalg_cholesky_ex(inp, upper=True)

    tu.assert_result_equal(res_info, ref_info)
    tu.assert_result_close(res_l, ref_l)


_PARAM_ROWS = [
    (upper, check_errors, dtype)
    for dtype in _DTYPES
    for upper in (False, True)
    for check_errors in (False, True)
]


@pytest.mark.linalg_cholesky_ex
@pytest.mark.parametrize(
    "upper,check_errors,dtype", tu.selected_cases(_PARAM_ROWS, quick=[])
)
def test_linalg_cholesky_ex_param_combos(upper, check_errors, dtype):
    # Parameter-coverage shape, square-adapted for this operator. The
    # supplement dtype list (bf16/fp16/int32/int64/int8/uint8/fp8) is rejected
    # by this kernel selection, so the probed supported dtypes carry both bool
    # parameters in every True/False combination.
    inp = _pd_input(dtype, (1024, 1024), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_l, ref_info = torch.ops.aten.linalg_cholesky_ex(
        ref_inp, upper=upper, check_errors=check_errors
    )
    res_l, res_info = flag_gems.linalg_cholesky_ex(
        inp, upper=upper, check_errors=check_errors
    )

    tu.assert_result_equal(res_info, ref_info)
    tu.assert_result_close(res_l, ref_l)


@pytest.mark.linalg_cholesky_ex
@pytest.mark.parametrize("dtype", tu.selected_cases(_DTYPES, quick=[]))
def test_linalg_cholesky_ex_default_arguments(dtype):
    # The schema defaults are upper=False and check_errors=False; omitting the
    # arguments checks that the candidate implements them, which passing the
    # same values explicitly does not.
    inp = _pd_input(dtype, (16, 16), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_l, ref_info = torch.ops.aten.linalg_cholesky_ex(ref_inp)
    res_l, res_info = flag_gems.linalg_cholesky_ex(inp)

    tu.assert_result_equal(res_info, ref_info)
    tu.assert_result_close(res_l, ref_l)


_LAYOUT_KINDS = ["stride", "offset", "batch"]
_LAYOUT_ROWS = [(kind, dtype) for dtype in _DTYPES for kind in _LAYOUT_KINDS]


@pytest.mark.linalg_cholesky_ex
@pytest.mark.parametrize("layout,dtype", tu.selected_cases(_LAYOUT_ROWS, quick=[]))
def test_linalg_cholesky_ex_layout(layout, dtype):
    inp = _layout_input(dtype, layout)
    ref_inp = tu.to_reference(inp)

    ref_l, ref_info = torch.ops.aten.linalg_cholesky_ex(ref_inp)
    res_l, res_info = flag_gems.linalg_cholesky_ex(inp)

    tu.assert_result_equal(res_info, ref_info)
    tu.assert_result_close(res_l, ref_l)


_OUT_SHAPES = [(16, 16), (2, 16, 16)]
_OUT_LAYOUTS = ["plain", "strided", "offset"]
_OUT_ROWS = [
    (shape, layout, upper, dtype)
    for dtype in _DTYPES
    for upper in (False, True)
    for layout in _OUT_LAYOUTS
    for shape in _OUT_SHAPES
]


@pytest.mark.linalg_cholesky_ex
@pytest.mark.parametrize(
    "shape,layout,upper,dtype", tu.selected_cases(_OUT_ROWS, quick=[])
)
def test_linalg_cholesky_ex_out_overload(shape, layout, upper, dtype):
    # The runtime schema is
    #   linalg_cholesky_ex.L(Tensor self, *, bool upper=False,
    #     bool check_errors=False, Tensor(a!) L, Tensor(b!) info)
    # so both write-out buffers are keyword-only and are named L and info.
    inp = _pd_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    # Reference buffers carry the reference dtype and live on the reference
    # tensor device; the candidate buffers use the candidate input device, so
    # nothing is compared against itself and a dtype or device mismatch cannot
    # pass unnoticed.
    ref_l, ref_info = _out_buffers(shape, ref_inp.dtype, ref_inp.device, layout)
    ref_out = torch.ops.aten.linalg_cholesky_ex.L(
        ref_inp, upper=upper, check_errors=False, L=ref_l, info=ref_info
    )

    res_l, res_info = _out_buffers(shape, inp.dtype, inp.device, layout)
    before = [_buffer_metadata(buf) for buf in (res_l, res_info)]
    res_out = flag_gems.linalg_cholesky_ex(
        inp, upper=upper, check_errors=False, L=res_l, info=res_info
    )

    # .L writes the buffers it was given and returns them, so both tuple
    # members must be those exact objects with unchanged layout metadata; a
    # correctly sized view must not be restrided.
    assert res_out[0] is res_l
    assert res_out[1] is res_info
    assert [_buffer_metadata(buf) for buf in res_out] == before
    assert res_out[0].device == inp.device
    assert res_out[1].device == inp.device
    tu.assert_result_equal(res_out[1], ref_out[1])
    tu.assert_result_close(res_out[0], ref_out[0])


_FAILURE_ROWS = [
    (kind, upper, dtype)
    for dtype in _DTYPES
    for kind in ("first-pivot", "last-pivot", "mixed-batch")
    for upper in (False, True)
]


@pytest.mark.linalg_cholesky_ex
@pytest.mark.parametrize("kind,upper,dtype", tu.selected_cases(_FAILURE_ROWS, quick=[]))
def test_linalg_cholesky_ex_failed_minor(kind, upper, dtype):
    # The negated diagonal entry fixes the failing minor: info is 1, 8 or
    # [0, 8, 1] depending on kind and is confirmed by the reference. The entries
    # past that minor are unspecified by the native contract yet are compared in
    # full here, so a divergence is visible instead of being masked away.
    inp = _non_pd_input(dtype, 8, kind)
    ref_inp = tu.to_reference(inp)

    ref_l, ref_info = torch.ops.aten.linalg_cholesky_ex(ref_inp, upper=upper)
    res_l, res_info = flag_gems.linalg_cholesky_ex(inp, upper=upper)

    tu.assert_result_equal(res_info, ref_info)
    tu.assert_result_close(res_l, ref_l)


_SPECIAL_PLACEMENTS = ["dense", "diagonal", "selected-triangle", "ignored-triangle"]
_SPECIAL_SCENARIOS = ["nan", "inf", "mixed", "asym"]
_SPECIAL_ROWS = [
    (placement, upper, dtype, scenario)
    for dtype in _DTYPES
    for scenario in _SPECIAL_SCENARIOS
    for placement in _SPECIAL_PLACEMENTS
    for upper in (False, True)
]


@pytest.mark.linalg_cholesky_ex
@pytest.mark.parametrize(
    "placement,upper,dtype,scenario", tu.selected_cases(_SPECIAL_ROWS, quick=[])
)
def test_linalg_cholesky_ex_special_values(placement, upper, dtype, scenario):
    # nan-only, inf-only, nan+inf-together and a finite asymmetric payload, each
    # placed in one known region. The ignored-triangle rows, in particular the
    # finite asymmetric one, fail a candidate that reads the wrong triangle.
    inp = _special_input(dtype, placement, upper, scenario)
    ref_inp = tu.to_reference(inp)

    ref_l, ref_info = torch.ops.aten.linalg_cholesky_ex(ref_inp, upper=upper)
    res_l, res_info = flag_gems.linalg_cholesky_ex(inp, upper=upper)

    tu.assert_result_equal(res_info, ref_info)
    tu.assert_result_close(res_l, ref_l)


@pytest.mark.linalg_cholesky_ex
@pytest.mark.parametrize("dtype", tu.selected_cases(_COMPLEX_DTYPES, quick=[]))
def test_linalg_cholesky_ex_complex_diagonal(dtype):
    # A Hermitian matrix has a real diagonal and the native kernel reads the
    # diagonal as real (measured info 0 with the imaginary part present), so an
    # imaginary diagonal perturbation must not change the returned factor.
    inp = _pd_input(dtype, (4, 4), ["-1", "1"])
    imag = torch.arange(1, 5, dtype=torch.float32, device=flag_gems.device) * 1j
    inp = inp + torch.diag(imag.to(dtype))
    ref_inp = tu.to_reference(inp)

    ref_l, ref_info = torch.ops.aten.linalg_cholesky_ex(ref_inp)
    res_l, res_info = flag_gems.linalg_cholesky_ex(inp)

    tu.assert_result_equal(res_info, ref_info)
    tu.assert_result_close(res_l, ref_l)


_BACKWARD_ROWS = [(upper, dtype) for dtype in _DTYPES for upper in (False, True)]


@pytest.mark.linalg_cholesky_ex
@pytest.mark.parametrize("upper,dtype", tu.selected_cases(_BACKWARD_ROWS, quick=[]))
def test_linalg_cholesky_ex_backward(upper, dtype):
    inp = _pd_input(dtype, (4, 4), ["-1", "1"])
    # Independent differentiable operands for the reference and the candidate,
    # and one non-constant upstream gradient of the same values on each side:
    # the candidate gradient is built on the candidate input device and the
    # reference gradient is its equivalent on the reference device, so neither
    # autograd call is handed a tensor from the other device.
    ref_inp = tu.to_reference(inp).detach().requires_grad_(True)
    res_inp = inp.detach().clone().requires_grad_(True)
    upstream = torch.randn(4, 4, dtype=dtype, device=flag_gems.device)
    ref_upstream = tu.to_reference(upstream)

    ref_l, ref_info = torch.ops.aten.linalg_cholesky_ex(ref_inp, upper=upper)
    res_l, res_info = flag_gems.linalg_cholesky_ex(res_inp, upper=upper)

    # The successful forward is compared before differentiating.
    tu.assert_result_equal(res_info, ref_info)
    tu.assert_result_close(res_l, ref_l)

    (ref_grad,) = torch.autograd.grad(ref_l, ref_inp, grad_outputs=ref_upstream)
    (res_grad,) = torch.autograd.grad(res_l, res_inp, grad_outputs=upstream)
    tu.assert_result_close(res_grad, ref_grad)


# Dtypes the native kernel selection rejects, measured on the CUDA/cuSOLVER
# backend: the fp8 pair reports 'cholesky_cusolver not implemented for
# Float8_e4m3fn / Float8_e5m2', half and bfloat16 report the same not
# implemented error, and bool, int8, uint8, int32 and int64 report
# 'linalg.cholesky: Expected a floating point or complex tensor as input'. The
# measurement is scoped to that vendor because another backend's kernel
# selection may well accept half or bfloat16. Construction is gated on the same
# static capability flags the shared helpers use - no runtime probe - because
# building a tensor of a dtype the device cannot represent would fail before the
# operator is reached.
_REJECTED_DTYPES = []
if flag_gems.vendor_name == "nvidia":
    _REJECTED_DTYPES = [torch.float16, torch.int32, torch.int8, torch.uint8, torch.bool]
    if utils.bf16_is_supported:
        _REJECTED_DTYPES.append(torch.bfloat16)
    if utils.int64_is_supported:
        _REJECTED_DTYPES.append(torch.int64)
    if utils.fp8_is_supported:
        _REJECTED_DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]


@pytest.mark.linalg_cholesky_ex
@pytest.mark.parametrize("dtype", _REJECTED_DTYPES)
def test_linalg_cholesky_ex_rejects_dtype(dtype):
    # tu.make_input constructs every dtype in _REJECTED_DTYPES on this device,
    # so the rejection below comes from the operator call, not the fixture.
    inp = tu.make_input(dtype, (4, 4), ["0", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.linalg_cholesky_ex(inp)


_REJECTED_SHAPES = [(), (8,), (4, 8), (2, 4, 8)]


@pytest.mark.linalg_cholesky_ex
@pytest.mark.parametrize("shape", _REJECTED_SHAPES)
def test_linalg_cholesky_ex_rejects_shape(shape):
    # Rank 0 and rank 1 carry no matrix and (4, 8) / (2, 4, 8) are not square.
    inp = torch.ones(shape, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.linalg_cholesky_ex(inp)


@pytest.mark.linalg_cholesky_ex
@pytest.mark.parametrize("kwargs", [{"upper": "x"}, {"check_errors": "y"}])
def test_linalg_cholesky_ex_rejects_param(kwargs):
    # The schema takes bools for both flags; a string does not convert, so the
    # candidate has to reject it as well.
    inp = _pd_input(torch.float32, (8, 8), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.linalg_cholesky_ex(inp, **kwargs)


_CHECK_ERROR_ROWS = [
    (kind, upper, dtype)
    for dtype in _DTYPES
    for upper in (False, True)
    for kind in ("first-pivot", "last-pivot")
]


@pytest.mark.linalg_cholesky_ex
@pytest.mark.parametrize("kind,upper,dtype", _CHECK_ERROR_ROWS)
def test_linalg_cholesky_ex_check_errors_raises(kind, upper, dtype):
    # check_errors=True raises instead of only reporting the failing minor
    # through info; measured for both failing pivots, both upper paths and all
    # supported dtypes.
    inp = _non_pd_input(dtype, 8, kind)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.linalg_cholesky_ex(inp, upper=upper, check_errors=True)
