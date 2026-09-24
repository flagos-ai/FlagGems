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

# linalg_inv only accepts floating point / complex input, requires rank >= 2
# with square trailing dimensions, and rejects singular matrices. float32 and
# complex64 are always available; float64 / complex128 follow the fp64 flag.
SUPPORTED_DTYPES = [torch.float32, torch.complex64]
if utils.fp64_is_supported:
    SUPPORTED_DTYPES += [torch.float64, torch.complex128]

_REAL_OF = {torch.complex64: torch.float32, torch.complex128: torch.float64}

# Descriptor-to-shape mapping for this operator:
#   * the 0-dim and 1-dim spec shapes use the smallest matrix the operator can
#     take, because every rank < 2 input is rejected;
#   * the higher-rank spec entries keep their batch axes and square the trailing
#     matrix dimensions:
#       (1024, 1024) unchanged;
#       (20, 320, 15) -> (20, 320, 320);
#       (16, 128, 64, 60) -> (16, 128, 64, 64);
#       (16, 7, 57, 32, 29) -> (16, 7, 57, 32, 32).
# The last two entries carry singleton batch axes.
_SPEC_SHAPES = [
    (1, 1),
    (256, 256),
    (1024, 1024),
    (20, 320, 320),
    (16, 128, 64, 64),
    (16, 7, 57, 32, 32),
    (1, 320, 320),
    (20, 1, 320, 320),
]

_GRID_SHAPES = tu.selected_cases(_SPEC_SHAPES, quick=[(2, 19, 19)])

# 'moderate' keeps a well-conditioned dense matrix for every range; 'diagonal'
# additionally keeps the requested signs and magnitudes.
_FAMILIES = ("moderate", "diagonal")

_MATRIX_ROWS = [
    (family, tuple(shape), tuple(value_range))
    for family in _FAMILIES
    for shape in _GRID_SHAPES
    for value_range in tu.selected_ranges()
]

# Layout and boundary geometries the native kernel accepts; default-only because
# they add no value-range information.
_BOUNDARY_ROWS = tu.selected_cases(
    [
        ("transpose", (2, 32, 32), ("-1", "1")),
        ("slice_offset", (32, 32), ("-1", "1")),
        ("moderate", (0, 0), ("-1", "1")),
        ("moderate", (0, 4, 4), ("-1", "1")),
        ("moderate", (3, 0, 0), ("-1", "1")),
        ("moderate", (1, 1), ("-1", "1")),
    ],
    quick=[],
)

_CASES = _MATRIX_ROWS + _BOUNDARY_ROWS


def _make_matrix(dtype, shape, value_range, family):
    """Square matrix of ``shape`` for one construction family.

    'moderate' scales the drawn values to unit size and shifts them by
    (n + 0.5) * I, so the reference stays well conditioned for every range.
    'diagonal' keeps the requested magnitudes and signs: the diagonal values
    span the range with zero excluded (that is what keeps the matrix
    invertible), and complex diagonals are drawn in the matching real dtype
    first because an extreme-range complex diagonal is natively singular.
    'transpose' / 'slice_offset' reuse the moderate values through a transposed
    or storage-offset strided view.
    """
    if any(size == 0 for size in shape):
        return torch.empty(shape, dtype=dtype, device=flag_gems.device)
    n = shape[-1]
    if family == "diagonal":
        real = _REAL_OF.get(dtype, dtype)
        values = torch.testing.make_tensor(
            tuple(shape[:-1]),
            dtype=real,
            device=flag_gems.device,
            low=tu.resolve_bound(value_range[0], real),
            high=tu.resolve_bound(value_range[1], real),
            exclude_zero=True,
        )
        return torch.diag_embed(values.to(dtype))
    if family == "transpose":
        return _make_matrix(dtype, shape, value_range, "moderate").transpose(-1, -2)
    if family == "slice_offset":
        rows, cols = shape[-2:]
        allocation = torch.empty(
            tuple(shape[:-2]) + (2 * rows + 3, 2 * cols + 3),
            dtype=dtype,
            device=flag_gems.device,
        )
        view = allocation[..., 1 : 1 + rows, 2 : 2 + cols]
        view.copy_(
            _make_matrix(
                dtype, tuple(shape[:-2]) + (rows, cols), value_range, "moderate"
            )
        )
        return view
    moderate = tu.make_input(dtype, shape, value_range)
    scale = moderate.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1)
    eye = torch.eye(n, dtype=dtype, device=flag_gems.device)
    return moderate / scale + (n + 0.5) * eye


@pytest.mark.linalg_inv
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("case", _CASES)
def test_linalg_inv_matches_reference(dtype, case):
    family, shape, value_range = case
    inp = _make_matrix(dtype, shape, value_range, family)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.linalg_inv(ref_inp)
    res_out = flag_gems.linalg_inv(inp)

    # The shared assertion transfers to the CPU, so the output device contract
    # is checked here against the input.
    assert res_out.device == inp.device
    tu.assert_result_close(res_out, ref_out)


_OUT_CASES = tu.selected_cases(
    [
        ("contiguous", (3, 16, 16)),
        ("strided", (2, 16, 16)),
        ("offset", (2, 16, 16)),
        ("transposed", (16, 16)),
    ],
    quick=[],
)


def _out_buffer(layout, shape, dtype):
    device = flag_gems.device
    if layout == "contiguous":
        return torch.empty(shape, dtype=dtype, device=device)
    if layout == "strided":
        # Every other row and column of a larger allocation.
        rows, cols = shape[-2:]
        allocation = torch.empty(
            tuple(shape[:-2]) + (2 * rows, 2 * cols), dtype=dtype, device=device
        )
        return allocation[..., ::2, ::2]
    if layout == "offset":
        # Contiguous rows at a nonzero storage offset.
        allocation = torch.empty(
            torch.Size(shape).numel() + 5, dtype=dtype, device=device
        )
        return allocation[5:].view(shape)
    if layout == "transposed":
        return torch.empty(tuple(reversed(shape)), dtype=dtype, device=device).t()
    raise ValueError(f"unknown out-buffer layout {layout}")


@pytest.mark.linalg_inv
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("case", _OUT_CASES)
def test_linalg_inv_out_overload(dtype, case):
    layout, shape = case
    inp = _make_matrix(dtype, shape, ("-1", "1"), "moderate")
    ref_inp = tu.to_reference(inp)
    ref_buf = torch.empty(shape, dtype=ref_inp.dtype, device=ref_inp.device)
    ref_out = torch.ops.aten.linalg_inv.out(ref_inp, out=ref_buf)

    out = _out_buffer(layout, shape, dtype)
    stride_before = tuple(out.stride())
    offset_before = out.storage_offset()
    res_out = flag_gems.linalg_inv(inp, out=out)

    # The candidate must write into the supplied buffer, not reallocate or
    # restride it.
    assert res_out is out
    assert tuple(out.stride()) == stride_before
    assert out.storage_offset() == offset_before
    tu.assert_result_close(res_out, ref_out)


_BACKWARD_SHAPES = tu.selected_cases([(5, 5), (2, 5, 5), (2, 3, 5, 5)], quick=[])


@pytest.mark.linalg_inv
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("shape", _BACKWARD_SHAPES)
def test_linalg_inv_backward(dtype, shape):
    inp = _make_matrix(dtype, shape, ("-1", "1"), "moderate").requires_grad_(True)
    ref_inp = tu.to_reference(inp)
    upstream = tu.make_input(dtype, shape, ("-1", "1"))

    ref_out = torch.ops.aten.linalg_inv(ref_inp)
    res_out = flag_gems.linalg_inv(inp)
    tu.assert_result_close(res_out, ref_out)

    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, tu.to_reference(upstream))
    (res_grad,) = torch.autograd.grad(res_out, inp, upstream)
    tu.assert_result_close(res_grad, ref_grad)


# Non-finite inputs: whether the native kernel rejects the input or returns an
# inverse with propagated nan / inf depends on the dtype, the scenario and the
# placement of the payload. The split below was measured on the nvidia CUDA
# backend (the payload sits on the diagonal of a 5 x 5 matrix, in row 0, in
# column 0, or on the diagonal of the second lane of a 2-element batch), so the
# two case lists are static and nothing is decided at run time. Both lists are
# derived from the same capsule, so on a vendor where this behaviour was not
# measured they are empty and no case is collected in either mode.
_MEASURED_VENDOR = flag_gems.runtime.device.vendor_name == "nvidia"

_SPECIAL_SCENARIOS = ("nan", "inf", "mixed")
_SPECIAL_PLACEMENTS = ("diag", "row", "col", "lane")

# Combinations the native kernel computes a result for. The nan entries of the
# batched-lane placement propagate a nan into the result.
_SPECIAL_RETURNED = {
    (torch.float32, "inf", "diag"),
    (torch.float32, "nan", "lane"),
    (torch.float32, "inf", "lane"),
    (torch.float32, "mixed", "lane"),
    (torch.float64, "inf", "diag"),
    (torch.float64, "nan", "lane"),
    (torch.float64, "inf", "lane"),
    (torch.float64, "mixed", "lane"),
    (torch.complex64, "nan", "lane"),
    (torch.complex64, "inf", "lane"),
    (torch.complex64, "mixed", "lane"),
    (torch.complex128, "nan", "lane"),
    (torch.complex128, "inf", "lane"),
    (torch.complex128, "mixed", "lane"),
}

_ALL_SPECIAL_CASES = (
    [
        (dtype, scenario, placement)
        for dtype in SUPPORTED_DTYPES
        for scenario in _SPECIAL_SCENARIOS
        for placement in _SPECIAL_PLACEMENTS
    ]
    if _MEASURED_VENDOR
    else []
)

_SPECIAL_RETURN_CASES = [
    case for case in _ALL_SPECIAL_CASES if case in _SPECIAL_RETURNED
]
_SPECIAL_RAISE_CASES = [
    case for case in _ALL_SPECIAL_CASES if case not in _SPECIAL_RETURNED
]

_SPECIAL_BASE = 5.5


def _special_matrix(dtype, scenario, placement):
    payload = tu.make_special_input(dtype, scenario)
    size = payload.numel()
    base = torch.eye(size, dtype=dtype, device=flag_gems.device) * _SPECIAL_BASE
    if placement == "lane":
        lane = base.clone()
        lane.diagonal().add_(payload)
        return torch.stack([base, lane])
    if placement == "diag":
        base.diagonal().add_(payload)
        return base
    if placement == "row":
        base[0, :] = payload
        return base
    base[:, 0] = payload
    return base


# Positive specials change the expected result classification, so they stay
# default-only.
@pytest.mark.linalg_inv
@pytest.mark.parametrize(
    "dtype,scenario,placement", tu.selected_cases(_SPECIAL_RETURN_CASES, quick=[])
)
def test_linalg_inv_nonfinite_input(dtype, scenario, placement):
    inp = _special_matrix(dtype, scenario, placement)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.linalg_inv(ref_inp)
    res_out = flag_gems.linalg_inv(inp)

    tu.assert_result_close(res_out, ref_out)


# The rejection list is statically vendor / dtype gated, so it is used verbatim
# in quick as well: it keeps the measured rejections in quick on the measured
# vendor and stays empty elsewhere, instead of introducing an unconditional
# vendor-specific expectation.
@pytest.mark.linalg_inv
@pytest.mark.parametrize(
    "dtype,scenario,placement",
    tu.selected_cases(_SPECIAL_RAISE_CASES, quick=_SPECIAL_RAISE_CASES),
)
def test_linalg_inv_nonfinite_input_rejected(dtype, scenario, placement):
    inp = _special_matrix(dtype, scenario, placement)
    with pytest.raises(RuntimeError):
        flag_gems.linalg_inv(inp)


# Native dtype support measured on the active backend: only floating point and
# complex input is accepted ('Expected a floating point or complex tensor as
# input' for the integer / bool dtypes), and the measured CUDA backend rejects
# the low precision types ('Low precision dtypes not supported'). The low
# precision rejection is a property of that backend, so it is scoped to it,
# while the dtype-kind rejection follows the operator's own signature.
_INTEGER_DTYPES = [
    pytest.param(torch.int8, id="int8"),
    pytest.param(torch.uint8, id="uint8"),
    pytest.param(torch.int32, id="int32"),
    pytest.param(torch.bool, id="bool"),
]
if utils.int64_is_supported:
    _INTEGER_DTYPES.append(pytest.param(torch.int64, id="int64"))

_LOW_PRECISION_DTYPES = []
if _MEASURED_VENDOR:
    _LOW_PRECISION_DTYPES.append(pytest.param(torch.float16, id="float16"))
    if utils.bf16_is_supported:
        _LOW_PRECISION_DTYPES.append(pytest.param(torch.bfloat16, id="bfloat16"))
    if utils.fp8_is_supported:
        _LOW_PRECISION_DTYPES += [
            pytest.param(torch.float8_e4m3fn, id="float8_e4m3fn"),
            pytest.param(torch.float8_e5m2, id="float8_e5m2"),
        ]


@pytest.mark.linalg_inv
@pytest.mark.parametrize("dtype", _INTEGER_DTYPES + _LOW_PRECISION_DTYPES)
def test_linalg_inv_unsupported_dtype(dtype):
    inp = torch.ones((4, 4), dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.linalg_inv(inp)


@pytest.mark.linalg_inv
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((), id="rank0"),
        pytest.param((256,), id="rank1"),
        pytest.param((3, 4), id="nonsquare"),
        pytest.param((2, 3, 5), id="nonsquare_batch"),
    ],
)
def test_linalg_inv_invalid_shape(shape):
    inp = torch.ones(shape, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.linalg_inv(inp)


_SINGULAR_CASES = [
    pytest.param("zeros", (4, 4), id="zeros"),
    pytest.param("ones", (3, 3), id="ones"),
    pytest.param("duplicate_rows", (3, 3), id="duplicate_rows"),
    pytest.param("batch_with_singular", (2, 4, 4), id="batch_with_singular"),
]


def _singular_matrix(kind, shape):
    n = shape[-1]
    device = flag_gems.device
    if kind == "zeros":
        return torch.zeros(shape, dtype=torch.float32, device=device)
    if kind == "ones":
        return torch.ones(shape, dtype=torch.float32, device=device)
    if kind == "duplicate_rows":
        matrix = torch.eye(n, dtype=torch.float32, device=device) * 2
        matrix[-1] = matrix[0]
        return matrix
    # A batch whose second element is exactly singular.
    eye = torch.eye(n, dtype=torch.float32, device=device) * 2
    return torch.stack([eye, torch.zeros_like(eye)])


@pytest.mark.linalg_inv
@pytest.mark.parametrize("kind,shape", _SINGULAR_CASES)
def test_linalg_inv_singular_input(kind, shape):
    inp = _singular_matrix(kind, shape)
    with pytest.raises(RuntimeError):
        flag_gems.linalg_inv(inp)
