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

# aten::linalg_eigvalsh(Tensor self, str UPLO="L") -> Tensor
# aten::linalg_eigvalsh.out(Tensor self, str UPLO="L", *, Tensor(a!) out) -> Tensor(a!)
#
# Native contract behind the static choices below (generation-time probes; the
# delivered tests never probe native support at collection or run time):
#   * float32/float64 and complex64/complex128 are implemented and return the
#     matching real dtype; every other dtype is rejected.
#   * rank >= 2 with a square trailing pair; UPLO is one of "L"/"l"/"U"/"u".
#   * only the triangle UPLO names is read, eigenvalues come back ascending, and
#     .out returns the buffer it was given with its strides and offset intact.
#   * every observed rejection is a RuntimeError.
#
# linalg_eigvalsh has a single tensor operand, so the spec's broadcast dimension
# does not apply.

LINALG_EIGVALSH_DTYPES = [torch.float32, torch.complex64]
if utils.fp64_is_supported:
    LINALG_EIGVALSH_DTYPES += [torch.float64, torch.complex128]

# A complex input still yields real eigenvalues.
_REAL_DTYPES = {torch.complex64: torch.float32, torch.complex128: torch.float64}


def _real_dtype(dtype):
    return _REAL_DTYPES.get(dtype, dtype)


# eigvalsh needs rank >= 2 with a square trailing pair, so the spec's 0-dim and
# 1-dim levels can only appear as the negative cases below. The rectangular spec
# levels keep their batch axes and square the trailing pair at a real trailing
# side: (20, 320, 320), (16, 128, 64, 64) and (16, 7, 57, 32, 32); (1024, 1024)
# is already square. 2-D and batched inputs are separate solver paths, so both
# appear at several sizes, with the empty and single-element boundaries.
LINALG_EIGVALSH_SHAPES = [
    (0, 0),  # empty matrix
    (1, 1),  # single eigenvalue
    (2, 2),
    (5, 5),
    (16, 16),
    (64, 64),
    (256, 256),
    (1024, 1024),  # spec 2-D level
    (0, 5, 5),  # empty batch
    (2, 1, 1),
    (2, 19, 19),
    (4, 8, 8),
    (20, 320, 320),  # spec 3-D level
    (2, 3, 8, 8),
    (16, 128, 64, 64),  # spec 4-D level
    (2, 3, 1, 8, 8),
    (16, 7, 57, 32, 32),  # spec 5-D level
]

# Quick mode uses the spec's quick geometry with a square trailing pair.
_QUICK_SHAPES = [(2, 19, 19)]

# Unresolved value-oracle gap, recorded locally and not reported as covered:
# these eight raw float32 descriptor/range draws make the native solver raise
# torch._C._LinAlgError, so they have no value oracle. Withheld only on the
# vendor where that was measured.
_MEASURED_UNRESOLVED_ROWS = [
    (shape, value_range)
    for shape in [(64, 64), (256, 256), (20, 320, 320), (16, 128, 64, 64)]
    for value_range in (["0", "max"], ["min", "0"])
]

_UNRESOLVED_VALUE_ROWS = (
    frozenset(
        (tuple(shape), tuple(value_range))
        for shape, value_range in _MEASURED_UNRESOLVED_ROWS
    )
    if flag_gems.vendor_name == "nvidia"
    else frozenset()
)

VALUE_CASES = tu.selected_cases(
    [
        (shape, value_range, dtype)
        for dtype in LINALG_EIGVALSH_DTYPES
        for shape in LINALG_EIGVALSH_SHAPES
        for value_range in tu.selected_ranges()
        if dtype is not torch.float32
        or (tuple(shape), tuple(value_range)) not in _UNRESOLVED_VALUE_ROWS
    ],
    quick=[
        (shape, ["-1", "1"], dtype)
        for dtype in LINALG_EIGVALSH_DTYPES
        for shape in _QUICK_SHAPES
    ],
)


@pytest.mark.linalg_eigvalsh
@pytest.mark.parametrize("shape,value_range,dtype", VALUE_CASES)
def test_linalg_eigvalsh_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.linalg_eigvalsh(ref_inp, "L")
    res_out = flag_gems.linalg_eigvalsh(inp, "L")

    tu.assert_result_close(res_out, ref_out)


# Endpoint fixtures, additional to the withheld raw draws above. Each is a
# genuine diagonal matrix (torch.diag_embed of a diagonal of shape[:-1]), so the
# off-diagonal entries are exactly zero and the spectrum is the diagonal:
#   diagonal_repeat    every diagonal entry at the dtype endpoint
#   diagonal_one_half  one endpoint, remaining diagonal 0.5
#   diagonal_one_zero  one endpoint, remaining diagonal 0
#   dense_one_half     dense 0.5 with one endpoint on the diagonal
# The repeated-endpoint float32 rows at the four large sides are withheld pending
# evidence: the native solver returns a non-finite result there for this finite
# input without raising (generation-time audit: 64/256/6400/131072 NaN,
# repeatable). That is an oracle limitation on an unresolved combination, never
# an expected candidate result, so those rows are excluded locally for this
# vendor only; the smaller sides stay and no other dtype is excluded.
_DIAGONAL_ENDPOINT_SHAPES = [
    (64, 64),
    (256, 256),
    (20, 320, 320),
    (16, 128, 64, 64),
]

# Additive smaller sides for the repeated-endpoint fixture only.
_DIAGONAL_ENDPOINT_SMALL_SHAPES = [(4, 4), (16, 16), (2, 3, 8, 8), (2, 3, 1, 8, 8)]

_ENDPOINT_SHAPE_FIXTURES = [
    (shape, fixture)
    for shape in _DIAGONAL_ENDPOINT_SHAPES
    for fixture in ("diagonal_one_half", "diagonal_one_zero", "dense_one_half")
] + [
    (shape, "diagonal_repeat")
    for shape in _DIAGONAL_ENDPOINT_SHAPES + _DIAGONAL_ENDPOINT_SMALL_SHAPES
]

# The withheld combinations, pending evidence (see the comment above).
_REPEATED_ENDPOINT_FLOAT32_ORACLE_GAP = (
    frozenset((tuple(shape), "diagonal_repeat") for shape in _DIAGONAL_ENDPOINT_SHAPES)
    if flag_gems.vendor_name == "nvidia"
    else frozenset()
)

_ENDPOINT_CASES = tu.selected_cases(
    [
        (shape, fixture, sign, dtype)
        for dtype in LINALG_EIGVALSH_DTYPES
        for sign in ("max", "min")
        for shape, fixture in _ENDPOINT_SHAPE_FIXTURES
        if not (
            dtype is torch.float32
            and (tuple(shape), fixture) in _REPEATED_ENDPOINT_FLOAT32_ORACLE_GAP
        )
    ],
    quick=[],
)


def _endpoint_matrix(dtype, shape, sign, fixture):
    """Endpoint fixture of exactly ``shape``; see the fixture list above."""
    if fixture == "dense_one_half":
        mat = torch.full(shape, 0.5, dtype=dtype, device=flag_gems.device)
        mat[..., 0, 0] = _endpoint_bound(dtype, sign)
        return mat
    if fixture == "diagonal_repeat":
        diag = torch.full(
            shape[:-1],
            _endpoint_bound(dtype, sign),
            dtype=dtype,
            device=flag_gems.device,
        )
    else:
        rest = 0.5 if fixture == "diagonal_one_half" else 0.0
        diag = torch.full(shape[:-1], rest, dtype=dtype, device=flag_gems.device)
        diag[..., 0] = _endpoint_bound(dtype, sign)
    return torch.diag_embed(diag)


def _endpoint_bound(dtype, sign):
    finfo = torch.finfo(_real_dtype(dtype))
    return finfo.max if sign == "max" else finfo.min


@pytest.mark.linalg_eigvalsh
@pytest.mark.parametrize("shape,fixture,sign,dtype", _ENDPOINT_CASES)
def test_linalg_eigvalsh_endpoint(shape, fixture, sign, dtype):
    inp = _endpoint_matrix(dtype, shape, sign, fixture)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.linalg_eigvalsh(ref_inp, "L")
    res_out = flag_gems.linalg_eigvalsh(inp, "L")

    tu.assert_result_close(res_out, ref_out)


# Deterministic magnitude ladder: extra coverage only, not a substitute for the
# endpoint families or for the withheld raw draws.
_EXTREME_MAGNITUDE_SHAPES = [(64, 64), (256, 256), (20, 320, 320), (16, 128, 64, 64)]
_EXTREME_MAGNITUDE_SCALES = [1e20, 1e34]


def _extremal_hermitian(dtype, shape, scale):
    """Deterministic Hermitian matrix whose entries have magnitude ``scale``."""
    real_dtype = _real_dtype(dtype)
    generator = torch.Generator(device=flag_gems.device).manual_seed(0)
    mat = torch.randn(
        shape, generator=generator, dtype=real_dtype, device=flag_gems.device
    )
    if dtype.is_complex:
        imag = torch.randn(
            shape, generator=generator, dtype=real_dtype, device=flag_gems.device
        )
        mat = torch.complex(mat, imag)
    return ((mat + mat.mH) / 2) * scale


_EXTREME_MAGNITUDE_CASES = tu.selected_cases(
    [
        (shape, scale, dtype)
        for dtype in LINALG_EIGVALSH_DTYPES
        for shape in _EXTREME_MAGNITUDE_SHAPES
        for scale in _EXTREME_MAGNITUDE_SCALES
    ],
    quick=[],
)


@pytest.mark.linalg_eigvalsh
@pytest.mark.parametrize("shape,scale,dtype", _EXTREME_MAGNITUDE_CASES)
def test_linalg_eigvalsh_extreme_magnitude(shape, scale, dtype):
    inp = _extremal_hermitian(dtype, shape, scale)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.linalg_eigvalsh(ref_inp, "L")
    res_out = flag_gems.linalg_eigvalsh(inp, "L")

    tu.assert_result_close(res_out, ref_out)


# UPLO selects the triangle that is read, so "L" and "U" hold different
# eigenvalues for a non-symmetric input; single letters are case-insensitive.
# The prescribed large square and batched scales are covered as well.
_UPLO_CASES = tu.selected_cases(
    [
        (shape, uplo)
        for shape in [(5, 5), (2, 7, 7), (256, 256), (16, 128, 64, 64)]
        for uplo in ("L", "U", "l", "u")
    ],
    quick=[],
)


@pytest.mark.linalg_eigvalsh
@pytest.mark.parametrize("shape,uplo", _UPLO_CASES)
@pytest.mark.parametrize("dtype", LINALG_EIGVALSH_DTYPES)
def test_linalg_eigvalsh_uplo(shape, uplo, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.linalg_eigvalsh(ref_inp, uplo)
    res_out = flag_gems.linalg_eigvalsh(inp, uplo)

    tu.assert_result_close(res_out, ref_out)


_DEFAULT_UPLO_SHAPES = tu.selected_cases([(5, 5)], quick=[])


@pytest.mark.linalg_eigvalsh
@pytest.mark.parametrize("shape", _DEFAULT_UPLO_SHAPES)
@pytest.mark.parametrize("dtype", LINALG_EIGVALSH_DTYPES)
def test_linalg_eigvalsh_default_uplo(shape, dtype):
    # UPLO is omitted, so the candidate has to implement the schema default "L".
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.linalg_eigvalsh(ref_inp)
    res_out = flag_gems.linalg_eigvalsh(inp)

    tu.assert_result_close(res_out, ref_out)


def _out_buffer(shape, dtype, layout, device):
    """Out buffer of exactly ``shape`` in one storage layout.

    ``contiguous`` is a fresh dense allocation; the other layouts are views of a
    wider parent. A wrong out shape would make the operator resize the buffer and
    invalidate the strides captured before the call.
    """
    if layout == "contiguous":
        return torch.empty(shape, dtype=dtype, device=device)
    if layout == "sliced":
        parent = torch.empty(
            tuple(shape[:-1]) + (2 * shape[-1],), dtype=dtype, device=device
        )
        return parent[..., ::2]
    if layout == "offset_sliced":
        # Last stride 2 with a non-zero storage offset.
        parent = torch.empty(
            tuple(shape[:-1]) + (2 * shape[-1],), dtype=dtype, device=device
        )
        return parent[..., 1::2]
    if len(shape) >= 2:
        parent = torch.empty(
            tuple(shape[:-2]) + (shape[-1], shape[-2]), dtype=dtype, device=device
        )
        return parent.transpose(-1, -2)
    # 1-D out: a slice of a wider, contiguous buffer.
    parent = torch.empty((shape[-1], shape[-1]), dtype=dtype, device=device)
    return parent.t()[:, 0]


_OUT_CASES = tu.selected_cases(
    [
        (shape, uplo, layout)
        for shape in [(5, 5), (2, 4, 4)]
        for uplo in ("L", "U")
        for layout in ("contiguous", "sliced", "offset_sliced", "transposed")
    ],
    quick=[],
)


@pytest.mark.linalg_eigvalsh
@pytest.mark.parametrize("shape,uplo,layout", _OUT_CASES)
@pytest.mark.parametrize("dtype", LINALG_EIGVALSH_DTYPES)
def test_linalg_eigvalsh_out(shape, uplo, layout, dtype):
    # .out writes into the supplied buffer and returns that same object with its
    # strides and storage offset untouched; each side gets its own buffer on its
    # own device.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    out_shape = shape[:-1]
    out_dtype = _real_dtype(dtype)

    ref_out = _out_buffer(out_shape, out_dtype, layout, ref_inp.device)
    torch.ops.aten.linalg_eigvalsh.out(ref_inp, uplo, out=ref_out)

    res_out = _out_buffer(out_shape, out_dtype, layout, inp.device)
    supplied_stride = res_out.stride()
    supplied_offset = res_out.storage_offset()
    res_returned = flag_gems.linalg_eigvalsh(inp, uplo, out=res_out)

    assert res_returned is res_out
    assert res_out.stride() == supplied_stride
    assert res_out.storage_offset() == supplied_offset
    tu.assert_result_close(res_out, ref_out)


_INPUT_LAYOUT_CASES = tu.selected_cases(
    [
        (shape, layout)
        for shape in [(16, 16), (2, 8, 8), (4, 8, 8)]
        for layout in ("strided", "offset_strided", "transposed")
    ],
    quick=[],
)


def _strided_input(dtype, shape, layout):
    """Square matrix view of a larger backing buffer.

    Only the matrix dimensions are enlarged, so a view keeps exactly the shape
    named by the case: (2, 8, 8) is backed by (2, 16, 16), not by a (4, 16, 16)
    buffer that would make it a batch-4 workload; (4, 8, 8) is its own row.
    """
    backing = tuple(shape[:-2]) + (2 * shape[-2], 2 * shape[-1])
    base = tu.make_input(dtype, backing, ["-1", "1"])
    if layout == "strided":
        return base[..., ::2, ::2]
    if layout == "offset_strided":
        return base[..., 1::2, 1::2]
    return base[..., ::2, ::2].transpose(-1, -2)


@pytest.mark.linalg_eigvalsh
@pytest.mark.parametrize("shape,layout", _INPUT_LAYOUT_CASES)
@pytest.mark.parametrize("dtype", LINALG_EIGVALSH_DTYPES)
def test_linalg_eigvalsh_strided_input(shape, layout, dtype):
    # A square view with non-unit strides, a non-zero storage offset or a
    # transposed layout is a valid input, so the operator has to follow the
    # input strides instead of assuming a dense, offset-zero matrix.
    inp = _strided_input(dtype, shape, layout)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.linalg_eigvalsh(ref_inp, "L")
    res_out = flag_gems.linalg_eigvalsh(inp, "L")

    tu.assert_result_close(res_out, ref_out)


_SPECIAL_CASES = tu.selected_cases(
    # The shared generator covers floating dtypes only, so the supported complex
    # dtypes are added explicitly; the native result is the oracle.
    tu.special_value_cases(LINALG_EIGVALSH_DTYPES)
    + [
        (dtype, scenario)
        for dtype in LINALG_EIGVALSH_DTYPES
        if dtype.is_complex
        for scenario in ("nan", "inf", "mixed")
    ],
    quick=[],
)


@pytest.mark.linalg_eigvalsh
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_linalg_eigvalsh_special_values(dtype, scenario):
    # Diagonal embedding of the shared payloads, with the native result as oracle.
    inp = torch.diag_embed(tu.make_special_input(dtype, scenario))
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.linalg_eigvalsh(ref_inp, "L")
    res_out = flag_gems.linalg_eigvalsh(inp, "L")

    tu.assert_result_close(res_out, ref_out)


_TRIANGLE_SIDE = 4


def _one_triangle_matrix(dtype, uplo, payload, payload_triangle):
    """Finite, distinct payloads in each triangle (0.25 below, 0.5 above, 1..side
    on the diagonal) with ``payload`` written into the triangle UPLO selects
    ("selected") or into the other one ("ignored").
    """
    device = flag_gems.device
    side = _TRIANGLE_SIDE
    mat = torch.zeros((side, side), dtype=dtype, device=device)
    square = torch.ones((side, side), dtype=torch.bool, device=device)
    lower = torch.tril(square, -1)
    upper = torch.triu(square, 1)
    mat[lower], mat[upper] = 0.25, 0.5
    mat.diagonal().copy_(
        torch.tensor([float(i) for i in range(1, side + 1)], dtype=dtype, device=device)
    )
    is_lower = uplo.upper() == "L"
    if payload_triangle == "selected":
        target = lower if is_lower else upper
    else:
        target = upper if is_lower else lower
    mat[target] = payload
    return mat


# nan or inf in the triangle UPLO ignores leaves the finite spectrum unchanged;
# inf inside the selected triangle yields non-finite eigenvalues.
_TRIANGLE_PAYLOAD_CASES = tu.selected_cases(
    [
        (dtype, uplo, payload, payload_triangle)
        for dtype in LINALG_EIGVALSH_DTYPES
        for uplo in ("L", "U")
        for payload, payload_triangle in (
            (float("nan"), "ignored"),
            (float("inf"), "ignored"),
            (float("inf"), "selected"),
        )
    ],
    quick=[],
)


@pytest.mark.linalg_eigvalsh
@pytest.mark.parametrize("dtype,uplo,payload,payload_triangle", _TRIANGLE_PAYLOAD_CASES)
def test_linalg_eigvalsh_triangle_payload(dtype, uplo, payload, payload_triangle):
    inp = _one_triangle_matrix(dtype, uplo, payload, payload_triangle)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.linalg_eigvalsh(ref_inp, uplo)
    res_out = flag_gems.linalg_eigvalsh(inp, uplo)

    tu.assert_result_close(res_out, ref_out)


# Native rejection, distinct from the finite-input oracle gap above: a nan inside
# the triangle UPLO names makes the solver raise torch._C._LinAlgError (a
# RuntimeError) for every supported dtype and both UPLO values, while a nan on
# the diagonal still returns a result. Measured on this vendor, and kept in quick
# mode like the other negatives.
_NAN_SELECTED_TRIANGLE_ROWS = (
    [(dtype, uplo) for dtype in LINALG_EIGVALSH_DTYPES for uplo in ("L", "U")]
    if flag_gems.vendor_name == "nvidia"
    else []
)
_NAN_SELECTED_TRIANGLE_CASES = tu.selected_cases(
    _NAN_SELECTED_TRIANGLE_ROWS, quick=_NAN_SELECTED_TRIANGLE_ROWS
)


@pytest.mark.linalg_eigvalsh
@pytest.mark.parametrize("dtype,uplo", _NAN_SELECTED_TRIANGLE_CASES)
def test_linalg_eigvalsh_nan_in_selected_triangle(dtype, uplo):
    inp = _one_triangle_matrix(dtype, uplo, float("nan"), "selected")

    with pytest.raises(RuntimeError):
        flag_gems.linalg_eigvalsh(inp, uplo)


# UPLO selects the triangle that is read, so the two values give different
# eigenvalues. The diagonal carries a non-zero imaginary part, so the matrix is
# not Hermitian and the native result for the selected triangle is the oracle.
_COMPLEX_DIAGONAL_CASES = tu.selected_cases(
    [
        (dtype, uplo)
        for dtype in LINALG_EIGVALSH_DTYPES
        if dtype.is_complex
        for uplo in ("L", "U")
    ],
    quick=[],
)


@pytest.mark.linalg_eigvalsh
@pytest.mark.parametrize("dtype,uplo", _COMPLEX_DIAGONAL_CASES)
def test_linalg_eigvalsh_complex_diagonal(dtype, uplo):
    side = 4
    device = flag_gems.device
    mat = torch.zeros((side, side), dtype=dtype, device=device)
    square = torch.ones((side, side), dtype=torch.bool, device=device)
    mat[torch.tril(square, -1)] = 0.25 + 0.1j
    mat[torch.triu(square, 1)] = 0.5 + 0.2j
    mat.diagonal().copy_(
        torch.tensor([1 + 5j, 2 + 7j, 3 + 9j, 4 + 11j], dtype=dtype, device=device)
    )
    ref_inp = tu.to_reference(mat)

    ref_out = torch.ops.aten.linalg_eigvalsh(ref_inp, uplo)
    res_out = flag_gems.linalg_eigvalsh(mat, uplo)

    tu.assert_result_close(res_out, ref_out)


_BACKWARD_CASES = tu.selected_cases(
    [(shape, uplo) for shape in [(4, 4), (2, 7, 7)] for uplo in ("L", "U")],
    quick=[],
)


@pytest.mark.linalg_eigvalsh
@pytest.mark.parametrize("shape,uplo", _BACKWARD_CASES)
@pytest.mark.parametrize("dtype", tu.selected_cases(LINALG_EIGVALSH_DTYPES, quick=[]))
def test_linalg_eigvalsh_backward(shape, uplo, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    ref_inp = tu.to_reference(inp)
    # eigvalsh returns real eigenvalues, so the upstream gradient is real even
    # for a complex input.
    upstream = tu.make_input(_real_dtype(dtype), shape[:-1], ["-1", "1"])
    ref_upstream = tu.to_reference(upstream)

    ref_out = torch.ops.aten.linalg_eigvalsh(ref_inp, uplo)
    res_out = flag_gems.linalg_eigvalsh(inp, uplo)
    tu.assert_result_close(res_out, ref_out)

    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)
    (res_grad,) = torch.autograd.grad(res_out, inp, grad_outputs=upstream)

    tu.assert_result_close(res_grad, ref_grad)


# Dtypes the native kernel does not implement (RuntimeError: "linalg_eigh_cuda"
# not implemented for ...), measured on this vendor and therefore listed only
# there: another vendor may implement more.
_MEASURED_UNSUPPORTED_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float16,
    torch.bfloat16,
    torch.int32,
    torch.int64,
    torch.bool,
]


def _dtype_is_constructible(dtype):
    """Static backend capability gate using the shared flags.

    Only float8/bfloat16/int64 have a dedicated field; the others need no flag,
    and a missing field is an error rather than a silently enabled capability.
    """
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return utils.fp8_is_supported
    if dtype is torch.bfloat16:
        return utils.bf16_is_supported
    if dtype is torch.int64:
        return utils.int64_is_supported
    return True


_UNSUPPORTED_DTYPES = (
    [dtype for dtype in _MEASURED_UNSUPPORTED_DTYPES if _dtype_is_constructible(dtype)]
    if flag_gems.vendor_name == "nvidia"
    else []
)

# Rank < 2 and rectangular trailing dimensions are rejected by the operator.
_MALFORMED_SHAPES = [(), (4,), (3, 4), (4, 3), (2, 3, 4)]


@pytest.mark.linalg_eigvalsh
@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)
def test_linalg_eigvalsh_unsupported_dtype(dtype):
    inp = tu.make_input(dtype, (4, 4), ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems.linalg_eigvalsh(inp, "L")


@pytest.mark.linalg_eigvalsh
@pytest.mark.parametrize("shape", _MALFORMED_SHAPES)
def test_linalg_eigvalsh_invalid_shape(shape):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems.linalg_eigvalsh(inp, "L")


@pytest.mark.linalg_eigvalsh
@pytest.mark.parametrize("uplo", ["X", "", "lower", "Upper", "lU"])
def test_linalg_eigvalsh_invalid_uplo(uplo):
    inp = tu.make_input(torch.float32, (4, 4), ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems.linalg_eigvalsh(inp, uplo)
