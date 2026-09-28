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

"""Correctness tests for ``aten::_convert_indices_from_coo_to_csr``.

The operator consumes a sorted COO row-index vector plus a row count (``size``) and
returns the CSR row pointers: an integral tensor of length ``size + 1`` with
``out[0] == 0`` and ``out[r + 1] == out[r] + count(r)`` -- int64 by default, int32 with
``out_int32=True``.

- Input rank 0 or 1 only; a rank >= 2 tensor is rejected natively, so ``_vector_shape``
  maps every spec shape onto the index vector with the same element count and only the
  rejected rank is dropped.
- Integral input dtypes only (int8/uint8/int16/int32/int64). Floating, bool, complex and
  fp8 dtypes have no kernel, which is what the unsupported-dtype and NaN/Inf negatives
  assert.
- The input holds row indices, so ``_indices`` maps each shared value range into the
  index domain. A non-empty vector needs ``size >= 1``; an empty vector is valid for
  any ``size >= 0``, and ``size = 0`` returns the single pointer ``[0]``.
- The default (int64) output form is asserted as it stands. The runtime's broad
  ``support_int64`` capability also covers auxiliary int64 allocations, so it does not
  settle the int64 *output* case; that question is left open rather than worked around
  by substituting ``out_int32=True`` for the schema default.
- Layout: a unit-stride view with a non-zero storage offset is honoured natively and is
  asserted. A strided view is not honoured -- the kernel reads the storage prefix and
  ignores the strides -- so strided inputs are neither asserted nor counted as covered.
- Sortedness is not validated natively and a candidate that sorted its input first would
  legitimately differ, so every asserted vector is already ascending.
- No broadcast and no backward: the only other operand is an int row count and no
  floating-point dtype is accepted, so there is no second shape to broadcast and no
  differentiable input.
"""

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Native integral input dtypes (int16 included). int64 *input* support is a device
# capability and is gated by the shared flag.
INTEGER_DTYPES = [torch.int8, torch.uint8, torch.int16, torch.int32] + (
    [torch.int64] if utils.int64_is_supported else []
)

# Dtypes the operator has no kernel for. Device-wide dtype availability comes from the
# shared flags, so a dtype the device cannot represent is not asserted as an
# operator-level rejection; complex128 carries its payload in float64 components and so
# follows the float64 capability.
_REJECTED_DTYPE_GATES = (
    (torch.float16, True),
    (torch.float32, True),
    (torch.bool, True),
    (torch.bfloat16, utils.bf16_is_supported),
    (torch.float64, utils.fp64_is_supported),
    (torch.complex128, utils.fp64_is_supported),
    (torch.float8_e4m3fn, utils.fp8_is_supported),
    (torch.float8_e5m2, utils.fp8_is_supported),
)

# Every rejection below was measured on this NVIDIA CUDA build, so the rows are selected
# on the actual vendor name rather than skipped at run time: other vendors also declare
# device_name='cuda' (runtime/backend/_amd, runtime/backend/_iluvatar), and a
# kernel-availability result of one vendor's build says nothing about the others. A
# non-NVIDIA vendor collects no rejection row; the measured rows are kept in both the
# default and the quick suite.
_NVIDIA = flag_gems.vendor_name == "nvidia"


def _measured_rows(rows):
    """Rows measured on this NVIDIA build; empty on any other vendor."""
    return tu.selected_cases(rows, quick=rows) if _NVIDIA else []


REJECTED_DTYPE_ROWS = _measured_rows(
    [dtype for dtype, representable in _REJECTED_DTYPE_GATES if representable]
)

# No floating dtype has a kernel, so the spec's special-value dimension is a negative
# family here: nan, inf and mixed payloads are all rejected, following the shared
# representable-special-value contract (e4m3fn carries nan only).
SPECIAL_DTYPES = (
    [torch.float16, torch.float32]
    + ([torch.bfloat16] if utils.bf16_is_supported else [])
    + ([torch.float64] if utils.fp64_is_supported else [])
    + ([torch.float8_e4m3fn, torch.float8_e5m2] if utils.fp8_is_supported else [])
)
SPECIAL_VALUE_ROWS = _measured_rows(tu.special_value_cases(SPECIAL_DTYPES))

# A spec shape holding a single element cannot express a multi-row sweep, so it
# accumulates into a fixed 32-row table instead of degenerating to ``[0, count]``.
_MIN_ROWS = 32

_INT32_LIMIT = int(torch.iinfo(torch.int32).max)


def _aux_int(dtype, bound):
    """Integer type for auxiliary index arithmetic; never the operator's output.

    The values are reduced into ``[0, bound)``, so int32 covers every bound used here --
    the 256 needed by uint8 included -- and these fixtures do not depend on the device's
    int64 capability, which is about the operator's int64 *output*. int64 arithmetic is
    used only when a bound really exceeds int32, or when the source is already int64 and
    the device publishes int64 support.
    """
    needs_int64 = bound > _INT32_LIMIT or dtype is torch.int64
    return torch.int64 if needs_int64 and utils.int64_is_supported else torch.int32


def _vector_shape(shape):
    """The rank-0/rank-1 index-vector shape a spec shape maps onto."""
    return shape if len(shape) <= 1 else (math.prod(shape),)


def _row_count(shape):
    """Row count (``size``) for a spec shape: its flattened element count."""
    length = math.prod(_vector_shape(shape))
    return length if length > 1 else _MIN_ROWS


def _indices(dtype, shape, value_range, size):
    """Sorted in-domain row indices for one spec (dtype, shape, value_range) case.

    ``tu.make_input`` supplies the requested value range. An index has to stay inside
    the ``size + 1`` result buffer and inside the input dtype, so the values are taken
    modulo ``min(size, dtype bound)`` and sorted. The bound is computed as a Python int
    because it need not be representable in the input dtype (256 is not, for uint8);
    the modulo itself needs no more than int32.
    """
    vector_shape = _vector_shape(shape)
    raw = tu.make_input(dtype, vector_shape, value_range)
    if raw.numel() == 0:
        return raw
    bound = max(min(size, int(torch.iinfo(dtype).max) + 1), 1)
    indices = torch.remainder(raw.to(_aux_int(dtype, bound)), bound)
    return indices.flatten().sort().values.reshape(vector_shape).to(dtype)


def _gathered_rows(nnz, size, dtype):
    """Indices taken in ascending order, wrapping around when nnz exceeds size."""
    rows = torch.arange(
        nnz, dtype=_aux_int(dtype, max(nnz, size)), device=flag_gems.device
    )
    return torch.remainder(rows, max(size, 1)).sort().values.to(dtype)


def _rows_in_row_zero(nnz, size, dtype):
    """Every index lands in row 0; the rest of the table stays empty."""
    return torch.zeros(nnz, dtype=dtype, device=flag_gems.device)


def _rows_in_last_row(nnz, size, dtype):
    """Every index lands in the last row."""
    return torch.full((nnz,), size - 1, dtype=dtype, device=flag_gems.device)


def _duplicate_skew(nnz, size, dtype):
    """Three quarters of the indices pile up in row 0, the rest spread evenly."""
    device = flag_gems.device
    heavy = (3 * nnz) // 4
    arith = _aux_int(dtype, max(heavy, size))
    zeros = torch.zeros(heavy, dtype=arith, device=device)
    tail = (
        torch.remainder(
            torch.arange(nnz - heavy, dtype=arith, device=device), max(size - 1, 1)
        )
        + 1
    )
    return torch.cat([zeros, tail]).sort().values.to(dtype)


def _every_third_row(nnz, size, dtype):
    """Only every third row is occupied; the rows in between stay empty."""
    arith = _aux_int(dtype, 3 * nnz)
    return (3 * torch.arange(nnz, dtype=arith, device=flag_gems.device)).to(dtype)


def _half_empty_table(nnz, size, dtype):
    """The first half of the rows receives no index at all."""
    start = size // 2
    arith = _aux_int(dtype, start + nnz)
    return (start + torch.arange(nnz, dtype=arith, device=flag_gems.device)).to(dtype)


def _spread_rows(nnz, size, dtype):
    """Far fewer indices than rows, spread with regular gaps."""
    step = max(size // max(nnz, 1), 1)
    arith = _aux_int(dtype, step * nnz)
    return (step * torch.arange(nnz, dtype=arith, device=flag_gems.device)).to(dtype)


@pytest.mark.convert_indices_from_coo_to_csr
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test__convert_indices_from_coo_to_csr(shape, value_range, dtype):
    size = _row_count(shape)
    inp = _indices(dtype, shape, value_range, size)
    ref_inp = tu.to_reference(inp)

    # out_int32 keeps its schema default here, so the candidate has to implement that
    # default on its own; the explicit False/True sweep is the next test.
    ref_out = torch.ops.aten._convert_indices_from_coo_to_csr(ref_inp, size)
    res_out = flag_gems._convert_indices_from_coo_to_csr(inp, size)

    tu.assert_result_equal(res_out, ref_out)


# Parameter coverage for the boolean ``out_int32`` flag: both values here, the omitted
# schema default above, on the spec shape (1024, 1024) with the [-1, 1] value range.
@pytest.mark.convert_indices_from_coo_to_csr
@pytest.mark.parametrize("shape", tu.selected_cases([(1024, 1024)], quick=[]))
@pytest.mark.parametrize("out_int32", tu.selected_cases([False, True], quick=[]))
@pytest.mark.parametrize(
    "dtype", [torch.int32] + ([torch.int64] if utils.int64_is_supported else [])
)
def test__convert_indices_from_coo_to_csr_out_int32(shape, out_int32, dtype):
    size = _row_count(shape)
    inp = _indices(dtype, shape, ["-1", "1"], size)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._convert_indices_from_coo_to_csr(
        ref_inp, size, out_int32=out_int32
    )
    res_out = flag_gems._convert_indices_from_coo_to_csr(inp, size, out_int32=out_int32)

    tu.assert_result_equal(res_out, ref_out)


# ``size`` boundaries over vectors that fit them: an empty vector with a zero-row and a
# three-row table, a single index, a dense full table, one trailing empty row, and many
# indices collapsing into a single row.
_SIZE_CASES = tu.selected_cases(
    [(0, 0), (0, 3), (1, 1), (4096, 4096), (4096, 4097), (65536, 1)], quick=[]
)


@pytest.mark.convert_indices_from_coo_to_csr
@pytest.mark.parametrize("nnz,size", _SIZE_CASES)
def test__convert_indices_from_coo_to_csr_size(nnz, size):
    inp = _gathered_rows(nnz, size, torch.int32)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._convert_indices_from_coo_to_csr(ref_inp, size)
    res_out = flag_gems._convert_indices_from_coo_to_csr(inp, size)

    tu.assert_result_equal(res_out, ref_out)


# A rank-0 index tensor is a valid input; the scalar operand form proper is the int
# ``size`` that every test passes.
_SCALAR_CASES = tu.selected_cases([(0, 1), (3, 4), (3, 32)], quick=[])


@pytest.mark.convert_indices_from_coo_to_csr
@pytest.mark.parametrize("value,size", _SCALAR_CASES)
def test__convert_indices_from_coo_to_csr_scalar_input(value, size):
    inp = torch.tensor(value, dtype=torch.int32, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._convert_indices_from_coo_to_csr(ref_inp, size)
    res_out = flag_gems._convert_indices_from_coo_to_csr(inp, size)

    tu.assert_result_equal(res_out, ref_out)


# Deterministic sorted index distributions that the value-range grid cannot express: a
# uniform vector reduced modulo the row count only ever reaches a few rows. Every
# ``size`` stays <= 128 so each index stays representable in int8/uint8 too.
_DISTRIBUTION_ROWS = tu.selected_cases(
    [
        (_gathered_rows, 64, 64),  # every row receives exactly one index
        (_rows_in_row_zero, 257, 64),  # a single occupied row, at the start
        (_rows_in_last_row, 257, 64),  # a single occupied row, at the end
        (_duplicate_skew, 4096, 64),  # one hot row plus a spread-out tail
        (_every_third_row, 40, 128),  # regular interior gaps
        (_half_empty_table, 32, 128),  # the leading rows stay empty
        (_gathered_rows, 32, 128),  # the trailing rows stay empty
        (_spread_rows, 17, 128),  # sparse over many rows
        (_gathered_rows, 4096, 17),  # more indices than rows, wrapping around
    ],
    quick=[],
)


@pytest.mark.convert_indices_from_coo_to_csr
@pytest.mark.parametrize("builder,nnz,size", _DISTRIBUTION_ROWS)
@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test__convert_indices_from_coo_to_csr_distribution(builder, nnz, size, dtype):
    inp = builder(nnz, size, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._convert_indices_from_coo_to_csr(ref_inp, size)
    res_out = flag_gems._convert_indices_from_coo_to_csr(inp, size)

    tu.assert_result_equal(res_out, ref_out)


# Largest storable row index per dtype: the last two pointers must step by one. Only
# int8/uint8/int16 are asserted -- a ``size + 1`` table at the int32 boundary needs
# ~8.6 GiB as int32 and ~17.2 GiB as int64, and the int64 boundary is not
# materializable at all.
_BOUNDARY_CASES = tu.selected_cases(
    [
        (dtype, int(torch.iinfo(dtype).max) + 2)
        for dtype in (torch.int8, torch.uint8, torch.int16)
    ],
    quick=[],
)


@pytest.mark.convert_indices_from_coo_to_csr
@pytest.mark.parametrize("dtype,size", _BOUNDARY_CASES)
def test__convert_indices_from_coo_to_csr_max_index(dtype, size):
    inp = torch.tensor(
        [int(torch.iinfo(dtype).max)], dtype=dtype, device=flag_gems.device
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._convert_indices_from_coo_to_csr(ref_inp, size)
    res_out = flag_gems._convert_indices_from_coo_to_csr(inp, size)

    tu.assert_result_equal(res_out, ref_out)


# A unit-stride view with a non-zero storage offset is honoured natively, unlike a
# strided view, so it is asserted here. ``tu.to_reference`` keeps the offset and the
# strides, so the reference is built from independent storage with the same geometry.
_OFFSET_CASES = tu.selected_cases([(3, 6, 200), (7, 33, 256)], quick=[])


@pytest.mark.convert_indices_from_coo_to_csr
@pytest.mark.parametrize("offset,length,size", _OFFSET_CASES)
def test__convert_indices_from_coo_to_csr_offset_input(offset, length, size):
    base = torch.arange(2 * size, dtype=torch.int32, device=flag_gems.device) + 1
    inp = base.narrow(0, offset, length)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._convert_indices_from_coo_to_csr(ref_inp, size)
    res_out = flag_gems._convert_indices_from_coo_to_csr(inp, size)

    tu.assert_result_equal(res_out, ref_out)


# ``.out``: the operator returns the buffer it was handed. The buffer is a unit-stride
# window inside a sentinel-filled parent, so a write outside the requested ``size + 1``
# elements shows up in the full-parent comparison; ``guard = 0`` is the plain
# contiguous buffer, ``guard = 4`` the non-zero storage offset.
_OUT_SENTINEL = -99
_OUT_GUARD = 4
_OUT_CASES = tu.selected_cases(
    [(guard, out_int32) for guard in (0, _OUT_GUARD) for out_int32 in (False, True)],
    quick=[],
)


@pytest.mark.convert_indices_from_coo_to_csr
@pytest.mark.parametrize("guard,out_int32", _OUT_CASES)
def test__convert_indices_from_coo_to_csr_out(guard, out_int32):
    size = 64
    inp = _gathered_rows(4 * size, size, torch.int32)
    ref_inp = tu.to_reference(inp)
    out_dtype = torch.int32 if out_int32 else torch.int64

    res_parent = torch.full(
        (size + 1 + 2 * guard,), _OUT_SENTINEL, dtype=out_dtype, device=inp.device
    )
    res_buf = res_parent.narrow(0, guard, size + 1)
    res_out = flag_gems._convert_indices_from_coo_to_csr(
        inp, size, out_int32=out_int32, out=res_buf
    )

    ref_buf = torch.empty(size + 1, dtype=out_dtype, device=ref_inp.device)
    ref_out = torch.ops.aten._convert_indices_from_coo_to_csr.out(
        ref_inp, size, out_int32=out_int32, out=ref_buf
    )

    expected = torch.full_like(res_parent, _OUT_SENTINEL)
    expected.narrow(0, guard, size + 1).copy_(ref_out)
    assert res_out is res_buf
    tu.assert_result_equal(res_parent, expected)


# Negative cases. Every form was measured on the native operator first; only the
# candidate exception is asserted.
@pytest.mark.convert_indices_from_coo_to_csr
@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4), (1, 1, 1, 1), (1, 1, 1, 1, 1)])
def test__convert_indices_from_coo_to_csr_non_vector_input(shape):
    inp = torch.zeros(shape, dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems._convert_indices_from_coo_to_csr(inp, 8)


@pytest.mark.convert_indices_from_coo_to_csr
@pytest.mark.parametrize("dtype", REJECTED_DTYPE_ROWS)
def test__convert_indices_from_coo_to_csr_unsupported_dtype(dtype):
    inp = torch.zeros(4, dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems._convert_indices_from_coo_to_csr(inp, 8)


@pytest.mark.convert_indices_from_coo_to_csr
@pytest.mark.parametrize("dtype,scenario", SPECIAL_VALUE_ROWS)
def test__convert_indices_from_coo_to_csr_nan_inf_input(dtype, scenario):
    # A float payload is rejected whatever its values, NaN and Inf included.
    inp = tu.make_special_input(dtype, scenario)
    with pytest.raises(RuntimeError):
        flag_gems._convert_indices_from_coo_to_csr(inp, 8)


@pytest.mark.convert_indices_from_coo_to_csr
@pytest.mark.parametrize("size", [-2, -3, 2.5, None, "4"])
def test__convert_indices_from_coo_to_csr_invalid_size(size):
    inp = torch.zeros(4, dtype=torch.int32, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._convert_indices_from_coo_to_csr(inp, size)


@pytest.mark.convert_indices_from_coo_to_csr
@pytest.mark.parametrize("out_int32", ["x"])
def test__convert_indices_from_coo_to_csr_invalid_out_int32(out_int32):
    inp = torch.zeros(4, dtype=torch.int32, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._convert_indices_from_coo_to_csr(inp, 4, out_int32=out_int32)
