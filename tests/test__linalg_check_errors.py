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
"""Correctness tests for ``aten::_linalg_check_errors``.

The operator scans the integer status codes of a factorization: it returns
nothing when every code is zero and otherwise reports the first failure it
finds. The observable results are that silent return and the exception class and
message, both compared against ``torch.ops.aten._linalg_check_errors``.
"""

import re

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

API_NAME = "torch.linalg.inv"

# ``info`` is validated against ``infos.scalar_type() == kInt`` before anything
# else: a probe on the active backend raised RuntimeError "infos.scalar_type() ==
# kInt INTERNAL ASSERT FAILED (BatchLinearAlgebra.cpp:1566)" for every other
# dtype, so int32 is the only accepted input dtype. No floating dtype reaches the
# kernel, which is why there is no fp8 and no positive nan/inf workload; those
# dtypes appear below as negative cases instead.
SUPPORTED_DTYPES = [torch.int32]

# Single-input operator returning ``()``: there is no second operand to
# broadcast, no differentiable output (hence no backward workload) and no ``.out``
# overload (``torch.ops.aten._linalg_check_errors.overloads() == ["default"]`` and
# the attribute does not exist).
_INTERNAL_ASSERT_PREFIX = re.compile(
    r"(?:false )?INTERNAL ASSERT FAILED at .*?, please report a bug to PyTorch[.]\s*"
)


def _normalized_message(message):
    """Strip the build-specific internal-assert preamble and normalize whitespace,
    keeping the whole semantic payload of the message.

    Only the ATen source path and line number inside the preamble vary between
    builds; the API name, the batch position and the wording that separates a
    singular diagonal element from a non-positive-definite factorization, an
    illegal argument value, an unknown status code or the scalar-conversion
    failure are all stable and are compared verbatim.
    """
    return " ".join(_INTERNAL_ASSERT_PREFIX.sub("", message).split())


def _capture(fn, info, api_name, is_matrix):
    """Call ``fn`` and return the exception it raised, or None when it returned
    silently.

    The operator produces no tensor, so a successful call must return None. Only
    RuntimeError, which ``torch._C._LinAlgError`` derives from, counts as a
    reported failure; anything else, including a missing candidate, propagates.
    """
    try:
        result = fn(info, api_name, is_matrix=is_matrix)
    except RuntimeError as exc:
        return exc
    assert result is None, f"a successful call returns nothing, got {result!r}"
    return None


def _assert_same_outcome(res_exc, ref_exc):
    """The candidate must either stay silent or classify and report exactly like
    the reference: same exception class and same semantic message payload."""
    if ref_exc is None:
        assert res_exc is None, f"candidate raised {res_exc!r}, reference was silent"
        return
    assert res_exc is not None, "reference reports a failure, candidate stayed silent"
    assert type(res_exc) is type(ref_exc), (
        f"candidate raised {type(res_exc).__name__}, "
        f"reference raised {type(ref_exc).__name__}"
    )
    assert _normalized_message(str(res_exc)) == _normalized_message(str(ref_exc)), (
        f"candidate message {str(res_exc)!r} differs from "
        f"reference message {str(ref_exc)!r}"
    )


def _compare_calls(info, api_name, is_matrix):
    """Run the reference and the candidate on the same codes and report both
    outcomes without asserting, so callers can add their own input checks."""
    ref_info = tu.to_reference(info)
    ref_exc = _capture(
        torch.ops.aten._linalg_check_errors, ref_info, api_name, is_matrix
    )
    res_exc = _capture(flag_gems._linalg_check_errors, info, api_name, is_matrix)
    return res_exc, ref_exc


_INFO_GRID_CASES = [
    (shape, value_range, is_matrix)
    for shape in tu.selected_shapes()
    for value_range in tu.selected_ranges()
    for is_matrix in (False, True)
]


@pytest.mark.linalg_check_errors
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("shape,value_range,is_matrix", _INFO_GRID_CASES)
def test__linalg_check_errors_info_grid(shape, value_range, is_matrix, dtype):
    """Codes drawn from the five spec ranges select the silent or the reporting
    branch of the reference; the candidate must land on the same one, with the
    same message, and must leave the codes untouched."""
    info = tu.make_input(dtype, shape, value_range)
    snapshot = info.clone()

    res_exc, ref_exc = _compare_calls(info, API_NAME, is_matrix)

    _assert_same_outcome(res_exc, ref_exc)
    tu.assert_result_equal(info, snapshot)


# The reference accepts all-zero codes of any shape, including empty 1-D and empty
# multi-dimensional tensors, and reports nothing.
_ZERO_INFO_SHAPES = tu.selected_cases(
    [
        (),
        (1,),
        (256,),
        (1024, 1024),
        (20, 320, 15),
        (16, 128, 64, 60),
        (16, 7, 57, 32, 29),
        (0,),
        (0, 3),
    ],
    quick=[(2, 19, 7)],
)


@pytest.mark.linalg_check_errors
@pytest.mark.parametrize("is_matrix", [False, True])
@pytest.mark.parametrize("shape", _ZERO_INFO_SHAPES)
def test__linalg_check_errors_zero_info(shape, is_matrix):
    """All-zero codes mean a completed factorization, so the call returns
    nothing, for both flag branches and for scalar and empty shapes alike."""
    info = torch.zeros(shape, dtype=torch.int32, device=flag_gems.device)
    snapshot = info.clone()

    res = flag_gems._linalg_check_errors(info, API_NAME, is_matrix=is_matrix)

    assert res is None
    tu.assert_result_equal(info, snapshot)


# For each name the same code 2 selects different wording and, for inv, cholesky
# and solve, a different exception class than for lu_factor or an unrecognized
# name; an empty name still reports the unknown code.
_API_CASES = tu.selected_cases(
    [
        ("torch.linalg.inv", (3,), False),
        ("torch.linalg.cholesky", (3,), False),
        ("torch.linalg.solve", (3,), False),
        ("torch.linalg.lu_factor", (3,), False),
        ("torch.linalg.qr", (3,), False),
        ("my.custom.op", (3,), False),
        ("", (3,), False),
        ("torch.linalg.inv", (1,), True),
        ("torch.linalg.cholesky", (1,), True),
        ("torch.linalg.solve", (1,), True),
        ("torch.linalg.lu_factor", (1,), True),
        ("torch.linalg.qr", (1,), True),
    ],
    quick=[("torch.linalg.inv", (3,), False), ("torch.linalg.inv", (1,), True)],
)


@pytest.mark.linalg_check_errors
@pytest.mark.parametrize("api_name,shape,is_matrix", _API_CASES)
def test__linalg_check_errors_api_name_dispatch(api_name, shape, is_matrix):
    """``api_name`` is observable only through the raised message and its class:
    a generic message that merely contains the name, or the wording of another
    factorization, fails the comparison."""
    info = torch.full(shape, 2, dtype=torch.int32, device=flag_gems.device)

    res_exc, ref_exc = _compare_calls(info, api_name, is_matrix)

    _assert_same_outcome(res_exc, ref_exc)


# The scan stops at the first non-zero code, so the reported batch position and
# code identify that element rather than the largest or the last one.
_BATCH_POSITION_CASES = tu.selected_cases(
    [
        ((3,), [1, 1, 1], False),
        ((3,), [0, 1, 1], False),
        ((3,), [0, 0, 1], False),
        ((3,), [2, 3, 4], False),
        ((3,), [0, 3, 4], False),
        ((3,), [0, 0, 4], False),
        ((3,), [0, -1, 2], False),
        ((3,), [2, 0, -1], False),
        ((2, 3), [0, 0, 0, 0, 1, 0], False),
        ((2, 3), [0, 1, 0, 0, 0, 0], False),
        ((), [1], False),
    ],
    quick=[((3,), [1, 1, 1], False), ((3,), [0, 1, 1], False)],
)


@pytest.mark.linalg_check_errors
@pytest.mark.parametrize("shape,values,is_matrix", _BATCH_POSITION_CASES)
def test__linalg_check_errors_reports_first_failure(shape, values, is_matrix):
    """Mixed and repeated codes make the reported position and code decisive: a
    candidate reporting the last or the largest failure lands elsewhere."""
    info = torch.tensor(values, dtype=torch.int32, device=flag_gems.device).reshape(
        shape
    )
    snapshot = info.clone()

    res_exc, ref_exc = _compare_calls(info, API_NAME, is_matrix)

    _assert_same_outcome(res_exc, ref_exc)
    tu.assert_result_equal(info, snapshot)


_SCALAR_INFO_CODES = tu.selected_cases(
    [1, 2, 7, 1024, -1, -2147483648],
    quick=[1, -1],
)


@pytest.mark.linalg_check_errors
@pytest.mark.parametrize("code", _SCALAR_INFO_CODES)
def test__linalg_check_errors_scalar_info(code):
    """With is_matrix=True the tensor is read as a single status code, so the
    message carries no batch position, and a code of any magnitude is reported:
    a positive code fails the factorization, a negative one is an illegal
    argument value."""
    info = torch.tensor([code], dtype=torch.int32, device=flag_gems.device)

    res_exc, ref_exc = _compare_calls(info, API_NAME, True)

    _assert_same_outcome(res_exc, ref_exc)


# Non-singleton tensors holding a non-zero code cannot be read as a scalar, so
# this is neither the batch nor the scalar branch.
_NON_SINGLETON_SCALAR_CASES = [[1, 0], [0, 1, 0], [0, 0, 0, 1]]


@pytest.mark.linalg_check_errors
@pytest.mark.parametrize("values", _NON_SINGLETON_SCALAR_CASES)
def test__linalg_check_errors_scalar_flag_needs_one_element(values):
    info = torch.tensor(values, dtype=torch.int32, device=flag_gems.device)

    res_exc, ref_exc = _compare_calls(info, API_NAME, True)

    _assert_same_outcome(res_exc, ref_exc)


# Dtypes are gated on the static device capability flags (no runtime probing):
# int64, bfloat16, float64/complex128 and both fp8 types are only constructed
# where the backend reports support. Every listed dtype is rejected natively
# before the codes are read, by the kInt assertion described above.
_NEGATIVE_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.float16,
    torch.float32,
    torch.bool,
    torch.complex64,
]
if utils.int64_is_supported:
    _NEGATIVE_DTYPES.append(torch.int64)
if utils.bf16_is_supported:
    _NEGATIVE_DTYPES.append(torch.bfloat16)
if utils.fp64_is_supported:
    _NEGATIVE_DTYPES.extend([torch.float64, torch.complex128])
if utils.fp8_is_supported:
    _NEGATIVE_DTYPES.extend([torch.float8_e4m3fn, torch.float8_e5m2])


@pytest.mark.linalg_check_errors
@pytest.mark.parametrize("bad_dtype", _NEGATIVE_DTYPES)
def test__linalg_check_errors_rejects_non_int32_dtype(bad_dtype):
    """Only int32 reaches the kernel; any other constructible dtype is an
    invalid input."""
    info = torch.zeros((2, 3), dtype=bad_dtype, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._linalg_check_errors(info, API_NAME, is_matrix=False)


_NON_CONTIGUOUS_VIEWS = [
    ((2, 6), (2, 3), (6, 2)),
    ((8,), (4,), (2,)),
]


@pytest.mark.linalg_check_errors
@pytest.mark.parametrize("buffer_shape,size,stride", _NON_CONTIGUOUS_VIEWS)
def test__linalg_check_errors_rejects_non_contiguous_info(buffer_shape, size, stride):
    """The reference asserts ``infos.is_contiguous()``, so a strided info view is
    rejected even though its dtype and values are valid, and stays unmodified."""
    buffer = torch.zeros(buffer_shape, dtype=torch.int32, device=flag_gems.device)
    info = torch.as_strided(buffer, size=size, stride=stride)
    snapshot = info.clone()

    with pytest.raises(RuntimeError):
        flag_gems._linalg_check_errors(info, API_NAME, is_matrix=False)

    tu.assert_result_equal(info, snapshot)


# A storage-offset slice is contiguous, so it is accepted; only its position
# reporting is checked here, and this positive family is default-only.
_OFFSET_VIEW_CASES = tu.selected_cases(
    [
        # (buffer length, view start, view stop, failing index inside the view)
        (8, 2, 6, None),
        (8, 2, 6, 0),
        (12, 3, 9, 5),
    ],
    quick=[],
)


@pytest.mark.linalg_check_errors
@pytest.mark.parametrize("length,start,stop,failing", _OFFSET_VIEW_CASES)
def test__linalg_check_errors_offset_view(length, start, stop, failing):
    """The reported position is relative to the contiguous view rather than to
    the backing buffer."""
    buffer = torch.zeros(length, dtype=torch.int32, device=flag_gems.device)
    if failing is not None:
        buffer[start + failing] = 1
    info = buffer[start:stop]
    snapshot = info.clone()

    res_exc, ref_exc = _compare_calls(info, API_NAME, False)

    _assert_same_outcome(res_exc, ref_exc)
    tu.assert_result_equal(info, snapshot)


_READ_ONLY_CASES = [[0, 0, 0], [0, 1, 0], [0, -1, 0], [4, 0, 0]]


@pytest.mark.linalg_check_errors
@pytest.mark.parametrize("values", _READ_ONLY_CASES)
def test__linalg_check_errors_leaves_info_unchanged(values):
    """Checking the codes is a read-only report: neither the silent nor the
    reporting path may modify the info tensor, and both must classify alike."""
    info = torch.tensor(values, dtype=torch.int32, device=flag_gems.device)
    snapshot = info.clone()

    res_exc, ref_exc = _compare_calls(info, API_NAME, False)

    _assert_same_outcome(res_exc, ref_exc)
    tu.assert_result_equal(info, snapshot)


@pytest.mark.linalg_check_errors
@pytest.mark.parametrize("extra", [(), (False,)], ids=["omitted", "positional"])
def test__linalg_check_errors_is_matrix_must_be_a_keyword(extra):
    """``is_matrix`` is keyword-only in the schema, so omitting it or passing it
    positionally is an invalid call rather than a default."""
    info = torch.zeros((3,), dtype=torch.int32, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._linalg_check_errors(info, API_NAME, *extra)


@pytest.mark.linalg_check_errors
def test__linalg_check_errors_requires_api_name():
    """``api_name`` has no default either: a lone info tensor is invalid."""
    info = torch.zeros((3,), dtype=torch.int32, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._linalg_check_errors(info)


_NON_BOOL_FLAGS = ["x", [1], b"x", ()]


@pytest.mark.linalg_check_errors
@pytest.mark.parametrize("bad_flag", _NON_BOOL_FLAGS)
def test__linalg_check_errors_rejects_non_bool_is_matrix(bad_flag):
    """Numbers, None and 0-dim bool tensors are coerced to bool by the schema,
    but a string, bytes, a sequence or an arbitrary object is not and is
    rejected; only the natively rejected values are listed here."""
    info = torch.zeros((3,), dtype=torch.int32, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._linalg_check_errors(info, API_NAME, is_matrix=bad_flag)


_NON_STR_API_NAMES = [123, None, 1.5]


@pytest.mark.linalg_check_errors
@pytest.mark.parametrize("bad_name", _NON_STR_API_NAMES)
def test__linalg_check_errors_rejects_non_str_api_name(bad_name):
    info = torch.zeros((3,), dtype=torch.int32, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._linalg_check_errors(info, bad_name, is_matrix=False)


_NON_TENSOR_INFO = [[0, 0], None]


@pytest.mark.linalg_check_errors
@pytest.mark.parametrize("bad_info", _NON_TENSOR_INFO)
def test__linalg_check_errors_rejects_non_tensor_info(bad_info):
    """A plain Python sequence or None is not an info tensor."""
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._linalg_check_errors(bad_info, API_NAME, is_matrix=False)
