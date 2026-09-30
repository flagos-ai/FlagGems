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

"""Correctness tests for ``aten::_is_zerotensor``.

The operator reads the ZeroTensor dispatch marker of the tensor impl and
returns a Python ``bool``; it is not an element-wise zero test, so an ordinary
dense tensor (``torch.zeros`` included) is reported ``False`` and only a
``_efficientzerotensor`` operand is reported ``True``.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# The nine spec dtypes plus the bool/complex/float64 types the native predicate
# accepts (probed: valid for every tested shape, range and construction).
_IS_ZEROTENSOR_DTYPES = tu.REQUIRED_DTYPES + [torch.bool, torch.complex64]
if utils.fp64_is_supported:
    _IS_ZEROTENSOR_DTYPES = _IS_ZEROTENSOR_DTYPES + [torch.float64]

_DTYPE_FLAGS = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.float64: utils.fp64_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
}
_IS_ZEROTENSOR_DTYPES = [
    dtype for dtype in _IS_ZEROTENSOR_DTYPES if _DTYPE_FLAGS.get(dtype, True)
]

# Flag preservation is rank sensitive (``t()``/``narrow``/``expand`` need rank),
# so the view kinds share one fixed 2-D shape.
_FLAG_SHAPE = (4, 6)

# Scalar and zero-extent operands keep the marker too.
_ZERO_EXTENT_SHAPES = [(0,), (0, 3), (2, 0, 4)]
_CONSTRUCTION_SHAPES = tu.selected_cases(
    tu.REQUIRED_SHAPES + _ZERO_EXTENT_SHAPES,
    quick=[(), (1,), (256,), (2, 19, 7)] + _ZERO_EXTENT_SHAPES,
)

# Both operands hold the same (zero) values; only the marker differs.
_ZERO_CONSTRUCTIONS = [
    (
        "zerotensor",
        lambda shape, dtype, device: torch.ops.aten._efficientzerotensor(
            shape, dtype=dtype, device=device
        ),
        True,
    ),
    (
        "dense_zero",
        lambda shape, dtype, device: torch.zeros(shape, dtype=dtype, device=device),
        False,
    ),
]

# Cheap single-tensor rewrites of a flagged operand. Views and the detach /
# contiguous aliases keep the marker; materializing ops (clone, zeros_like)
# drop it. The expected bool was probed natively for shape (4, 6).
_FLAG_KINDS = [
    ("identity", lambda t: t, True),
    ("detach", lambda t: t.detach(), True),
    ("slice", lambda t: t[1:3, 2:5], True),
    ("view", lambda t: t.view(-1), True),
    ("reshape", lambda t: t.reshape(3, 8), True),
    ("transpose", lambda t: t.t(), True),
    ("unsqueeze", lambda t: t.unsqueeze(0), True),
    ("narrow", lambda t: t.narrow(1, 1, 3), True),
    ("expand", lambda t: t.expand(8, 4, 6), True),
    ("contiguous", lambda t: t.contiguous(), True),
    ("clone", lambda t: t.clone(), False),
    ("zeros_like", lambda t: torch.zeros_like(t), False),
]

_SPECIAL_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]
if utils.fp64_is_supported:
    _SPECIAL_DTYPES.append(torch.float64)

_GRAD_DTYPES = [
    dtype
    for dtype in _IS_ZEROTENSOR_DTYPES
    if dtype.is_floating_point or dtype.is_complex
]

# The schema takes a single Tensor, so any non-tensor argument must fail;
# there is no unsupported-dtype or unsupported-rank form to reject.
_NON_TENSOR_ARGUMENTS = [1, 2.5, "tensor", [1, 2, 3], (1, 2), {"a": 1}, torch.float32]
_NON_TENSOR_IDS = ["int", "float", "str", "list", "tuple", "dict", "dtype"]


def _reference_device():
    """Device the native oracle runs on (``--ref cpu`` moves it to the host)."""
    return "cpu" if utils.TO_CPU else flag_gems.device


def _zerotensor(shape, dtype, device):
    """Build a flagged operand natively.

    ``_efficientzerotensor`` is the only public ZeroTensor constructor, and such
    an operand has no storage for ``tu.to_reference`` to copy, so each side of
    the comparison is built independently.
    """
    return torch.ops.aten._efficientzerotensor(shape, dtype=dtype, device=device)


def _assert_flag(res, ref):
    """The predicate contract is a Python ``bool``, not a 0-dim tensor."""
    assert type(res) is bool
    assert res == ref


@pytest.mark.is_zerotensor
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _IS_ZEROTENSOR_DTYPES)
def test__is_zerotensor_dense_input(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    res = flag_gems._is_zerotensor(inp)
    # A dense operand is never flagged, whatever its values or shape.
    assert res is False
    _assert_flag(res, torch.ops.aten._is_zerotensor(ref_inp))


@pytest.mark.is_zerotensor
@pytest.mark.parametrize(
    "expected_flag,construction",
    [(expected, build) for _, build, expected in _ZERO_CONSTRUCTIONS],
    ids=[name for name, _, _ in _ZERO_CONSTRUCTIONS],
)
@pytest.mark.parametrize("shape", _CONSTRUCTION_SHAPES)
@pytest.mark.parametrize("dtype", _IS_ZEROTENSOR_DTYPES)
def test__is_zerotensor_marker_vs_dense_zero(shape, dtype, expected_flag, construction):
    inp = construction(shape, dtype, flag_gems.device)
    ref_inp = construction(shape, dtype, _reference_device())

    res = flag_gems._is_zerotensor(inp)
    assert res is expected_flag
    _assert_flag(res, torch.ops.aten._is_zerotensor(ref_inp))


@pytest.mark.is_zerotensor
@pytest.mark.parametrize(
    "expected_flag,transform",
    [(expected, transform) for _, transform, expected in _FLAG_KINDS],
    ids=[name for name, _, _ in _FLAG_KINDS],
)
@pytest.mark.parametrize("dtype", _IS_ZEROTENSOR_DTYPES)
def test__is_zerotensor_marker_through_view(transform, expected_flag, dtype):
    inp = _zerotensor(_FLAG_SHAPE, dtype, flag_gems.device)
    ref_inp = _zerotensor(_FLAG_SHAPE, dtype, _reference_device())

    res = flag_gems._is_zerotensor(transform(inp))
    assert res is expected_flag
    _assert_flag(res, torch.ops.aten._is_zerotensor(transform(ref_inp)))


@pytest.mark.is_zerotensor
@pytest.mark.parametrize(
    "expected_flag,construction",
    [(expected, build) for _, build, expected in _ZERO_CONSTRUCTIONS],
    ids=[name for name, _, _ in _ZERO_CONSTRUCTIONS],
)
@pytest.mark.parametrize("dtype", _IS_ZEROTENSOR_DTYPES)
def test__is_zerotensor_keeps_input_state(expected_flag, construction, dtype):
    inp = construction(_FLAG_SHAPE, dtype, flag_gems.device)
    shape, stride = tuple(inp.shape), inp.stride()
    offset, device = inp.storage_offset(), inp.device

    flag_gems._is_zerotensor(inp)

    # The query must not materialize, detach or otherwise rewrite its operand.
    assert tuple(inp.shape) == shape
    assert inp.stride() == stride
    assert inp.storage_offset() == offset
    assert inp.device == device
    assert inp.dtype == dtype
    assert torch.ops.aten._is_zerotensor(inp) is expected_flag


@pytest.mark.is_zerotensor
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(
        tu.special_value_cases(
            [dtype for dtype in _SPECIAL_DTYPES if _DTYPE_FLAGS.get(dtype, True)]
        ),
        quick=[],
    ),
)
def test__is_zerotensor_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    res = flag_gems._is_zerotensor(inp)
    # nan/inf/-0.0 payloads are dense values, so the marker stays unset.
    assert res is False
    _assert_flag(res, torch.ops.aten._is_zerotensor(ref_inp))


@pytest.mark.is_zerotensor
@pytest.mark.parametrize("dtype", _GRAD_DTYPES)
def test__is_zerotensor_differentiable_input(dtype):
    # The result is a Python bool with no grad_fn, so the operator has no
    # backward workload (autograd on it raises "'bool' object is not
    # iterable"); this case covers the grad-requiring operand form instead.
    inp = _zerotensor(_FLAG_SHAPE, dtype, flag_gems.device).requires_grad_(True)
    ref_inp = _zerotensor(_FLAG_SHAPE, dtype, _reference_device()).requires_grad_(True)

    _assert_flag(flag_gems._is_zerotensor(inp), torch.ops.aten._is_zerotensor(ref_inp))
    assert inp.requires_grad is True


@pytest.mark.is_zerotensor
def test__is_zerotensor_undefined_tensor():
    # ``None`` is a valid undefined-tensor operand for the native operator.
    _assert_flag(flag_gems._is_zerotensor(None), torch.ops.aten._is_zerotensor(None))


@pytest.mark.is_zerotensor
@pytest.mark.parametrize("value", _NON_TENSOR_ARGUMENTS, ids=_NON_TENSOR_IDS)
def test__is_zerotensor_rejects_non_tensor(value):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._is_zerotensor(value)


@pytest.mark.is_zerotensor
def test__is_zerotensor_rejects_missing_argument():
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._is_zerotensor()
