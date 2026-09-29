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


def _support_gated(dtypes):
    """Keep only the dtypes the active device declares support for.

    These are static device capability flags, so the gate runs at collection time and
    never probes the operator or allocates a tensor.
    """
    gated = []
    for dtype in dtypes:
        if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            if not utils.fp8_is_supported:
                continue
        elif dtype is torch.bfloat16 and not utils.bf16_is_supported:
            continue
        elif dtype is torch.int64 and not utils.int64_is_supported:
            continue
        elif dtype is torch.float64 and not utils.fp64_is_supported:
            continue
        gated.append(dtype)
    return gated


MSG = "functional_assert_scalar accuracy check"

# aten::_functional_assert_scalar(Scalar self, str assert_msg, Tensor dep_token) -> Tensor
# is value preserving: dep_token's values are copied into a fresh result that mirrors its
# shape, dtype and device. A truthy ``self`` performs the copy; a falsy ``self`` raises
# RuntimeError(assert_msg) instead of returning.
#
# Probed exemptions and native evidence on the active backend:
# * backward: torch.autograd.grad(out, x, grad_outputs=torch.ones_like(out)) raises
#   RuntimeError('derivative for aten::_functional_assert_scalar is not implemented')
#   for float16, bfloat16, float32 and float64 tokens, and the result carries a
#   <NotImplemented> grad_fn, so there is no autograd formula to assert.
# * broadcast: one tensor operand plus a Scalar leaves no operand pair to broadcast.
# * dim / unsupported-dtype negatives: the schema has no dim parameter and dep_token
#   accepts every dtype probed here. The schema negatives that do exist - a non-number
#   ``self``, a non-str ``assert_msg``, and a non-Tensor or absent ``dep_token`` - are
#   covered by the dedicated negative tests below.

# dep_token dtype grid. bool, complex64 and float64 are supported as well; the extra
# dtypes only leave the quick smoke subset.
TOKEN_DTYPES = _support_gated(tu.REQUIRED_DTYPES) + tu.selected_cases(
    [torch.bool, torch.complex64] + _support_gated([torch.float64]), quick=[]
)

# ``self`` accepts Python numbers and bools; every truthy value copies dep_token.
# Coverage mapping, one row per dimension:
#   int boundaries   -> torch.iinfo(torch.int64).min / .max
#   float boundaries -> torch.finfo(torch.float64).max / .tiny
#   bool             -> True
#   complex          -> 1 + 2j
#   special scalars  -> nan / inf / -inf, all truthy so the copy still happens
# Falsy scalars are the operator's invalid parameter value and live in ZERO_SCALARS.
SCALAR_VALUES = tu.selected_cases(
    [
        1,
        -7,
        0.5,
        -2.25,
        True,
        torch.iinfo(torch.int64).max,
        torch.iinfo(torch.int64).min,
        torch.finfo(torch.float64).max,
        torch.finfo(torch.float64).tiny,
        float("nan"),
        float("inf"),
        float("-inf"),
        1 + 2j,
    ],
    quick=[],
)

ZERO_SCALARS = [0, 0.0, -0.0, False, 0j]

# Native propagates assert_msg verbatim and substitutes the fallback below for an empty
# message; both behaviors were probed with torch.ops.aten._functional_assert_scalar.
EMPTY_MSG_FALLBACK = "Assertion is failed"
UNICODE_MSG = "断言失败 \U0001f600"
ESCAPED_MSG = "line1\nline2\tend"
MESSAGE_CASES = [
    (MSG, MSG),
    ("", EMPTY_MSG_FALLBACK),
    (UNICODE_MSG, UNICODE_MSG),
    (ESCAPED_MSG, ESCAPED_MSG),
]

# tu.special_value_cases() skips non-floating dtypes, so the complex64 rows are appended
# here and built by the shared tu.make_special_input(). That helper also omits inf/mixed
# for float8_e4m3fn, which cannot represent infinity; those e4m3fn finite extremes
# (448 / -448) are therefore covered by the ["0", "max"] and ["min", "0"] rows of the
# main grid below, not by this family.
SPECIAL_TOKEN_CASES = tu.selected_cases(
    tu.special_value_cases(
        _support_gated(tu.REQUIRED_DTYPES) + _support_gated([torch.float64])
    )
    + [(torch.complex64, scenario) for scenario in ("nan", "inf", "mixed")],
    quick=[],
)

# Named dep_token layouts over the coordinate-varying base torch.arange(1, 73).reshape(4, 6, 3)
# on flag_gems.device. The copy must resolve every geometry. Strides are not asserted: the
# probed native copy compacts gaps (narrowed and sliced come back dense at (9, 3, 1) and an
# expanded token comes back dense at (15, 3, 1)), so only the values plus the
# storage/identity contract are compared.
LAYOUTS = [
    "contiguous",  # (4, 6, 3), strides (18, 3, 1), offset 0
    "transposed",  # (6, 4, 3), strides (3, 18, 1): dim 0 has stride 3, dim 2 is unit
    "narrowed",  # (4, 3, 3), strides (18, 3, 1) leave gaps, offset 3
    "sliced",  # (2, 3, 3), strides (18, 6, 1), offset 18
    "unit_offset",  # (69,) flat unit-stride view at offset 3
    "expanded",  # (2, 5, 3) zero strides from expand()
]

# Empty dep_token geometries. The native copy and the shared assertion handle inherited
# strides and storage offsets on empty views (probed), so these cases are kept; an empty
# result is still required to be a distinct tensor.
EMPTY_VIEWS = [
    "empty_flat",  # (0,) flat slice at offset 5
    "empty_rows",  # base[:0], shape (0, 6, 3)
    "empty_cols_offset",  # base[2:2, ::2], shape (0, 3, 3) at offset 36
    "empty_transposed",  # base[:0].transpose(0, 1), shape (6, 0, 3)
    "empty_expanded",  # expand() to a zero extent, shape (2, 0, 3)
]

CONJ_LAYOUTS = ["conj_contiguous", "conj_transposed"]


def _base_token(dtype):
    base = torch.arange(1, 4 * 6 * 3 + 1, dtype=torch.float32, device=flag_gems.device)
    return base.reshape(4, 6, 3).to(dtype)


def _layout_token(layout, dtype):
    """Build one named dep_token layout with coordinate-varying values."""
    base = _base_token(dtype)
    if layout == "contiguous":
        return base
    if layout == "transposed":
        return base.transpose(0, 1)
    if layout == "narrowed":
        return base.narrow(1, 1, 3)
    if layout == "sliced":
        return base[1:3, ::2, ::1]
    if layout == "unit_offset":
        return base.reshape(-1)[3:]
    if layout == "expanded":
        return base[:1, :1, :3].expand(2, 5, 3)
    raise ValueError(f"unknown dep_token layout {layout!r}")


def _empty_token(view, dtype):
    """Build one empty dep_token geometry."""
    base = _base_token(dtype)
    if view == "empty_flat":
        return base.reshape(-1)[5:5]
    if view == "empty_rows":
        return base[:0]
    if view == "empty_cols_offset":
        return base[2:2, ::2]
    if view == "empty_transposed":
        return base[:0].transpose(0, 1)
    if view == "empty_expanded":
        return base[:1, :1, :3].expand(2, 0, 3)
    raise ValueError(f"unknown empty dep_token view {view!r}")


def _conj_token(layout, dtype):
    """Build a lazily conjugated dep_token with a nonzero imaginary component.

    Both parts vary by coordinate, so the lazy conjugate bit is observable: a
    candidate that reads raw storage flips a nonzero imaginary sign and fails value
    equality instead of coincidentally matching a real-valued fixture.
    """
    coords = torch.arange(
        1, 4 * 6 * 3 + 1, dtype=torch.float32, device=flag_gems.device
    )
    base = torch.complex(coords, coords * 0.5).reshape(4, 6, 3).to(dtype)
    if layout == "conj_contiguous":
        return base.conj()
    if layout == "conj_transposed":
        return base.transpose(0, 1).conj()
    raise ValueError(f"unknown conjugate layout {layout!r}")


def _assert_token_copy(res_out, token, ref_out, before):
    """A truthy ``self`` returns a fresh tensor holding dep_token's values.

    assert_result_equal already covers dtype, shape and values, so this helper adds only
    the identity/storage and input-mutation contract it does not cover. A view of
    independent owned storage is fine; aliasing dep_token's own storage is not.
    """
    assert res_out is not token
    assert res_out.device == token.device
    if token.numel() > 0:
        # A view into dep_token's storage keeps that storage pointer even when the copy
        # is offset, so this rejects storage sharing that value equality would accept.
        assert (
            res_out.untyped_storage().data_ptr() != token.untyped_storage().data_ptr()
        )
    tu.assert_result_equal(res_out, ref_out)
    # A candidate that copies dep_token and then corrupts the input is still wrong.
    tu.assert_result_equal(token, before)


@pytest.mark.functional_assert_scalar
@pytest.mark.parametrize("dtype", TOKEN_DTYPES)
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__functional_assert_scalar(dtype, shape, value_range):
    token = tu.make_input(dtype, shape, value_range)
    ref_token = tu.to_reference(token)
    before = token.clone()

    ref_out = torch.ops.aten._functional_assert_scalar(1.0, MSG, ref_token)
    res_out = flag_gems._functional_assert_scalar(1.0, MSG, token)

    _assert_token_copy(res_out, token, ref_out, before)


@pytest.mark.functional_assert_scalar
@pytest.mark.parametrize(
    "dtype",
    tu.selected_cases(_support_gated([torch.float32, torch.int64]), quick=[]),
)
@pytest.mark.parametrize("layout", LAYOUTS)
def test__functional_assert_scalar_token_layout(layout, dtype):
    token = _layout_token(layout, dtype)
    ref_token = tu.to_reference(token)
    before = token.clone()

    ref_out = torch.ops.aten._functional_assert_scalar(-2.5, MSG, ref_token)
    res_out = flag_gems._functional_assert_scalar(-2.5, MSG, token)

    _assert_token_copy(res_out, token, ref_out, before)


@pytest.mark.functional_assert_scalar
@pytest.mark.parametrize("layout", tu.selected_cases(CONJ_LAYOUTS, quick=[]))
def test__functional_assert_scalar_conj_token(layout):
    # The lazy conjugate bit must be resolved by the copy: the fixture's imaginary
    # parts are all nonzero, so reading raw storage yields the wrong sign.
    token = _conj_token(layout, torch.complex64)
    ref_token = tu.to_reference(token)
    before = token.clone()

    ref_out = torch.ops.aten._functional_assert_scalar(3.0, MSG, ref_token)
    res_out = flag_gems._functional_assert_scalar(3.0, MSG, token)

    _assert_token_copy(res_out, token, ref_out, before)


@pytest.mark.functional_assert_scalar
@pytest.mark.parametrize(
    "dtype",
    tu.selected_cases(_support_gated([torch.float32, torch.int8]), quick=[]),
)
@pytest.mark.parametrize("view", EMPTY_VIEWS)
def test__functional_assert_scalar_empty_token(view, dtype):
    token = _empty_token(view, dtype)
    ref_token = tu.to_reference(token)
    before = token.clone()

    ref_out = torch.ops.aten._functional_assert_scalar(1.0, MSG, ref_token)
    res_out = flag_gems._functional_assert_scalar(1.0, MSG, token)

    _assert_token_copy(res_out, token, ref_out, before)


@pytest.mark.functional_assert_scalar
@pytest.mark.parametrize(
    "dtype",
    tu.selected_cases(_support_gated([torch.float32, torch.int64]), quick=[]),
)
@pytest.mark.parametrize("scalar", SCALAR_VALUES)
def test__functional_assert_scalar_scalar_operand(scalar, dtype):
    token = tu.make_input(dtype, (4, 8, 6), ["-1", "1"])
    ref_token = tu.to_reference(token)
    before = token.clone()

    ref_out = torch.ops.aten._functional_assert_scalar(scalar, MSG, ref_token)
    res_out = flag_gems._functional_assert_scalar(scalar, MSG, token)

    _assert_token_copy(res_out, token, ref_out, before)


@pytest.mark.functional_assert_scalar
@pytest.mark.parametrize(
    "dtype",
    tu.selected_cases(_support_gated([torch.float32, torch.int64]), quick=[]),
)
def test__functional_assert_scalar_keyword_arguments(dtype):
    # The schema requires all three arguments; naming them is the second valid call form.
    token = tu.make_input(dtype, (4, 8, 6), ["-1", "1"])
    ref_token = tu.to_reference(token)
    before = token.clone()

    ref_out = torch.ops.aten._functional_assert_scalar(
        self=1.0, assert_msg=MSG, dep_token=ref_token
    )
    res_out = flag_gems._functional_assert_scalar(
        self=1.0, assert_msg=MSG, dep_token=token
    )

    _assert_token_copy(res_out, token, ref_out, before)


@pytest.mark.functional_assert_scalar
@pytest.mark.parametrize("dtype,scenario", SPECIAL_TOKEN_CASES)
def test__functional_assert_scalar_special_token_values(dtype, scenario):
    # dep_token values are copied verbatim, so nan/inf must survive the copy.
    token = tu.make_special_input(dtype, scenario)
    ref_token = tu.to_reference(token)
    before = token.clone()

    ref_out = torch.ops.aten._functional_assert_scalar(1.0, MSG, ref_token)
    res_out = flag_gems._functional_assert_scalar(1.0, MSG, token)

    _assert_token_copy(res_out, token, ref_out, before)


@pytest.mark.functional_assert_scalar
@pytest.mark.parametrize("scalar", ZERO_SCALARS)
def test__functional_assert_scalar_zero_scalar_raises(scalar):
    token = tu.make_input(torch.float32, (4,), ["-1", "1"])

    with pytest.raises(RuntimeError) as excinfo:
        flag_gems._functional_assert_scalar(scalar, MSG, token)

    assert MSG in str(excinfo.value)


@pytest.mark.functional_assert_scalar
@pytest.mark.parametrize("msg,expected", MESSAGE_CASES)
def test__functional_assert_scalar_assert_message(msg, expected):
    token = tu.make_input(torch.float32, (4,), ["-1", "1"])

    with pytest.raises(RuntimeError) as excinfo:
        flag_gems._functional_assert_scalar(0.0, msg, token)

    assert expected in str(excinfo.value)


@pytest.mark.functional_assert_scalar
def test__functional_assert_scalar_rejects_tensor_self():
    # A 0-dim Tensor is rejected by schema validation ('Expected a value of type
    # number for argument self'), so the Scalar operand is always a Python number.
    token = tu.make_input(torch.float32, (4,), ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems._functional_assert_scalar(
            torch.ones((), device=flag_gems.device), MSG, token
        )


@pytest.mark.functional_assert_scalar
def test__functional_assert_scalar_rejects_non_number_self():
    token = tu.make_input(torch.float32, (4,), ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems._functional_assert_scalar("1.0", MSG, token)


@pytest.mark.functional_assert_scalar
@pytest.mark.parametrize("msg", [None, 123, 1.5, ["m"]])
def test__functional_assert_scalar_rejects_non_str_message(msg):
    # bytes are accepted natively as a str argument, so they are not a negative case.
    token = tu.make_input(torch.float32, (4,), ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems._functional_assert_scalar(1.0, msg, token)


@pytest.mark.functional_assert_scalar
@pytest.mark.parametrize("dep_token", [None, 7, "token", [1.0, 2.0]])
def test__functional_assert_scalar_rejects_non_tensor_token(dep_token):
    with pytest.raises(RuntimeError):
        flag_gems._functional_assert_scalar(1.0, MSG, dep_token)


@pytest.mark.functional_assert_scalar
def test__functional_assert_scalar_rejects_missing_token():
    # A missing required argument is a TypeError for a Python-level candidate and a
    # RuntimeError for the native schema; both are valid rejections.
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._functional_assert_scalar(1.0, MSG)
