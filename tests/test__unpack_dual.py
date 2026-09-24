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
from _pytest.mark.structures import Mark, MarkDecorator
from torch.autograd.forward_ad import dual_level

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Register underscore-prefixed pytest markers explicitly.
setattr(
    pytest.mark,
    "_unpack_dual",
    MarkDecorator(Mark("_unpack_dual", (), {}, _ispytest=True), _ispytest=True),
)

# Return an alias of the primal and its tangent (None for plain tensors).
# Complex32 is excluded because the comparison helpers cannot materialize it.
DUAL_DTYPES = (
    [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float32,
        torch.bfloat16,
        torch.float16,
    ]
    + ([torch.float64] if utils.fp64_is_supported else [])
    + [torch.complex64]
)

PLAIN_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool, torch.complex64]
)

UNPACK_DUAL_LEVELS = [0, 1, 3]

_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
_MUTATION_SHAPES = [(16, 32), (4, 8, 16)]
_EMPTY_SHAPES = [(0,), (2, 0, 3)]


def _assert_primal_view(res_primal, ref_primal, dual):
    assert res_primal.stride() == ref_primal.stride()
    assert res_primal.storage_offset() == ref_primal.storage_offset()
    assert res_primal.data_ptr() == dual.data_ptr()


@pytest.mark._unpack_dual
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", DUAL_DTYPES)
def test__unpack_dual_dual_tensor(shape, dtype):
    primal = tu.make_input(dtype, shape, ["-1", "1"])
    tangent = tu.make_input(dtype, shape, ["-1", "1"])
    ref_primal = tu.to_reference(primal)
    ref_tangent = tu.to_reference(tangent)

    with dual_level() as level:
        ref_dual = torch.ops.aten._make_dual(ref_primal, ref_tangent, level)
        ref_primal_out, ref_tangent_out = torch.ops.aten._unpack_dual(ref_dual, level)

        dual = torch.ops.aten._make_dual(primal, tangent, level)
        res_primal_out, res_tangent_out = flag_gems._unpack_dual(dual, level)

        # A dual tensor created with a tangent must yield a tensor tangent, not
        # None.
        assert isinstance(res_tangent_out, torch.Tensor)
        tu.assert_result_equal(res_primal_out, ref_primal_out)
        tu.assert_result_equal(res_tangent_out, ref_tangent_out)
        _assert_primal_view(res_primal_out, ref_primal_out, dual)


@pytest.mark._unpack_dual
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", DUAL_DTYPES)
def test__unpack_dual_dual_tensor_value_ranges(shape, value_range, dtype):
    primal = tu.make_input(dtype, shape, value_range)
    tangent = tu.make_input(dtype, shape, value_range)
    ref_primal = tu.to_reference(primal)
    ref_tangent = tu.to_reference(tangent)

    with dual_level() as level:
        ref_dual = torch.ops.aten._make_dual(ref_primal, ref_tangent, level)
        ref_primal_out, ref_tangent_out = torch.ops.aten._unpack_dual(ref_dual, level)

        dual = torch.ops.aten._make_dual(primal, tangent, level)
        res_primal_out, res_tangent_out = flag_gems._unpack_dual(dual, level)

        assert isinstance(res_tangent_out, torch.Tensor)
        tu.assert_result_equal(res_primal_out, ref_primal_out)
        tu.assert_result_equal(res_tangent_out, ref_tangent_out)
        _assert_primal_view(res_primal_out, ref_primal_out, dual)


@pytest.mark._unpack_dual
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("level", UNPACK_DUAL_LEVELS)
@pytest.mark.parametrize("dtype", PLAIN_DTYPES)
def test__unpack_dual_plain_tensor(shape, level, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_primal_out, _ = torch.ops.aten._unpack_dual(ref_inp, level)
    res_primal_out, res_tangent_out = flag_gems._unpack_dual(inp, level)

    assert res_tangent_out is None
    tu.assert_result_equal(res_primal_out, ref_primal_out)
    _assert_primal_view(res_primal_out, ref_primal_out, inp)


@pytest.mark._unpack_dual
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", PLAIN_DTYPES)
def test__unpack_dual_plain_tensor_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_primal_out, _ = torch.ops.aten._unpack_dual(ref_inp, 0)
    res_primal_out, res_tangent_out = flag_gems._unpack_dual(inp, 0)

    assert res_tangent_out is None
    tu.assert_result_equal(res_primal_out, ref_primal_out)
    _assert_primal_view(res_primal_out, ref_primal_out, inp)


@pytest.mark._unpack_dual
@pytest.mark.parametrize("shape", _NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", DUAL_DTYPES)
def test__unpack_dual_non_contiguous(shape, dtype):
    # Forward AD may materialize tangent storage; only primal layout must match.
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    primal = base[..., ::2]
    ref_primal = ref_base[..., ::2]
    tangent = tu.make_input(dtype, primal.shape, ["-1", "1"])
    ref_tangent = tu.to_reference(tangent)
    assert not primal.is_contiguous()

    with dual_level() as level:
        ref_dual = torch.ops.aten._make_dual(ref_primal, ref_tangent, level)
        ref_primal_out, ref_tangent_out = torch.ops.aten._unpack_dual(ref_dual, level)

        dual = torch.ops.aten._make_dual(primal, tangent, level)
        res_primal_out, res_tangent_out = flag_gems._unpack_dual(dual, level)

        assert res_primal_out.stride() == ref_primal_out.stride()
        assert res_primal_out.storage_offset() == ref_primal_out.storage_offset()
        assert res_primal_out.data_ptr() == primal.data_ptr()
        tu.assert_result_equal(res_primal_out, ref_primal_out)
        tu.assert_result_equal(res_tangent_out, ref_tangent_out)


@pytest.mark._unpack_dual
@pytest.mark.parametrize("shape", _MUTATION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__unpack_dual_mutation(shape, dtype):
    primal = tu.make_input(dtype, shape, ["-1", "1"])
    ref_primal = tu.to_reference(primal)
    tangent = tu.make_input(dtype, shape, ["-1", "1"])
    ref_tangent = tu.to_reference(tangent)

    with dual_level() as level:
        ref_dual = torch.ops.aten._make_dual(ref_primal, ref_tangent, level)
        ref_primal_out, _ = torch.ops.aten._unpack_dual(ref_dual, level)

        dual = torch.ops.aten._make_dual(primal, tangent, level)
        res_primal_out, _ = flag_gems._unpack_dual(dual, level)

        ref_primal_out.fill_(2.5)
        res_primal_out.fill_(2.5)

        assert res_primal_out.data_ptr() == dual.data_ptr()
        tu.assert_result_equal(res_primal_out, ref_primal_out)
        utils.gems_assert_equal(primal, ref_primal)


@pytest.mark._unpack_dual
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES))
def test__unpack_dual_special_values(dtype):
    values = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), 1.5, -1.5, float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_primal = tu.to_reference(values)
    tangent = torch.ones_like(values)
    ref_tangent = tu.to_reference(tangent)

    with dual_level() as level:
        ref_dual = torch.ops.aten._make_dual(ref_primal, ref_tangent, level)
        ref_primal_out, ref_tangent_out = torch.ops.aten._unpack_dual(ref_dual, level)

        dual = torch.ops.aten._make_dual(values, tangent, level)
        res_primal_out, res_tangent_out = flag_gems._unpack_dual(dual, level)

        utils.gems_assert_equal(res_primal_out, ref_primal_out, equal_nan=True)
        utils.gems_assert_equal(res_tangent_out, ref_tangent_out, equal_nan=True)
        assert (
            torch.signbit(res_primal_out[0]).item() == torch.signbit(values[0]).item()
        )
        assert (
            torch.signbit(res_primal_out[1]).item() == torch.signbit(values[1]).item()
        )


@pytest.mark._unpack_dual
@pytest.mark.parametrize("shape", _EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", DUAL_DTYPES)
def test__unpack_dual_empty(shape, dtype):
    primal = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    tangent = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_primal = tu.to_reference(primal)
    ref_tangent = tu.to_reference(tangent)

    with dual_level() as level:
        ref_dual = torch.ops.aten._make_dual(ref_primal, ref_tangent, level)
        ref_primal_out, ref_tangent_out = torch.ops.aten._unpack_dual(ref_dual, level)

        dual = torch.ops.aten._make_dual(primal, tangent, level)
        res_primal_out, res_tangent_out = flag_gems._unpack_dual(dual, level)

        tu.assert_result_equal(res_primal_out, ref_primal_out)
        tu.assert_result_equal(res_tangent_out, ref_tangent_out)
        _assert_primal_view(res_primal_out, ref_primal_out, dual)


@pytest.mark._unpack_dual
def test__unpack_dual_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten._unpack_dual(3.14, 0)
    # The generated wrapper may fail on the first touch of the input (attribute
    # lookup, triton input validation or a dispatcher cast), so accept the
    # plausible Python failure modes; the point is that it must fail rather
    # than silently accept the scalar.
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._unpack_dual(3.14, 0)


@pytest.mark._unpack_dual
def test__unpack_dual_rejects_non_int_level():
    inp = tu.make_input(torch.float32, (8,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten._unpack_dual(ref_inp, 1.5)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._unpack_dual(inp, 1.5)


@pytest.mark._unpack_dual
@pytest.mark.parametrize("bad_level", [-1, 1])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__unpack_dual_rejects_inactive_level(dtype, bad_level):
    primal = tu.make_input(dtype, (4, 5), ["-1", "1"])
    tangent = tu.make_input(dtype, (4, 5), ["-1", "1"])
    ref_primal = tu.to_reference(primal)
    ref_tangent = tu.to_reference(tangent)

    with dual_level() as level:
        ref_dual = torch.ops.aten._make_dual(ref_primal, ref_tangent, level)
        dual = torch.ops.aten._make_dual(primal, tangent, level)
        inactive = bad_level if bad_level != level else level + 1

        with pytest.raises(RuntimeError):
            torch.ops.aten._unpack_dual(ref_dual, inactive)
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            flag_gems._unpack_dual(dual, inactive)


@pytest.mark._unpack_dual
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(DUAL_DTYPES))
)
def test__unpack_dual_special_scenarios(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    reference = tu.to_reference(inp)
    tangent = tu.make_special_input(dtype, scenario)
    ref_tangent = tu.to_reference(tangent)
    with torch.autograd.forward_ad.dual_level() as level:
        ref_dual = torch.ops.aten._make_dual(reference, ref_tangent, level)
        dual = torch.ops.aten._make_dual(inp, tangent, level)
        actual = flag_gems._unpack_dual(dual, level)
        expected = torch.ops.aten._unpack_dual(ref_dual, level)
        for actual_part, expected_part in zip(actual, expected):
            tu.assert_result_equal(actual_part, expected_part)
