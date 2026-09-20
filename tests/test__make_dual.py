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
    "_make_dual",
    MarkDecorator(Mark("_make_dual", (), {}, _ispytest=True), _ispytest=True),
)

# Attach a tangent at the active forward-AD level while aliasing the primal.
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

_MAKE_DUAL_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
_MAKE_DUAL_MUTATION_SHAPES = [(16, 32), (4, 8, 16)]
_MAKE_DUAL_EMPTY_SHAPES = [(0,), (2, 0, 3)]
_INACTIVE_LEVELS = [-1, 0, 1, 3]
_MISMATCHED_SHAPES = [((4, 5), (3, 7)), ((4, 5), (5,)), ((16,), (8,))]
_NON_FLOAT_PRIMAL_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
    torch.bool,
]

_SPECIAL_DTYPES = utils.ALL_FLOAT_DTYPES + [torch.float8_e4m3fn, torch.float8_e5m2]


def _assert_view_semantics(res_out, ref_out, inp):
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.data_ptr() == inp.data_ptr()


def _assert_dual_semantics(res_out, ref_out):
    res_primal, res_tangent = torch.autograd.forward_ad.unpack_dual(res_out)
    ref_primal, ref_tangent_out = torch.autograd.forward_ad.unpack_dual(ref_out)
    assert isinstance(res_tangent, torch.Tensor)
    tu.assert_result_equal(res_primal, ref_primal)
    tu.assert_result_equal(res_tangent, ref_tangent_out)


@pytest.mark._make_dual
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", DUAL_DTYPES)
def test__make_dual(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    tangent = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_tangent = tu.to_reference(tangent)

    with dual_level() as level:
        ref_out = torch.ops.aten._make_dual(ref_inp, ref_tangent, level)
        res_out = flag_gems._make_dual(inp, tangent, level)

        _assert_view_semantics(res_out, ref_out, inp)
        _assert_dual_semantics(res_out, ref_out)


@pytest.mark._make_dual
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", DUAL_DTYPES)
def test__make_dual_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    tangent = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    ref_tangent = tu.to_reference(tangent)

    with dual_level() as level:
        ref_out = torch.ops.aten._make_dual(ref_inp, ref_tangent, level)
        res_out = flag_gems._make_dual(inp, tangent, level)

        _assert_view_semantics(res_out, ref_out, inp)
        res_primal, res_tangent = torch.autograd.forward_ad.unpack_dual(res_out)
        ref_primal, ref_tangent_out = torch.autograd.forward_ad.unpack_dual(ref_out)
        tu.assert_result_equal(res_primal, ref_primal)
        tu.assert_result_equal(res_tangent, ref_tangent_out)


@pytest.mark._make_dual
@pytest.mark.parametrize("shape", _MAKE_DUAL_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", DUAL_DTYPES)
def test__make_dual_non_contiguous(shape, dtype):
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    tangent = tu.make_input(dtype, inp.shape, ["-1", "1"])
    ref_tangent = tu.to_reference(tangent)
    assert not inp.is_contiguous()

    with dual_level() as level:
        ref_out = torch.ops.aten._make_dual(ref_inp, ref_tangent, level)
        res_out = flag_gems._make_dual(inp, tangent, level)

        _assert_view_semantics(res_out, ref_out, inp)
        _assert_dual_semantics(res_out, ref_out)


@pytest.mark._make_dual
@pytest.mark.parametrize("shape", _MAKE_DUAL_EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", DUAL_DTYPES)
def test__make_dual_empty(shape, dtype):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    tangent = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)
    ref_tangent = tu.to_reference(tangent)

    with dual_level() as level:
        ref_out = torch.ops.aten._make_dual(ref_inp, ref_tangent, level)
        res_out = flag_gems._make_dual(inp, tangent, level)

        _assert_view_semantics(res_out, ref_out, inp)
        _assert_dual_semantics(res_out, ref_out)


@pytest.mark._make_dual
@pytest.mark.parametrize("shape", _MAKE_DUAL_MUTATION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__make_dual_mutation(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    tangent = tu.make_input(dtype, shape, ["-1", "1"])
    ref_tangent = tu.to_reference(tangent)

    with dual_level() as level:
        ref_out = torch.ops.aten._make_dual(ref_inp, ref_tangent, level)
        res_out = flag_gems._make_dual(inp, tangent, level)

        ref_out.fill_(2.5)
        res_out.fill_(2.5)

        tu.assert_result_equal(res_out, ref_out)
        assert res_out.data_ptr() == inp.data_ptr()
        tu.assert_result_equal(inp, ref_inp)
        tu.assert_result_equal(tangent, ref_tangent)


@pytest.mark._make_dual
@pytest.mark.parametrize("dtype", tu.selected_cases(_SPECIAL_DTYPES))
def test__make_dual_special_values(dtype):
    values = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), 1.5, -1.5, float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = tu.to_reference(values)
    tangent = torch.ones_like(values)
    ref_tangent = tu.to_reference(tangent)

    with dual_level() as level:
        ref_out = torch.ops.aten._make_dual(ref_inp, ref_tangent, level)
        res_out = flag_gems._make_dual(values, tangent, level)

        tu.assert_result_equal(res_out, ref_out)
        # signbit has no fp8 kernel, so the sign check goes through float32.
        res_f = res_out.to(torch.float32)
        values_f = values.to(torch.float32)
        assert torch.signbit(res_f[0]).item() == torch.signbit(values_f[0]).item()
        assert torch.signbit(res_f[1]).item() == torch.signbit(values_f[1]).item()


@pytest.mark._make_dual
@pytest.mark.parametrize("dtype", _NON_FLOAT_PRIMAL_DTYPES)
def test__make_dual_rejects_non_float_primal(dtype):
    with dual_level() as level:
        inp = tu.make_input(dtype, (4, 5), ["-1", "1"])
        tangent = tu.make_input(torch.float32, (4, 5), ["-1", "1"])
        with pytest.raises(RuntimeError):
            torch.ops.aten._make_dual(
                tu.to_reference(inp),
                tu.to_reference(tangent),
                level,
            )
        # The generated wrapper may fail on the first touch of the input
        # (attribute lookup, triton input validation or a dispatcher cast), so
        # accept the plausible Python failure modes; the point is that it must
        # fail rather than silently accept the int/bool primal.
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            flag_gems._make_dual(inp, tangent, level)


@pytest.mark._make_dual
def test__make_dual_rejects_non_tensor_primal():
    with dual_level() as level:
        tangent = tu.make_input(torch.float32, (4, 5), ["-1", "1"])
        with pytest.raises(RuntimeError):
            torch.ops.aten._make_dual(3.14, tu.to_reference(tangent), level)
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            flag_gems._make_dual(3.14, tangent, level)


@pytest.mark._make_dual
@pytest.mark.parametrize("primal_shape,tangent_shape", _MISMATCHED_SHAPES)
def test__make_dual_rejects_tangent_size_mismatch(primal_shape, tangent_shape):
    with dual_level() as level:
        inp = tu.make_input(torch.float32, primal_shape, ["-1", "1"])
        tangent = tu.make_input(torch.float32, tangent_shape, ["-1", "1"])
        with pytest.raises(RuntimeError):
            torch.ops.aten._make_dual(
                tu.to_reference(inp),
                tu.to_reference(tangent),
                level,
            )
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            flag_gems._make_dual(inp, tangent, level)


@pytest.mark._make_dual
@pytest.mark.parametrize("level", _INACTIVE_LEVELS)
def test__make_dual_rejects_inactive_level(level):
    inp = tu.make_input(torch.float32, (4, 5), ["-1", "1"])
    tangent = tu.make_input(torch.float32, (4, 5), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._make_dual(
            tu.to_reference(inp),
            tu.to_reference(tangent),
            level,
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._make_dual(inp, tangent, level)


@pytest.mark._make_dual
def test__make_dual_rejects_non_int_level():
    with dual_level():
        inp = tu.make_input(torch.float32, (4, 5), ["-1", "1"])
        tangent = tu.make_input(torch.float32, (4, 5), ["-1", "1"])
        with pytest.raises(RuntimeError):
            torch.ops.aten._make_dual(
                tu.to_reference(inp),
                tu.to_reference(tangent),
                1.5,
            )
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            flag_gems._make_dual(inp, tangent, 1.5)


@pytest.mark._make_dual
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(DUAL_DTYPES))
)
def test__make_dual_special_scenarios(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    reference = tu.to_reference(inp)
    tangent = tu.make_special_input(dtype, scenario)
    ref_tangent = tu.to_reference(tangent)
    with torch.autograd.forward_ad.dual_level() as level:
        ref_dual = torch.ops.aten._make_dual(reference, ref_tangent, level)
        result = flag_gems._make_dual(inp, tangent, level)
        actual = torch.ops.aten._unpack_dual(result, level)
        expected = torch.ops.aten._unpack_dual(ref_dual, level)
        for actual_part, expected_part in zip(actual, expected):
            tu.assert_result_equal(actual_part, expected_part)
