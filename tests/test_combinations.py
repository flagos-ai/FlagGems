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

# Gather length-r combinations from a 1-D input, optionally with replacement.
_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool]
)

_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]

_FLOAT_DTYPES = [dtype for dtype in _DTYPES if dtype.is_floating_point]

# FP8 nonempty backward requires unavailable masked_scatter/add kernels.
_BACKWARD_DTYPES = [
    dtype for dtype in _DTYPES if dtype.is_floating_point and dtype not in _FP8_DTYPES
]

_GRID_DTYPES = [torch.float32, torch.float16, torch.int32, torch.bool]

_EMPTY_DTYPES = [torch.float32, torch.int32, torch.bool, torch.int8]

# Quick shapes have no 1-D entry; use the same two local representatives.
_SPEC_1D_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) == 1] or [
    (4,),
    (8,),
]

_LEVEL_SHAPES = tu.selected_cases(
    [(1,), (2,), (4,), (8,), (16,), (64,), (96,)], quick=[(4,), (8,)]
)

_R_VALUES = [1, 2, 3]

_REPLACEMENT_MODES = [False, True]


@pytest.mark.combinations
@pytest.mark.parametrize("shape", _SPEC_1D_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_combinations_spec_shapes_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.combinations(ref_inp, 2, False)
    res_out = flag_gems.combinations(inp, 2, False)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize("shape", _LEVEL_SHAPES)
@pytest.mark.parametrize("r", _R_VALUES)
@pytest.mark.parametrize("with_replacement", _REPLACEMENT_MODES)
@pytest.mark.parametrize("dtype", _GRID_DTYPES)
def test_combinations_shapes_r_replacement(shape, r, with_replacement, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.combinations(ref_inp, r, with_replacement)
    res_out = flag_gems.combinations(inp, r, with_replacement)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize("r", _R_VALUES)
@pytest.mark.parametrize("with_replacement", _REPLACEMENT_MODES)
@pytest.mark.parametrize("dtype", _EMPTY_DTYPES)
def test_combinations_empty_input(r, with_replacement, dtype):
    inp = tu.make_input(dtype, (0,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.combinations(ref_inp, r, with_replacement)
    res_out = flag_gems.combinations(inp, r, with_replacement)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize("r", [0, 5, 10])
@pytest.mark.parametrize("with_replacement", _REPLACEMENT_MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_combinations_r_boundaries(r, with_replacement, dtype):
    inp = tu.make_input(dtype, (4,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.combinations(ref_inp, r, with_replacement)
    res_out = flag_gems.combinations(inp, r, with_replacement)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize("dtype", _DTYPES)
def test_combinations_non_contiguous(dtype):
    base = tu.make_input(dtype, (32,), ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base[::2]
    ref_inp = ref_base[::2]

    ref_out = torch.ops.aten.combinations(ref_inp, 2, False)
    res_out = flag_gems.combinations(inp, 2, False)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(_DTYPES))
)
def test_combinations_nan_inf(dtype, scenario):
    values = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(values)

    ref_out = torch.ops.aten.combinations(ref_inp, 2, False)
    res_out = flag_gems.combinations(values, 2, False)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize("dtype", _GRID_DTYPES)
def test_combinations_does_not_mutate_input(dtype):
    inp = tu.make_input(dtype, (16,), ["-1", "1"])
    before = tu.to_reference(inp)

    flag_gems.combinations(inp, 2, False)

    tu.assert_result_equal(inp, before)


@pytest.mark.combinations
@pytest.mark.parametrize("n", [0, 1, 8])
@pytest.mark.parametrize("r", _R_VALUES)
@pytest.mark.parametrize("with_replacement", _REPLACEMENT_MODES)
@pytest.mark.parametrize("dtype", tu.selected_cases(_BACKWARD_DTYPES))
def test_combinations_backward(n, r, with_replacement, dtype):
    inp = tu.make_input(dtype, (n,), ["-1", "1"]).requires_grad_()
    ref_inp = tu.to_reference(inp)
    ref_out = torch.ops.aten.combinations(ref_inp, r, with_replacement)
    grad = tu.make_input(dtype, ref_out.shape, ["-1", "1"])
    ref_grad = tu.to_reference(grad)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    res_out = flag_gems.combinations(inp, r, with_replacement)
    tu.assert_result_equal(res_out, ref_out)

    assert res_out.requires_grad
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
    # Singleton gathers copy gradients; empty results contribute exact zeros.
    if r == 1 or n == 0 or (r > n and not with_replacement):
        tu.assert_result_equal(res_in_grad, ref_in_grad)
    else:
        tu.assert_result_close(res_in_grad, ref_in_grad)


@pytest.mark.combinations
@pytest.mark.parametrize("n", [0, 1, 8])
@pytest.mark.parametrize("with_replacement", _REPLACEMENT_MODES)
@pytest.mark.parametrize("dtype", tu.selected_cases(_FLOAT_DTYPES))
def test_combinations_zero_r_no_autograd(n, with_replacement, dtype):
    inp = tu.make_input(dtype, (n,), ["-1", "1"]).requires_grad_()
    ref_inp = tu.to_reference(inp)
    ref_out = torch.ops.aten.combinations(ref_inp, 0, with_replacement)
    res_out = flag_gems.combinations(inp, 0, with_replacement)
    tu.assert_result_equal(res_out, ref_out)
    assert not res_out.requires_grad
    assert res_out.grad_fn is None


@pytest.mark.combinations
@pytest.mark.parametrize("shape", [(), (4, 4), (2, 3, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_combinations_raises_on_non_1d(shape, dtype):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.combinations(ref_inp, 2, False)
    with pytest.raises((RuntimeError, TypeError, ValueError, IndexError)):
        flag_gems.combinations(inp, 2, False)


@pytest.mark.combinations
def test_combinations_raises_on_negative_r():
    inp = torch.arange(4, dtype=torch.float32, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.combinations(ref_inp, -1, False)
    with pytest.raises((RuntimeError, TypeError, ValueError, IndexError)):
        flag_gems.combinations(inp, -1, False)


@pytest.mark.combinations
def test_combinations_raises_on_non_int_r():
    inp = torch.arange(4, dtype=torch.float32, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.combinations(ref_inp, 2.0, False)
    with pytest.raises((RuntimeError, TypeError, ValueError, IndexError)):
        flag_gems.combinations(inp, 2.0, False)
