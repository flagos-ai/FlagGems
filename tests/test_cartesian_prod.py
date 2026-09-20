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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Gather one row per combination, with the first input varying slowest.
_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]

_SUPPORTED_DTYPES = (
    _FP8_DTYPES
    + [torch.int8, torch.uint8]
    + utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [torch.bool]
)

_RANGE_PAIRS = [
    (dtype, value_range)
    for dtype in _SUPPORTED_DTYPES
    if dtype != torch.bool
    for value_range in tu.selected_ranges()
]

_CARTESIAN_PROD_SIZES = tu.selected_cases(
    [
        [8],
        [1],
        [0],
        [3, 5],
        [16, 16],
        [1, 7],
        [64, 128],
        [256, 256],
        [2, 4, 3],
        [3, 1, 3],
        [8, 16, 32],
        [2, 5, 8, 3],
        [0, 3],
        [5, 0],
    ],
    quick=[[8], [3, 5], [2, 4, 3]],
)

_BACKWARD_SIZES = tu.selected_cases(
    [[8], [3, 5], [2, 4, 3], [16, 16]], quick=[[8], [3, 5]]
)


@pytest.mark.cartesian_prod
@pytest.mark.parametrize("sizes", _CARTESIAN_PROD_SIZES)
@pytest.mark.parametrize("dtype,value_range", _RANGE_PAIRS)
def test_cartesian_prod(sizes, dtype, value_range):
    inp = [tu.make_input(dtype, (size,), value_range) for size in sizes]
    ref_inp = [tu.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.cartesian_prod(ref_inp)
    res_out = flag_gems.cartesian_prod(inp)

    assert res_out.dtype == ref_out.dtype == dtype
    tu.assert_result_equal(res_out, ref_out)

    # cartesian_prod is a pure gather: the inputs must not be mutated.
    for t, ref_t in zip(inp, ref_inp):
        tu.assert_result_equal(t, ref_t)


@pytest.mark.cartesian_prod
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_cartesian_prod_row_order(dtype):
    a = torch.tensor([0, 1, 2], dtype=dtype, device=flag_gems.device)
    b = torch.tensor([10, 20], dtype=dtype, device=flag_gems.device)
    expected = torch.tensor(
        [[0, 10], [0, 20], [1, 10], [1, 20], [2, 10], [2, 20]],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = [
        tu.to_reference(a),
        tu.to_reference(b),
    ]

    ref_out = torch.ops.aten.cartesian_prod(ref_inp)
    res_out = flag_gems.cartesian_prod([a, b])

    assert res_out.shape == ref_out.shape == (6, 2)
    utils.gems_assert_equal(res_out, tu.to_reference(expected))
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.cartesian_prod
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_cartesian_prod_non_contiguous(dtype):
    base = tu.make_input(dtype, (16,), ["-1", "1"])
    ref_base = tu.to_reference(base)
    other = tu.make_input(dtype, (5,), ["-1", "1"])
    inp = [base[::2], other]
    ref_inp = [ref_base[::2], tu.to_reference(other)]
    assert not inp[0].is_contiguous()

    ref_out = torch.ops.aten.cartesian_prod(ref_inp)
    res_out = flag_gems.cartesian_prod(inp)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cartesian_prod
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(_SUPPORTED_DTYPES))
)
def test_cartesian_prod_nan_inf(dtype, scenario):
    values = tu.make_special_input(dtype, scenario)
    other = torch.tensor([1.0, -1.0], dtype=dtype, device=flag_gems.device)
    ref_inp = [
        tu.to_reference(values),
        tu.to_reference(other),
    ]

    ref_out = torch.ops.aten.cartesian_prod(ref_inp)
    res_out = flag_gems.cartesian_prod([values, other])

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cartesian_prod
# A single input returns a view and supports FP8 backward. Multiple inputs
# repeat values; their backward uses sum, whose CUDA kernel rejects FP8.
@pytest.mark.parametrize(
    "sizes,dtype",
    tu.selected_cases(
        [
            (sizes, dtype)
            for sizes in _BACKWARD_SIZES
            for dtype in utils.ALL_FLOAT_DTYPES + _FP8_DTYPES
            if len(sizes) == 1 or dtype not in _FP8_DTYPES
        ]
    ),
)
def test_cartesian_prod_backward(sizes, dtype):
    out_shape = (sizes[0],) if len(sizes) == 1 else (math.prod(sizes), len(sizes))
    inp = [
        tu.make_input(dtype, (size,), ["-1", "1"]).requires_grad_() for size in sizes
    ]
    grad = tu.make_input(dtype, out_shape, ["-1", "1"])
    ref_inp = [tu.to_reference(t) for t in inp]
    ref_grad = tu.to_reference(grad)

    ref_out = torch.ops.aten.cartesian_prod(ref_inp)
    ref_in_grads = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)

    res_out = flag_gems.cartesian_prod(inp)
    tu.assert_result_equal(res_out, ref_out)

    # One input is a view; multiple inputs sum repeated appearances.
    assert res_out.requires_grad
    res_in_grads = torch.autograd.grad(res_out, inp, grad_outputs=grad)
    for res_grad, ref_grad in zip(res_in_grads, ref_in_grads):
        if len(sizes) == 1:
            tu.assert_result_equal(res_grad, ref_grad)
        else:
            tu.assert_result_close(res_grad, ref_grad)


@pytest.mark.cartesian_prod
def test_cartesian_prod_rejects_empty_list():
    with pytest.raises(RuntimeError):
        torch.ops.aten.cartesian_prod([])
    with pytest.raises((TypeError, ValueError, RuntimeError, IndexError)):
        flag_gems.cartesian_prod([])


@pytest.mark.cartesian_prod
@pytest.mark.parametrize("shape", [(3, 4), ()])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_cartesian_prod_rejects_multidim_input(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    with pytest.raises(RuntimeError):
        torch.ops.aten.cartesian_prod([ref_inp])
    with pytest.raises(RuntimeError):
        flag_gems.cartesian_prod([inp])


@pytest.mark.cartesian_prod
def test_cartesian_prod_rejects_mixed_dtype():
    a = tu.make_input(torch.float32, (4,), ["-1", "1"])
    b = tu.make_input(torch.int32, (4,), ["-1", "1"])
    ref_inp = [
        tu.to_reference(a),
        tu.to_reference(b),
    ]
    with pytest.raises(RuntimeError):
        torch.ops.aten.cartesian_prod(ref_inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.cartesian_prod([a, b])


@pytest.mark.cartesian_prod
def test_cartesian_prod_rejects_non_tensor():
    a = tu.make_input(torch.float32, (4,), ["-1", "1"])
    ref_inp = tu.to_reference(a)
    with pytest.raises(RuntimeError):
        torch.ops.aten.cartesian_prod([ref_inp, 3.14])
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.cartesian_prod([a, 3.14])
