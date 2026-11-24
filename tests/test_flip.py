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

FLIP_DIMS = [(0,), (-2,), (2,), (0, 2), (2, 1), (0, -1, 1)]


@pytest.mark.flip_backward
@pytest.mark.parametrize("dims", [(), (0,), (-1,), (0, 2)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "sliced"])
def test_flip_backward(dims, dtype, layout):
    inp = torch.randn((4, 5, 6), dtype=dtype, device=flag_gems.device)
    if layout == "transpose":
        inp = inp.transpose(0, 2)
    elif layout == "sliced":
        inp = inp[::2]
    grad = torch.randn_like(inp)
    ref = utils.to_reference(inp.detach(), False).requires_grad_()
    actual = inp.detach().requires_grad_()
    expected = torch.autograd.grad(
        torch.flip(ref, dims), ref, utils.to_reference(grad, False)
    )[0]
    result = torch.autograd.grad(flag_gems.flip(actual, dims), actual, grad)[0]
    utils.gems_assert_equal(result, expected)


@pytest.mark.flip_backward
@pytest.mark.parametrize("shape", [(0, 3), (1, 3)])
def test_flip_backward_empty_or_singleton(shape):
    inp = torch.empty(shape, device=flag_gems.device, requires_grad=True)
    grad = torch.randn_like(inp)
    result = torch.autograd.grad(flag_gems.flip(inp, (0,)), inp, grad)[0]
    utils.gems_assert_equal(result, utils.to_reference(grad, False))


def get_max_ndim(shape, dims):
    max_ndim = max(len(shape), len(dims))
    for dim in dims:
        dim = dim + 1 if dim >= 0 else -dim
        if dim > max_ndim:
            max_ndim = dim

    return max_ndim


@pytest.mark.flip
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dims", FLIP_DIMS)
def test_flip(shape, dtype, dims):
    if dtype in utils.ALL_FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-1000, 1000, shape, device=flag_gems.device).to(dtype)
    max_ndim = get_max_ndim(shape, dims)
    inp = utils.unsqueeze_tensor(inp, max_ndim)
    ref_inp = utils.to_reference(inp, False)

    res_out = flag_gems.flip(inp, dims)
    ref_out = torch.flip(ref_inp, dims)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.flip
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES)
@pytest.mark.parametrize("dims", FLIP_DIMS)
def test_flip_with_non_dense_input(shape, dtype, dims):
    max_ndim = get_max_ndim(shape, dims)
    shape = utils.unsqueeze_tuple(shape, max(max_ndim, 2))

    shape_dialted = tuple(item * 2 for item in shape)
    if dtype in utils.ALL_FLOAT_DTYPES:
        inp = torch.randn(shape_dialted, dtype=dtype, device=flag_gems.device)[::2, ::2]
    else:
        inp = torch.randint(-1000, 1000, shape_dialted, device=flag_gems.device).to(
            dtype
        )[::2, ::2]
    ref_inp = utils.to_reference(inp, False)

    res_out = flag_gems.flip(inp, dims)
    ref_out = torch.flip(ref_inp, dims)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.flip
@pytest.mark.flip_backward
@pytest.mark.parametrize("dims", [(0, 2), (1, 3), (0, -1, 2)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("layout", ["contiguous", "transpose"])
def test_flip_4d_multi_axis_forward_and_backward(dims, dtype, layout):
    inp = torch.randn((2, 3, 4, 5), dtype=dtype, device=flag_gems.device)
    if layout == "transpose":
        inp = inp.transpose(1, 2)
    grad = torch.randn_like(inp)

    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()
    res_inp = inp.detach().requires_grad_()

    ref_out = torch.flip(ref_inp, dims)
    res_out = flag_gems.flip(res_inp, dims)
    utils.gems_assert_equal(res_out, ref_out)

    (ref_dx,) = torch.autograd.grad(ref_out, ref_inp, utils.to_reference(grad, False))
    (res_dx,) = torch.autograd.grad(res_out, res_inp, grad)
    utils.gems_assert_equal(res_dx, ref_dx)
