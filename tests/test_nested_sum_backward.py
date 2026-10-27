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

# Nested tensor shapes covering 1D-to-4D inputs; reduction dims 0/1/-1 exercise
# the broadcast-back gradient kernel across different inner/outer sizes.
NESTED_SUM_BACKWARD_SHAPES = [
    (10,),
    (4, 8),
    (4, 8, 16),
    (2, 3, 4, 5),
    (8, 16, 32),
    (3, 7, 11),
    (64, 512),
    (32, 256, 256),
    (1, 8192),
    (32, 50257),
]


def _ref_nested_sum_backward(grad, self, dim, keepdim):
    if not keepdim:
        grad = grad.unsqueeze(dim)
    return grad.expand_as(self).contiguous()


@pytest.mark.nested_sum_backward
@pytest.mark.parametrize("shape", NESTED_SUM_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_nested_sum_backward(shape, dtype, dim, keepdim):
    ndim = len(shape)
    actual_dim = dim + ndim if dim < 0 else dim

    if actual_dim >= ndim:
        # Invalid dim: out of range for this shape.
        return

    res_self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_self = utils.to_reference(res_self)

    if keepdim:
        grad_shape = list(shape)
        grad_shape[actual_dim] = 1
    else:
        grad_shape = list(shape)
        grad_shape.pop(actual_dim)

    res_grad = torch.randn(grad_shape, dtype=dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(res_grad)

    ref_out = _ref_nested_sum_backward(ref_grad, ref_self, actual_dim, keepdim)

    res_out = flag_gems._nested_sum_backward(res_grad, res_self, [actual_dim], keepdim)

    assert res_out.shape == tuple(shape)
    assert res_out.dtype == res_grad.dtype

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.nested_sum_backward
@pytest.mark.parametrize("shape", [(1, 1, 1), (1,), (2, 1, 4), (1, 3, 1, 5)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_sum_backward_edge(shape, dtype):
    dim = 0

    res_self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_self = utils.to_reference(res_self)

    for keepdim in (True, False):
        if keepdim:
            grad_shape = list(shape)
            grad_shape[dim] = 1
        else:
            grad_shape = list(shape)
            grad_shape.pop(dim)

        res_grad = torch.randn(grad_shape, dtype=dtype, device=flag_gems.device)
        ref_grad = utils.to_reference(res_grad)

        ref_out = _ref_nested_sum_backward(ref_grad, ref_self, dim, keepdim)

        res_out = flag_gems._nested_sum_backward(res_grad, res_self, [dim], keepdim)

        assert res_out.shape == tuple(shape)
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.nested_sum_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_sum_backward_non_contiguous(dtype):
    base_shape = (8, 16, 32)

    res_self = torch.randn(base_shape, dtype=dtype, device=flag_gems.device)
    res_self = res_self.transpose(0, 1)  # non-contiguous

    shape = list(res_self.shape)
    dim = 1

    grad_shape = list(shape)
    grad_shape[dim] = 1
    res_grad = torch.randn(grad_shape, dtype=dtype, device=flag_gems.device)

    ref_self = utils.to_reference(res_self)
    ref_grad = utils.to_reference(res_grad)
    ref_out = _ref_nested_sum_backward(ref_grad, ref_self, dim, True)

    res_out = flag_gems._nested_sum_backward(res_grad, res_self, [dim], True)

    assert res_out.shape == tuple(shape)
    utils.gems_assert_equal(res_out, ref_out)
