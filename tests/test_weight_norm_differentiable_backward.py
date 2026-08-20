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
from . import conftest as cfg

SHAPES = [(4, 8), (32, 257), (3, 5, 17)]
FLOAT_DTYPES = [torch.float32] if cfg.QUICK_MODE else utils.ALL_FLOAT_DTYPES


def _make_inputs(shape, dim, dtype, device):
    broadcast_shape = [1] * len(shape)
    broadcast_shape[dim] = shape[dim]
    grad_w = torch.randn(shape, dtype=dtype, device=device)
    saved_v = torch.randn(shape, dtype=dtype, device=device)
    saved_g = torch.randn(broadcast_shape, dtype=dtype, device=device)
    _, saved_norms = torch._weight_norm_interface(saved_v, saved_g, dim)
    return grad_w, saved_v, saved_g, saved_norms, dim


@pytest.mark.weight_norm_differentiable_backward
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("first_dim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_weight_norm_differentiable_backward(shape, first_dim, dtype):
    dim = 0 if first_dim else len(shape) - 1
    result_args = _make_inputs(shape, dim, dtype, flag_gems.device)
    reference_args = tuple(
        utils.to_reference(arg, True) if isinstance(arg, torch.Tensor) else arg
        for arg in result_args
    )

    reference = torch.ops.aten._weight_norm_differentiable_backward(*reference_args)
    with flag_gems.use_gems():
        result = torch.ops.aten._weight_norm_differentiable_backward(*result_args)

    reduce_size = torch.tensor(shape).prod().item() // shape[dim]
    for actual, expected in zip(result, reference):
        utils.gems_assert_close(
            actual,
            expected,
            dtype,
            reduce_dim=reduce_size,
            equal_nan=True,
        )


@pytest.mark.weight_norm_differentiable_backward
@pytest.mark.parametrize("arg_index", range(4))
def test_weight_norm_differentiable_backward_requires_contiguous(arg_index):
    args = list(_make_inputs((4, 6), 0, torch.float32, flag_gems.device))
    args[arg_index] = torch.empty_strided(
        args[arg_index].shape,
        tuple(stride * 2 for stride in args[arg_index].stride()),
        dtype=args[arg_index].dtype,
        device=args[arg_index].device,
    )
    names = ("grad_w", "saved_v", "saved_g", "saved_norms")
    with (
        flag_gems.use_gems(),
        pytest.raises(RuntimeError, match=rf"{names[arg_index]} must be contiguous"),
    ):
        torch.ops.aten._weight_norm_differentiable_backward(*args)


@pytest.mark.weight_norm_differentiable_backward
@pytest.mark.parametrize("dim", [-1, 1])
def test_weight_norm_differentiable_backward_rejects_invalid_dim(dim):
    args = _make_inputs((3, 4, 5), 0, torch.float32, flag_gems.device)
    with (
        flag_gems.use_gems(),
        pytest.raises(
            RuntimeError, match="Expected dim to be the first or last dimension"
        ),
    ):
        torch.ops.aten._weight_norm_differentiable_backward(*args[:-1], dim)


@pytest.mark.weight_norm_differentiable_backward
def test_weight_norm_differentiable_backward_empty_reduction():
    args = _make_inputs((4, 0), 0, torch.float32, flag_gems.device)
    reference_args = tuple(
        utils.to_reference(arg, True) if isinstance(arg, torch.Tensor) else arg
        for arg in args
    )
    reference = torch.ops.aten._weight_norm_differentiable_backward(*reference_args)
    with flag_gems.use_gems():
        result = torch.ops.aten._weight_norm_differentiable_backward(*args)
    for actual, expected in zip(result, reference):
        utils.gems_assert_close(actual, expected, torch.float32, equal_nan=True)


@pytest.mark.weight_norm_differentiable_backward
def test_weight_norm_differentiable_backward_special_norms():
    args = list(_make_inputs((3, 4), 0, torch.float32, flag_gems.device))
    args[3] = torch.tensor(
        [[0.0], [torch.inf], [torch.nan]],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    reference_args = tuple(
        utils.to_reference(arg) if isinstance(arg, torch.Tensor) else arg
        for arg in args
    )
    reference = torch.ops.aten._weight_norm_differentiable_backward(*reference_args)
    with flag_gems.use_gems():
        result = torch.ops.aten._weight_norm_differentiable_backward(*args)
    for actual, expected in zip(result, reference):
        utils.gems_assert_close(actual, expected, torch.float32, equal_nan=True)


@pytest.mark.weight_norm_differentiable_backward
def test_weight_norm_differentiable_backward_double_backward():
    shape = (3, 5)
    dim = 0
    args = list(_make_inputs(shape, dim, torch.float32, flag_gems.device))
    for index in range(3):
        args[index].requires_grad_(True)

    with flag_gems.use_gems():
        grad_v, grad_g = torch.ops.aten._weight_norm_differentiable_backward(*args)
        first_grads = torch.autograd.grad(
            grad_v.sum() + grad_g.sum(), args[:3], create_graph=True
        )
        second_grads = torch.autograd.grad(
            sum(grad.sum() for grad in first_grads), args[:3]
        )
    assert all(grad is not None for grad in second_grads)
