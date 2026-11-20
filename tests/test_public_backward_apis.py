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

from collections.abc import Callable

import flag_gems
import pytest
import torch
from flag_gems import ops

from . import accuracy_utils as utils


def _public_backward(name: str) -> Callable:
    assert name in flag_gems.__all__
    assert name in ops.__all__
    implementation = getattr(flag_gems, name)
    assert callable(implementation)
    return implementation


def _check_backward(backward, args, ref_output, ref_input, grad_output):
    ref_grad_output = utils.to_reference(grad_output)
    (expected,) = torch.autograd.grad(ref_output, ref_input, ref_grad_output)
    actual = backward(grad_output, *args)
    utils.gems_assert_close(actual, expected, ref_input.dtype)


@pytest.mark.parametrize(
    "operator",
    [
        pytest.param("log10", marks=pytest.mark.log10_backward),
        pytest.param("log1p", marks=pytest.mark.log1p_backward),
        pytest.param("log2", marks=pytest.mark.log2_backward),
    ],
)
def test_public_log_backward(operator):
    x = torch.rand((5, 7), device=flag_gems.device).t() + 0.5
    ref_x = utils.to_reference(x).requires_grad_()
    output = getattr(torch, operator)(ref_x)
    grad_output = torch.randn_like(x)
    _check_backward(
        _public_backward(operator + "_backward"), (x,), output, ref_x, grad_output
    )


@pytest.mark.parametrize(
    "operator",
    [
        pytest.param("min", marks=pytest.mark.min_backward),
        pytest.param("max", marks=pytest.mark.max_backward),
        pytest.param("median", marks=pytest.mark.median_backward),
    ],
)
@pytest.mark.parametrize("dim", [None, -1])
def test_public_extrema_backward(operator, dim):
    x = torch.randn((5, 7), device=flag_gems.device)
    ref_x = utils.to_reference(x).requires_grad_()
    forward = getattr(torch, operator)
    if dim is None:
        ref_output = forward(ref_x)
        args = (x, ref_output.detach().to(x.device))
    else:
        ref_output, indices = forward(ref_x, dim=dim, keepdim=True)
        args = (x, ref_output.detach().to(x.device), indices.to(x.device), dim)
    grad_output = torch.randn(ref_output.shape, device=x.device)
    _check_backward(
        _public_backward(operator + "_backward"), args, ref_output, ref_x, grad_output
    )


@pytest.mark.logsumexp_backward
@pytest.mark.parametrize("dim", [-1, (0, 1), ()])
def test_public_logsumexp_backward(dim):
    x = torch.randn((5, 7), device=flag_gems.device)
    ref_x = utils.to_reference(x).requires_grad_()
    # FlagGems treats an empty dimension tuple as an identity copy.
    output = torch.logsumexp(ref_x, dim) if dim else ref_x
    grad_output = torch.randn(output.shape, device=x.device)
    args = (x, output.detach().to(x.device), dim)
    _check_backward(
        _public_backward("logsumexp_backward"), args, output, ref_x, grad_output
    )


@pytest.mark.logcumsumexp_backward
def test_public_logcumsumexp_backward():
    x = torch.randn((5, 7), device=flag_gems.device)
    ref_x = utils.to_reference(x).requires_grad_()
    output = torch.logcumsumexp(ref_x, dim=-1)
    grad_output = torch.randn_like(x)
    args = (x, output.detach().to(x.device), -1)
    _check_backward(
        _public_backward("logcumsumexp_backward"),
        args,
        output,
        ref_x,
        grad_output,
    )


@pytest.mark.parametrize(
    "ndim",
    [
        pytest.param(2, marks=pytest.mark.max_unpool2d_backward),
        pytest.param(3, marks=pytest.mark.max_unpool3d_backward),
    ],
)
def test_public_max_unpool_backward(ndim):
    x = torch.randn((1, 2) + (6,) * ndim, device=flag_gems.device)
    pool = getattr(torch.nn.functional, f"max_pool{ndim}d")
    unpool = getattr(torch.nn.functional, f"max_unpool{ndim}d")
    ref_pooled, ref_indices = pool(utils.to_reference(x), 2, return_indices=True)
    ref_pooled.requires_grad_()
    output = unpool(ref_pooled, ref_indices, 2)
    indices = ref_indices.to(x.device)
    if ndim == 2:
        args = (ref_pooled.detach().to(x.device), indices, output.shape[2:])
    else:
        args = (indices,)
    grad_output = torch.randn(output.shape, device=x.device)
    _check_backward(
        _public_backward(f"max_unpool{ndim}d_backward"),
        args,
        output,
        ref_pooled,
        grad_output,
    )


@pytest.mark.lu_unpack_backward
def test_public_lu_unpack_backward():
    factors = torch.randn((5, 7), device=flag_gems.device)
    pivots = torch.arange(1, 6, device=factors.device, dtype=torch.int32)
    ref_factors = utils.to_reference(factors).requires_grad_()
    _, l_factor, u_factor = torch.lu_unpack(
        ref_factors, utils.to_reference(pivots), unpack_pivots=False
    )
    grad_l = torch.randn(l_factor.shape, device=factors.device)
    grad_u = torch.randn(u_factor.shape, device=factors.device)
    (expected,) = torch.autograd.grad(
        (l_factor, u_factor),
        ref_factors,
        (utils.to_reference(grad_l), utils.to_reference(grad_u)),
    )
    grad = _public_backward("lu_unpack_backward")(grad_l, grad_u, *factors.shape)
    utils.gems_assert_close(grad, expected, factors.dtype)
