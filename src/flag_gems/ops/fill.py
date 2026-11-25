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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")], num_outputs=1
)
@triton.jit
def fill_scalar_func(inp, value_scalar):
    return tl.full(inp.shape, value_scalar, dtype=inp.dtype)


@pointwise_dynamic(
    is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")], num_outputs=1
)
@triton.jit
def fill_tensor_func(inp, value):
    return value


def _fill_scalar_(input, value):
    with torch_device_fn.device(input.device):
        fill_scalar_func(input, value, out0=input)
    return input


def _fill_tensor_(input, value):
    if not value.is_cuda:
        return _fill_scalar_(input, value.item())
    with torch_device_fn.device(input.device):
        fill_tensor_func(input, value, out0=input)
    return input


def _fill_scalar(input, value):
    out = torch.empty_like(input)
    with torch_device_fn.device(input.device):
        return fill_scalar_func(input, value, out0=out)


def _fill_tensor(input, value):
    if not value.is_cuda:
        return _fill_scalar(input, value.item())
    out = torch.empty_like(input)
    with torch_device_fn.device(input.device):
        return fill_tensor_func(input, value, out0=out)


class _GradSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        from flag_gems.ops.sum import sum as gems_sum

        ctx.shape = input.shape
        ctx.dtype = input.dtype
        ctx.device = input.device
        return gems_sum(input)

    @staticmethod
    def backward(ctx, grad_output):
        empty = torch.empty(ctx.shape, dtype=ctx.dtype, device=ctx.device)
        return _FillTensor.apply(empty, grad_output)


def _value_gradient(grad_output, value_device):
    from flag_gems.ops.sum import sum as gems_sum

    grad_value = (
        _GradSum.apply(grad_output)
        if torch.is_grad_enabled()
        else gems_sum(grad_output)
    )
    if grad_value.device != value_device:
        grad_value = grad_value.to(value_device)
    return grad_value


class _FillScalarInplace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, value):
        ctx.mark_dirty(input)
        return _fill_scalar_(input, value)

    @staticmethod
    def backward(ctx, grad_output):
        return torch.zeros_like(grad_output), None


class _FillTensorInplace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, value):
        ctx.value_device = value.device
        ctx.mark_dirty(input)
        return _fill_tensor_(input, value)

    @staticmethod
    def backward(ctx, grad_output):
        return torch.zeros_like(grad_output), _value_gradient(
            grad_output, ctx.value_device
        )


class _FillScalar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, value):
        return _fill_scalar(input, value)

    @staticmethod
    def backward(ctx, grad_output):
        return torch.zeros_like(grad_output), None


class _FillTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, value):
        ctx.value_device = value.device
        return _fill_tensor(input, value)

    @staticmethod
    def backward(ctx, grad_output):
        return torch.zeros_like(grad_output), _value_gradient(
            grad_output, ctx.value_device
        )


def fill_scalar(input, value):
    logger.debug("GEMS FILL (Dynamic)")
    if torch.is_grad_enabled() and input.requires_grad:
        return _FillScalar.apply(input, value)
    return _fill_scalar(input, value)


def fill_scalar_out(input, value, *, out=None):
    logger.debug("GEMS FILL_SCALAR_OUT")
    if out is None:
        return fill_scalar(input, value)
    with torch_device_fn.device(input.device):
        fill_scalar_func(input, value, out0=out)
    return out


def fill_tensor(input, value):
    logger.debug("GEMS FILL (Dynamic)")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    if torch.is_grad_enabled() and (input.requires_grad or value.requires_grad):
        return _FillTensor.apply(input, value)
    return _fill_tensor(input, value)


def fill_tensor_out(input, value, *, out=None):
    logger.debug("GEMS FILL_TENSOR_OUT")
    if out is None:
        return fill_tensor(input, value)
    if not value.is_cuda:
        return fill_scalar_out(input, value.item(), out=out)
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    with torch_device_fn.device(input.device):
        fill_tensor_func(input, value, out0=out)
    return out


def fill_tensor_(self, value):
    logger.debug("GEMS FILL_TENSOR_")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    if torch.is_grad_enabled() and (self.requires_grad or value.requires_grad):
        return _FillTensorInplace.apply(self, value)
    _fill_tensor_(self, value)
    torch.autograd.graph.increment_version(self)
    return self


def fill_scalar_(self, value):
    logger.debug("GEMS FILL_SCALAR_")
    if torch.is_grad_enabled() and self.requires_grad:
        return _FillScalarInplace.apply(self, value)
    _fill_scalar_(self, value)
    torch.autograd.graph.increment_version(self)
    return self
