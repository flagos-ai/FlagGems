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

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def lerp_tensor_kernel(input, end, weight):
    return tl.where(
        tl.abs(weight) < 0.5,
        input + weight * (end - input),
        end - (end - input) * (1 - weight),
    )


@pointwise_dynamic(
    is_tensor=[True, True, False],
    dtypes=[None, None, float],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit(do_not_specialize=["weight"])
def lerp_scalar_kernel_head(input, end, weight):
    return input + weight * (end - input)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    dtypes=[None, None, float],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit(do_not_specialize=["weight"])
def lerp_scalar_kernel_tail(input, end, weight):
    return end - (end - input) * (1 - weight)


def lerp_backward(grad_output, input, end, weight):
    """Backward of ``lerp``.

    ``d/input = grad * (1 - weight)``, ``d/end = grad * weight`` and, for a
    tensor weight, ``d/weight = grad * (end - input)``. A scalar weight has no
    gradient.
    """
    logger.debug("GEMS LERP_BACKWARD")
    grad_input = grad_output * (1 - weight)
    grad_end = grad_output * weight
    if torch.is_tensor(weight):
        grad_weight = grad_output * (end - input)
        return grad_input, grad_end, grad_weight
    return grad_input, grad_end, None


class LerpTensorOpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, end, weight):
        ctx.save_for_backward(input, end, weight)
        return _lerp_tensor_forward(input, end, weight)

    @staticmethod
    def backward(ctx, grad_output):
        input, end, weight = ctx.saved_tensors
        return lerp_backward(grad_output, input, end, weight)


class LerpScalarOpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, end, weight):
        ctx.weight = weight
        ctx.save_for_backward(input, end)
        return _lerp_scalar_forward(input, end, weight)

    @staticmethod
    def backward(ctx, grad_output):
        input, end = ctx.saved_tensors
        grad_input, grad_end, _ = lerp_backward(grad_output, input, end, ctx.weight)
        return grad_input, grad_end, None


def lerp_tensor(input, end, weight):
    logger.debug("GEMS LERP_TENSOR")
    return LerpTensorOpFunction.apply(input, end, weight)


def lerp_tensor_(input, end, weight):
    logger.debug("GEMS LERP_INPLACE_TENSOR")
    return lerp_tensor_kernel(input, end, weight, out0=input)


def lerp_scalar(input, end, weight):
    logger.debug("GEMS LERP_TENSOR")
    return LerpScalarOpFunction.apply(input, end, weight)


def _lerp_tensor_forward(input, end, weight):
    return lerp_tensor_kernel(input, end, weight)


def _lerp_scalar_forward(input, end, weight):
    if weight < 0.5:
        return lerp_scalar_kernel_head(input, end, weight)
    return lerp_scalar_kernel_tail(input, end, weight)


def lerp_scalar_(input, end, weight):
    logger.debug("GEMS LERP_INPLACE_TENSOR")
    if weight < 0.5:
        return lerp_scalar_kernel_head(input, end, weight, out0=input)
    else:
        return lerp_scalar_kernel_tail(input, end, weight, out0=input)
