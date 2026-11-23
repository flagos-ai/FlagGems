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
from flag_gems.utils.triton_lang_extension import fmod as _fmod
from flag_gems.utils.triton_lang_extension import trunc as _trunc

from .sum import sum as gems_sum
from .sum import sum_dim

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def fmod_func(x, y):
    # Convert to float32 for computation to avoid libdevice float16/bfloat16 issues
    dtype = x.dtype
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    result = _fmod(x_fp32, y_fp32)
    return result.to(dtype)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def fmod_func_tensor_scalar(x, y):
    # Convert to float32 for computation to avoid libdevice float16/bfloat16 issues
    dtype = x.dtype
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    result = _fmod(x_fp32, y_fp32)
    return result.to(dtype)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def fmod_other_grad(grad, x, y):
    # ATen rounds division to the forward result dtype before truncating.
    quotient = _trunc(
        (x.to(tl.float32) / y.to(tl.float32)).to(grad.dtype).to(tl.float32)
    )
    return -grad.to(tl.float32) * quotient


def _sum_to_shape(value, shape):
    # An empty broadcast can originate from a size-one input; reducing its
    # empty gradient must produce initialized zeros for that input.
    if value.numel() == 0 and tuple(value.shape) != tuple(shape):
        return torch.zeros(shape, dtype=value.dtype, device=value.device)
    # Reduce leading broadcast dimensions first, then retained singleton axes.
    while value.ndim > len(shape):
        if value.ndim == 1:
            value = gems_sum(value)
        else:
            value = sum_dim(value, dim=[0], keepdim=False)
    for axis, size in enumerate(shape):
        if size == 1 and value.shape[axis] != 1:
            value = sum_dim(value, dim=[axis], keepdim=True)
    return value


class _FmodTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return fmod_func(x, y)

    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.saved_tensors
        grad_x = _sum_to_shape(grad, x.shape) if ctx.needs_input_grad[0] else None
        grad_y = None
        if ctx.needs_input_grad[1]:
            if grad.numel() == 0:
                grad_y = torch.zeros_like(y)
            else:
                grad_y = _sum_to_shape(fmod_other_grad(grad, x, y), y.shape)
                if grad_y.dtype != y.dtype:
                    grad_y = grad_y.to(y.dtype)
        return grad_x, grad_y


class _FmodScalar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        return fmod_func_tensor_scalar(x, y)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def fmod_tensor(A, B):
    logger.debug("GEMS FMOD_TENSOR")
    if torch.is_grad_enabled() and (A.requires_grad or B.requires_grad):
        return _FmodTensor.apply(A, B)
    return fmod_func(A, B)


def fmod_scalar(A, B):
    logger.debug("GEMS FMOD_SCALAR")
    if torch.is_grad_enabled() and A.requires_grad:
        return _FmodScalar.apply(A, B)
    return fmod_func_tensor_scalar(A, B)


def fmod_tensor_(A, B):
    logger.debug("GEMS FMOD_TENSOR_")
    return fmod_func(A, B, out0=A)


def fmod_scalar_(A, B):
    logger.debug("GEMS FMOD_SCALAR_")
    return fmod_func_tensor_scalar(A, B, out0=A)
