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


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def log10_func(x):
    return tl.log(x.to(tl.float32)) * 0.4342944819032518


@triton.jit
def _log10_grad(grad_output, x):
    # ATen rounds the denominator to the input dtype before division.
    value = x if x.dtype == tl.float64 else x.to(tl.float32)
    denominator = (value * 2.302585092994046).to(x.dtype)
    return grad_output / denominator


_log10_backward_strided = pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])(
    _log10_grad
)


@triton.jit
def _log10_backward_kernel(
    grad_output, x, grad_input, n, BLOCK: tl.constexpr, LARGE: tl.constexpr
):
    pid = tl.program_id(0)
    if LARGE:
        pid = pid.to(tl.int64)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    grad = tl.load(grad_output + offsets, mask, other=0)
    value = tl.load(x + offsets, mask, other=1)
    tl.store(grad_input + offsets, _log10_grad(grad, value), mask)


class Log10Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return log10_func(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        if not (x.is_contiguous() and grad_output.is_contiguous()):
            return _log10_backward_strided(grad_output, x)
        grad_input = torch.empty_like(x)
        n = x.numel()
        if n:
            block_size = 512
            with torch_device_fn.device(x.device):
                _log10_backward_kernel[(triton.cdiv(n, block_size),)](
                    grad_output,
                    x,
                    grad_input,
                    n,
                    BLOCK=block_size,
                    LARGE=n > 2147483647,
                )
        return grad_input


def log10(A):
    logger.debug("GEMS LOG10")
    if torch.is_grad_enabled() and A.requires_grad:
        return Log10Function.apply(A)
    return log10_func(A)


def log10_(A):
    logger.debug("GEMS LOG10_")
    return log10_func(A, out0=A)


def log10_out(A, out):
    logger.debug("GEMS LOG10_OUT")
    return log10_func(A, out0=out)
